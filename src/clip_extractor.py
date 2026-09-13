"""
Clip window planning and ffmpeg input construction.

Two responsibilities, deliberately separated from rendering:

* `plan_window` decides WHICH seconds of the source become a short. It is a pure
  function, which matters because the duration clamping has more edge cases than
  it looks like it does — matches near the start of a file, matches longer than
  the maximum clip length, and media shorter than the minimum.
* `build_input_args` produces the seek arguments shared by every ffmpeg call, so
  the "-ss before -i" ordering lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Config
from src.subtitle_utils import Match


@dataclass(frozen=True)
class ClipWindow:
    """A span of source media, in seconds from the start of the file."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: "ClipWindow") -> bool:
        return self.start < other.end and other.start < self.end


def plan_window(
    match: Match, media_duration: float | None, config: Config
) -> ClipWindow:
    """
    Turn a matched quote into a clip window.

    Padding first, then clamping to the configured duration range. The match
    itself is kept inside the window wherever possible — a clip that cuts away
    before the line finishes is worse than one that runs slightly long.

    When the media is shorter than `min_duration` the whole file is returned:
    there is nothing else to give, and failing here would reject short content
    for a reason the user can't act on.
    """
    limits = config.clip
    upper = media_duration if media_duration else match.end + limits.pad_after

    start = match.start - limits.pad_before
    end = match.end + limits.pad_after

    start, end = _clamp(start, end, 0.0, upper)

    if end - start < limits.min_duration:
        start, end = _expand(start, end, limits.min_duration, 0.0, upper)

    if end - start > limits.max_duration:
        start, end = _shrink(start, end, match, limits.max_duration)

    return ClipWindow(start=round(start, 3), end=round(end, 3))


def _clamp(start: float, end: float, lower: float, upper: float) -> tuple[float, float]:
    return max(lower, start), min(upper, end)


def _expand(
    start: float, end: float, target: float, lower: float, upper: float
) -> tuple[float, float]:
    """
    Grow a window to `target` seconds, centred where possible.

    Whatever one side can't absorb (because it hit the start or end of the
    file) is handed to the other side, so a match 2 seconds into a film still
    produces a full-length clip.
    """
    if upper - lower <= target:
        return lower, upper

    needed = target - (end - start)
    half = needed / 2.0

    new_start = start - half
    new_end = end + half

    if new_start < lower:
        new_end += lower - new_start
        new_start = lower
    if new_end > upper:
        new_start -= new_end - upper
        new_end = upper

    return max(lower, new_start), min(upper, new_end)


def _shrink(
    start: float, end: float, match: Match, target: float
) -> tuple[float, float]:
    """
    Trim a window to `target` seconds without cutting the line off.

    The matched span is what the clip exists for, so it is preserved and the
    padding absorbs the trim. When the match itself is longer than the maximum
    clip length there is no way to keep all of it — it is truncated from the
    end, keeping the beginning of the line, which is the part that makes sense
    without context.
    """
    match_length = match.end - match.start

    if match_length >= target:
        return match.start, match.start + target

    # Centre the matched span in the allowed length.
    slack = target - match_length
    new_start = match.start - slack / 2.0
    new_end = new_start + target

    if new_start < start:
        new_start, new_end = start, start + target
    if new_end > end:
        new_end, new_start = end, end - target

    return new_start, new_end


def build_input_args(source, window: ClipWindow) -> list[str]:
    """
    The seek arguments for a clip.

    `-ss` goes BEFORE `-i` so ffmpeg seeks by keyframe before decoding, which is
    the difference between instant and reading the whole file. `-t` after the
    input bounds the duration.

    Deliberately no `-c copy` anywhere: stream copying cannot cut on an exact
    frame, so it drifts audio and leaves captions out of sync with the picture.
    Everything is re-encoded.
    """
    return [
        "-ss", f"{window.start:.3f}",
        "-i", str(source),
        "-t", f"{window.duration:.3f}",
    ]


def encoder_args(config: Config) -> list[str]:
    """
    Video encoder settings.

    NVENC is materially faster on the GTX 1080, but note it only accelerates the
    ENCODE — the blur and caption filters are CPU-only regardless, since gblur
    has no CUDA equivalent and libass renders on the CPU.
    """
    video = config.video

    if video.encoder == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-preset", video.nvenc_preset,
            "-rc", "vbr",
            "-cq", str(video.nvenc_cq),
            "-b:v", "0",
        ]

    return [
        "-c:v", "libx264",
        "-preset", video.preset,
        "-crf", str(video.crf),
    ]
