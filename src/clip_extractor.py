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
    match: Match,
    media_duration: float | None,
    config: Config,
    cues: list | None = None,
) -> ClipWindow:
    """
    Turn a matched quote into a clip window.

    Two modes, set by `clip.anchor`:

    **punchline** (default) — the matched line lands near the END, with the
    preceding dialogue running into it. This is the one that makes clips make
    sense. A quote is almost always the line that LANDS, not the line that sets
    it up, so a clip should play setup then payoff and stop. Padding the match
    symmetrically instead puts the punchline in the middle and then runs on
    into whatever unrelated conversation follows, which is how you get a clip
    that is technically correct and completely pointless.

    **center** — the old behaviour, kept for quotes that are the beginning of
    something rather than the end of it.

    When `cues` are supplied the start is snapped to a gap in the dialogue, so
    the clip opens on a natural beat rather than halfway through a sentence.
    """
    limits = config.clip
    upper = media_duration if media_duration else match.end + limits.pad_after

    if limits.anchor == "punchline":
        return _plan_punchline(match, upper, config, cues)

    start = match.start - limits.pad_before
    end = match.end + limits.pad_after

    start, end = _clamp(start, end, 0.0, upper)

    if end - start < limits.min_duration:
        start, end = _expand(start, end, limits.min_duration, 0.0, upper)

    if end - start > limits.max_duration:
        start, end = _shrink(start, end, match, limits.max_duration)

    return ClipWindow(start=round(start, 3), end=round(end, 3))


def _plan_punchline(
    match: Match, upper: float, config: Config, cues: list | None
) -> ClipWindow:
    """
    End just after the line, and take the setup from before it.

    The end is fixed first — the quote plus a short beat to let it land — and
    the start is then whatever gives a legal duration, preferring a point where
    the dialogue actually pauses.
    """
    limits = config.clip
    match_length = match.end - match.start

    # A "quote" longer than the maximum clip has no room for setup. Keep the
    # BEGINNING of the line: opening mid-sentence is worse than ending early,
    # and the start is the part that makes sense without context.
    if match_length >= limits.max_duration:
        start = match.start
        return ClipWindow(
            start=round(start, 3),
            end=round(min(upper, start + limits.max_duration), 3),
        )

    # Media shorter than the minimum clip: give back all of it. Failing here
    # would reject short content for a reason nobody can act on.
    if upper <= limits.min_duration:
        return ClipWindow(start=0.0, end=round(upper, 3))

    end = min(upper, match.end + limits.pad_after)

    # The legal range for a start, working backwards from a fixed end.
    earliest = max(0.0, end - limits.max_duration)
    latest = max(0.0, end - limits.min_duration)
    # A clip must contain its own quote...
    latest = min(latest, match.start)
    # ...and must never be longer than the maximum.
    latest = max(latest, earliest)

    start = _snap_to_pause(cues, earliest, latest, limits.boundary_gap)
    if start is None:
        start = latest

    # Not enough room before the line — near the start of the file, say. Take
    # the remainder from after it rather than shipping an under-length clip.
    if end - start < limits.min_duration:
        end = min(upper, start + limits.min_duration)

    return ClipWindow(start=round(start, 3), end=round(end, 3))


def _snap_to_pause(
    cues: list | None, earliest: float, latest: float, min_gap: float
) -> float | None:
    """
    Find the best place to start inside [earliest, latest].

    Candidates are cue starts preceded by a pause. A gap in the dialogue is the
    cheapest available proxy for a scene change or a beat, and it costs nothing
    because the cues are already parsed.

    The LATEST qualifying pause wins, not the longest. The aim is the tightest
    clip that still opens cleanly — reaching further back for a bigger pause
    just buys dead air, and the longest gap in range is often an artifact
    anyway (the run-up to the first line of an episode measures as an enormous
    one). If more setup is wanted, that's what `min_duration` is for.

    Returns None when there's nothing to snap to, leaving the caller to fall
    back to a plain duration-based start.
    """
    if not cues:
        return None

    chosen: float | None = None
    previous_end = 0.0

    for index, cue in enumerate(cues):
        gap = cue.start - previous_end
        previous_end = max(previous_end, cue.end)

        # The first cue has no real predecessor; the "gap" in front of it is
        # measured from zero and means nothing.
        if index == 0:
            continue
        if cue.start < earliest or cue.start > latest:
            continue
        if gap >= min_gap:
            chosen = cue.start

    return chosen


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
