"""
Synthetic media, built by ffmpeg at test time.

There is no real library on this machine, so every fixture is manufactured:
`testsrc` video, a sine tone, and subtitle tracks written here with KNOWN text
at KNOWN timestamps. That makes timing assertions exact rather than approximate,
and lets each failure mode be reproduced deliberately instead of hoped for.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Kept small and low frame-rate: these exist to carry subtitle tracks and to be
# cut, not to be looked at. Full-size fixtures would make the suite slow for no
# extra coverage.
WIDTH, HEIGHT, FPS = 640, 360, 10
DURATION = 120

# The line every timing assertion is anchored to.
KNOWN_QUOTE = "I am the smartest man alive"
KNOWN_QUOTE_START = 42.0
KNOWN_QUOTE_END = 45.0

# Lines used to build a multipart series: spread across the runtime so the
# separation and ordering rules have something real to work on.
SERIES_QUOTES = [
    ("This is where it begins, right here on this street", 8.0, 11.0),
    ("You never told me it would cost this much", 33.0, 36.0),
    (KNOWN_QUOTE, KNOWN_QUOTE_START, KNOWN_QUOTE_END),
    ("Everything we built is gone and nobody noticed", 72.0, 75.0),
    ("So that is how the whole thing finally ended", 108.0, 111.0),
]

_FILLER = [
    "We should probably talk about what happened last night",
    "There is no version of this where everyone walks away happy",
    "He said the same thing to me about the money",
    "Nobody is coming to help us with any of this",
    "I told you that road was closed after the storm",
    "She kept the letters in a box under the floor",
    "That was the last time anyone saw him downtown",
    "Put it back exactly where you found it please",
]


def _timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(path: Path, cues: list[tuple[str, float, float]]) -> Path:
    blocks = []
    for index, (text, start, end) in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n{_timestamp(start)} --> {_timestamp(end)}\n{text}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def full_transcript_cues() -> list[tuple[str, float, float]]:
    """~30 cues spanning the whole runtime — a healthy, usable track."""
    cues = list(SERIES_QUOTES)
    slot = 3.0
    for index, line in enumerate(_FILLER * 3):
        start = 14.0 + index * 4.0
        end = start + slot
        if end > DURATION - 4:
            break
        # Don't collide with the anchored quotes.
        if any(abs(start - qs) < 4.0 for _, qs, _ in SERIES_QUOTES):
            continue
        cues.append((line, start, end))
    return sorted(cues, key=lambda cue: cue[1])


def forced_track_cues() -> list[tuple[str, float, float]]:
    """
    Plenty of cues and plenty of text, but all crammed into the first third.

    This is what a forced track looks like: it passes the cue-count and
    character-count gates and only the coverage gate catches it.
    """
    cues = []
    for index, line in enumerate(_FILLER * 4):
        start = 1.0 + index * 1.3
        end = start + 1.2
        if start > DURATION * 0.35:
            break
        cues.append((line, start, end))
    return cues


def sparse_track_cues() -> list[tuple[str, float, float]]:
    """Four cues. Should be rejected on count, before coverage is considered."""
    return [
        ("Uno momento por favor", 5.0, 7.0),
        ("Nos vemos manana", 40.0, 42.0),
        ("Esta bien", 80.0, 82.0),
        ("Adios amigo", 115.0, 117.0),
    ]


def _base_input() -> list[str]:
    return [
        "-f", "lavfi",
        "-i", f"testsrc=duration={DURATION}:size={WIDTH}x{HEIGHT}:rate={FPS}",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={DURATION}",
    ]


# NOTE: no "-shortest" anywhere below. It counts the SUBTITLE stream as an
# input, so muxing a track that ends at 0:42 truncates a 2-minute video to 42
# seconds — which silently destroys the coverage fixture, since a short track in
# a correspondingly short file looks like full coverage. The lavfi inputs carry
# explicit durations, so it was never needed.


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "fixture ffmpeg failed:\n"
            + " ".join(command)
            + "\n"
            + (result.stderr or "")[-2000:]
        )


def build_with_text_subs(directory: Path) -> Path:
    """Healthy English subrip track."""
    out = directory / "with_text_subs.mkv"
    if out.exists():
        return out
    srt = write_srt(directory / "_full.srt", full_transcript_cues())
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *_base_input(), "-i", str(srt),
        "-map", "0:v", "-map", "1:a", "-map", "2:s",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-c:s", "srt",
        "-metadata:s:s:0", "language=eng",
        str(out),
    ])
    return out


def build_no_subs(directory: Path) -> Path:
    """No subtitle stream at all."""
    out = directory / "no_subs.mkv"
    if out.exists():
        return out
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *_base_input(),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(out),
    ])
    return out


def build_forced_subs(directory: Path) -> Path:
    """A forced-disposition track that stops a third of the way in."""
    out = directory / "forced_subs.mkv"
    if out.exists():
        return out
    srt = write_srt(directory / "_forced.srt", forced_track_cues())
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *_base_input(), "-i", str(srt),
        "-map", "0:v", "-map", "1:a", "-map", "2:s",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-c:s", "srt",
        "-metadata:s:s:0", "language=eng",
        "-disposition:s:0", "forced",
        str(out),
    ])
    return out


def build_multi_lang(directory: Path) -> Path:
    """
    Spanish first, English second.

    Deliberately ordered so that a hardcoded 0:s:0 would pick the WRONG track —
    the whole point of language-aware selection.
    """
    out = directory / "multi_lang.mkv"
    if out.exists():
        return out
    spa = write_srt(directory / "_spa.srt", sparse_track_cues())
    eng = write_srt(directory / "_eng.srt", full_transcript_cues())
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *_base_input(), "-i", str(spa), "-i", str(eng),
        "-map", "0:v", "-map", "1:a", "-map", "2:s", "-map", "3:s",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-c:s", "srt",
        "-metadata:s:s:0", "language=spa",
        "-metadata:s:s:1", "language=eng",
        str(out),
    ])
    return out


def build_sidecar_movie(directory: Path) -> Path:
    """No embedded subs, but a .en.srt sitting beside the file."""
    out = directory / "sidecar_movie.mkv"
    if out.exists():
        return out
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *_base_input(),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(out),
    ])
    write_srt(directory / "sidecar_movie.en.srt", full_transcript_cues())
    return out


def build_all(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "with_text_subs": build_with_text_subs(directory),
        "no_subs": build_no_subs(directory),
        "forced_subs": build_forced_subs(directory),
        "multi_lang": build_multi_lang(directory),
        "sidecar_movie": build_sidecar_movie(directory),
    }
