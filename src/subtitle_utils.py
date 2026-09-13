"""
Subtitle parsing and quote -> timestamp matching.

This is where the project's core invariant is enforced. A quote is text we were
told to look for; the timestamp it resolves to comes from the subtitle file and
nowhere else. Nothing here asks a model when something happened.

The matching has to be fuzzy because hand-written quotes never align to cue
boundaries and rarely match word-for-word: subtitles compress dialogue, split
lines mid-sentence, and carry markup. So candidates are compared against sliding
windows of consecutive cues rather than single cues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pysrt
from rapidfuzz import fuzz

from src.subtitle_source import clean_cue_text


@dataclass(frozen=True)
class Cue:
    """One subtitle line, times in seconds from the start of the media."""

    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Match:
    """
    A quote located in the transcript.

    `score` is reported even when it falls below the threshold, so near-misses
    can be logged. A run that produces nothing should be able to say "closest
    was 68 on this line" rather than only "no match".
    """

    quote: str
    start: float
    end: float
    score: float
    cue_span: tuple[int, int]
    matched_text: str

    def clears(self, threshold: float) -> bool:
        return self.score >= threshold


_SPEAKER = re.compile(r"^\s*[-–—]?\s*[A-Z][A-Z .'`-]{1,20}:\s*")
_SOUND_EFFECT = re.compile(r"[\[(][^\])]*[\])]")
_PUNCTUATION = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """
    Reduce a line to comparable words.

    Strips the things subtitles add that dialogue doesn't have: markup,
    bracketed sound effects ("[DOOR SLAMS]"), and speaker labels ("DEWEY:").
    Leaving those in drags the score down on otherwise perfect matches.
    """
    text = clean_cue_text(text)
    text = _SOUND_EFFECT.sub(" ", text)
    text = _SPEAKER.sub("", text)
    text = _PUNCTUATION.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def load_cues(srt_path: Path) -> list[Cue]:
    """
    Parse an .srt into cues, dropping any that carry no dialogue.

    Uses pysrt.ERROR_PASS — note that ERROR_IGNORE does not exist and raises
    AttributeError at parse time.
    """
    try:
        subs = pysrt.open(
            str(srt_path), encoding="utf-8", error_handling=pysrt.ERROR_PASS
        )
    except UnicodeDecodeError:
        subs = pysrt.open(
            str(srt_path), encoding="latin-1", error_handling=pysrt.ERROR_PASS
        )

    cues: list[Cue] = []
    for item in subs:
        text = clean_cue_text(item.text)
        if not text:
            continue
        cues.append(
            Cue(
                index=len(cues),
                start=item.start.ordinal / 1000.0,
                end=item.end.ordinal / 1000.0,
                text=text,
            )
        )
    return cues


def best_match(
    cues: list[Cue], quote: str, *, window_max_cues: int = 3
) -> Match | None:
    """
    Find where `quote` occurs in `cues`.

    Every window of 1..window_max_cues consecutive cues is scored with
    token_sort_ratio, which ignores word order and so survives the reordering
    that happens when a line is split across cues.

    Returns the best window regardless of score — the caller applies the
    threshold, so a near-miss can be reported rather than silently discarded.
    Returns None only when there is nothing to search.
    """
    target = normalize(quote)
    if not target or not cues:
        return None

    best: Match | None = None

    for size in range(1, window_max_cues + 1):
        for start_index in range(len(cues) - size + 1):
            window = cues[start_index : start_index + size]
            joined = normalize(" ".join(cue.text for cue in window))
            if not joined:
                continue

            score = fuzz.token_sort_ratio(target, joined)

            # Strictly greater, so that on a tie the SMALLER window wins — it
            # was found first. A tighter window means a tighter clip, without
            # neighbouring dialogue dragged in.
            if best is None or score > best.score:
                best = Match(
                    quote=quote,
                    start=window[0].start,
                    end=window[-1].end,
                    score=float(score),
                    cue_span=(window[0].index, window[-1].index),
                    matched_text=" ".join(cue.text for cue in window),
                )

    return best


def find_matches(
    cues: list[Cue],
    quotes: list[str],
    *,
    threshold: float,
    window_max_cues: int = 3,
) -> tuple[list[Match], list[Match]]:
    """
    Match a batch of quotes, splitting results into (accepted, rejected).

    Rejected matches are returned rather than dropped so callers can print
    "closest was 68/100 on <line>" — the single most useful thing to know when
    a title produces nothing.
    """
    accepted: list[Match] = []
    rejected: list[Match] = []

    for quote in quotes:
        match = best_match(cues, quote, window_max_cues=window_max_cues)
        if match is None:
            continue
        (accepted if match.clears(threshold) else rejected).append(match)

    return accepted, rejected


def cues_in_window(cues: list[Cue], start: float, end: float) -> list[Cue]:
    """
    Every cue overlapping [start, end], shifted to be relative to `start`.

    The shift is what makes burned-in captions line up: cue times are in
    source-file time, and after cutting, the clip's timeline begins at zero.
    Skipping this puts every caption at the wrong moment, or past the end of the
    clip entirely.

    Cues that straddle an edge are clamped rather than dropped, so a line that
    starts just before the cut is still readable.
    """
    shifted: list[Cue] = []

    for cue in cues:
        if cue.end <= start or cue.start >= end:
            continue
        shifted.append(
            Cue(
                index=len(shifted),
                start=max(0.0, cue.start - start),
                end=min(end - start, cue.end - start),
                text=cue.text,
            )
        )

    return shifted
