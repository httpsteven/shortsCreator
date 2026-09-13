"""
Quote -> timestamp matching.

The central assertion of the whole project is here: a quote resolves to the
timestamp the subtitle file says it occurs at. Everything downstream trusts
that number, so it is asserted exactly rather than approximately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.subtitle_utils import (
    best_match,
    cues_in_window,
    find_matches,
    load_cues,
    normalize,
)
from tests.fixtures import (
    KNOWN_QUOTE,
    KNOWN_QUOTE_END,
    KNOWN_QUOTE_START,
    full_transcript_cues,
    write_srt,
)

THRESHOLD = 72.0


@pytest.fixture
def cues(tmp_path: Path):
    return load_cues(write_srt(tmp_path / "t.srt", full_transcript_cues()))


# --------------------------------------------------------------------------
# The core guarantee
# --------------------------------------------------------------------------


def test_known_quote_resolves_to_known_timestamp(cues) -> None:
    match = best_match(cues, KNOWN_QUOTE)

    assert match is not None
    assert match.score == pytest.approx(100, abs=1)
    assert match.start == pytest.approx(KNOWN_QUOTE_START, abs=0.05)
    assert match.end == pytest.approx(KNOWN_QUOTE_END, abs=0.05)


def test_paraphrased_quote_still_matches(cues) -> None:
    """
    Hand-written quotes are written from memory and rarely match word-for-word.
    Punctuation, capitalisation and a missing article must not break the match.
    """
    match = best_match(cues, "i'm the smartest man alive!")

    assert match.clears(THRESHOLD)
    assert match.start == pytest.approx(KNOWN_QUOTE_START, abs=0.05)


def test_unrelated_quote_does_not_clear_threshold(cues) -> None:
    match = best_match(cues, "the mitochondria is the powerhouse of the cell")

    assert match is not None  # a best window always exists...
    assert not match.clears(THRESHOLD)  # ...but it must not be accepted


def test_quote_split_across_two_cues(tmp_path: Path) -> None:
    """
    Subtitles break sentences mid-thought. A quote written as one sentence has
    to match the pair, and the window must span BOTH cues so the clip contains
    the whole line.
    """
    cues = load_cues(
        write_srt(
            tmp_path / "split.srt",
            [
                ("Filler line to pad the transcript out", 1.0, 3.0),
                ("You never told me", 20.0, 21.5),
                ("it would cost this much", 21.5, 23.0),
                ("Another unrelated line here", 30.0, 32.0),
            ],
        )
    )

    match = best_match(cues, "You never told me it would cost this much")

    assert match.clears(THRESHOLD)
    assert match.start == pytest.approx(20.0, abs=0.05)
    assert match.end == pytest.approx(23.0, abs=0.05)
    assert match.cue_span == (1, 2)


def test_smaller_window_preferred_on_tie(tmp_path: Path) -> None:
    """A tighter window means a tighter clip, with no neighbouring dialogue."""
    cues = load_cues(
        write_srt(
            tmp_path / "tie.srt",
            [
                ("Something completely different", 1.0, 3.0),
                ("This is the exact line", 10.0, 12.0),
                ("", 12.5, 13.0),
            ],
        )
    )

    match = best_match(cues, "This is the exact line")
    assert match.cue_span == (1, 1)


def test_window_size_is_respected(tmp_path: Path) -> None:
    cues = load_cues(
        write_srt(
            tmp_path / "long.srt",
            [(f"fragment number {i} of the sentence", i * 2.0, i * 2.0 + 1.8)
             for i in range(1, 8)],
        )
    )

    match = best_match(cues, " ".join(f"fragment number {i} of the sentence"
                                      for i in range(1, 8)),
                       window_max_cues=2)

    # Only 2 cues may be joined, so the span can never exceed that.
    assert match.cue_span[1] - match.cue_span[0] <= 1


# --------------------------------------------------------------------------
# Batch matching and near-miss reporting
# --------------------------------------------------------------------------


def test_rejected_matches_are_returned_not_discarded(cues) -> None:
    """
    "Produced nothing" is undiagnosable; "closest was 41 on this line" is not.
    """
    accepted, rejected = find_matches(
        cues,
        [KNOWN_QUOTE, "a line that appears nowhere in this transcript"],
        threshold=THRESHOLD,
    )

    assert len(accepted) == 1
    assert len(rejected) == 1
    assert rejected[0].score < THRESHOLD
    assert rejected[0].matched_text  # the closest line is named


# --------------------------------------------------------------------------
# Caption timing
# --------------------------------------------------------------------------


def test_cues_are_shifted_relative_to_clip_start(tmp_path: Path) -> None:
    """
    Cue times are in source-file time. After cutting, the clip starts at zero —
    so every cue must be rebased, or the captions appear at the wrong moment
    (or past the end of the clip, i.e. not at all).
    """
    cues = load_cues(
        write_srt(
            tmp_path / "shift.srt",
            [
                ("Before the clip", 10.0, 12.0),
                ("First line in clip", 41.0, 43.0),
                ("Second line in clip", 44.0, 46.0),
                ("After the clip", 80.0, 82.0),
            ],
        )
    )

    window = cues_in_window(cues, start=40.0, end=50.0)

    assert [c.text for c in window] == ["First line in clip", "Second line in clip"]
    assert window[0].start == pytest.approx(1.0)
    assert window[0].end == pytest.approx(3.0)
    assert window[1].start == pytest.approx(4.0)


def test_cues_straddling_the_edge_are_clamped_not_dropped(tmp_path: Path) -> None:
    """A line starting just before the cut should still be readable."""
    cues = load_cues(
        write_srt(tmp_path / "edge.srt", [("Straddles the start", 38.0, 42.0)])
    )

    window = cues_in_window(cues, start=40.0, end=50.0)

    assert len(window) == 1
    assert window[0].start == 0.0            # clamped, not negative
    assert window[0].end == pytest.approx(2.0)


def test_normalize_strips_subtitle_artifacts() -> None:
    assert normalize("<i>DEWEY:</i> [DOOR SLAMS] I am the king!") == "i am the king"


# --------------------------------------------------------------------------
# Structure: the theme song, and therefore the cold open
# --------------------------------------------------------------------------


def music(start: float, text: str) -> tuple[float, str]:
    return (start, text)


def build(rows: list[tuple[float, str]]):
    from src.subtitle_utils import Cue

    return [Cue(i, s, s + 2.0, t) for i, (s, t) in enumerate(rows)]


def test_theme_is_found_whatever_glyph_the_rip_uses() -> None:
    """Malcolm's first season uses "*" in some rips and "♪" in others."""
    from src.subtitle_utils import find_theme_start

    asterisk = build([
        (2.0, "MALCOLM: This is the world--"), (34.6, "At some point, it stops."),
        (37.0, "* Yes, no, maybe"), (40.7, "* I don't know"),
        (42.7, "* Can you repeat the question?"),
    ])
    glyph = build([
        (14.4, "What'd you do?"), (68.2, "Either she's losing her touch"),
        (72.0, "♪ Yes, no, maybe"), (76.2, "♪ I don't know"),
        (78.6, "♪ Can you repeat the question?"),
    ])

    assert find_theme_start(asterisk) == pytest.approx(37.0)
    assert find_theme_start(glyph) == pytest.approx(72.0)


def test_cold_opens_are_not_a_fixed_length() -> None:
    """
    The reason detection exists at all: across three episodes of one show the
    theme starts at 37s, 72s and 129s. Any fixed cutoff is wrong for two of
    them.
    """
    from src.subtitle_utils import find_theme_start

    late = build([
        (3.1, "And then there's a big explosion"), (7.5, "(laughing)"),
        (99.3, "What do you think you're doing?"), (126.5, "Yep, this is a good dinner."),
        (129.5, "* Yes, no, maybe"), (133.1, "* I don't know"),
        (135.6, "* Can you repeat the question?"),
    ])

    assert find_theme_start(late) == pytest.approx(129.5)


def test_sound_effects_are_not_mistaken_for_music() -> None:
    from src.subtitle_utils import find_theme_start, is_music_cue

    assert not is_music_cue("(laughing)")
    assert not is_music_cue("(squeaking)")
    assert not is_music_cue("MALCOLM: This is the world--")
    assert is_music_cue("(gentle music playing)")

    assert find_theme_start(build([
        (2.0, "(squeaking)"), (5.0, "(laughing)"), (9.0, "(door slams)"),
    ])) is None


def test_a_single_sung_line_is_not_a_title_sequence() -> None:
    """A character humming, or a radio — one line is not a theme."""
    from src.subtitle_utils import find_theme_start

    assert find_theme_start(build([
        (20.0, "♪ la la la"), (24.0, "Would you stop that?"),
        (28.0, "Never."),
    ])) is None


def test_an_episode_opening_on_a_song_is_not_truncated() -> None:
    """min_start guards the case where the cold open itself is musical."""
    from src.subtitle_utils import find_theme_start

    rows = build([
        (1.0, "♪ opening number"), (3.0, "♪ still singing"), (5.0, "♪ and again"),
        (40.0, "Dialogue resumes."),
        (80.0, "* Yes, no, maybe"), (84.0, "* I don't know"),
        (88.0, "* Can you repeat the question?"),
    ])

    assert find_theme_start(rows) == pytest.approx(80.0)
