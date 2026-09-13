"""
Multipart series planning.

The failure this guards against is a series that looks broken to a viewer:
gap-numbered parts, two shorts showing the same scene, or parts out of order.
"""

from __future__ import annotations

import pytest

from src.clip_extractor import ClipWindow
from src.config import Config
from src.segmenter import plan_series, resolve_overlaps, suppress_nearby
from src.subtitle_utils import Match


def m(start: float, score: float = 90.0, quote: str | None = None) -> Match:
    return Match(
        quote=quote or f"quote at {start}",
        start=start, end=start + 3.0, score=score,
        cue_span=(0, 0), matched_text="text",
    )


# --------------------------------------------------------------------------
# Proximity suppression
# --------------------------------------------------------------------------


def test_nearby_matches_are_collapsed_to_the_best_one() -> None:
    """Two quotes from one scene would otherwise be two near-identical shorts."""
    kept, dropped = suppress_nearby([m(100, 80), m(140, 95), m(600, 85)], 180.0)

    assert dropped == 1
    assert sorted(match.start for match in kept) == [140, 600]
    # The stronger of the pair survived, not the earlier.
    assert any(match.score == 95 for match in kept)


def test_suppression_works_best_first_not_earliest_first() -> None:
    weak_early, strong_late = m(10, 74), m(120, 99)
    kept, _ = suppress_nearby([weak_early, strong_late], 180.0)

    assert [match.start for match in kept] == [120]


def test_distant_matches_all_survive() -> None:
    kept, dropped = suppress_nearby([m(0), m(300), m(600), m(900)], 180.0)

    assert dropped == 0
    assert len(kept) == 4


# --------------------------------------------------------------------------
# Series planning
# --------------------------------------------------------------------------


def test_parts_are_ordered_chronologically_not_by_input_order(
    config: Config,
) -> None:
    """
    quotes.json order is not trusted. Matched subtitle time decides, the same
    way it decides everything else in this pipeline.
    """
    plan = plan_series([m(900), m(300), m(1500), m(600)], 2000.0, config)

    assert plan.ok
    starts = [part.match.start for part in plan.parts]
    assert starts == sorted(starts)
    assert [part.index for part in plan.parts] == [1, 2, 3, 4]


def test_shortfall_renumbers_against_actual_count(config: Config) -> None:
    """
    Asked for 5, only 3 distinct moments matched. The series must read
    1/3, 2/3, 3/3 — never 1 of 5, 3 of 5, 5 of 5.
    """
    plan = plan_series([m(300), m(900), m(1500)], 2000.0, config, requested_parts=5)

    assert [part.badge for part in plan.parts] == ["Part 1/3", "Part 2/3", "Part 3/3"]
    assert all(part.total == 3 for part in plan.parts)


def test_below_minimum_skips_the_series_with_a_reason(config: Config) -> None:
    plan = plan_series([m(300), m(900)], 2000.0, config)

    assert not plan.ok
    assert plan.parts == []
    assert "min_parts" in plan.reason


def test_capped_at_max_parts_keeping_the_strongest(config: Config) -> None:
    matches = [m(300, 75), m(900, 99), m(1500, 98), m(2100, 97), m(2700, 96), m(3300, 95)]
    plan = plan_series(matches, 4000.0, config)

    assert len(plan.parts) == config.multipart.max_parts
    # The weakest was dropped, and order is still chronological.
    assert 300 not in [part.match.start for part in plan.parts]
    starts = [part.match.start for part in plan.parts]
    assert starts == sorted(starts)


def test_requested_parts_narrows_below_the_configured_max(config: Config) -> None:
    matches = [m(300), m(900), m(1500), m(2100), m(2700)]
    plan = plan_series(matches, 4000.0, config, requested_parts=3)

    assert len(plan.parts) == 3
    assert all(part.total == 3 for part in plan.parts)


def test_no_matches_reports_a_reason(config: Config) -> None:
    plan = plan_series([], 2000.0, config)

    assert not plan.ok
    assert plan.reason == "no quotes matched"


def test_labels_are_carried_onto_parts(config: Config) -> None:
    matches = [m(300, quote="the diner"), m(900, quote="the heist"), m(1500, quote="the end")]
    plan = plan_series(
        matches, 2000.0, config,
        labels={"the diner": "The diner scene", "the heist": "The heist"},
    )

    assert plan.parts[0].label == "The diner scene"
    assert plan.parts[2].label is None


def test_part_filename_suffix(config: Config) -> None:
    plan = plan_series([m(300), m(900), m(1500)], 2000.0, config)
    assert plan.parts[1].suffix == "Part 2 of 3"


# --------------------------------------------------------------------------
# Overlap resolution
# --------------------------------------------------------------------------


def test_overlapping_windows_are_split_at_the_midpoint() -> None:
    """Two parts must never show the viewer the same seconds twice."""
    resolved = resolve_overlaps([ClipWindow(0, 40), ClipWindow(30, 70)])

    assert resolved[0].end == resolved[1].start == 35.0
    assert not resolved[0].overlaps(resolved[1])


def test_non_overlapping_windows_are_untouched() -> None:
    windows = [ClipWindow(0, 20), ClipWindow(200, 220)]
    assert resolve_overlaps(windows) == windows


def test_default_config_never_produces_overlap(config: Config) -> None:
    """
    With min_separation (180s) far wider than max_duration (60s), overlap is
    impossible — this pins that relationship so a config change can't quietly
    break it.
    """
    plan = plan_series([m(300), m(500), m(900), m(1500)], 2000.0, config)

    windows = [part.window for part in plan.parts]
    for earlier, later in zip(windows, windows[1:]):
        assert not earlier.overlaps(later)
