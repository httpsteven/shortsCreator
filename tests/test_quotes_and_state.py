"""
quotes.json lookup and the produced-clips ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.quote_finder import QuoteStore, QuotesError
from src.state import ClipRecord, State, clip_id, series_id

SAMPLE = {
    "Heat": [
        "I do what I do best, I take scores.",
        {"label": "The diner", "quote": "A guy told me one time."},
    ],
    "Malcolm in the Middle - S01E03 - Home Alone 4": {
        "dewey_chaos": ["I am the smartest man alive!"],
        "malcolm_normal": ["I just want one normal day."],
    },
    "Empty Movie": [],
}


@pytest.fixture
def store(tmp_path: Path) -> QuoteStore:
    path = tmp_path / "quotes.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return QuoteStore.load(path)


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------


def test_flat_list_returns_all_quotes(store: QuoteStore) -> None:
    candidates = store.candidates_for("Heat")

    assert len(candidates) == 2
    assert candidates[1].label == "The diner"
    assert candidates[1].quote == "A guy told me one time."


def test_category_pulls_one_bucket(store: QuoteStore) -> None:
    key = "Malcolm in the Middle - S01E03 - Home Alone 4"
    candidates = store.candidates_for(key, "dewey_chaos")

    assert [c.quote for c in candidates] == ["I am the smartest man alive!"]
    assert candidates[0].category == "dewey_chaos"


def test_omitting_category_pools_every_bucket(store: QuoteStore) -> None:
    key = "Malcolm in the Middle - S01E03 - Home Alone 4"
    candidates = store.candidates_for(key)

    assert len(candidates) == 2
    assert {c.category for c in candidates} == {"dewey_chaos", "malcolm_normal"}


def test_unknown_category_returns_nothing_rather_than_the_whole_pool(
    store: QuoteStore,
) -> None:
    """
    A typo in --category should look like a typo. Falling back to every bucket
    would quietly produce the wrong clips and look like success.
    """
    key = "Malcolm in the Middle - S01E03 - Home Alone 4"

    assert store.candidates_for(key, "dewey_chos") == []
    assert store.categories_for(key) == ["dewey_chaos", "malcolm_normal"]


def test_key_resolution_tolerates_title_drift(store: QuoteStore) -> None:
    """quotes.json is assembled by hand, so keys drift from parsed filenames."""
    assert store.resolve("heat") == "Heat"
    assert store.resolve("malcolm in the middle s01e03 home alone 4") == (
        "Malcolm in the Middle - S01E03 - Home Alone 4"
    )
    assert store.resolve("Some Film That Is Not Here") is None


def test_entry_present_but_empty_is_distinguishable(store: QuoteStore) -> None:
    assert store.resolve("Empty Movie") == "Empty Movie"
    assert store.candidates_for("Empty Movie") == []


def test_missing_file_explains_the_workflow(tmp_path: Path) -> None:
    with pytest.raises(QuotesError) as exc:
        QuoteStore.load(tmp_path / "nope.json")

    assert "export_titles" in str(exc.value)


def test_malformed_json_reports_the_location(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"Heat": ["unterminated]', encoding="utf-8")

    with pytest.raises(QuotesError) as exc:
        QuoteStore.load(path)

    assert "line" in str(exc.value)


# --------------------------------------------------------------------------
# State ledger
# --------------------------------------------------------------------------


def make_record(quote: str = "a quote", **overrides) -> ClipRecord:
    base = dict(
        clip_id=clip_id(Path("/srv/x.mkv"), quote),
        source="/srv/x.mkv", lookup_key="X", quote=quote, category=None,
        start=10.0, end=30.0, score=95.0, output="/out/x.mp4",
    )
    base.update(overrides)
    return ClipRecord(**base)


def test_clip_identity_survives_formatting_differences() -> None:
    """
    Keyed on source + quote, so re-running after a config change still
    recognises the clip rather than producing a duplicate.
    """
    assert clip_id(Path("/srv/x.mkv"), "A Quote.") == clip_id(
        Path("/srv/x.mkv"), "  a quote.  "
    )
    assert clip_id(Path("/srv/x.mkv"), "one") != clip_id(Path("/srv/y.mkv"), "one")


def test_series_identity_is_order_independent() -> None:
    quotes = ["one", "two", "three"]
    assert series_id(Path("/srv/x.mkv"), quotes) == series_id(
        Path("/srv/x.mkv"), list(reversed(quotes))
    )


def test_state_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = State.load(path)
    state.record(make_record())
    state.save()

    reloaded = State.load(path)
    assert len(reloaded) == 1
    assert reloaded.has_clip(Path("/srv/x.mkv"), "a quote")


def test_corrupt_state_is_quarantined_not_fatal(tmp_path: Path) -> None:
    """
    Re-producing clips is recoverable; refusing to run is not. A corrupt ledger
    is moved aside so the next run can proceed.
    """
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")

    state = State.load(path)

    assert len(state) == 0
    assert (tmp_path / "state.corrupt.json").exists()


def test_series_parts_come_back_in_order(tmp_path: Path) -> None:
    state = State.load(tmp_path / "state.json")
    for index in (3, 1, 2):
        state.record(
            make_record(
                quote=f"q{index}",
                clip_id=clip_id(Path("/srv/x.mkv"), f"q{index}"),
                series="abc", part_index=index, part_total=3,
            )
        )

    assert [r.part_index for r in state.series_parts("abc")] == [1, 2, 3]
    assert state.has_series("abc")
