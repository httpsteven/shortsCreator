"""
Recaps: several moments stitched into one clip.

The concat filter graph is the risky part — it composes N segments, joins them
and burns one caption track over the join — so this renders a real file and
inspects it rather than only checking the plan.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.caption_renderer import build_recap_filter, render_recap
from src.config import Config
from src.recap import Moment, plan_recap, select_by_beats
from src.subtitle_utils import Match, cues_in_window, load_cues
from tests.fixtures import SERIES_QUOTES, full_transcript_cues, write_srt


def make_match(quote: str, start: float, end: float, score: float = 95.0) -> Match:
    return Match(
        quote=quote, start=start, end=end, score=score,
        cue_span=(0, 0), matched_text=quote,
    )


@pytest.fixture
def matches() -> list[Moment]:
    return [
        Moment(source=Path("/srv/x.mkv"), label="X", match=make_match(q, s, e),
               runtime=120.0)
        for q, s, e in SERIES_QUOTES
    ]


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def test_segments_are_chronological_and_fill_the_target(
    config: Config, matches: list[Match]
) -> None:
    plan = plan_recap(matches, config, runtime=120.0, target_duration=20.0)

    assert plan.ok
    assert plan.duration == pytest.approx(20.0, abs=0.1)

    offsets = [s.offset for s in plan.segments]
    starts = [s.window.start for s in plan.segments]
    assert offsets == sorted(offsets)
    assert starts == sorted(starts)


def test_offsets_are_contiguous(config: Config, matches: list[Match]) -> None:
    """A gap or overlap in the output timeline puts every caption off."""
    plan = plan_recap(matches, config, runtime=120.0, target_duration=20.0)

    expected = 0.0
    for segment in plan.segments:
        assert segment.offset == pytest.approx(expected, abs=0.01)
        expected += segment.duration


def test_each_segment_ends_on_its_own_line(
    config: Config, matches: list[Match]
) -> None:
    """Same rule as a standalone clip: the moment lands, it doesn't trail off."""
    plan = plan_recap(matches, config, runtime=120.0, target_duration=20.0)

    for segment in plan.segments:
        assert segment.window.end == pytest.approx(
            min(120.0, segment.match.end + config.clip.pad_after), abs=0.01
        )


def test_nearby_moments_are_collapsed(config: Config, matches: list[Match]) -> None:
    """Two quotes seconds apart would be the same moment twice."""
    plan = plan_recap(matches, config, runtime=120.0, target_duration=20.0)
    starts = [s.window.start for s in plan.segments]

    for earlier, later in zip(starts, starts[1:]):
        assert later - earlier >= config.recap.min_separation - 5


def test_too_few_moments_is_refused_with_a_reason(config: Config) -> None:
    plan = plan_recap(
        [
            Moment(Path("/srv/x.mkv"), "X", make_match("one", 10, 12), 120.0),
            Moment(Path("/srv/x.mkv"), "X", make_match("two", 60, 62), 120.0),
        ],
        config, runtime=120.0, target_duration=45.0,
    )

    assert not plan.ok
    assert "min_segments" in plan.reason


def test_target_too_short_for_the_minimum_segment(
    config: Config, matches: list[Match]
) -> None:
    """Ten seconds cannot hold four moments worth watching."""
    plan = plan_recap(matches, config, runtime=120.0, target_duration=10.0)

    assert not plan.ok
    assert "segment" in plan.reason


# --------------------------------------------------------------------------
# The filter graph
# --------------------------------------------------------------------------


def test_filter_graph_joins_every_segment(config: Config) -> None:
    chain = build_recap_filter(3, Path("/tmp/x.ass"), config, with_audio=True)

    assert "concat=n=3:v=1:a=1" in chain
    # setsar on the panel and again after the overlay, for each segment:
    # concat refuses inputs whose aspect ratios disagree, and seeking into
    # different parts of a file can report them differently.
    assert chain.count("setsar=1") == 6
    # Captions applied once, over the joined timeline.
    assert chain.count("ass=") == 1


def test_filter_graph_without_audio(config: Config) -> None:
    chain = build_recap_filter(2, Path("/tmp/x.ass"), config, with_audio=False)

    assert "concat=n=2:v=1:a=0" in chain
    assert "[0:a]" not in chain


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_renders_a_real_recap(
    media: dict[str, Path], config: Config, tmp_path: Path, matches: list[Match]
) -> None:
    source = media["with_text_subs"]
    cues = load_cues(write_srt(tmp_path / "t.srt", full_transcript_cues()))

    plan = plan_recap(matches, config, runtime=120.0, target_duration=20.0)
    assert plan.ok
    plan.segments = [
        type(s)(s.index, Moment(source, s.moment.label, s.match, 120.0), s.window, s.offset)
        for s in plan.segments
    ]

    destination = tmp_path / "recap.mp4"
    ok, message = render_recap(
        plan.segments,
        [cues_in_window(cues, s.window.start, s.window.end) for s in plan.segments],
        destination,
        config,
    )
    assert ok, message

    info = dict(
        line.split("=", 1)
        for line in subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-show_entries", "format=duration",
             "-of", "default=nw=1", str(destination)],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
    )

    assert (info["width"], info["height"]) == ("1080", "1920")
    assert float(info["duration"]) == pytest.approx(plan.duration, abs=1.0)


def test_recap_captions_span_the_whole_output(
    media: dict[str, Path], config: Config, tmp_path: Path, matches: list[Match]
) -> None:
    """
    Cues are rebased twice — once to their segment, once onto the output
    timeline. Get the second wrong and every caption after the first segment
    appears at the wrong moment, or past the end of the clip entirely.
    """
    source = media["with_text_subs"]
    cues = load_cues(write_srt(tmp_path / "t.srt", full_transcript_cues()))

    plan = plan_recap(matches, config, runtime=120.0, target_duration=20.0)
    plan.segments = [
        type(s)(s.index, Moment(source, s.moment.label, s.match, 120.0), s.window, s.offset)
        for s in plan.segments
    ]
    destination = tmp_path / "recap.mp4"

    ok, message = render_recap(
        plan.segments,
        [cues_in_window(cues, s.window.start, s.window.end) for s in plan.segments],
        destination, config,
    )
    assert ok, message

    ass = destination.with_suffix(".ass").read_text()
    times = []
    for line in ass.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        hours, minutes, seconds = line.split(",")[1].split(":")
        times.append(int(hours) * 3600 + int(minutes) * 60 + float(seconds))

    assert times, "no captions were written"
    assert max(times) <= plan.duration + 0.5
    # Captions must appear beyond the first segment, or the offsets were dropped.
    assert max(times) > plan.segments[0].duration


# --------------------------------------------------------------------------
# Beat selection — covering the story, not just the best lines
# --------------------------------------------------------------------------


def moment_at(start: float, score: float, source: str = "/srv/x.mkv") -> Moment:
    return Moment(
        source=Path(source), label="X",
        match=make_match(f"q{start}", start, start + 2.0, score), runtime=1200.0,
    )


def test_beats_cover_the_whole_runtime_not_just_the_best_lines() -> None:
    """
    Ranking purely by score clusters wherever the strongest lines fall —
    usually the middle, because that is where most of an episode is. A rundown
    that skips the opening and the ending is not a rundown.
    """
    # Five strong moments bunched in the middle, two weak ones at the edges.
    moments = [
        moment_at(30, 74),
        moment_at(560, 99), moment_at(580, 98), moment_at(600, 97),
        moment_at(620, 96), moment_at(640, 95),
        moment_at(1150, 76),
    ]

    chosen = select_by_beats(moments, 4, runtime=1200.0)
    starts = [m.start for m in chosen]

    assert len(chosen) == 4
    assert starts == sorted(starts)
    # The opening and the ending are represented despite scoring lowest.
    assert min(starts) < 300
    assert max(starts) > 900


def test_beats_take_the_best_within_each_act() -> None:
    moments = [moment_at(50, 80), moment_at(100, 95), moment_at(900, 90)]

    chosen = select_by_beats(moments, 2, runtime=1200.0)

    assert [m.start for m in chosen] == [100, 900]


def test_beats_backfill_when_an_act_is_empty() -> None:
    """An episode with nothing quotable in its third act still fills the slot."""
    moments = [moment_at(50, 80), moment_at(100, 95), moment_at(150, 90)]

    chosen = select_by_beats(moments, 3, runtime=1200.0)

    assert len(chosen) == 3


def test_score_spread_is_used_when_there_is_no_single_runtime(
    config: Config,
) -> None:
    """Timestamps from different episodes are not comparable."""
    moments = [
        moment_at(100, 99, "/srv/a.mkv"),
        moment_at(100, 98, "/srv/b.mkv"),
        moment_at(100, 97, "/srv/c.mkv"),
        moment_at(100, 96, "/srv/d.mkv"),
        moment_at(100, 60, "/srv/e.mkv"),
    ]

    plan = plan_recap(
        moments, config, target_duration=40.0, spread="score", suppress=False
    )

    assert plan.ok
    # All five survive: moments at the same minute of DIFFERENT episodes are not
    # near-duplicates, and suppression would wrongly collapse them to one.
    assert len(plan.segments) == 5
    assert len({s.source for s in plan.segments}) == 5

    # And when a count IS imposed, ranking decides — the weakest goes.
    capped = plan_recap(
        moments, config, target_duration=40.0, segments=4,
        spread="score", suppress=False,
    )
    assert len(capped.segments) == 4
    assert all(s.match.score >= 96 for s in capped.segments)


# --------------------------------------------------------------------------
# Compilations — one clip, many sources
# --------------------------------------------------------------------------


def test_renders_a_compilation_from_two_different_files(
    media: dict[str, Path], config: Config, tmp_path: Path
) -> None:
    """
    The whole point of a compilation: segments come from different episodes.
    A single-source assumption anywhere in the graph breaks this.
    """
    first, second = media["with_text_subs"], media["multi_lang"]
    cues = load_cues(write_srt(tmp_path / "t.srt", full_transcript_cues()))

    moments = [
        Moment(first, "Episode one", make_match("a", 33.0, 36.0, 95.0), 120.0),
        Moment(second, "Episode two", make_match("b", 72.0, 75.0, 94.0), 120.0),
        Moment(first, "Episode one", make_match("c", 108.0, 111.0, 93.0), 120.0),
        Moment(second, "Episode two", make_match("d", 8.0, 11.0, 92.0), 120.0),
    ]

    plan = plan_recap(
        moments, config, target_duration=20.0, spread="score", suppress=False
    )
    assert plan.ok
    assert len({s.source for s in plan.segments}) == 2, "both sources represented"

    destination = tmp_path / "compilation.mp4"
    ok, message = render_recap(
        plan.segments,
        [cues_in_window(cues, s.window.start, s.window.end) for s in plan.segments],
        destination, config,
    )
    assert ok, message

    probed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-show_entries", "format=duration", "-of", "default=nw=1",
         str(destination)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = dict(line.split("=", 1) for line in probed.strip().splitlines())

    assert (info["width"], info["height"]) == ("1080", "1920")
    assert float(info["duration"]) == pytest.approx(plan.duration, abs=1.0)
