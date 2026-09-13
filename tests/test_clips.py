"""
Clip window planning and the vertical render.

Window planning is a pure function with more edge cases than it appears to
have — matches near the start of a file, matches longer than the maximum clip
length, media shorter than the minimum — so it is tested exhaustively here
rather than discovered in production.

The render tests assert what can be asserted mechanically (dimensions, duration,
ASS structure). Whether the captions are LEGIBLE is a visual question and was
checked by extracting frames and looking at them; see README.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.caption_renderer import (
    ass_text,
    ass_time,
    build_ass,
    build_filter_chain,
    escape_filter_path,
    render,
)
from src.clip_extractor import ClipWindow, build_input_args, encoder_args, plan_window
from src.config import Config
from src.subtitle_utils import Cue, Match, best_match, cues_in_window, load_cues
from tests.fixtures import KNOWN_QUOTE, full_transcript_cues, write_srt


def make_match(start: float, end: float, score: float = 90.0) -> Match:
    return Match(
        quote="q", start=start, end=end, score=score,
        cue_span=(0, 0), matched_text="q",
    )


def probe_video(path: Path) -> dict[str, str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-show_entries", "format=duration",
         "-of", "default=nw=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.strip().splitlines())


# --------------------------------------------------------------------------
# Window planning
# --------------------------------------------------------------------------


def test_padding_is_applied_then_clamped_to_minimum(config: Config) -> None:
    window = plan_window(make_match(50.0, 53.0), media_duration=600.0, config=config)

    # 3s match + 1.5 + 2.0 padding = 6.5s, which is under the 20s minimum.
    assert window.duration == pytest.approx(config.clip.min_duration)
    # The match stays inside the window.
    assert window.start <= 50.0 and window.end >= 53.0


def test_match_near_start_does_not_produce_negative_window(config: Config) -> None:
    """
    A match 2 seconds in still has to yield a full-length clip. Whatever the
    left side can't absorb is handed to the right.
    """
    window = plan_window(make_match(2.0, 4.0), media_duration=600.0, config=config)

    assert window.start == 0.0
    assert window.duration == pytest.approx(config.clip.min_duration)


def test_match_near_end_does_not_overrun_media(config: Config) -> None:
    window = plan_window(make_match(595.0, 598.0), media_duration=600.0, config=config)

    assert window.end <= 600.0
    assert window.duration == pytest.approx(config.clip.min_duration)


def test_window_never_exceeds_maximum(config: Config) -> None:
    window = plan_window(make_match(100.0, 200.0), media_duration=600.0, config=config)

    assert window.duration == pytest.approx(config.clip.max_duration)


def test_match_longer_than_max_keeps_the_start_of_the_line(config: Config) -> None:
    """
    Nothing can hold a 100-second line in a 60-second clip. Keeping the
    beginning is the choice that still makes sense without context.
    """
    window = plan_window(make_match(100.0, 200.0), media_duration=600.0, config=config)

    assert window.start == pytest.approx(100.0)
    assert window.end == pytest.approx(160.0)


def test_media_shorter_than_minimum_returns_whole_file(config: Config) -> None:
    """Rejecting short media would be a failure the user can do nothing about."""
    window = plan_window(make_match(2.0, 5.0), media_duration=8.0, config=config)

    assert window.start == 0.0
    assert window.end == pytest.approx(8.0)


def test_windows_can_be_tested_for_overlap() -> None:
    assert ClipWindow(0, 30).overlaps(ClipWindow(25, 60))
    assert not ClipWindow(0, 30).overlaps(ClipWindow(30, 60))


# --------------------------------------------------------------------------
# ffmpeg argument construction
# --------------------------------------------------------------------------


def test_seek_comes_before_input(config: Config) -> None:
    """
    "-ss" before "-i" is fast keyframe seeking. After "-i" it decodes the whole
    file up to that point, which on a 2-hour film is the difference between
    instant and minutes.
    """
    args = build_input_args(Path("/tmp/x.mkv"), ClipWindow(40.0, 60.0))

    assert args.index("-ss") < args.index("-i")
    assert args[args.index("-t") + 1] == "20.000"


def test_no_stream_copy_anywhere(config: Config) -> None:
    """Stream copy can't cut on an exact frame; it drifts audio and captions."""
    args = build_input_args(Path("/tmp/x.mkv"), ClipWindow(0, 20)) + encoder_args(config)

    assert "copy" not in args


def test_nvenc_encoder_args(config: Config) -> None:
    config.video.encoder = "h264_nvenc"
    args = encoder_args(config)

    assert args[:2] == ["-c:v", "h264_nvenc"]
    assert "-cq" in args


# --------------------------------------------------------------------------
# ASS generation
# --------------------------------------------------------------------------


def test_commas_in_dialogue_are_not_escaped(config: Config) -> None:
    """
    The bug this project was warned about. The Text field is everything after
    the last fixed comma, so escaping commas renders a literal backslash on
    screen — invisible to every check except looking at a frame.
    """
    cues = [Cue(0, 1.0, 3.0, "Well, that's the thing, isn't it?")]
    ass = build_ass(cues, config)

    assert r"\," not in ass
    assert "Well, that's the thing, isn't it?" in ass


def test_dialogue_text_field_keeps_everything_after_the_ninth_comma(
    config: Config,
) -> None:
    """A Dialogue line has 9 fixed fields; the rest is text, commas and all."""
    cues = [Cue(0, 1.0, 3.0, "a, b, c, d")]
    line = [l for l in build_ass(cues, config).splitlines() if l.startswith("Dialogue:")][0]

    assert line.split(",", 9)[9] == "a, b, c, d"


def test_newlines_become_ass_hard_breaks() -> None:
    assert ass_text("one\ntwo") == r"one\Ntwo"


def test_ass_time_format() -> None:
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(65.25) == "0:01:05.25"
    assert ass_time(3661.5) == "1:01:01.50"
    assert ass_time(-5) == "0:00:00.00"


def test_playres_matches_output_resolution(config: Config) -> None:
    """
    Without this libass assumes 384x288 and every font size and margin is
    wrong by roughly a factor of three.
    """
    ass = build_ass([Cue(0, 0, 1, "x")], config)

    assert f"PlayResX: {config.video.width}" in ass
    assert f"PlayResY: {config.video.height}" in ass


def test_part_badge_spans_the_whole_clip(config: Config) -> None:
    ass = build_ass([Cue(0, 0, 2, "x")], config, part_label="Part 2/4", clip_duration=30.0)

    badge = [l for l in ass.splitlines() if "PartBadge," in l and l.startswith("Dialogue")]
    assert len(badge) == 1
    assert badge[0].endswith("Part 2/4")
    assert "0:00:00.00,0:00:30.00" in badge[0]


def test_no_badge_style_when_no_label(config: Config) -> None:
    ass = build_ass([Cue(0, 0, 2, "x")], config)
    assert "PartBadge" not in ass


def test_filter_path_escaping() -> None:
    """Colons are ffmpeg filter argument separators; unescaped they break it."""
    assert escape_filter_path(Path("/tmp/a:b/c.ass")) == r"/tmp/a\:b/c.ass"


def test_filter_chain_composes_blur_behind_a_cropped_panel(
    config: Config,
) -> None:
    """The default 4:5 panel is cropped to fill, not fitted to the width."""
    chain = build_filter_chain(Path("/tmp/x.ass"), config)

    assert "gblur" in chain
    assert "overlay=(W-w)/2:(H-h)/2" in chain
    assert "scale=1080:1350:force_original_aspect_ratio=increase" in chain
    assert "crop=1080:1350" in chain


def test_filter_chain_uses_even_height_scale_when_uncropped(
    config: Config,
) -> None:
    """
    With no panel aspect the source fits the full width, and the height must be
    EVEN — "-2" not "-1". Odd dimensions are rejected by libx264 with an error
    that points nowhere near the scale filter.
    """
    config.video.foreground_aspect = None
    chain = build_filter_chain(Path("/tmp/x.ass"), config)

    assert f"scale={config.video.width}:-2" in chain


def test_panel_height_is_always_even(config: Config) -> None:
    from src.caption_renderer import foreground_size

    for aspect in ("4:5", "16:9", "3:4", "1:1", "5:7"):
        config.video.foreground_aspect = aspect
        width, height = foreground_size(config)
        assert height % 2 == 0, f"{aspect} gave an odd height"
        assert width == config.video.width


def test_panel_is_clamped_to_the_canvas(config: Config) -> None:
    """A panel taller than the frame is just a full-frame crop."""
    from src.caption_renderer import foreground_size

    config.video.foreground_aspect = "9:16"
    assert foreground_size(config) == (1080, 1920)

    config.video.foreground_aspect = "1:4"
    assert foreground_size(config)[1] <= config.video.height


def test_unparseable_aspect_falls_back_rather_than_crashing(
    config: Config,
) -> None:
    from src.caption_renderer import foreground_size

    for bad in ("nonsense", "4:0", "", "   ", "-1:5"):
        config.video.foreground_aspect = bad
        assert foreground_size(config) is None


# --------------------------------------------------------------------------
# End-to-end render
# --------------------------------------------------------------------------


def test_render_produces_vertical_clip_of_the_right_length(
    media: dict[str, Path], config: Config, tmp_path: Path
) -> None:
    source = media["with_text_subs"]
    cues = load_cues(write_srt(tmp_path / "t.srt", full_transcript_cues()))
    match = best_match(cues, KNOWN_QUOTE)
    window = plan_window(match, media_duration=120.0, config=config)

    destination = tmp_path / "clip.mp4"
    ok, message = render(
        source, window, cues_in_window(cues, window.start, window.end),
        destination, config, part_label="Part 1/3",
    )

    assert ok, message
    info = probe_video(destination)
    assert (info["width"], info["height"]) == ("1080", "1920")
    assert float(info["duration"]) == pytest.approx(window.duration, abs=0.3)


def test_rendered_captions_are_shifted_into_clip_time(
    media: dict[str, Path], config: Config, tmp_path: Path
) -> None:
    """
    The .ass written beside the clip must contain clip-relative times. A cue at
    42s in the source appearing at 42s in a 20s clip would never be seen.
    """
    source = media["with_text_subs"]
    cues = load_cues(write_srt(tmp_path / "t.srt", full_transcript_cues()))
    match = best_match(cues, KNOWN_QUOTE)
    window = plan_window(match, media_duration=120.0, config=config)

    destination = tmp_path / "clip.mp4"
    ok, message = render(
        source, window, cues_in_window(cues, window.start, window.end),
        destination, config,
    )
    assert ok, message

    ass = destination.with_suffix(".ass").read_text()
    dialogue = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert dialogue

    # Every cue must start within the clip's own duration.
    for line in dialogue:
        start = line.split(",")[1]
        hours, minutes, seconds = start.split(":")
        offset = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        assert offset <= window.duration + 0.5


# --------------------------------------------------------------------------
# Punchline anchoring — where the quote sits in the finished clip
# --------------------------------------------------------------------------


def dialogue(*spans: tuple[float, float]) -> list[Cue]:
    return [Cue(i, s, e, f"line {i}") for i, (s, e) in enumerate(spans)]


def test_quote_lands_at_the_end_not_the_middle(config: Config) -> None:
    """
    A quote is almost always the line that LANDS, not the line that sets it up.
    Padding it symmetrically puts the punchline in the middle and then runs on
    into unrelated conversation — technically correct, editorially pointless.
    """
    window = plan_window(make_match(600.0, 603.0), media_duration=1320.0, config=config)

    # The clip ends just after the line, not ~9 seconds later.
    assert window.end == pytest.approx(603.0 + config.clip.pad_after)
    # And the line sits in the last fifth of the clip.
    position = (603.0 - window.start) / window.duration
    assert position > 0.8


def test_start_snaps_to_a_pause_in_the_dialogue(config: Config) -> None:
    """Clips should open on a beat, not halfway through someone's sentence."""
    cues = dialogue(
        (570.0, 572.0),   # ...then a 4s pause, the natural place to start
        (576.0, 578.0),
        (578.5, 580.0),
        (580.2, 582.0),
        (598.0, 600.0),
        (600.0, 603.0),   # the quote
    )

    window = plan_window(
        make_match(600.0, 603.0), media_duration=1320.0, config=config, cues=cues
    )

    assert window.start == pytest.approx(576.0)


def test_falls_back_cleanly_without_cues(config: Config) -> None:
    a = plan_window(make_match(600.0, 603.0), 1320.0, config)
    b = plan_window(make_match(600.0, 603.0), 1320.0, config, cues=[])

    assert a == b
    assert config.clip.min_duration <= a.duration <= config.clip.max_duration


def test_a_pause_is_ignored_if_it_would_break_the_duration_limits(
    config: Config,
) -> None:
    """A huge gap far too early must not produce a 200-second clip."""
    cues = dialogue((10.0, 12.0), (400.0, 402.0), (600.0, 603.0))

    window = plan_window(make_match(600.0, 603.0), 1320.0, config, cues=cues)

    assert window.duration <= config.clip.max_duration
    assert window.start >= 603.0 + config.clip.pad_after - config.clip.max_duration


def test_centre_anchor_still_available(config: Config) -> None:
    config.clip.anchor = "center"
    window = plan_window(make_match(600.0, 603.0), 1320.0, config)

    position = (603.0 - window.start) / window.duration
    assert position < 0.8  # not pinned to the end any more


def test_punchline_anchor_respects_both_duration_bounds(config: Config) -> None:
    for start in (5.0, 50.0, 600.0, 1300.0):
        window = plan_window(make_match(start, start + 3.0), 1320.0, config)
        assert window.duration <= config.clip.max_duration + 0.01
        assert window.start >= 0.0
        assert window.end <= 1320.0 + 0.01
