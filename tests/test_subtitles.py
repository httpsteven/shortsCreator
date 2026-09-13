"""
Subtitle detection and verification.

The point of these tests is that each failure mode produces its OWN reason code.
A pipeline that reports "skipped" without saying why is undiagnosable, and the
distinction between "no subtitles", "image-only subtitles" and "a forced track
that looks fine but isn't" is the difference between knowing what to fix and
guessing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config
from src.media_probe import probe, select_stream
from src.subtitle_source import (
    Provenance,
    Reason,
    acquire,
    diagnose_level1,
    validate_srt,
)
from tests.fixtures import KNOWN_QUOTE, full_transcript_cues, write_srt


# --------------------------------------------------------------------------
# Level 1 — what the container declares
# --------------------------------------------------------------------------


def test_detects_embedded_english_text_stream(media: dict[str, Path]) -> None:
    result = probe(media["with_text_subs"])

    assert result.duration == pytest.approx(120, abs=1)
    assert len(result.text_streams) == 1
    assert result.text_streams[0].language == "eng"
    assert result.text_streams[0].is_text
    assert not result.text_streams[0].is_image


def test_no_subtitle_streams_is_detected(media: dict[str, Path]) -> None:
    result = probe(media["no_subs"])

    assert result.subtitle_streams == []
    assert result.sidecars == []
    assert diagnose_level1(result) is Reason.NO_SUB_STREAMS


def test_forced_disposition_is_read(media: dict[str, Path]) -> None:
    result = probe(media["forced_subs"])

    assert len(result.subtitle_streams) == 1
    assert result.subtitle_streams[0].forced is True


def test_sidecar_is_found_with_language_suffix(media: dict[str, Path]) -> None:
    result = probe(media["sidecar_movie"])

    assert result.subtitle_streams == []
    assert len(result.sidecars) == 1
    assert result.sidecars[0].language == "en"


# --------------------------------------------------------------------------
# Stream selection — never hardcoded 0:s:0
# --------------------------------------------------------------------------


def test_selects_english_over_first_stream(media: dict[str, Path]) -> None:
    """
    The fixture puts Spanish FIRST on purpose. A hardcoded 0:s:0 would pick it.
    """
    result = probe(media["multi_lang"])
    languages = [s.language for s in result.subtitle_streams]
    assert languages == ["spa", "eng"]

    ranked = select_stream(result.subtitle_streams, ["eng", "en"])
    assert ranked[0].language == "eng"


def test_forced_streams_rank_last() -> None:
    from src.media_probe import SubtitleStream

    forced_english = SubtitleStream(0, "subrip", "eng", None, forced=True, default=True)
    plain_spanish = SubtitleStream(1, "subrip", "spa", None, forced=False, default=False)

    ranked = select_stream([forced_english, plain_spanish], ["eng"])

    # Even though the forced track is the preferred language AND the default,
    # a non-forced track of any language is the better bet.
    assert ranked[0] is plain_spanish


# --------------------------------------------------------------------------
# Level 2 — proving the track is usable
# --------------------------------------------------------------------------


def test_healthy_track_validates(media: dict[str, Path], config: Config) -> None:
    result = acquire(probe(media["with_text_subs"]), config, allow_whisper=False)

    assert result.ok
    assert result.provenance is Provenance.TEXT_EMBEDDED
    assert result.reason is Reason.OK
    assert result.validation.cue_count >= 20
    assert result.validation.coverage > 0.8


def test_forced_track_rejected_on_coverage(
    media: dict[str, Path], config: Config
) -> None:
    """
    The gate that earns its keep.

    This track has plenty of cues and plenty of text — it clears every cheaper
    check. Only the coverage test catches that it stops a third of the way in.
    """
    result = acquire(probe(media["forced_subs"]), config, allow_whisper=False)

    assert not result.ok
    assert result.reason is Reason.LOW_COVERAGE_LIKELY_FORCED
    # Proves the cheaper gates really did pass first.
    assert result.validation.cue_count >= config.subtitles.min_cues
    assert result.validation.char_count >= config.subtitles.min_chars
    assert result.validation.coverage < 0.5


def test_no_subs_reports_no_sub_streams(
    media: dict[str, Path], config: Config
) -> None:
    result = acquire(probe(media["no_subs"]), config, allow_whisper=False)

    assert not result.ok
    assert result.reason is Reason.NO_SUB_STREAMS
    assert result.needs_whisper


def test_sidecar_is_used_when_no_embedded_track(
    media: dict[str, Path], config: Config
) -> None:
    result = acquire(probe(media["sidecar_movie"]), config, allow_whisper=False)

    assert result.ok
    assert result.provenance is Provenance.TEXT_SIDECAR
    assert KNOWN_QUOTE.lower() in result.srt_path.read_text().lower()


def test_sparse_track_rejected_on_cue_count(
    tmp_path: Path, config: Config
) -> None:
    srt = write_srt(
        tmp_path / "sparse.srt",
        [("Hola amigo como estas hoy", 5.0, 7.0), ("Adios", 90.0, 92.0)],
    )
    validation = validate_srt(srt, duration=120.0, config=config)

    assert not validation.ok
    # Count is checked before coverage, so this is the reason, not LOW_COVERAGE.
    assert validation.reason is Reason.TOO_FEW_CUES


def test_empty_file_reports_empty_output(tmp_path: Path, config: Config) -> None:
    empty = tmp_path / "empty.srt"
    empty.write_text("")

    validation = validate_srt(empty, duration=120.0, config=config)
    assert validation.reason is Reason.EMPTY_OUTPUT


def test_coverage_gate_skipped_when_duration_unknown(
    tmp_path: Path, config: Config
) -> None:
    """Without a runtime, coverage can't be computed — so it isn't guessed at."""
    srt = write_srt(tmp_path / "full.srt", full_transcript_cues())

    validation = validate_srt(srt, duration=None, config=config)

    assert validation.ok
    assert validation.coverage is None


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def test_whisper_tier_used_when_no_text_track(
    media: dict[str, Path], config: Config, tmp_path: Path
) -> None:
    """
    Tier 3 is reached only after embedded and sidecar routes fail.

    The transcriber is injected, so the ladder is testable without
    faster-whisper installed.
    """
    transcript = write_srt(tmp_path / "whisper.srt", full_transcript_cues())
    calls: list[Path] = []

    def fake_transcriber(path: Path) -> tuple[Path, bool]:
        calls.append(path)
        return transcript, False

    result = acquire(
        probe(media["no_subs"]), config, transcriber=fake_transcriber
    )

    assert calls == [media["no_subs"]]
    assert result.ok
    assert result.provenance is Provenance.WHISPER_GENERATED


def test_whisper_not_called_when_embedded_track_works(
    media: dict[str, Path], config: Config
) -> None:
    """Transcribing a file that already has subtitles would be pure waste."""
    calls: list[Path] = []

    def fake_transcriber(path: Path):
        calls.append(path)
        return None, False

    result = acquire(
        probe(media["with_text_subs"]), config, transcriber=fake_transcriber
    )

    assert result.provenance is Provenance.TEXT_EMBEDDED
    assert calls == []


def test_whisper_unavailable_is_its_own_reason(
    media: dict[str, Path], config: Config
) -> None:
    result = acquire(probe(media["no_subs"]), config, transcriber=None)

    assert result.reason is Reason.WHISPER_UNAVAILABLE


def test_manual_override_takes_precedence(
    media: dict[str, Path], config: Config, tmp_path: Path
) -> None:
    """An uploaded caption source outranks a perfectly good embedded track."""
    override = write_srt(tmp_path / "override.srt", full_transcript_cues())

    result = acquire(
        probe(media["with_text_subs"]), config, override=override, allow_whisper=False
    )

    assert result.provenance is Provenance.MANUAL_OVERRIDE


def test_bad_override_fails_loudly_rather_than_falling_through(
    media: dict[str, Path], config: Config, tmp_path: Path
) -> None:
    """
    A user explicitly supplied this file. Silently ignoring it and using the
    embedded track would leave them wondering why their upload did nothing.
    """
    override = write_srt(tmp_path / "bad.srt", [("Hi", 1.0, 2.0)])

    result = acquire(
        probe(media["with_text_subs"]), config, override=override, allow_whisper=False
    )

    assert not result.ok
    assert result.reason is Reason.TOO_FEW_CUES
