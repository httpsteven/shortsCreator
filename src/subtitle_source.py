"""
Subtitle acquisition and validation — Level 2 of "do I have subtitles?".

Level 1 (`media_probe`) reports what a container *declares*. This module proves
whether that declaration is worth anything, by extracting the track and putting
it through a series of gates. A stream can exist, convert without error, parse
cleanly, and still be useless.

The gate that earns its keep is coverage. A *forced* subtitle track carries only
foreign-language lines — a handful of cues scattered across a two-hour film. It
extracts fine and parses fine, so every cheaper check passes, and it would then
match almost nothing while looking completely healthy.

Every failure returns a specific reason code rather than a bare False, because a
run that produces zero shorts has to be diagnosable.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pysrt

from src.config import Config
from src.media_probe import MediaProbe, Sidecar, SubtitleStream, select_stream


class Provenance(StrEnum):
    """Where a usable transcript came from. Recorded for every generated clip."""

    MANUAL_OVERRIDE = "manual_override"
    TEXT_EMBEDDED = "text_embedded"
    TEXT_SIDECAR = "text_sidecar"
    WHISPER_CACHED = "whisper_cached"
    WHISPER_GENERATED = "whisper_generated"
    UNUSABLE = "unusable"


class Reason(StrEnum):
    """
    Why an item has no usable transcript.

    These strings are surfaced in the dashboard and in the audit CSV, so they
    are part of the contract — don't rename them casually.
    """

    OK = "OK"
    NO_SUB_STREAMS = "NO_SUB_STREAMS"
    IMAGE_ONLY = "IMAGE_ONLY"
    NO_TEXT_TRACK = "NO_TEXT_TRACK"
    EXTRACT_FAILED = "EXTRACT_FAILED"
    EMPTY_OUTPUT = "EMPTY_OUTPUT"
    PARSE_FAILED = "PARSE_FAILED"
    TOO_FEW_CUES = "TOO_FEW_CUES"
    TOO_LITTLE_TEXT = "TOO_LITTLE_TEXT"
    LOW_COVERAGE_LIKELY_FORCED = "LOW_COVERAGE_LIKELY_FORCED"
    WHISPER_DISABLED = "WHISPER_DISABLED"
    WHISPER_UNAVAILABLE = "WHISPER_UNAVAILABLE"
    WHISPER_FAILED = "WHISPER_FAILED"


HUMAN_REASONS: dict[Reason, str] = {
    Reason.OK: "usable transcript",
    Reason.NO_SUB_STREAMS: "no subtitle streams and no sidecar file",
    Reason.IMAGE_ONLY: "only image-based subtitles (PGS/VobSub) — needs OCR",
    Reason.NO_TEXT_TRACK: "subtitle streams exist but none are text-based",
    Reason.EXTRACT_FAILED: "ffmpeg could not extract the subtitle track",
    Reason.EMPTY_OUTPUT: "extracted subtitle file was empty",
    Reason.PARSE_FAILED: "extracted subtitle file could not be parsed",
    Reason.TOO_FEW_CUES: "too few subtitle cues to be a real transcript",
    Reason.TOO_LITTLE_TEXT: "subtitle track contains almost no text",
    Reason.LOW_COVERAGE_LIKELY_FORCED: (
        "subtitles stop well before the end — most likely a forced track "
        "covering only foreign-language lines"
    ),
    Reason.WHISPER_DISABLED: "no text subtitles and whisper is disabled",
    Reason.WHISPER_UNAVAILABLE: "no text subtitles and faster-whisper is not installed",
    Reason.WHISPER_FAILED: "whisper transcription failed",
}


@dataclass
class Validation:
    """The outcome of putting a candidate .srt through the gates."""

    ok: bool
    reason: Reason
    cue_count: int = 0
    char_count: int = 0
    coverage: float | None = None

    def describe(self) -> str:
        coverage = "n/a" if self.coverage is None else f"{self.coverage:.0%}"
        return f"{self.reason} (cues={self.cue_count}, coverage={coverage})"


@dataclass
class SubtitleResult:
    """What the acquisition ladder produced for one media item."""

    provenance: Provenance
    reason: Reason
    srt_path: Path | None = None
    stream: SubtitleStream | None = None
    validation: Validation | None = None

    @property
    def ok(self) -> bool:
        return self.srt_path is not None and self.provenance is not Provenance.UNUSABLE

    @property
    def needs_whisper(self) -> bool:
        """True when the only remaining route is transcription."""
        return self.reason in {
            Reason.NO_SUB_STREAMS,
            Reason.IMAGE_ONLY,
            Reason.NO_TEXT_TRACK,
            Reason.LOW_COVERAGE_LIKELY_FORCED,
            Reason.TOO_FEW_CUES,
            Reason.TOO_LITTLE_TEXT,
            Reason.EMPTY_OUTPUT,
            Reason.EXTRACT_FAILED,
            Reason.PARSE_FAILED,
        }


# --------------------------------------------------------------------------
# Validation — the actual verification
# --------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")
_ASS_OVERRIDE = re.compile(r"\{[^}]*\}")


def clean_cue_text(text: str) -> str:
    """Strip markup that carries no dialogue: <i>, {\\an8}, and line breaks."""
    text = _TAG.sub("", text)
    text = _ASS_OVERRIDE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def validate_srt(
    srt_path: Path, duration: float | None, config: Config
) -> Validation:
    """
    Put a candidate transcript through every gate, cheapest first.

    `duration` is the source media's runtime; without it the coverage gate is
    skipped rather than guessed at.
    """
    settings = config.subtitles

    if not srt_path.exists():
        return Validation(False, Reason.EXTRACT_FAILED)

    if srt_path.stat().st_size == 0:
        return Validation(False, Reason.EMPTY_OUTPUT)

    try:
        # pysrt.ERROR_PASS — NOT ERROR_IGNORE, which does not exist and raises
        # AttributeError at parse time.
        subs = pysrt.open(str(srt_path), encoding="utf-8", error_handling=pysrt.ERROR_PASS)
    except UnicodeDecodeError:
        try:
            subs = pysrt.open(
                str(srt_path), encoding="latin-1", error_handling=pysrt.ERROR_PASS
            )
        except Exception:
            return Validation(False, Reason.PARSE_FAILED)
    except Exception:
        return Validation(False, Reason.PARSE_FAILED)

    cues = [cue for cue in subs if clean_cue_text(cue.text)]
    cue_count = len(cues)
    char_count = sum(
        len(re.sub(r"[^A-Za-z]", "", clean_cue_text(cue.text))) for cue in cues
    )

    coverage: float | None = None
    if duration and cues:
        last_end = max(cue.end.ordinal for cue in cues) / 1000.0
        coverage = min(last_end / duration, 1.0)

    # No cues at all is the signature of an image-based track: ffmpeg exits 0
    # and writes a file, but there was never any text in it to convert.
    if cue_count == 0:
        return Validation(False, Reason.EMPTY_OUTPUT, cue_count, char_count, coverage)

    if cue_count < settings.min_cues:
        return Validation(False, Reason.TOO_FEW_CUES, cue_count, char_count, coverage)

    if char_count < settings.min_chars:
        return Validation(
            False, Reason.TOO_LITTLE_TEXT, cue_count, char_count, coverage
        )

    # The forced-track detector. Only applied when the runtime is known.
    if coverage is not None and coverage < settings.min_coverage:
        return Validation(
            False, Reason.LOW_COVERAGE_LIKELY_FORCED, cue_count, char_count, coverage
        )

    return Validation(True, Reason.OK, cue_count, char_count, coverage)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def extract_stream(media_path: Path, stream_index: int, destination: Path) -> bool:
    """
    Convert one embedded subtitle stream to .srt.

    The trailing "?" on the map spec makes the stream optional, so ffmpeg
    doesn't hard-fail when it isn't there. That means success has to be judged
    by inspecting the output — which is what `validate_srt` is for, and why
    image-based tracks (which produce nothing convertible) surface as an empty
    file rather than an error.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(media_path),
        "-map", f"0:{stream_index}?",
        "-c:s", "srt",
        str(destination),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=600, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    return result.returncode == 0 and destination.exists()


def convert_sidecar(sidecar: Path, destination: Path) -> bool:
    """Normalize a sidecar to .srt. A .srt sidecar is simply copied."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if sidecar.suffix.lower() == ".srt":
        try:
            shutil.copyfile(sidecar, destination)
            return True
        except OSError:
            return False

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(sidecar), str(destination),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    return result.returncode == 0 and destination.exists()


# --------------------------------------------------------------------------
# The acquisition ladder
# --------------------------------------------------------------------------


def acquire(
    probe_result: MediaProbe,
    config: Config,
    *,
    override: Path | None = None,
    force_whisper: bool = False,
    allow_whisper: bool = True,
    transcriber=None,
) -> SubtitleResult:
    """
    Find a usable transcript for one media item.

    Tiers, first success wins:

      0. manual override   — an .srt supplied from the dashboard
      1. embedded text     — best-ranked text stream that passes validation
      2. sidecar file      — .srt/.ass sitting next to the media
      3. whisper           — transcription (cached, then generated)

    Every tier is validated with the same gates, so an uploaded file that turns
    out to be junk is rejected exactly like a bad embedded track.

    `transcriber` is injected rather than imported so this module stays usable
    (and testable) without faster-whisper installed.
    """
    media = probe_result.path
    duration = probe_result.duration
    work_dir = config.extracted_subs_dir
    stem = _cache_stem(media)

    # -- tier 0: manual override ------------------------------------------
    if override is not None:
        destination = work_dir / f"{stem}.override.srt"
        if convert_sidecar(override, destination):
            validation = validate_srt(destination, duration, config)
            if validation.ok:
                return SubtitleResult(
                    Provenance.MANUAL_OVERRIDE, Reason.OK, destination,
                    validation=validation,
                )
            # An override that fails validation is worth reporting loudly: the
            # user explicitly supplied it, so silently falling through would be
            # confusing.
            return SubtitleResult(
                Provenance.UNUSABLE, validation.reason, validation=validation
            )

    last_failure: Validation | None = None

    # "Always transcribe this one", set from the dashboard. Skips the embedded
    # and sidecar tiers entirely — the point of the flag is that the text track
    # present in the file is known to be wrong (mistimed, wrong language, or a
    # transcript of a different cut), so validating it again would just accept
    # it again.
    ranked = (
        []
        if force_whisper
        else select_stream(
            probe_result.subtitle_streams, config.subtitles.preferred_languages
        )
    )

    for stream in ranked:
        destination = work_dir / f"{stem}.s{stream.index}.srt"
        if not extract_stream(media, stream.index, destination):
            last_failure = Validation(False, Reason.EXTRACT_FAILED)
            continue

        validation = validate_srt(destination, duration, config)
        if validation.ok:
            return SubtitleResult(
                Provenance.TEXT_EMBEDDED, Reason.OK, destination,
                stream=stream, validation=validation,
            )
        last_failure = validation

    # -- tier 2: sidecars --------------------------------------------------
    sidecars = [] if force_whisper else _rank_sidecars(probe_result.sidecars, config)
    for sidecar in sidecars:
        destination = work_dir / f"{stem}.sidecar{sidecar.path.suffix}.srt"
        if not convert_sidecar(sidecar.path, destination):
            last_failure = Validation(False, Reason.EXTRACT_FAILED)
            continue

        validation = validate_srt(destination, duration, config)
        if validation.ok:
            return SubtitleResult(
                Provenance.TEXT_SIDECAR, Reason.OK, destination,
                validation=validation,
            )
        last_failure = validation

    # -- nothing text-based worked ----------------------------------------
    fallback_reason = _diagnose(probe_result, last_failure)

    if not allow_whisper or not config.whisper.enabled:
        # Report what is actually wrong with the FILE (no subtitle streams, a
        # forced track, image-only subs). "Whisper is disabled" describes our
        # configuration, not the item, and is the less useful of the two.
        return SubtitleResult(
            Provenance.UNUSABLE, fallback_reason, validation=last_failure
        )

    # -- tier 3: whisper ---------------------------------------------------
    if transcriber is None:
        return SubtitleResult(
            Provenance.UNUSABLE, Reason.WHISPER_UNAVAILABLE, validation=last_failure
        )

    transcript, was_cached = transcriber(media)
    if transcript is None:
        return SubtitleResult(
            Provenance.UNUSABLE, Reason.WHISPER_FAILED, validation=last_failure
        )

    validation = validate_srt(transcript, duration, config)
    if not validation.ok:
        return SubtitleResult(
            Provenance.UNUSABLE, validation.reason, validation=validation
        )

    return SubtitleResult(
        Provenance.WHISPER_CACHED if was_cached else Provenance.WHISPER_GENERATED,
        Reason.OK,
        transcript,
        validation=validation,
    )


def diagnose_level1(probe_result: MediaProbe) -> Reason:
    """
    The Level-1 verdict, without extracting anything.

    Used by the audit's fast path (`--sample` / no `--deep`) to report what a
    library looks like before paying for extraction on every file.
    """
    if not probe_result.subtitle_streams and not probe_result.sidecars:
        return Reason.NO_SUB_STREAMS
    if probe_result.has_any_subtitle_candidate:
        return Reason.OK
    if probe_result.image_streams:
        return Reason.IMAGE_ONLY
    return Reason.NO_TEXT_TRACK


def _diagnose(probe_result: MediaProbe, last: Validation | None) -> Reason:
    """Pick the most informative reason when every text route has failed."""
    if last is not None:
        # A real validation failure says more than "no text track".
        return last.reason
    if not probe_result.subtitle_streams and not probe_result.sidecars:
        return Reason.NO_SUB_STREAMS
    if probe_result.image_streams and not probe_result.text_streams:
        return Reason.IMAGE_ONLY
    return Reason.NO_TEXT_TRACK


def _rank_sidecars(sidecars: list[Sidecar], config: Config) -> list[Sidecar]:
    """Preferred-language sidecars first; unlabelled next; others last."""
    preferred = [lang.strip().lower() for lang in config.subtitles.preferred_languages]

    def rank(sidecar: Sidecar) -> tuple[int, str]:
        if sidecar.language and sidecar.language in preferred:
            group = 0
        elif sidecar.language is None:
            group = 1
        else:
            group = 2
        return (group, sidecar.path.name)

    return sorted(sidecars, key=rank)


def _cache_stem(media: Path) -> str:
    """
    A collision-proof, readable cache name.

    Two different shows can both have "S01E01.mkv", so the name alone is not
    enough; a short hash of the full path disambiguates without making the
    cache directory unreadable.
    """
    import hashlib

    digest = hashlib.sha1(str(media.resolve()).encode()).hexdigest()[:10]
    safe = re.sub(r"[^A-Za-z0-9]+", "_", media.stem)[:60].strip("_")
    return f"{safe}.{digest}"
