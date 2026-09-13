"""
ffprobe wrapper and subtitle-stream classification.

This is Level 1 of the "do I have subtitles?" question: what the container
*declares*. It is deliberately cheap — metadata only, no decoding — and it is
deliberately not trusted on its own. A declared stream is not a usable one;
`subtitle_source` proves that separately.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Codecs that convert cleanly to .srt, and therefore carry the text we need to
# match quotes against.
TEXT_SUBTITLE_CODECS = {
    "subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "eia_608", "subviewer",
}

# Codecs that are pictures of text. They have perfectly good timing but no
# readable string, so they need OCR before this pipeline can use them.
IMAGE_SUBTITLE_CODECS = {
    "hdmv_pgs_subtitle", "pgssub", "dvd_subtitle", "dvdsub",
    "dvb_subtitle", "dvbsub", "xsub",
}

SIDECAR_EXTENSIONS = (".srt", ".ass", ".ssa", ".vtt")


class ProbeError(Exception):
    """ffprobe could not read the file at all."""


@dataclass(frozen=True)
class SubtitleStream:
    index: int
    codec: str
    language: str | None
    title: str | None
    forced: bool
    default: bool

    @property
    def is_text(self) -> bool:
        return self.codec in TEXT_SUBTITLE_CODECS

    @property
    def is_image(self) -> bool:
        return self.codec in IMAGE_SUBTITLE_CODECS

    def describe(self) -> str:
        bits = [f"#{self.index}", self.codec]
        if self.language:
            bits.append(self.language)
        if self.forced:
            bits.append("forced")
        return " ".join(bits)


@dataclass(frozen=True)
class Sidecar:
    path: Path
    language: str | None


@dataclass
class MediaProbe:
    path: Path
    duration: float | None
    subtitle_streams: list[SubtitleStream] = field(default_factory=list)
    sidecars: list[Sidecar] = field(default_factory=list)

    @property
    def text_streams(self) -> list[SubtitleStream]:
        return [s for s in self.subtitle_streams if s.is_text]

    @property
    def image_streams(self) -> list[SubtitleStream]:
        return [s for s in self.subtitle_streams if s.is_image]

    @property
    def has_any_subtitle_candidate(self) -> bool:
        return bool(self.text_streams or self.sidecars)


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def probe(path: Path, *, find_sidecars: bool = True) -> MediaProbe:
    """
    Read a file's duration and subtitle streams.

    Raises ProbeError when ffprobe fails — that means the file is unreadable or
    not media, which is a different problem from "has no subtitles" and must not
    be conflated with it.
    """
    raw = _run_ffprobe(path)

    duration = _parse_duration(raw)
    streams = [
        _parse_subtitle_stream(s)
        for s in raw.get("streams", [])
        if s.get("codec_type") == "subtitle"
    ]

    sidecars = _find_sidecars(path) if find_sidecars else []
    return MediaProbe(
        path=path, duration=duration, subtitle_streams=streams, sidecars=sidecars
    )


def _run_ffprobe(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        # stream_tags and stream_disposition are what make language-aware
        # selection and forced-track detection possible.
        "stream=index,codec_type,codec_name:stream_tags=language,title"
        ":stream_disposition=forced,default:format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except FileNotFoundError as exc:
        raise ProbeError("ffprobe not found on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"ffprobe timed out reading {path.name}.") from exc

    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit {result.returncode}"
        raise ProbeError(f"ffprobe failed on {path.name}: {reason}")

    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned unparseable JSON for {path.name}.") from exc


def _parse_duration(raw: dict) -> float | None:
    value = (raw.get("format") or {}).get("duration")
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def _parse_subtitle_stream(raw: dict) -> SubtitleStream:
    tags = raw.get("tags") or {}
    disposition = raw.get("disposition") or {}

    language = (tags.get("language") or "").strip().lower() or None
    # "und" is ffmpeg's explicit "unknown"; treating it as a real tag would
    # make it look like a language we could match against.
    if language == "und":
        language = None

    return SubtitleStream(
        index=int(raw.get("index", -1)),
        codec=(raw.get("codec_name") or "").strip().lower(),
        language=language,
        title=(tags.get("title") or "").strip() or None,
        forced=bool(disposition.get("forced")),
        default=bool(disposition.get("default")),
    )


def _find_sidecars(path: Path) -> list[Sidecar]:
    """
    Subtitle files sitting beside the media: "Movie.srt", "Movie.en.srt".

    Matched by stem prefix rather than exact stem so language-suffixed names are
    picked up, with the suffix read back as the language tag.
    """
    found: list[Sidecar] = []
    stem = path.stem.lower()

    try:
        neighbours = list(path.parent.iterdir())
    except OSError:
        return found

    for candidate in sorted(neighbours):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in SIDECAR_EXTENSIONS:
            continue

        candidate_stem = candidate.stem.lower()
        if candidate_stem == stem:
            found.append(Sidecar(path=candidate, language=None))
        elif candidate_stem.startswith(stem + "."):
            # "Movie.en.srt" -> "en"; "Movie.forced.eng.srt" -> "eng"
            suffix = candidate_stem[len(stem) + 1 :]
            parts = [p for p in suffix.split(".") if p]
            language = parts[-1] if parts else None
            found.append(Sidecar(path=candidate, language=language))

    return found


def select_stream(
    streams: list[SubtitleStream], preferred_languages: list[str]
) -> list[SubtitleStream]:
    """
    Rank text subtitle streams best-first.

    Never hardcodes 0:s:0. Order of preference:
      1. a preferred language, not forced
      2. no language tag at all, not forced  (very common on single-language rips)
      3. any other language, not forced
      4. anything forced, last

    Forced tracks sink to the bottom because they only carry foreign-language
    lines — they parse perfectly and match almost nothing.
    """
    preferred = [lang.strip().lower() for lang in preferred_languages]

    def rank(stream: SubtitleStream) -> tuple[int, int, int]:
        if stream.forced:
            group = 3
        elif stream.language and stream.language in preferred:
            group = 0
        elif stream.language is None:
            group = 1
        else:
            group = 2
        # Prefer the default track within a group, then earliest index, so the
        # ordering is stable across runs.
        return (group, 0 if stream.default else 1, stream.index)

    return sorted([s for s in streams if s.is_text], key=rank)
