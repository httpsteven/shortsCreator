"""
Configuration: typed dataclasses loaded from config.yaml.

Two deliberate choices here:

* Every section is optional and every field has a default, so a minimal
  config.yaml (just roots + output) works and the file stays readable.
* Unknown keys are a hard error rather than being ignored. A typo like
  `min_seperation` would otherwise silently fall back to the default and
  produce baffling behaviour hours later.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised with an actionable message; the CLI prints it without a traceback."""


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


@dataclass
class Roots:
    movies: Path | None = None
    tv: Path | None = None


@dataclass
class Subtitles:
    preferred_languages: list[str] = field(
        default_factory=lambda: ["eng", "en", "english"]
    )
    min_cues: int = 20
    min_chars: int = 500
    min_coverage: float = 0.5


@dataclass
class Matching:
    threshold: float = 72.0
    window_max_cues: int = 3


@dataclass
class Clip:
    min_duration: float = 20.0
    max_duration: float = 60.0
    pad_before: float = 1.5
    pad_after: float = 2.0

    # Where the matched line sits in the finished clip.
    #   "punchline" — the line lands near the END, with setup running into it.
    #   "center"    — the line sits in the middle (the old behaviour).
    # Punchline is the default because that is what a quote usually IS: you
    # write down the line that lands, not the line that sets it up.
    anchor: str = "punchline"

    # A start is snapped to a gap in the dialogue at least this long, so clips
    # begin at a natural beat instead of halfway through someone's sentence.
    boundary_gap: float = 0.8

    # Ignore matches starting before this point — opening credits, recaps,
    # "previously on". 0 disables it.
    skip_first_seconds: float = 0.0

    # Start clips at a real shot change rather than a pause in the dialogue.
    # Costs a short decode of the few seconds where the clip might begin.
    scene_detect: bool = True
    scene_threshold: float = 0.3


@dataclass
class Video:
    width: int = 1080
    height: int = 1920
    blur_sigma: float = 20.0

    # Aspect ratio of the sharp centre panel, as "W:H". The source is cropped
    # to fill it, so 4:5 from a 16:9 original loses about half the frame width
    # in exchange for a much larger subject — which is the trade the format
    # wants, since a full-width 16:9 strip on a phone is a small picture
    # surrounded by blur. Set to null to fit the full width and crop nothing.
    foreground_aspect: str | None = "4:5"
    encoder: str = "libx264"
    crf: int = 18
    preset: str = "medium"
    nvenc_cq: int = 19
    nvenc_preset: str = "p5"


@dataclass
class CaptionStyle:
    font: str = "Arial"
    font_size: int = 64
    primary_colour: str = "&H00FFFFFF"
    outline_colour: str = "&H00000000"
    back_colour: str = "&H80000000"
    bold: bool = True
    border_style: int = 1
    outline: float = 4.0
    shadow: float = 1.0
    alignment: int = 2
    margin_v: int = 320
    margin_l: int = 60
    margin_r: int = 60


@dataclass
class PartBadgeStyle:
    enabled: bool = True
    font: str = "Arial"
    font_size: int = 44
    primary_colour: str = "&H00FFFFFF"
    outline_colour: str = "&H00000000"
    back_colour: str = "&H00000000"
    bold: bool = True
    border_style: int = 1
    outline: float = 3.0
    shadow: float = 0.0
    alignment: int = 9
    margin_v: int = 90
    margin_l: int = 50
    margin_r: int = 50


@dataclass
class Multipart:
    min_parts: int = 3
    max_parts: int = 5
    min_separation: float = 180.0


@dataclass
class Recap:
    """A single clip that stitches several moments into a rundown."""

    duration: float = 45.0
    min_segments: int = 4
    max_segments: int = 8
    # Below this a segment is a flash rather than a moment.
    segment_min: float = 4.0
    # Moments closer together than this are near-duplicates in a recap.
    min_separation: float = 20.0

    # How one moment gives way to the next.
    #   none      - hard cut
    #   dip       - a quick fade through black (default)
    #   crossfade - the two moments dissolve into each other
    #
    # dip is the default because a montage cuts between unrelated scenes, and
    # dissolving two unrelated shots together reads as mush rather than as a
    # transition. It also leaves the timeline untouched, so captions cannot
    # drift; crossfade overlaps the segments and shortens the total.
    transition: str = "dip"
    transition_duration: float = 0.35


@dataclass
class Whisper:
    enabled: bool = True
    model: str = "distil-large-v3"
    device: str = "cuda"
    # Not float16: Pascal runs fp16 at 1/64 rate, so it is slower than int8 here.
    compute_type: str = "int8_float32"
    cpu_compute_type: str = "int8"
    beam_size: int = 5
    vad_filter: bool = True
    chunk_seconds: float = 600.0


@dataclass
class YieldToViewers:
    enabled: bool = True
    pause_on_any_stream: bool = True
    resume_after_idle: float = 300.0
    heartbeat_stale_after: float = 60.0


@dataclass
class Worker:
    # Deliberately serial. This machine is also the Plex server on four shared
    # cores, and gblur at 1080x1920 is memory-bandwidth bound — a second
    # concurrent render buys little and costs playback headroom. Politeness
    # settings rather than parallelism are what matter here.
    nice: int = 10
    ionice_class: int = 3


@dataclass
class Quotes:
    path: Path = Path("quotes.json")
    categories: Path = Path("categories.yaml")


@dataclass
class Config:
    roots: Roots = field(default_factory=Roots)
    output_dir: Path = Path("./output")
    cache_dir: Path = Path("~/.cache/shortscreator")
    db_path: Path | None = None

    quotes: Quotes = field(default_factory=Quotes)
    subtitles: Subtitles = field(default_factory=Subtitles)
    matching: Matching = field(default_factory=Matching)
    clip: Clip = field(default_factory=Clip)
    video: Video = field(default_factory=Video)
    captions: CaptionStyle = field(default_factory=CaptionStyle)
    part_badge: PartBadgeStyle = field(default_factory=PartBadgeStyle)
    multipart: Multipart = field(default_factory=Multipart)
    recap: Recap = field(default_factory=Recap)
    whisper: Whisper = field(default_factory=Whisper)
    yield_to_viewers: YieldToViewers = field(default_factory=YieldToViewers)
    worker: Worker = field(default_factory=Worker)

    # Convenience paths derived from output_dir, so callers don't rebuild them.
    @property
    def clips_dir(self) -> Path:
        return self.output_dir / "clips"

    @property
    def thumbs_dir(self) -> Path:
        return self.output_dir / "thumbs"

    @property
    def transcripts_dir(self) -> Path:
        return self.cache_dir / "transcripts"

    @property
    def extracted_subs_dir(self) -> Path:
        return self.cache_dir / "subtitles"

    @property
    def uploads_dir(self) -> Path:
        """Caption sources uploaded from the dashboard."""
        return self.cache_dir / "uploads"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _expand(value: str | os.PathLike[str]) -> Path:
    """`~` and `$VARS` both appear in hand-written configs; honour both."""
    return Path(os.path.expandvars(str(value))).expanduser()


def _build(cls: type, raw: Any, where: str) -> Any:
    """
    Construct a dataclass from a mapping, rejecting unknown keys.

    Path-typed fields are expanded here rather than at every use site.
    """
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")

    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        known = ", ".join(sorted(fields))
        raise ConfigError(
            f"{where}: unknown option(s) {sorted(unknown)}. Valid keys: {known}"
        )

    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        annotation = str(fields[key].type)
        kwargs[key] = _expand(value) if "Path" in annotation else value
    return cls(**kwargs)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """
    Load config.yaml.

    Falls back to config.example.yaml's defaults (i.e. the dataclass defaults)
    for anything absent, so the file only needs to carry what differs.
    """
    config_path = Path(path) if path else Path("config.yaml")
    if not config_path.exists():
        raise ConfigError(
            f"No config at {config_path}. Copy config.example.yaml to "
            f"{config_path} and set at least `roots` and `output`."
        )

    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: expected a mapping at the top level.")

    top_level = {
        "roots", "output", "cache", "database", "quotes", "subtitles",
        "matching", "clip", "video", "captions", "part_badge", "multipart",
        "recap",
        "whisper", "yield_to_viewers", "worker",
    }
    unknown = set(raw) - top_level
    if unknown:
        raise ConfigError(
            f"{config_path}: unknown top-level section(s) {sorted(unknown)}. "
            f"Valid sections: {', '.join(sorted(top_level))}"
        )

    output = raw.get("output") or {}
    cache = raw.get("cache") or {}
    database = raw.get("database") or {}

    config = Config(
        roots=_build(Roots, raw.get("roots"), "roots"),
        output_dir=_expand(output.get("dir", "./output")),
        cache_dir=_expand(cache.get("dir", "~/.cache/shortscreator")),
        # Env override so the dashboard's DB can move without editing the file.
        db_path=_expand(
            os.environ.get("SHORTS_DB_PATH") or database.get("path") or ""
        )
        if (os.environ.get("SHORTS_DB_PATH") or database.get("path"))
        else None,
        quotes=_build(Quotes, raw.get("quotes"), "quotes"),
        subtitles=_build(Subtitles, raw.get("subtitles"), "subtitles"),
        matching=_build(Matching, raw.get("matching"), "matching"),
        clip=_build(Clip, raw.get("clip"), "clip"),
        video=_build(Video, raw.get("video"), "video"),
        captions=_build(CaptionStyle, raw.get("captions"), "captions"),
        part_badge=_build(PartBadgeStyle, raw.get("part_badge"), "part_badge"),
        multipart=_build(Multipart, raw.get("multipart"), "multipart"),
        recap=_build(Recap, raw.get("recap"), "recap"),
        whisper=_build(Whisper, raw.get("whisper"), "whisper"),
        yield_to_viewers=_build(
            YieldToViewers, raw.get("yield_to_viewers"), "yield_to_viewers"
        ),
        worker=_build(Worker, raw.get("worker"), "worker"),
    )

    _validate(config)
    return config


def _validate(config: Config) -> None:
    """Catch the misconfigurations that would otherwise fail deep in ffmpeg."""
    if config.roots.movies is None and config.roots.tv is None:
        raise ConfigError(
            "No library roots configured. Set roots.movies and/or roots.tv."
        )

    if config.clip.min_duration > config.clip.max_duration:
        raise ConfigError(
            f"clip.min_duration ({config.clip.min_duration}s) exceeds "
            f"clip.max_duration ({config.clip.max_duration}s)."
        )

    if not 0 <= config.matching.threshold <= 100:
        raise ConfigError(
            f"matching.threshold must be 0-100, got {config.matching.threshold}."
        )

    if config.multipart.min_parts > config.multipart.max_parts:
        raise ConfigError(
            f"multipart.min_parts ({config.multipart.min_parts}) exceeds "
            f"multipart.max_parts ({config.multipart.max_parts})."
        )

    if config.video.encoder not in {"libx264", "h264_nvenc"}:
        raise ConfigError(
            f"video.encoder must be libx264 or h264_nvenc, "
            f"got {config.video.encoder!r}."
        )

    # Odd dimensions break libx264, and the blur-fill chain assumes even ones.
    if config.video.width % 2 or config.video.height % 2:
        raise ConfigError(
            f"video dimensions must both be even "
            f"({config.video.width}x{config.video.height})."
        )

    if config.whisper.compute_type == "float16" and config.whisper.device == "cuda":
        # Not fatal — someone may be on Ampere or newer, where fp16 is correct.
        # But on the Pascal card this project targets it is a large regression,
        # so it should never pass silently.
        print(
            "  warning: whisper.compute_type=float16 on CUDA. On Pascal cards "
            "(GTX 10xx) fp16 runs at 1/64 rate and is much slower than "
            "int8_float32. Only correct on Turing or newer."
        )
