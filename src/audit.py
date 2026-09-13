"""
Library-wide subtitle audit.

Answers "do I have subtitles?" for every item before any clipping is attempted,
and says what it would take to fix the ones that don't. Two depths:

* shallow — metadata only. Fast enough to sweep a whole library, and enough to
  see the shape of the problem.
* deep — actually extracts and validates each track. Slower, and the only way to
  catch a track that LOOKS fine and isn't (the forced-track case).

Whisper is never run from here. The audit's job is to tell you what it would
cost before you commit to it.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

from src.config import Config
from src.media_probe import MediaProbe, ProbeError, probe, select_stream
from src.sources.base import MediaItem
from src.subtitle_source import (
    HUMAN_REASONS,
    Provenance,
    Reason,
    Validation,
    convert_sidecar,
    diagnose_level1,
    sample_track,
    validate_srt,
)

# Rough transcription cost, as a multiple of real time. distil-large-v3 in int8
# on a GTX 1080 lands around 10x faster than realtime with VAD trimming silence;
# CPU-only is closer to 1x. Used only to give an order-of-magnitude estimate
# before committing to a batch, never to make decisions.
GPU_REALTIME_FACTOR = 0.10
CPU_REALTIME_FACTOR = 1.0


@dataclass
class AuditRow:
    item: MediaItem
    duration: float | None = None
    stream_count: int = 0
    codecs: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    forced_flags: int = 0
    sidecars: int = 0

    level1: Reason = Reason.NO_SUB_STREAMS
    level2: Reason | None = None
    provenance: Provenance | None = None
    cue_count: int = 0
    coverage: float | None = None
    error: str | None = None

    @property
    def usable(self) -> bool:
        if self.error:
            return False
        if self.level2 is not None:
            return self.level2 is Reason.OK
        return self.level1 is Reason.OK

    @property
    def needs_whisper(self) -> bool:
        return not self.usable and not self.error

    @property
    def whisper_minutes(self) -> float:
        if not self.needs_whisper or not self.duration:
            return 0.0
        return (self.duration / 60.0) * GPU_REALTIME_FACTOR

    @property
    def reason(self) -> Reason | None:
        return self.level2 if self.level2 is not None else self.level1

    def explain(self) -> str:
        if self.error:
            return self.error
        reason = self.reason
        return HUMAN_REASONS.get(reason, str(reason)) if reason else ""


def audit_item(item: MediaItem, config: Config, *, deep: bool) -> AuditRow:
    """Audit one file. Never raises — a probe failure becomes a row with an error."""
    row = AuditRow(item=item)

    try:
        result: MediaProbe = probe(item.path)
    except ProbeError as exc:
        row.error = str(exc)
        return row

    row.duration = result.duration
    row.stream_count = len(result.subtitle_streams)
    row.codecs = sorted({s.codec for s in result.subtitle_streams})
    row.languages = sorted({s.language or "und" for s in result.subtitle_streams})
    row.forced_flags = sum(1 for s in result.subtitle_streams if s.forced)
    row.sidecars = len(result.sidecars)
    row.level1 = diagnose_level1(result)

    if not deep:
        return row

    validation, provenance = deep_check(result, config)
    row.level2 = validation.reason
    row.provenance = provenance
    row.cue_count = validation.cue_count
    row.coverage = validation.coverage
    return row


def deep_check(
    result: MediaProbe, config: Config
) -> tuple[Validation, Provenance]:
    """
    Prove whether an item has a usable transcript — without reading whole files.

    Embedded tracks are SAMPLED (see subtitle_source.sample_track): a full
    extraction demuxes the entire file, which across a library means hundreds of
    gigabytes of reading and an evening of the machine sitting at 5% CPU and 30%
    iowait.

    Sidecars are validated in full, because they're already small text files
    sitting on disk — there's nothing to save.

    Whisper is never invoked. The audit reports what transcription WOULD cost;
    it doesn't silently spend an hour of GPU time behind a progress bar.
    """
    worst: Validation | None = None

    # -- embedded text streams, sampled --------------------------------
    ranked = select_stream(
        result.subtitle_streams, config.subtitles.preferred_languages
    )
    for stream in ranked:
        if not result.duration:
            # No runtime means no window positions and no coverage. Rare enough
            # to simply report rather than fall back to a full read.
            worst = Validation(False, Reason.PARSE_FAILED)
            break

        validation = sample_track(
            result.path, stream.index, result.duration, config
        )
        if validation.ok:
            return validation, Provenance.TEXT_EMBEDDED
        worst = validation

    # -- sidecars, validated in full -----------------------------------
    for sidecar in result.sidecars:
        destination = (
            config.extracted_subs_dir / f"audit.{sidecar.path.stem}.srt"
        )
        if not convert_sidecar(sidecar.path, destination):
            worst = worst or Validation(False, Reason.EXTRACT_FAILED)
            continue

        validation = validate_srt(destination, result.duration, config)
        destination.unlink(missing_ok=True)
        if validation.ok:
            return validation, Provenance.TEXT_SIDECAR
        worst = validation

    if worst is not None:
        return worst, Provenance.UNUSABLE

    # Nothing text-based to try at all — the Level 1 verdict is the answer.
    if result.image_streams and not result.text_streams:
        return Validation(False, Reason.IMAGE_ONLY), Provenance.UNUSABLE
    if not result.subtitle_streams and not result.sidecars:
        return Validation(False, Reason.NO_SUB_STREAMS), Provenance.UNUSABLE
    return Validation(False, Reason.NO_TEXT_TRACK), Provenance.UNUSABLE


def audit_library(
    items: list[MediaItem],
    config: Config,
    *,
    deep: bool = False,
    sample: int | None = None,
    seed: int | None = None,
    on_progress=None,
) -> list[AuditRow]:
    selected = list(items)
    if sample is not None and sample < len(selected):
        selected = random.Random(seed).sample(selected, sample)
        selected.sort(key=lambda item: str(item.path))

    rows: list[AuditRow] = []
    for position, item in enumerate(selected, start=1):
        rows.append(audit_item(item, config, deep=deep))
        if on_progress:
            on_progress(position, len(selected), item)
    return rows


def write_csv(rows: list[AuditRow], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "path", "kind", "display_name", "lookup_key", "duration_seconds",
            "sub_streams", "codecs", "languages", "forced_streams", "sidecars",
            "level1_status", "level2_status", "reason", "explanation",
            "provenance", "cue_count", "coverage_pct", "usable",
            "needs_whisper", "est_whisper_minutes", "error",
        ])
        for row in rows:
            writer.writerow([
                row.item.path,
                row.item.kind,
                row.item.display_name,
                row.item.lookup_key,
                f"{row.duration:.1f}" if row.duration else "",
                row.stream_count,
                "|".join(row.codecs),
                "|".join(row.languages),
                row.forced_flags,
                row.sidecars,
                row.level1,
                row.level2 or "",
                row.reason or "",
                row.explain(),
                row.provenance or "",
                row.cue_count,
                f"{row.coverage * 100:.0f}" if row.coverage is not None else "",
                "yes" if row.usable else "no",
                "yes" if row.needs_whisper else "no",
                f"{row.whisper_minutes:.1f}" if row.needs_whisper else "",
                row.error or "",
            ])


def summarize(rows: list[AuditRow], *, deep: bool) -> str:
    """A console summary that makes the transcription bill visible up front."""
    if not rows:
        return "  No items found."

    total = len(rows)
    usable = sum(1 for row in rows if row.usable)
    errors = sum(1 for row in rows if row.error)
    needing = [row for row in rows if row.needs_whisper]

    by_reason: dict[str, int] = {}
    for row in rows:
        if row.usable:
            continue
        key = str(row.reason or "UNKNOWN")
        by_reason[key] = by_reason.get(key, 0) + 1

    lines = [
        f"  Scanned {total} item(s)  ({'deep' if deep else 'shallow'} audit)",
        f"    usable subtitles : {usable} ({usable / total:.0%})",
        f"    need another route: {len(needing)}",
    ]
    if errors:
        lines.append(f"    unreadable files : {errors}")

    if by_reason:
        lines.append("")
        lines.append("  Why items aren't usable:")
        for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            explanation = HUMAN_REASONS.get(Reason(reason), "") if reason in Reason.__members__ else ""
            lines.append(f"    {count:>5}  {reason:<28} {explanation}")

    if needing:
        minutes = sum(row.whisper_minutes for row in needing)
        lines += [
            "",
            f"  Transcribing those {len(needing)} item(s) would take roughly "
            f"{minutes / 60:.1f} GPU-hour(s)",
            "    (rough estimate at ~10x realtime; measure on the real box)",
        ]

    if not deep:
        lines += [
            "",
            "  This was a shallow audit — it trusts what the container declares.",
            "  Re-run with --deep to actually extract and validate each track;",
            "  that is the only way a forced track is caught.",
        ]

    return "\n".join(lines)
