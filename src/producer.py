"""
Per-item production: quotes in, finished shorts out.

Separated from the CLI so the same logic serves both a terminal run and the
dashboard's job worker.

Every path out of here carries a reason. A run that produces nothing must be
able to say WHY for each item it passed over — that is the difference between
"tune the threshold" and "your quotes file has the wrong episode titles".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.caption_renderer import extract_thumbnail, render
from src.clip_extractor import plan_window
from src.config import Config
from src.media_probe import ProbeError, probe
from src.quote_finder import Candidate, QuoteStore
from src.segmenter import Part, plan_series
from src.sources.base import MediaItem
from src.state import ClipRecord, State, clip_id, series_id
from src.subtitle_source import HUMAN_REASONS, acquire
from src.subtitle_utils import Match, cues_in_window, find_matches, load_cues


@dataclass
class Outcome:
    """What happened to one item."""

    item: MediaItem
    produced: list[ClipRecord] = field(default_factory=list)
    skipped: str | None = None
    near_misses: list[Match] = field(default_factory=list)
    already_done: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.produced)


def produce(
    item: MediaItem,
    config: Config,
    store: QuoteStore,
    state: State,
    *,
    category: str | None = None,
    multipart: bool = False,
    parts: int | None = None,
    dry_run: bool = False,
    transcriber=None,
    force: bool = False,
    gate=None,
    override: Path | None = None,
    force_whisper: bool = False,
) -> Outcome:
    outcome = Outcome(item=item)

    # -- 1. what are we looking for? ---------------------------------------
    candidates = store.candidates_for(item.lookup_key, category)
    if not candidates:
        resolved = store.resolve(item.lookup_key)
        if resolved is None:
            outcome.skipped = (
                f"no quotes entry matching key {item.lookup_key!r} "
                f"(check the title spelling in quotes.json)"
            )
        elif category:
            available = store.categories_for(item.lookup_key)
            outcome.skipped = (
                f"no quotes in category {category!r}"
                + (f" (available: {', '.join(available)})" if available else "")
            )
        else:
            outcome.skipped = f"entry {resolved!r} exists but contains no quotes"
        return outcome

    # -- 2. get a transcript ----------------------------------------------
    try:
        probed = probe(item.path)
    except ProbeError as exc:
        outcome.skipped = str(exc)
        return outcome

    subtitles = acquire(
        probed, config, transcriber=transcriber,
        override=override, force_whisper=force_whisper,
    )
    if not subtitles.ok:
        outcome.skipped = HUMAN_REASONS.get(subtitles.reason, str(subtitles.reason))
        return outcome

    cues = load_cues(subtitles.srt_path)
    if not cues:
        outcome.skipped = "transcript parsed to zero usable cues"
        return outcome

    # -- 3. locate the quotes ---------------------------------------------
    labels = {c.quote: c.label for c in candidates if c.label}
    categories = {c.quote: c.category for c in candidates}

    accepted, rejected = find_matches(
        cues,
        [c.quote for c in candidates],
        threshold=config.matching.threshold,
        window_max_cues=config.matching.window_max_cues,
    )
    outcome.near_misses = rejected

    if not accepted:
        best = max((m.score for m in rejected), default=0.0)
        outcome.skipped = (
            f"none of {len(candidates)} quote(s) cleared the match threshold "
            f"({config.matching.threshold:g}); best was {best:.0f}"
        )
        return outcome

    # -- 4. arrange into clips --------------------------------------------
    if multipart:
        plan = plan_series(
            accepted, probed.duration, config,
            requested_parts=parts, labels=labels,
        )
        if not plan.ok:
            outcome.skipped = plan.reason
            return outcome
        planned = plan.parts
        identifier = series_id(item.path, [p.match.quote for p in planned])
    else:
        best = max(accepted, key=lambda m: m.score)
        window = plan_window(best, probed.duration, config)
        planned = [Part(index=1, total=1, match=best, window=window)]
        identifier = None

    # -- 5. render ---------------------------------------------------------
    for part in planned:
        identity = clip_id(item.path, part.match.quote)

        if not force and identity in state.clips:
            outcome.already_done += 1
            continue

        stem = item.slug if part.total == 1 else f"{item.slug} - {part.suffix}"
        destination = config.clips_dir / f"{stem}.mp4"

        if dry_run:
            outcome.produced.append(
                _record(
                    identity, item, part, destination, None,
                    subtitles, categories, identifier,
                )
            )
            continue

        window_cues = cues_in_window(cues, part.window.start, part.window.end)
        label = part.badge if part.total > 1 else None

        ok, message = render(
            item.path, part.window, window_cues, destination, config,
            part_label=label,
            ass_path=config.cache_dir / "ass" / f"{stem}.ass",
            gate=gate,
        )
        if not ok:
            outcome.skipped = f"render failed: {message}"
            return outcome

        thumbnail = config.thumbs_dir / f"{stem}.jpg"
        if not extract_thumbnail(destination, thumbnail):
            thumbnail = None

        record = _record(
            identity, item, part, destination, thumbnail,
            subtitles, categories, identifier,
        )
        state.record(record)
        outcome.produced.append(record)

    if identifier and outcome.produced and not dry_run:
        _write_series_sidecar(item, outcome.produced, config)

    return outcome


def _record(
    identity: str,
    item: MediaItem,
    part: Part,
    destination: Path,
    thumbnail: Path | None,
    subtitles,
    categories: dict[str, str | None],
    identifier: str | None,
) -> ClipRecord:
    return ClipRecord(
        clip_id=identity,
        source=str(item.path),
        lookup_key=item.lookup_key,
        quote=part.match.quote,
        category=categories.get(part.match.quote),
        start=part.window.start,
        end=part.window.end,
        score=part.match.score,
        output=str(destination),
        thumbnail=str(thumbnail) if thumbnail else None,
        provenance=str(subtitles.provenance),
        series=identifier,
        part_index=part.index if part.total > 1 else None,
        part_total=part.total if part.total > 1 else None,
    )


def _write_series_sidecar(
    item: MediaItem, records: list[ClipRecord], config: Config
) -> None:
    """
    A manifest for upload time: what order the parts go in, and what each says.

    Filenames alone carry the ordering, but not the quote or the source
    timestamp, which is what you want when writing descriptions.
    """
    destination = config.output_dir / f"{item.slug}.series.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "title": item.display_name,
                "lookup_key": item.lookup_key,
                "source": str(item.path),
                "parts": [
                    {
                        "part": record.part_index,
                        "of": record.part_total,
                        "quote": record.quote,
                        "source_start": record.start,
                        "source_end": record.end,
                        "match_score": record.score,
                        "file": Path(record.output).name,
                    }
                    for record in sorted(records, key=lambda r: r.part_index or 0)
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
