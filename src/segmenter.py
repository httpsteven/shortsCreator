"""
Multipart selection: turning a pool of matched quotes into a 3-5 part series.

The hard part isn't picking moments, it's picking moments that make sense as a
SERIES. Three rules do the work:

* Two quotes from the same scene would become two near-identical shorts, so
  matches close together are clustered and only the best survives.
* The order quotes appear in quotes.json is not trusted. Matched subtitle time
  is the only ordering authority — the same principle the rest of the pipeline
  runs on.
* A series is renumbered against what actually matched. Asking for 5 and getting
  3 produces "Part 1/3, 2/3, 3/3", never a gap-numbered "Part 1 of 5, Part 4
  of 5".
"""

from __future__ import annotations

from dataclasses import dataclass

from src.clip_extractor import ClipWindow, plan_window
from src.config import Config
from src.subtitle_utils import Match


@dataclass(frozen=True)
class Part:
    """One short within a series."""

    index: int          # 1-based
    total: int
    match: Match
    window: ClipWindow
    label: str | None = None

    @property
    def badge(self) -> str:
        return f"Part {self.index}/{self.total}"

    @property
    def suffix(self) -> str:
        """Filename fragment: 'Part 2 of 4'."""
        return f"Part {self.index} of {self.total}"


@dataclass
class SeriesPlan:
    """The outcome of planning a series — parts, or a reason there are none."""

    parts: list[Part]
    considered: int
    dropped_for_proximity: int
    dropped_for_cap: int
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.parts)


def suppress_nearby(matches: list[Match], min_separation: float) -> tuple[list[Match], int]:
    """
    Keep the best match in each cluster of nearby matches.

    Greedy non-maximum suppression: take the highest-scoring match, discard
    everything within `min_separation` of it, repeat. Working best-first (rather
    than earliest-first) means a strong match is never discarded in favour of a
    weaker one that merely happened to come earlier in the film.
    """
    kept: list[Match] = []
    dropped = 0

    for match in sorted(matches, key=lambda m: (-m.score, m.start)):
        if any(abs(match.start - chosen.start) < min_separation for chosen in kept):
            dropped += 1
            continue
        kept.append(match)

    return kept, dropped


def plan_series(
    matches: list[Match],
    media_duration: float | None,
    config: Config,
    *,
    requested_parts: int | None = None,
    labels: dict[str, str] | None = None,
    cues: list | None = None,
) -> SeriesPlan:
    """
    Build a series from already-accepted matches.

    `matches` must already have cleared the fuzzy threshold — selection here is
    about arrangement, not quality. `requested_parts` caps the series below the
    configured maximum for a single run without changing config.
    """
    settings = config.multipart
    labels = labels or {}
    considered = len(matches)

    if not matches:
        return SeriesPlan([], 0, 0, 0, reason="no quotes matched")

    kept, dropped_for_proximity = suppress_nearby(matches, settings.min_separation)

    cap = min(settings.max_parts, requested_parts or settings.max_parts)
    dropped_for_cap = 0
    if len(kept) > cap:
        # Keep the strongest, then restore chronological order below.
        kept = sorted(kept, key=lambda m: -m.score)[:cap]
        dropped_for_cap = len(matches) - dropped_for_proximity - cap

    if len(kept) < settings.min_parts:
        return SeriesPlan(
            [], considered, dropped_for_proximity, dropped_for_cap,
            reason=(
                f"only {len(kept)} distinct moment(s) matched, "
                f"below multipart.min_parts={settings.min_parts}"
            ),
        )

    # Matched time is the ordering authority, not the order in quotes.json.
    ordered = sorted(kept, key=lambda m: m.start)
    windows = [plan_window(m, media_duration, config, cues) for m in ordered]
    windows = resolve_overlaps(windows)

    total = len(ordered)
    parts = [
        Part(
            index=position,
            total=total,
            match=match,
            window=window,
            label=labels.get(match.quote),
        )
        for position, (match, window) in enumerate(zip(ordered, windows), start=1)
    ]

    return SeriesPlan(parts, considered, dropped_for_proximity, dropped_for_cap)


def resolve_overlaps(windows: list[ClipWindow]) -> list[ClipWindow]:
    """
    Ensure no two parts share footage.

    With the default settings this never fires — min_separation (180s) is far
    wider than max_duration (60s). It exists for configurations that narrow the
    separation, where padding can make adjacent windows touch. Overlapping parts
    would show the viewer the same seconds twice across two shorts.

    The shared region is split down the middle, which trims padding rather than
    dropping a part.
    """
    if len(windows) < 2:
        return windows

    resolved = list(windows)
    for position in range(len(resolved) - 1):
        current, following = resolved[position], resolved[position + 1]
        if current.end <= following.start:
            continue

        midpoint = (current.end + following.start) / 2.0
        resolved[position] = ClipWindow(current.start, round(midpoint, 3))
        resolved[position + 1] = ClipWindow(round(midpoint, 3), following.end)

    return resolved
