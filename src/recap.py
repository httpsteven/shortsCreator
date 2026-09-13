"""
Recaps: several moments stitched into one clip.

A multipart series gives you N separate files. A recap is one file that cuts
between N moments — a rundown of an episode rather than a single joke.

The selection rules are the same ones the series planner uses, because the
problems are the same: moments too close together are near-duplicates, and the
order in quotes.json is not evidence of anything. Matched subtitle time is the
only ordering authority here as everywhere else.

Segments are a FIXED length and are not snapped to pauses, unlike standalone
clips. Two reasons: a montage cuts hard by convention, and a fixed length keeps
the total predictable — "a 45 second rundown" should come out at 45 seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.clip_extractor import ClipWindow
from src.config import Config
from src.segmenter import suppress_nearby
from src.subtitle_utils import Match


@dataclass(frozen=True)
class Segment:
    """One moment inside a recap."""

    index: int
    match: Match
    window: ClipWindow
    #: Where this segment begins in the FINISHED clip, not in the source.
    offset: float

    @property
    def duration(self) -> float:
        return self.window.duration


@dataclass
class RecapPlan:
    segments: list[Segment] = field(default_factory=list)
    considered: int = 0
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.segments)

    @property
    def duration(self) -> float:
        return sum(segment.duration for segment in self.segments)


def plan_recap(
    matches: list[Match],
    media_duration: float | None,
    config: Config,
    *,
    target_duration: float | None = None,
    segments: int | None = None,
) -> RecapPlan:
    """
    Choose the moments for a rundown and lay them out on the output timeline.

    Each segment ends just after its line, exactly as a standalone clip does —
    the moment should land, not trail off — and the preceding seconds carry
    whatever setup fits.
    """
    settings = config.recap
    total = target_duration or settings.duration
    upper = media_duration or 0.0

    if not matches:
        return RecapPlan(reason="no quotes matched")

    kept, _dropped = suppress_nearby(matches, settings.min_separation)

    if len(kept) < settings.min_segments:
        return RecapPlan(
            considered=len(matches),
            reason=(
                f"only {len(kept)} distinct moment(s) matched, below "
                f"recap.min_segments={settings.min_segments}. Write more "
                f"quotes for this title."
            ),
        )

    wanted = min(segments or settings.max_segments, settings.max_segments)

    # A 45-second recap cannot hold ten moments without each becoming a flash.
    # Fit is decided by segment_min, and the strongest matches survive.
    affordable = int(total // settings.segment_min)

    # Note the order: clamp DOWN to what fits, then check the floor. Forcing
    # the count up to min_segments first would make the check unreachable and
    # silently ship four two-second flashes instead of refusing.
    count = min(wanted, len(kept), affordable)

    if count < settings.min_segments:
        return RecapPlan(
            considered=len(matches),
            reason=(
                f"{total:g}s only fits {affordable} segment(s) of "
                f"{settings.segment_min:g}s, below "
                f"recap.min_segments={settings.min_segments}"
            ),
        )

    chosen = sorted(sorted(kept, key=lambda m: -m.score)[:count], key=lambda m: m.start)
    each = total / len(chosen)

    built: list[Segment] = []
    offset = 0.0
    for index, match in enumerate(chosen, start=1):
        end = match.end + config.clip.pad_after
        if upper:
            end = min(upper, end)
        start = max(0.0, end - each)

        window = ClipWindow(start=round(start, 3), end=round(end, 3))
        built.append(Segment(index=index, match=match, window=window, offset=offset))
        offset += window.duration

    return RecapPlan(segments=built, considered=len(matches))
