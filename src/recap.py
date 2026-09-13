"""
Recaps and compilations: several moments stitched into one clip.

Two shapes, one mechanism:

* a RECAP draws its moments from one episode — a rundown of that episode;
* a COMPILATION draws one moment from each of many episodes — "top ten cold
  opens", "every time Dewey wins".

Both are a list of moments, each knowing its own source file, laid out on a
single output timeline. The only difference is how the moments are chosen.

Segments are a FIXED length and are not snapped to pauses, unlike standalone
clips. A montage cuts hard by convention, and a fixed length keeps the total
predictable — "a 45 second rundown" should come out at 45 seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.clip_extractor import ClipWindow
from src.config import Config
from src.segmenter import suppress_nearby
from src.subtitle_utils import Match


@dataclass(frozen=True)
class Moment:
    """A matched line, and the file it lives in."""

    source: Path
    label: str
    match: Match
    runtime: float | None = None

    @property
    def start(self) -> float:
        return self.match.start

    @property
    def score(self) -> float:
        return self.match.score


@dataclass(frozen=True)
class Segment:
    """One moment, placed on the output timeline."""

    index: int
    moment: Moment
    window: ClipWindow
    #: Where this segment begins in the FINISHED clip, not in its source.
    offset: float

    @property
    def source(self) -> Path:
        return self.moment.source

    @property
    def duration(self) -> float:
        return self.window.duration

    @property
    def match(self) -> Match:
        return self.moment.match


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
        """
        Length of the finished clip.

        Measured from the last segment's position rather than by summing, since
        a crossfade overlaps its neighbours and the sum would over-count every
        overlap.
        """
        if not self.segments:
            return 0.0
        last = self.segments[-1]
        return last.offset + last.duration


def select_by_beats(moments: list[Moment], count: int, runtime: float) -> list[Moment]:
    """
    Pick the best moment from each act, rather than the best N overall.

    Ranking purely by score clusters wherever the strongest lines happen to
    fall — usually the middle, because that is where most of an episode is. A
    rundown that skips the opening and the ending is not a rundown.

    So the runtime is divided into `count` equal stretches and the best moment
    in each is taken. Empty stretches give their slot back to the strongest
    moments left over, which is what happens when an episode simply has nothing
    quotable in its third act.
    """
    if runtime <= 0 or count <= 0:
        return []

    buckets: dict[int, list[Moment]] = {}
    for moment in moments:
        index = min(int(moment.start / runtime * count), count - 1)
        buckets.setdefault(index, []).append(moment)

    chosen: list[Moment] = []
    for index in range(count):
        candidates = buckets.get(index)
        if candidates:
            chosen.append(max(candidates, key=lambda m: m.score))

    # Backfill from whatever is left, strongest first.
    if len(chosen) < count:
        taken = {id(m) for m in chosen}
        leftovers = sorted(
            (m for m in moments if id(m) not in taken),
            key=lambda m: -m.score,
        )
        chosen += leftovers[: count - len(chosen)]

    return sorted(chosen, key=lambda m: m.start)


def plan_recap(
    moments: list[Moment],
    config: Config,
    *,
    runtime: float | None = None,
    target_duration: float | None = None,
    segments: int | None = None,
    spread: str = "beats",
    suppress: bool = True,
) -> RecapPlan:
    """
    Choose the moments and lay them out on the output timeline.

    `spread="beats"` divides one episode's runtime into acts and takes the best
    of each. `spread="score"` simply ranks — which is what a compilation wants,
    since its moments come from different files and their timestamps are not
    comparable.

    Each segment ends just after its line, exactly as a standalone clip does:
    the moment should land, not trail off.
    """
    settings = config.recap
    total = target_duration or settings.duration

    if not moments:
        return RecapPlan(reason="no quotes matched")

    if suppress:
        kept_matches, _ = suppress_nearby(
            [m.match for m in moments], settings.min_separation
        )
        keep = {id(match) for match in kept_matches}
        pool = [m for m in moments if id(m.match) in keep]
    else:
        pool = list(moments)

    if len(pool) < settings.min_segments:
        return RecapPlan(
            considered=len(moments),
            reason=(
                f"only {len(pool)} distinct moment(s) matched, below "
                f"recap.min_segments={settings.min_segments}. Write more quotes."
            ),
        )

    wanted = min(segments or settings.max_segments, settings.max_segments)
    affordable = int(total // settings.segment_min)

    # Clamp DOWN to what fits, then check the floor. Forcing the count up to
    # min_segments first would make the check unreachable and silently ship
    # four two-second flashes instead of refusing.
    count = min(wanted, len(pool), affordable)

    if count < settings.min_segments:
        return RecapPlan(
            considered=len(moments),
            reason=(
                f"{total:g}s only fits {affordable} segment(s) of "
                f"{settings.segment_min:g}s, below "
                f"recap.min_segments={settings.min_segments}"
            ),
        )

    if spread == "beats" and runtime:
        chosen = select_by_beats(pool, count, runtime)
    else:
        chosen = sorted(pool, key=lambda m: -m.score)[:count]
        chosen = sorted(chosen, key=lambda m: (str(m.source), m.start))

    each = total / len(chosen)

    # A crossfade overlaps each pair, so every segment after the first starts
    # earlier than the plain running total. Getting this wrong puts every
    # caption after the first segment at the wrong moment — which is the whole
    # reason the offset lives here rather than being recomputed at render time.
    overlap = (
        settings.transition_duration if settings.transition == "crossfade" else 0.0
    )

    built: list[Segment] = []
    offset = 0.0
    for index, moment in enumerate(chosen, start=1):
        end = moment.match.end + config.clip.pad_after
        if moment.runtime:
            end = min(moment.runtime, end)
        start = max(0.0, end - each)

        window = ClipWindow(start=round(start, 3), end=round(end, 3))
        built.append(
            Segment(index=index, moment=moment, window=window, offset=round(offset, 3))
        )
        offset += window.duration - overlap

    return RecapPlan(segments=built, considered=len(moments))
