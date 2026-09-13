"""
The media-source seam.

Discovery is filesystem-based today. Everything downstream — auditing, matching,
cutting, rendering — consumes `MediaItem` and never touches a path layout or an
API, so a Plex-backed source can be added later as one new class implementing
`MediaSource` without changes anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Protocol, runtime_checkable

from src.naming import build_movie_key, build_tv_key

MediaKind = Literal["movie", "episode"]


@dataclass(frozen=True)
class MediaItem:
    """One playable file, with whatever identity could be parsed for it."""

    kind: MediaKind
    path: Path

    # For a movie this is the film's title; for an episode, the episode title
    # (which may be empty — plenty of libraries don't carry one).
    title: str

    show: str | None = None
    season: int | None = None
    episode: int | None = None
    year: int | None = None

    @property
    def lookup_key(self) -> str:
        """The quotes.json key for this item. Always built via `naming`."""
        if self.kind == "episode":
            return build_tv_key(
                self.show or "",
                self.season or 0,
                self.episode or 0,
                self.title,
            )
        return build_movie_key(self.title, self.year)

    @property
    def display_name(self) -> str:
        """For logs and the dashboard — identity at a glance."""
        if self.kind == "episode":
            base = f"{self.show} S{self.season:02d}E{self.episode:02d}"
            return f"{base} - {self.title}" if self.title else base
        return f"{self.title} ({self.year})" if self.year else self.title

    @property
    def slug(self) -> str:
        """Filesystem-safe stem for generated artifacts."""
        import re

        return re.sub(r"[^A-Za-z0-9]+", "_", self.display_name).strip("_")


@runtime_checkable
class MediaSource(Protocol):
    """Anything that can enumerate a library."""

    def iter_items(self) -> Iterator[MediaItem]:
        ...
