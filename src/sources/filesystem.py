"""
Filesystem media discovery.

Walks the configured movie and TV roots and parses identity out of paths. This
is inherently heuristic — libraries are named inconsistently — so the `scan`
command prints everything it parsed. Eyeball that output before trusting a run;
a mis-parsed show name means a lookup key that matches nothing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from src.naming import clean_title, extract_year
from src.sources.base import MediaItem

VIDEO_EXTENSIONS = {
    ".mkv", ".mp4", ".m4v", ".avi", ".mov", ".ts", ".m2ts",
    ".wmv", ".mpg", ".mpeg", ".webm", ".flv",
}

# Directories that hold bonus content rather than the feature itself. Including
# them would generate shorts from behind-the-scenes footage.
EXTRA_DIRS = {
    "extras", "featurettes", "behind the scenes", "deleted scenes",
    "interviews", "scenes", "shorts", "trailers", "other", "specials",
    "sample", "samples", "subs", "subtitles",
}

# "Sample" files are a few seconds long and would silently produce garbage.
_SAMPLE = re.compile(r"(^|[.\s_-])sample([.\s_-]|$)", re.IGNORECASE)

# S01E03 / s01e03 / S01E03E04 / S01E03-E04 — the dominant convention.
_SXXEXX = re.compile(r"[sS](\d{1,2})[\s._-]*[eE](\d{1,3})")
# 1x03 — the other common one.
_NXNN = re.compile(r"(?<![a-zA-Z0-9])(\d{1,2})[xX](\d{1,3})(?![0-9])")
# "Season 01" / "Season 1" / "S01" as a directory name.
_SEASON_DIR = re.compile(r"^(?:season[\s._-]*|s)(\d{1,2})$", re.IGNORECASE)


class FilesystemSource:
    """Enumerates movies and episodes from two library roots."""

    def __init__(self, movies_root: Path | None, tv_root: Path | None) -> None:
        self.movies_root = movies_root
        self.tv_root = tv_root

    # -- discovery ---------------------------------------------------------

    def iter_items(self) -> Iterator[MediaItem]:
        if self.movies_root:
            yield from self._iter_movies(self.movies_root)
        if self.tv_root:
            yield from self._iter_episodes(self.tv_root)

    def missing_roots(self) -> list[Path]:
        """
        Roots that are configured but absent.

        The guard against running where the library doesn't resolve: if this
        returns everything, the pipeline should fail loudly rather than report
        an empty library, which looks identical to "nothing to do".
        """
        configured = [r for r in (self.movies_root, self.tv_root) if r]
        return [root for root in configured if not root.is_dir()]

    # -- movies ------------------------------------------------------------

    def _iter_movies(self, root: Path) -> Iterator[MediaItem]:
        for path in _walk_videos(root):
            # A movie usually lives in its own folder: "Heat (1995)/Heat.mkv".
            # The folder name is the better title source — it carries the year
            # and lacks the release junk the filename often has.
            parent = path.parent
            basis = parent.name if parent != root else path.stem

            title, year = extract_year(basis)
            title = clean_title(title)

            if not title:
                # Folder name was pure junk; fall back to the filename.
                title, year_from_file = extract_year(path.stem)
                title = clean_title(title)
                year = year or year_from_file

            if not title:
                continue

            yield MediaItem(kind="movie", path=path, title=title, year=year)

    # -- television --------------------------------------------------------

    def _iter_episodes(self, root: Path) -> Iterator[MediaItem]:
        for path in _walk_videos(root):
            parsed = _parse_episode(path, root)
            if parsed is None:
                continue
            yield parsed


def _walk_videos(root: Path) -> Iterator[Path]:
    """Every video file under `root`, minus samples and bonus-content folders."""
    if not root.is_dir():
        return

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if path.name.startswith("."):
            continue
        if _SAMPLE.search(path.stem):
            continue
        # Any ancestor being an extras folder disqualifies the file.
        relative_parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
        if any(part in EXTRA_DIRS for part in relative_parts):
            continue
        yield path


def _parse_episode(path: Path, root: Path) -> MediaItem | None:
    """
    Pull show / season / episode / title out of an episode path.

    The show name comes from the top-level directory under the TV root rather
    than the filename: directory naming is consistent within a library, while
    filenames vary per release.
    """
    stem = path.stem

    season, episode, remainder = _parse_season_episode(stem)

    if season is None:
        # Some libraries put the numbering only in the folder: "Season 02/03 -
        # Title.mkv". Recover the season from the directory, the episode from a
        # leading number in the filename.
        season = _season_from_dirs(path, root)
        if season is not None:
            leading = re.match(r"^\s*(\d{1,3})\b[\s._-]*(.*)$", stem)
            if leading:
                episode = int(leading.group(1))
                remainder = leading.group(2)

    if season is None or episode is None:
        return None

    relative = path.relative_to(root).parts
    show = clean_title(relative[0]) if len(relative) > 1 else ""
    if not show:
        # Flat layout: the show name is whatever precedes the SxxExx token.
        prefix = _SXXEXX.split(stem)[0] if _SXXEXX.search(stem) else ""
        show = clean_title(prefix)

    show, _ = extract_year(show)
    show = show.strip(" -–—")

    title = clean_title(remainder)
    # A title that's just a repeat of the numbering carries no information.
    if re.fullmatch(r"[sS]?\d{1,3}([eE]\d{1,3})?", title or ""):
        title = ""

    return MediaItem(
        kind="episode",
        path=path,
        title=title,
        show=show or None,
        season=season,
        episode=episode,
    )


def _parse_season_episode(stem: str) -> tuple[int | None, int | None, str]:
    """
    Find the season/episode marker and return it plus everything after it.

    Multi-episode files (S01E01E02) resolve to the FIRST episode: the file is a
    single video, so one identity is all that can be represented, and the first
    number is the one the episode title belongs to.
    """
    match = _SXXEXX.search(stem) or _NXNN.search(stem)
    if not match:
        return None, None, stem

    season = int(match.group(1))
    episode = int(match.group(2))

    remainder = stem[match.end():]
    # Consume a trailing "E04" / "-E04" from a multi-episode marker so it
    # doesn't end up in the title.
    remainder = re.sub(r"^[\s._-]*[eE]\d{1,3}", "", remainder)
    return season, episode, remainder.lstrip(" .-_")


def _season_from_dirs(path: Path, root: Path) -> int | None:
    """Look for a 'Season NN' directory between the root and the file."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        match = _SEASON_DIR.match(part.strip())
        if match:
            return int(match.group(1))
    return None
