"""
Filename parsing.

Libraries are named inconsistently, so this covers the conventions that
actually show up rather than one idealised layout. A mis-parsed show name
produces a lookup key that matches nothing, and the failure looks exactly like
"no quotes written for this episode" — so it's worth pinning down here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.naming import build_tv_key, find_key, normalize_key
from src.sources.filesystem import FilesystemSource


def build_tree(root: Path, paths: list[str]) -> None:
    for rel in paths:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")


# --------------------------------------------------------------------------
# Television
# --------------------------------------------------------------------------


@pytest.fixture
def tv_root(tmp_path: Path) -> Path:
    root = tmp_path / "local_tvshows"
    build_tree(
        root,
        [
            # The canonical layout.
            "Malcolm in the Middle/Season 01/Malcolm in the Middle - S01E03 - Home Alone 4.mkv",
            # Scene naming: dots, release junk, group tag.
            "Malcolm in the Middle/Season 01/Malcolm.in.the.Middle.S01E04.Shame.1080p.BluRay.x264-GROUP.mkv",
            # The other numbering convention.
            "The Bear/Season 02/The Bear - 2x05 - Pop.mkv",
            # Multi-episode file.
            "The Bear/Season 02/The Bear - S02E06E07 - Fishes.mkv",
            # Season in the folder only, episode as a leading number.
            "Frasier/Season 03/07 - The Adventures of Bad Boy and Dirty Girl.mkv",
            # Lowercase marker, no episode title.
            "Frasier/Season 03/frasier.s03e08.mkv",
            # Must be ignored.
            "The Bear/Season 02/The Bear - S02E05 - Pop-sample.mkv",
            "The Bear/Extras/Behind the scenes.mkv",
            "The Bear/Season 02/poster.jpg",
        ],
    )
    return root


def test_parses_canonical_episode(tv_root: Path) -> None:
    items = {i.path.name: i for i in FilesystemSource(None, tv_root).iter_items()}
    item = items["Malcolm in the Middle - S01E03 - Home Alone 4.mkv"]

    assert item.kind == "episode"
    assert item.show == "Malcolm in the Middle"
    assert item.season == 1
    assert item.episode == 3
    assert item.title == "Home Alone 4"
    assert item.lookup_key == "Malcolm in the Middle - S01E03 - Home Alone 4"


def test_strips_release_junk_from_title(tv_root: Path) -> None:
    items = {i.path.name: i for i in FilesystemSource(None, tv_root).iter_items()}
    item = items["Malcolm.in.the.Middle.S01E04.Shame.1080p.BluRay.x264-GROUP.mkv"]

    assert item.show == "Malcolm in the Middle"
    assert (item.season, item.episode) == (1, 4)
    # "1080p BluRay x264-GROUP" must not survive into the key.
    assert item.title == "Shame"


def test_parses_nxnn_convention(tv_root: Path) -> None:
    items = {i.path.name: i for i in FilesystemSource(None, tv_root).iter_items()}
    item = items["The Bear - 2x05 - Pop.mkv"]

    assert (item.show, item.season, item.episode) == ("The Bear", 2, 5)
    assert item.title == "Pop"


def test_multi_episode_file_resolves_to_first_episode(tv_root: Path) -> None:
    """One file is one identity; the title belongs to the first episode."""
    items = {i.path.name: i for i in FilesystemSource(None, tv_root).iter_items()}
    item = items["The Bear - S02E06E07 - Fishes.mkv"]

    assert (item.season, item.episode) == (2, 6)
    # The trailing E07 must not leak into the title.
    assert item.title == "Fishes"


def test_season_from_directory_with_leading_episode_number(tv_root: Path) -> None:
    items = {i.path.name: i for i in FilesystemSource(None, tv_root).iter_items()}
    item = items["07 - The Adventures of Bad Boy and Dirty Girl.mkv"]

    assert (item.show, item.season, item.episode) == ("Frasier", 3, 7)
    assert item.title == "The Adventures of Bad Boy and Dirty Girl"


def test_episode_without_title(tv_root: Path) -> None:
    items = {i.path.name: i for i in FilesystemSource(None, tv_root).iter_items()}
    item = items["frasier.s03e08.mkv"]

    assert (item.season, item.episode) == (3, 8)
    assert item.title == ""
    # No trailing separator when there's no title.
    assert item.lookup_key == "Frasier - S03E08"


def test_skips_samples_extras_and_non_video(tv_root: Path) -> None:
    names = [i.path.name for i in FilesystemSource(None, tv_root).iter_items()]

    assert not any("sample" in n.lower() for n in names)
    assert "Behind the scenes.mkv" not in names
    assert "poster.jpg" not in names


# --------------------------------------------------------------------------
# Movies
# --------------------------------------------------------------------------


@pytest.fixture
def movies_root(tmp_path: Path) -> Path:
    root = tmp_path / "local_movies"
    build_tree(
        root,
        [
            "Heat (1995)/Heat (1995) Bluray-1080p.mkv",
            "Blade Runner 2049 (2017)/Blade Runner 2049.mkv",
            "The Empire Strikes Back (1980).mkv",
            "Spider-Man (2002)/Spider-Man.2002.1080p.BluRay.x264-AMIABLE.mkv",
        ],
    )
    return root


def test_movie_title_and_year_from_folder(movies_root: Path) -> None:
    items = {i.title: i for i in FilesystemSource(movies_root, None).iter_items()}

    assert items["Heat"].year == 1995
    assert items["Heat"].kind == "movie"
    # The year is deliberately not part of the lookup key.
    assert items["Heat"].lookup_key == "Heat"


def test_movie_title_with_trailing_number_survives(movies_root: Path) -> None:
    """'2049' is part of the title, not a year — the year is in parentheses."""
    items = {i.title: i for i in FilesystemSource(movies_root, None).iter_items()}

    assert "Blade Runner 2049" in items
    assert items["Blade Runner 2049"].year == 2017


def test_movie_at_root_without_folder(movies_root: Path) -> None:
    items = {i.title: i for i in FilesystemSource(movies_root, None).iter_items()}

    assert items["The Empire Strikes Back"].year == 1980


def test_hyphenated_movie_title_not_truncated(movies_root: Path) -> None:
    """A trailing '-GROUP' tag must not eat the hyphen in 'Spider-Man'."""
    items = {i.title: i for i in FilesystemSource(movies_root, None).iter_items()}

    assert "Spider-Man" in items


# --------------------------------------------------------------------------
# Key resolution
# --------------------------------------------------------------------------


def test_build_tv_key_is_zero_padded() -> None:
    assert build_tv_key("Show", 1, 3, "Title") == "Show - S01E03 - Title"


def test_find_key_exact_then_normalized_then_fuzzy() -> None:
    available = ["Malcolm in the Middle - S01E03 - Home Alone 4"]

    # Exact.
    assert find_key(available[0], available) == available[0]
    # Punctuation / spacing drift.
    assert find_key("malcolm in the middle s01e03 home alone 4", available) == available[0]
    # Wording drift within threshold.
    assert find_key("Malcolm in the Middle - S01E03 - Home Alone IV", available) == available[0]
    # Genuinely different episode must NOT match.
    assert find_key("Malcolm in the Middle - S04E11 - Company Picnic", available) is None


def test_normalize_key_collapses_punctuation() -> None:
    assert normalize_key("Show - S01E03 - Title!") == normalize_key("show s01e03 title")


def test_missing_roots_are_reported(tmp_path: Path) -> None:
    """
    The guard against running where the library doesn't resolve. An empty
    result and a wrong mount look identical without this.
    """
    source = FilesystemSource(tmp_path / "nope", None)
    assert source.missing_roots() == [tmp_path / "nope"]


def test_fuzzy_key_never_crosses_episodes() -> None:
    """
    The show name is most of the string, so the numbering barely moves the
    similarity ratio: S04E01 scores 0.857 against S01E01, over the 0.85
    threshold. Fuzzy matching absorbs drift in the TITLE; the numbering is
    exact by construction and must be treated that way, or one episode's
    quotes get silently applied to another.
    """
    available = ["Malcolm in the Middle - S01E01 - Pilot"]

    assert find_key("Malcolm in the Middle - S04E01 - Zoo", available) is None
    assert find_key("Malcolm in the Middle - S01E02 - Red Dress", available) is None

    # The same episode with a drifted title still resolves.
    assert find_key("Malcolm in the Middle - S01E01 - The Pilot", available) == available[0]


def test_fuzzy_key_does_not_cross_movies_and_episodes() -> None:
    episodes = ["Malcolm in the Middle - S01E01 - Pilot"]
    movies = ["Malcolm in the Middle"]

    assert find_key("Malcolm in the Middle", episodes) is None
    assert find_key("Malcolm in the Middle - S01E01 - Pilot", movies) is None
