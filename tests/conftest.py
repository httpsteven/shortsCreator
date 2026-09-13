"""Shared test fixtures. Synthetic media is built once per session and reused."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import tests.fixtures as fx
from src.config import Config, Roots

# Built into the repo rather than a tmp dir so repeated runs don't re-encode.
MEDIA_DIR = Path(__file__).parent / "fixtures" / "media"


def pytest_configure(config):  # noqa: ARG001
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.exit("ffmpeg and ffprobe are required to build test fixtures.", 1)


@pytest.fixture(scope="session")
def media() -> dict[str, Path]:
    return fx.build_all(MEDIA_DIR)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """A config whose caches and outputs land in the test's tmp dir."""
    return Config(
        roots=Roots(movies=MEDIA_DIR, tv=None),
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
    )
