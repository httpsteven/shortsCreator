"""
The produced-clips ledger.

Stops a repeat run from making the same short twice. Keyed on the source file
plus the quote rather than on the output filename, so renaming or moving output
doesn't cause silent duplicates, and re-running after a config change (different
padding, different caption style) is still recognised as the same clip.

This is a plain JSON file so it stays readable and hand-editable. The SQLite
database is the dashboard's view of the same facts; this remains the pipeline's
own source of truth so the pipeline works with no database configured at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_VERSION = 1


def clip_id(source: Path, quote: str) -> str:
    """Stable identity for one produced clip."""
    digest = hashlib.sha1(
        f"{Path(source).resolve()}|{quote.strip().casefold()}".encode()
    ).hexdigest()
    return digest[:16]


def series_id(source: Path, quotes: list[str]) -> str:
    """Stable identity for a multipart set, independent of part order."""
    joined = "|".join(sorted(quote.strip().casefold() for quote in quotes))
    digest = hashlib.sha1(f"{Path(source).resolve()}|series|{joined}".encode())
    return digest.hexdigest()[:16]


@dataclass
class ClipRecord:
    clip_id: str
    source: str
    lookup_key: str
    quote: str
    category: str | None
    start: float
    end: float
    score: float
    output: str
    thumbnail: str | None = None
    provenance: str | None = None
    series: str | None = None
    part_index: int | None = None
    part_total: int | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class State:
    def __init__(self, path: Path, clips: dict[str, ClipRecord] | None = None) -> None:
        self.path = path
        self.clips: dict[str, ClipRecord] = clips or {}

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "State":
        if not path.exists():
            return cls(path)

        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A corrupt ledger must not stop the pipeline. Worst case it
            # re-produces clips, which is recoverable; refusing to run is not.
            backup = path.with_suffix(".corrupt.json")
            path.replace(backup)
            print(f"  warning: state file was unreadable, moved to {backup}")
            return cls(path)

        clips = {
            key: ClipRecord(**value)
            for key, value in (raw.get("clips") or {}).items()
        }
        return cls(path, clips)

    def save(self) -> None:
        """
        Write atomically.

        A run interrupted mid-write would otherwise leave a truncated ledger,
        and the pipeline would re-make everything it had already produced.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STATE_VERSION,
            "clips": {key: asdict(record) for key, record in self.clips.items()},
        }

        handle = tempfile.NamedTemporaryFile(
            "w", dir=self.path.parent, delete=False, encoding="utf-8", suffix=".tmp"
        )
        try:
            with handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    # -- queries -----------------------------------------------------------

    def has_clip(self, source: Path, quote: str) -> bool:
        return clip_id(source, quote) in self.clips

    def has_series(self, identifier: str) -> bool:
        return any(record.series == identifier for record in self.clips.values())

    def series_parts(self, identifier: str) -> list[ClipRecord]:
        return sorted(
            (r for r in self.clips.values() if r.series == identifier),
            key=lambda r: r.part_index or 0,
        )

    def record(self, record: ClipRecord) -> None:
        self.clips[record.clip_id] = record

    def forget(self, identifier: str) -> bool:
        return self.clips.pop(identifier, None) is not None

    def __len__(self) -> int:
        return len(self.clips)
