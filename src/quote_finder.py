"""
quotes.json lookup.

Two entry shapes coexist in one file:

    "The Empire Strikes Back": [
      "No, I am your father.",
      {"label": "The duel", "quote": "Do or do not. There is no try."}
    ],

    "Malcolm in the Middle - S01E03 - Home Alone 4": {
      "dewey_chaos":   ["I am the smartest man alive!"],
      "malcolm_normal": ["I just want one normal day."]
    }

A list is a flat pool (movie-style). A dict is categorised (TV-style):
`--category dewey_chaos` pulls one bucket, omitting it pools them all.

The same flat list serves single-short and multipart modes — single takes the
first match, multipart matches everything and arranges it. That is why
multipart needed no schema change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.naming import find_key


class QuotesError(Exception):
    """Raised with an actionable message; the CLI prints it without a traceback."""


@dataclass(frozen=True)
class Candidate:
    """One thing to look for, with where it came from."""

    quote: str
    label: str | None = None
    category: str | None = None


class QuoteStore:
    def __init__(self, data: dict[str, Any], source: Path | None = None) -> None:
        self._data = data
        self.source = source

    @classmethod
    def load(cls, path: Path) -> "QuoteStore":
        if not path.exists():
            raise QuotesError(
                f"No quotes file at {path}. Generate prompts with "
                f"`scripts/export_titles.py`, paste the replies back, then merge "
                f"them with `scripts/merge_quotes.py`."
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise QuotesError(
                f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno}): "
                f"{exc.msg}"
            ) from exc

        if not isinstance(data, dict):
            raise QuotesError(f"{path}: expected an object mapping titles to quotes.")

        return cls(data, source=path)

    # -- lookup ------------------------------------------------------------

    def keys(self) -> list[str]:
        return list(self._data)

    def resolve(self, lookup_key: str) -> str | None:
        """The key as written in the file, or None. Exact, then normalized, then fuzzy."""
        return find_key(lookup_key, self.keys())

    def categories_for(self, lookup_key: str) -> list[str]:
        """Category names for a categorised entry; empty for a flat list."""
        entry = self._entry(lookup_key)
        return sorted(entry) if isinstance(entry, dict) else []

    def candidates_for(
        self, lookup_key: str, category: str | None = None
    ) -> list[Candidate]:
        """
        Everything worth looking for in this item.

        An unknown category returns nothing rather than silently falling back to
        the whole pool — a typo in `--category` should look like a typo, not
        like a run that mysteriously produced the wrong clips.
        """
        entry = self._entry(lookup_key)
        if entry is None:
            return []

        if isinstance(entry, list):
            return [_candidate(raw, None) for raw in entry if _usable(raw)]

        if isinstance(entry, dict):
            if category is not None:
                bucket = entry.get(category)
                if bucket is None:
                    return []
                return [_candidate(raw, category) for raw in bucket if _usable(raw)]

            pooled: list[Candidate] = []
            for name in sorted(entry):
                bucket = entry[name] or []
                pooled += [_candidate(raw, name) for raw in bucket if _usable(raw)]
            return pooled

        return []

    def _entry(self, lookup_key: str) -> Any:
        resolved = self.resolve(lookup_key)
        return self._data.get(resolved) if resolved else None


def _usable(raw: Any) -> bool:
    if isinstance(raw, str):
        return bool(raw.strip())
    return isinstance(raw, dict) and bool(str(raw.get("quote", "")).strip())


def _candidate(raw: Any, category: str | None) -> Candidate:
    if isinstance(raw, str):
        return Candidate(quote=raw.strip(), label=None, category=category)

    return Candidate(
        quote=str(raw.get("quote", "")).strip(),
        label=(str(raw["label"]).strip() if raw.get("label") else None),
        category=category,
    )
