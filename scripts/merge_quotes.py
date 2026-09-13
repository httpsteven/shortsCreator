#!/usr/bin/env python3
"""
Merge pasted-back JSON replies into quotes.json.

Tolerant of what actually comes out of a chat window: replies wrapped in
```json fences, or with a sentence of preamble before the object. Anything it
still can't parse is reported by filename rather than silently skipped.

Merging is additive and de-duplicating. Re-running with the same reply twice
changes nothing, and a second batch for a title already present adds to it
rather than replacing it — so you can build the file up over several sessions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull the JSON object out of a chat reply."""
    fenced = FENCE.search(text)
    if fenced:
        text = fenced.group(1)

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost braces, for replies with prose around them.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def _quote_text(entry: Any) -> str | None:
    if isinstance(entry, str) and entry.strip():
        return entry.strip()
    if isinstance(entry, dict) and str(entry.get("quote", "")).strip():
        return str(entry["quote"]).strip()
    return None


def _merge_list(existing: list, incoming: list) -> list:
    """Append what's new, preserving order and the richer object form."""
    seen = {(_quote_text(entry) or "").casefold() for entry in existing}
    merged = list(existing)
    for entry in incoming:
        text = _quote_text(entry)
        if text and text.casefold() not in seen:
            merged.append(entry)
            seen.add(text.casefold())
    return merged


def merge(target: dict, incoming: dict) -> tuple[dict, dict[str, int]]:
    stats = {"titles_added": 0, "titles_updated": 0, "quotes_added": 0}

    for key, value in incoming.items():
        if key not in target:
            target[key] = value
            stats["titles_added"] += 1
            stats["quotes_added"] += _count(value)
            continue

        before = _count(target[key])
        current = target[key]

        # Flat list + flat list.
        if isinstance(current, list) and isinstance(value, list):
            target[key] = _merge_list(current, value)

        # Categorised + categorised: merge bucket by bucket.
        elif isinstance(current, dict) and isinstance(value, dict):
            for category, quotes in value.items():
                if not isinstance(quotes, list):
                    continue
                current[category] = _merge_list(current.get(category, []), quotes)

        else:
            # Shape changed between batches — keep what's on disk and say so
            # rather than quietly discarding one of them.
            print(
                f"  ! {key}: existing entry is "
                f"{'categorised' if isinstance(current, dict) else 'a flat list'} "
                f"but the reply is not — left unchanged"
            )
            continue

        added = _count(target[key]) - before
        if added:
            stats["titles_updated"] += 1
            stats["quotes_added"] += added

    return target, stats


def _count(value: Any) -> int:
    if isinstance(value, list):
        return sum(1 for entry in value if _quote_text(entry))
    if isinstance(value, dict):
        return sum(_count(bucket) for bucket in value.values())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replies", nargs="+", help="JSON reply files (or - for stdin)")
    parser.add_argument("--out", default="quotes.json", help="quotes file to update")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    destination = Path(args.out)
    if destination.exists():
        try:
            target = json.loads(destination.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"{destination} is not valid JSON: {exc}", file=sys.stderr)
            return 2
    else:
        target = {}

    totals = {"titles_added": 0, "titles_updated": 0, "quotes_added": 0}
    failures = 0

    for name in args.replies:
        text = sys.stdin.read() if name == "-" else Path(name).read_text(encoding="utf-8")
        try:
            incoming = extract_json(text)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  ! {name}: could not parse ({exc})")
            failures += 1
            continue

        if not isinstance(incoming, dict):
            print(f"  ! {name}: expected a JSON object at the top level")
            failures += 1
            continue

        target, stats = merge(target, incoming)
        for key, value in stats.items():
            totals[key] += value
        print(f"  + {name}: {stats['quotes_added']} quote(s)")

    print(
        f"\n  {totals['quotes_added']} quote(s) across "
        f"{totals['titles_added']} new and {totals['titles_updated']} existing title(s)"
    )
    if failures:
        print(f"  {failures} file(s) could not be parsed")

    if args.dry_run:
        print("  Dry run — nothing written.")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(target, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  Wrote {destination} ({len(target)} title(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
