#!/usr/bin/env python3
"""
Generate copy-paste prompts for building quotes.json.

The workflow this supports, which costs nothing per run:

    1. python scripts/export_titles.py --batch-size 12 --out prompts/
    2. paste each prompt into a Claude chat, copy the JSON reply into a file
    3. python scripts/merge_quotes.py replies/*.json

The lookup keys written into these prompts come from `src.naming`, the SAME
function the pipeline uses to look quotes up. That is the whole reason it lives
in one module: if the key in the prompt and the key in the pipeline ever differ
by so much as a space, every lookup misses and the failure looks like an empty
quotes file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.config import ConfigError, load_config  # noqa: E402
from src.sources.base import MediaItem  # noqa: E402
from src.sources.filesystem import FilesystemSource  # noqa: E402

MOVIE_PROMPT = """\
For each film below, give the most memorable, self-contained quotes — lines that
land without context and would work as a short vertical clip.

Rules:
- Quote the line as spoken, as closely as you can remember it. The pipeline
  fuzzy-matches these against the film's actual subtitle track, so near-enough
  wording is fine, but invented lines will simply never match.
- {count} quotes per film, in the order they occur in the film where you can.
- Skip any film you don't genuinely know. An omission costs nothing; a
  hallucinated quote silently produces no clip.

Reply with JSON only, in exactly this shape:

{{
  "Film Title": ["first quote", "second quote"],
  "Another Film": ["..."]
}}

Films:
{titles}
"""

TV_PROMPT = """\
For each episode below, find quotes that fit these categories:

{categories}

Rules:
- Quote the line as spoken, as closely as you can remember it. These are
  fuzzy-matched against the episode's real subtitle track, so approximate
  wording is fine; invented lines never match.
- Up to {count} quotes per category. Omit a category entirely if nothing in the
  episode fits — a thin category is better than a forced one.
- Skip any episode you don't genuinely know.

Reply with JSON only, in exactly this shape:

{{
  "{example_key}": {{
    {example_categories}
  }}
}}

Episodes:
{titles}
"""


def load_categories(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def movie_prompt(items: list[MediaItem], count: int) -> str:
    titles = "\n".join(f"- {item.lookup_key}" for item in items)
    return MOVIE_PROMPT.format(count=count, titles=titles)


def tv_prompt(
    show: str, items: list[MediaItem], categories: dict[str, str], count: int
) -> str:
    if not categories:
        # Without descriptions the model has nothing to aim at, and the quotes
        # come back generic. Better to say so than to emit a useless prompt.
        categories = {
            "memorable": "the most quotable, self-contained lines in the episode"
        }

    described = "\n".join(f"- {key}: {text}" for key, text in categories.items())
    example_categories = ",\n    ".join(
        f'"{key}": ["a quote", "another quote"]' for key in list(categories)[:2]
    )

    return TV_PROMPT.format(
        categories=described,
        count=count,
        titles="\n".join(f"- {item.lookup_key}" for item in items),
        example_key=items[0].lookup_key,
        example_categories=example_categories,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="titles per prompt (keep it small enough to answer well)")
    parser.add_argument("--count", type=int, default=4,
                        help="quotes requested per title or per category")
    parser.add_argument("--kind", choices=("movie", "episode"))
    parser.add_argument("--show", help="limit to one show")
    parser.add_argument("--out", help="write prompts here instead of stdout")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    source = FilesystemSource(config.roots.movies, config.roots.tv)
    missing = source.missing_roots()
    if missing:
        print(f"Library root(s) missing: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2

    items = list(source.iter_items())
    if args.kind:
        items = [item for item in items if item.kind == args.kind]
    if args.show:
        wanted = args.show.casefold()
        items = [i for i in items if i.show and wanted in i.show.casefold()]

    if not items:
        print("No media matched.", file=sys.stderr)
        return 1

    categories_by_show = load_categories(config.quotes.categories)
    prompts: list[tuple[str, str]] = []

    movies = [item for item in items if item.kind == "movie"]
    for index, batch in enumerate(batched(movies, args.batch_size), start=1):
        prompts.append((f"movies-{index:02d}", movie_prompt(batch, args.count)))

    episodes = [item for item in items if item.kind == "episode"]
    for show in sorted({item.show or "unknown" for item in episodes}):
        in_show = sorted(
            (i for i in episodes if (i.show or "unknown") == show),
            key=lambda i: (i.season or 0, i.episode or 0),
        )
        categories = categories_by_show.get(show, {})
        slug = "".join(ch if ch.isalnum() else "-" for ch in show).strip("-").lower()
        for index, batch in enumerate(batched(in_show, args.batch_size), start=1):
            prompts.append(
                (f"{slug}-{index:02d}", tv_prompt(show, batch, categories, args.count))
            )

    if args.out:
        destination = Path(args.out)
        destination.mkdir(parents=True, exist_ok=True)
        for name, body in prompts:
            (destination / f"{name}.txt").write_text(body, encoding="utf-8")
        print(f"Wrote {len(prompts)} prompt(s) to {destination}")
        print("Paste each into a chat, save the JSON replies, then run merge_quotes.py")
    else:
        for name, body in prompts:
            print(f"\n{'=' * 70}\n== {name}\n{'=' * 70}\n")
            print(body)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
