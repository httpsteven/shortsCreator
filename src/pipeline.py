"""
CLI orchestrator.

Subcommands are ordered the way you'd actually use them:

    scan       what did the filename parser make of my library?
    preflight  is this machine able to do the work?
    audit      do I have subtitles, and what would fixing the gaps cost?
    make       produce shorts

Every skip prints a reason, because a run that produces nothing has to be
diagnosable without adding logging after the fact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.audit import audit_library, summarize, write_csv
from src.config import Config, ConfigError, load_config
from src.preflight import report, run_checks
from src.gate import ViewerGate
from src.producer import Outcome, produce
from src.quote_finder import QuoteStore, QuotesError
from src.state import State
from src.sources.base import MediaItem
from src.sources.filesystem import FilesystemSource


def build_source(config: Config) -> FilesystemSource:
    return FilesystemSource(config.roots.movies, config.roots.tv)


def collect_items(config: Config, args: argparse.Namespace) -> list[MediaItem]:
    """
    Enumerate the library, applying whatever filters the command was given.

    Fails loudly when no configured root resolves. An empty list and a wrong
    mount point look identical otherwise, and the second is far more likely.
    """
    source = build_source(config)

    missing = source.missing_roots()
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise ConfigError(
            f"Library root(s) do not exist: {joined}\n"
            f"  This usually means the pipeline isn't running where the library "
            f"lives, or the path in config.yaml is wrong."
        )

    items = list(source.iter_items())

    kind = getattr(args, "kind", None)
    if kind:
        items = [item for item in items if item.kind == kind]

    show = getattr(args, "show", None)
    if show:
        wanted = show.casefold()
        items = [
            item for item in items
            if item.show and wanted in item.show.casefold()
        ]

    title = getattr(args, "title", None)
    if title:
        wanted = title.casefold()
        items = [item for item in items if wanted in item.title.casefold()]

    limit = getattr(args, "limit", None)
    if limit:
        items = items[:limit]

    return items


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------



def open_database(config: Config):
    """
    Open the dashboard database, or return None.

    Entirely optional: with no database.path configured the pipeline works
    exactly as before and simply isn't visible in the dashboard. A database
    that fails to open is a warning, never a reason to abandon a run that
    would otherwise succeed.
    """
    if not config.db_path:
        return None
    try:
        from src.db import Database

        return Database(config.db_path)
    except Exception as exc:
        print(f"  warning: could not open {config.db_path}: {exc}")
        return None



def cmd_preflight(config: Config, args: argparse.Namespace) -> int:
    print("Preflight")
    return 0 if report(run_checks(config)) else 1


def cmd_scan(config: Config, args: argparse.Namespace) -> int:
    """
    Print what the parser made of every file.

    Worth actually reading before a first run: filename parsing is heuristic,
    and a mis-parsed show name produces a lookup key that matches nothing —
    which looks exactly like "no quotes written for this episode".
    """
    items = collect_items(config, args)

    if not items:
        print("  No media found. Check roots in config.yaml, or your filters.")
        return 1

    movies = [item for item in items if item.kind == "movie"]
    episodes = [item for item in items if item.kind == "episode"]

    if movies:
        print(f"\nMovies ({len(movies)})")
        for item in movies:
            year = item.year or "----"
            print(f"  {year}  {item.title}")
            if args.verbose:
                print(f"        key: {item.lookup_key}")
                print(f"       path: {item.path}")

    if episodes:
        print(f"\nEpisodes ({len(episodes)})")
        for show in sorted({item.show or "(unknown)" for item in episodes}):
            in_show = [item for item in episodes if (item.show or "(unknown)") == show]
            print(f"  {show} — {len(in_show)} episode(s)")
            for item in sorted(in_show, key=lambda i: (i.season or 0, i.episode or 0)):
                title = item.title or "(no episode title)"
                print(f"    S{item.season:02d}E{item.episode:02d}  {title}")
                if args.verbose:
                    print(f"           key: {item.lookup_key}")

    print(f"\n  {len(items)} item(s) total.")
    return 0


def cmd_audit(config: Config, args: argparse.Namespace) -> int:
    items = collect_items(config, args)
    if not items:
        print("  No media found.")
        return 1

    def progress(position: int, total: int, item: MediaItem) -> None:
        if args.quiet:
            return
        print(f"  [{position:>4}/{total}] {item.display_name}", flush=True)

    rows = audit_library(
        items, config, deep=args.deep, sample=args.sample, on_progress=progress
    )

    database = open_database(config)
    if database is not None:
        for row in rows:
            database.upsert_audit(row)
        database.close()

    print()
    print(summarize(rows, deep=args.deep))

    if database is not None:
        print(f"  Recorded in {config.db_path}")

    if args.out:
        write_csv(rows, Path(args.out))
        print(f"\n  Wrote {len(rows)} row(s) to {args.out}")

    return 0


def build_transcriber(config: Config, quiet: bool = False):
    """
    A transcriber, or None when Whisper isn't usable.

    Returning None rather than raising means the ladder degrades to "skip items
    without text subtitles" instead of failing the whole run — which is the
    right behaviour when most of a library has subtitles and a few items don't.
    """
    if not config.whisper.enabled:
        return None

    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return None

    from src.transcribe import Transcriber

    def progress(done: int, total: int) -> None:
        if not quiet:
            print(f"        transcribing chunk {done}/{total}", flush=True)

    return Transcriber(config, on_progress=progress)


def report_outcome(outcome: Outcome, verbose: bool) -> None:
    """Print what happened to one item, always including the reason."""
    name = outcome.item.display_name

    if outcome.produced:
        print(f"  [made] {name}")
        for record in outcome.produced:
            part = ""
            if record.part_index:
                part = f" (part {record.part_index}/{record.part_total})"
            print(
                f"         {record.start:7.1f}-{record.end:<7.1f} "
                f"score {record.score:3.0f}{part}  {Path(record.output).name}"
            )
        return

    if outcome.already_done:
        print(f"  [skip] {name} — already produced ({outcome.already_done} clip(s))")
        return

    print(f"  [skip] {name} — {outcome.skipped}")

    # The single most useful thing to know when a title yields nothing.
    if verbose and outcome.near_misses:
        for match in sorted(outcome.near_misses, key=lambda m: -m.score)[:3]:
            print(f"         closest {match.score:3.0f}: {match.matched_text[:70]!r}")


def cmd_make(config: Config, args: argparse.Namespace) -> int:
    items = collect_items(config, args)
    if not items:
        print("  No media matched. Try `scan` to see what was found.")
        return 1

    try:
        store = QuoteStore.load(config.quotes.path)
    except QuotesError as exc:
        print(f"Quotes error: {exc}", file=sys.stderr)
        return 2

    state_path = config.output_dir / "state.json"
    state = State.load(state_path)
    transcriber = build_transcriber(config, quiet=args.quiet)

    database = open_database(config)
    gate = ViewerGate(config, database)
    if args.ignore_viewers:
        gate.force_open()

    # Wait rather than fail: an overnight batch should start when the house
    # goes quiet, not give up because someone was mid-episode when it launched.
    status = gate.status()
    if status.paused:
        print(f"  {status.reason}")
        if args.wait_timeout == 0:
            print("  Not waiting (--wait-timeout 0). Nothing produced.")
            return 0
        limit = None if args.wait_timeout < 0 else args.wait_timeout
        print(
            "  Waiting for the lab to go quiet"
            + (f", up to {limit / 60:.0f}m" if limit else "")
            + " (Ctrl-C to stop)..."
        )
        if not gate.wait_until_clear(
            announce=lambda message: print(f"  {message}", flush=True),
            timeout=limit,
        ):
            print("  Gave up waiting. Nothing produced.")
            return 0

    if args.dry_run:
        print("  Dry run — nothing will be rendered.\n")

    produced_clips = 0
    outcomes: list[Outcome] = []

    run_id = None
    if database is not None and not args.dry_run:
        from src.worker import new_run_id

        run_id = new_run_id()
        database.start_run(run_id, _describe_run(args))

    try:
        for item in items:
            if args.count and produced_clips >= args.count:
                break

            outcome = produce(
                item, config, store, state,
                category=args.category,
                multipart=args.multipart,
                parts=args.parts,
                recap=args.recap,
                recap_duration=args.recap_duration,
                recap_segments=args.recap_segments,
                dry_run=args.dry_run,
                transcriber=transcriber,
                force=args.force,
                gate=gate,
            )
            outcomes.append(outcome)

            if database is not None and not args.dry_run:
                for record in outcome.produced:
                    database.record_short(record, item)

            produced_clips += len(outcome.produced)
            report_outcome(outcome, verbose=not args.quiet)
    finally:
        if not args.dry_run:
            state.save()
        if database is not None:
            if run_id is not None:
                # The skip reasons are the useful part of a run record: a run
                # that produced nothing should be able to say why from the
                # dashboard, without going back to the terminal scrollback.
                reasons: dict[str, int] = {}
                for outcome in outcomes:
                    if outcome.skipped:
                        key = _reason_key(outcome.skipped)
                        reasons[key] = reasons.get(key, 0) + 1
                database.finish_run(
                    run_id,
                    items_seen=len(outcomes),
                    produced=sum(len(o.produced) for o in outcomes),
                    skipped=sum(1 for o in outcomes if not o.ok),
                    reasons=reasons,
                )
            database.close()

    made = sum(len(outcome.produced) for outcome in outcomes)
    skipped = sum(1 for outcome in outcomes if not outcome.ok)
    print(f"\n  {made} clip(s) from {len(outcomes)} item(s); {skipped} skipped.")
    if not args.dry_run and made:
        print(f"  Output: {config.clips_dir}")
    return 0



def _describe_run(args: argparse.Namespace) -> str:
    """A readable label for the run history — what was asked for, not argv."""
    parts = ["make"]
    for name in ("title", "show", "category"):
        value = getattr(args, name, None)
        if value:
            parts.append(f"--{name} {value}")
    if getattr(args, "multipart", False):
        parts.append("--multipart")
    if getattr(args, "parts", None):
        parts.append(f"--parts {args.parts}")
    if getattr(args, "count", None):
        parts.append(f"--count {args.count}")
    return " ".join(parts)


def _reason_key(message: str) -> str:
    """
    Collapse a specific skip message into a countable category.

    "no quotes entry matching key 'Heat'" and the same for 400 other titles are
    one fact, not four hundred.
    """
    lowered = message.lower()
    for prefix, key in (
        ("no quotes entry", "no quotes for this title"),
        ("no quotes in category", "category empty for this title"),
        ("entry ", "quotes entry is empty"),
        ("none of", "no quote cleared the threshold"),
        ("already produced", "already produced"),
        ("render failed", "render failed"),
        ("only ", "too few moments for a series"),
    ):
        if lowered.startswith(prefix):
            return key
    return message[:60]


def cmd_worker(config: Config, args: argparse.Namespace) -> int:
    """Drain the job queue the dashboard writes to."""
    if not config.db_path:
        print(
            "  No database.path configured — there is no queue to drain.\n"
            "  Point database.path at the dashboard's data/shorts.db.",
            file=sys.stderr,
        )
        return 2

    from src.worker import Worker

    worker = Worker(config, verbose=not args.quiet)
    if args.once:
        job = worker.database.claim_job()
        if job is None:
            print("  No queued jobs.")
            return 0
        worker.run_job(job)
        return 0

    worker.run_forever(poll=args.poll)
    return 0



def cmd_compile(config: Config, args: argparse.Namespace) -> int:
    """
    Pool moments from MANY episodes into one clip.

    "Top ten cold opens", "every time Dewey wins". Where a recap rounds up one
    episode, this rounds up a show — so by default it takes the single best
    moment per episode, because ten moments from one episode is a recap, not a
    top ten.
    """
    from src.caption_renderer import render_recap
    from src.media_probe import ProbeError, probe
    from src.recap import Moment, plan_recap
    from src.state import ClipRecord, clip_id
    from src.subtitle_source import HUMAN_REASONS, acquire
    from src.subtitle_utils import cues_in_window, find_matches, load_cues

    items = collect_items(config, args)
    if not items:
        print("  No media matched.")
        return 1

    try:
        store = QuoteStore.load(config.quotes.path)
    except QuotesError as exc:
        print(f"Quotes error: {exc}", file=sys.stderr)
        return 2

    database = open_database(config)
    gate = ViewerGate(config, database)
    if args.ignore_viewers:
        gate.force_open()

    transcriber = build_transcriber(config, quiet=args.quiet)

    moments: list[Moment] = []
    cues_by_source: dict[Path, list] = {}
    scanned = 0

    print(f"  Scanning {len(items)} item(s) for moments...")
    for item in items:
        if not store.candidates_for(item.lookup_key, args.category):
            continue
        scanned += 1

        try:
            probed = probe(item.path)
        except ProbeError as exc:
            print(f"  [skip] {item.display_name} — {exc}")
            continue

        subtitles = acquire(probed, config, transcriber=transcriber)
        if not subtitles.ok:
            print(
                f"  [skip] {item.display_name} — "
                f"{HUMAN_REASONS.get(subtitles.reason, subtitles.reason)}"
            )
            continue

        cues = load_cues(subtitles.srt_path)
        candidates = store.candidates_for(item.lookup_key, args.category)
        accepted, _rejected = find_matches(
            cues,
            [c.quote for c in candidates],
            threshold=config.matching.threshold,
            window_max_cues=config.matching.window_max_cues,
        )

        # A time window is how you target the cold open — everything before the
        # theme song — without needing to detect the theme itself.
        if args.after is not None:
            accepted = [m for m in accepted if m.start >= args.after]
        if args.before is not None:
            accepted = [m for m in accepted if m.start <= args.before]

        if not accepted:
            continue

        cues_by_source[item.path] = cues
        for match in sorted(accepted, key=lambda m: -m.score)[: args.per_item]:
            moments.append(
                Moment(
                    source=item.path,
                    label=item.display_name,
                    match=match,
                    runtime=probed.duration,
                )
            )
        print(f"  [take] {item.display_name} — {len(accepted)} candidate(s)")

    if not moments:
        print(f"\n  No moments found across {scanned} item(s) with quotes.")
        return 1

    plan = plan_recap(
        moments, config,
        target_duration=args.duration,
        segments=args.segments,
        # Timestamps from different episodes are not comparable, so ranking by
        # score is the only sensible order here — and suppression, which exists
        # to stop two moments from ONE scene, would wrongly collapse moments
        # that merely happen to sit at the same minute of different episodes.
        spread="score",
        suppress=False,
    )
    if not plan.ok:
        print(f"\n  {plan.reason}")
        return 1

    print(f"\n  {len(plan.segments)} moment(s), {plan.duration:.1f}s")
    for segment in plan.segments:
        print(
            f"    {segment.offset:5.1f}s  {segment.moment.label[:44]:<44} "
            f"score {segment.match.score:3.0f}"
        )

    if args.dry_run:
        print("\n  Dry run — nothing rendered.")
        return 0

    destination = config.clips_dir / f"{args.out}.mp4"
    cues_by_segment = [
        cues_in_window(
            cues_by_source[segment.source], segment.window.start, segment.window.end
        )
        for segment in plan.segments
    ]

    ok, message = render_recap(
        plan.segments, cues_by_segment, destination, config,
        ass_path=config.cache_dir / "ass" / f"{args.out}.ass",
        gate=gate,
    )
    if not ok:
        print(f"\n  Render failed: {message}", file=sys.stderr)
        return 1

    if database is not None:
        record = ClipRecord(
            clip_id=clip_id(destination, args.out),
            source=str(plan.segments[0].source),
            lookup_key=args.out,
            quote=" / ".join(s.moment.label[:30] for s in plan.segments[:4]),
            category=args.category,
            start=0.0,
            end=round(plan.duration, 3),
            score=round(
                sum(s.match.score for s in plan.segments) / len(plan.segments), 1
            ),
            output=str(destination),
        )
        from src.caption_renderer import extract_thumbnail

        thumbnail = config.thumbs_dir / f"{args.out}.jpg"
        if extract_thumbnail(destination, thumbnail):
            record.thumbnail = str(thumbnail)
        database.record_short(record)
        database.close()

    print(f"\n  Wrote {destination}")
    return 0



# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Turn a local media library into vertical shorts.",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="path to config.yaml"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    def add_filters(target: argparse.ArgumentParser) -> None:
        target.add_argument("--kind", choices=("movie", "episode"))
        target.add_argument("--show", help="substring match on show name")
        target.add_argument("--title", help="substring match on title")
        target.add_argument("--limit", type=int, help="stop after N items")

    preflight = sub.add_parser("preflight", help="check this machine can do the work")
    preflight.set_defaults(handler=cmd_preflight)

    scan = sub.add_parser("scan", help="show how the library was parsed")
    add_filters(scan)
    scan.add_argument("-v", "--verbose", action="store_true",
                      help="also print lookup keys and paths")
    scan.set_defaults(handler=cmd_scan)

    audit = sub.add_parser("audit", help="report subtitle availability")
    add_filters(audit)
    audit.add_argument("--deep", action="store_true",
                       help="extract and validate each track (slower, catches "
                            "forced tracks)")
    audit.add_argument("--sample", type=int,
                       help="audit a random N items rather than everything")
    audit.add_argument("--out", help="write a CSV report here")
    audit.add_argument("--quiet", action="store_true", help="suppress per-item output")
    audit.set_defaults(handler=cmd_audit)

    make = sub.add_parser("make", help="produce shorts")
    add_filters(make)
    make.add_argument("--category", help="pull quotes from one category bucket")
    make.add_argument("--multipart", action="store_true",
                      help="split each title into a multi-part series")
    make.add_argument("--recap", action="store_true",
                      help="one clip cutting between several moments — a "
                           "rundown rather than a single joke")
    make.add_argument("--recap-duration", type=float,
                      help="target recap length in seconds (default 45)")
    make.add_argument("--recap-segments", type=int,
                      help="how many moments to cut between")
    make.add_argument("--parts", type=int,
                      help="cap the series length for this run")
    make.add_argument("--count", type=int,
                      help="stop once N clips have been produced")
    make.add_argument("--dry-run", action="store_true",
                      help="plan everything, render nothing")
    make.add_argument("--force", action="store_true",
                      help="re-make clips already in the ledger")
    make.add_argument("--quiet", action="store_true")
    make.add_argument("--ignore-viewers", action="store_true",
                      help="run even while someone is streaming (don't)")
    make.add_argument("--wait-timeout", type=float, default=-1,
                      help="seconds to wait for viewers to finish; "
                           "0 = don't wait, -1 = wait indefinitely (default)")
    make.set_defaults(handler=cmd_make)

    worker = sub.add_parser("worker", help="drain jobs queued by the dashboard")
    worker.add_argument("--once", action="store_true",
                        help="run a single job and exit")
    worker.add_argument("--poll", type=float, default=5.0,
                        help="seconds between queue checks")
    worker.add_argument("--quiet", action="store_true")
    worker.set_defaults(handler=cmd_worker)

    compile_ = sub.add_parser(
        "compile", help="pool moments from many episodes into one clip"
    )
    add_filters(compile_)
    compile_.add_argument("--out", default="Compilation",
                          help="output filename, without extension")
    compile_.add_argument("--category", help="pull from one category bucket")
    compile_.add_argument("--duration", type=float, default=90.0,
                          help="target total length in seconds")
    compile_.add_argument("--segments", type=int,
                          help="how many moments to include")
    compile_.add_argument("--per-item", type=int, default=1,
                          help="moments to take from each episode (default 1)")
    compile_.add_argument("--after", type=float,
                          help="ignore moments before this second")
    compile_.add_argument("--before", type=float,
                          help="ignore moments after this second — "
                               "e.g. --before 90 for cold opens")
    compile_.add_argument("--dry-run", action="store_true")
    compile_.add_argument("--quiet", action="store_true")
    compile_.add_argument("--ignore-viewers", action="store_true")
    compile_.set_defaults(handler=cmd_compile)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    try:
        return args.handler(config, args)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
