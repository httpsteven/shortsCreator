#!/usr/bin/env python3
"""
Run the test suite and record the result in the dashboard database.

Exists so the dashboard's Tests panel reflects reality rather than being a
placeholder — and so a failing suite is visible from the same page as the
clips it would affect.

    python scripts/record_tests.py            # run and record
    python scripts/record_tests.py --quiet    # record only the summary
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError, load_config  # noqa: E402
from src.db import Database  # noqa: E402

SUMMARY = re.compile(r"(\d+) (passed|failed|error|errors|skipped)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--suite", default="pytest")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True, text=True, check=False,
    )
    duration = time.monotonic() - started

    output = (result.stdout or "") + (result.stderr or "")
    if not args.quiet:
        print(output)

    counts = {kind: int(number) for number, kind in SUMMARY.findall(output)}
    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)

    print(f"  {passed} passed, {failed} failed in {duration:.1f}s")

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"  not recorded: {exc}")
        return result.returncode

    if not config.db_path:
        print("  not recorded: no database.path configured")
        return result.returncode

    # The last lines of output are what's useful on a failure; the whole log
    # would bloat a row nobody reads in full.
    detail = "\n".join(output.strip().splitlines()[-25:]) if failed else None

    with Database(config.db_path) as database:
        with database.transaction() as connection:
            connection.execute(
                "INSERT INTO test_runs"
                " (run_id, suite, passed, failed, duration, detail, created_at)"
                " VALUES (?,?,?,?,?,?,datetime('now'))",
                (uuid.uuid4().hex[:12], args.suite, passed, failed, duration, detail),
            )
    print(f"  recorded in {config.db_path}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
