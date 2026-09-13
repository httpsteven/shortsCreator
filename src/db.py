"""
The SQLite contract with the Home Lab Dashboard.

One file, two processes: this pipeline writes, the dashboard reads (and writes
only to `jobs`, `review` and `lab_state`). Two pragmas make that safe:

* journal_mode=WAL — readers don't block the writer and the writer doesn't block
  readers. Without it the dashboard's polling would intermittently lock the
  pipeline out mid-render.
* busy_timeout — wait for a lock rather than failing instantly with SQLITE_BUSY.

Schema changes are owned here. The dashboard never migrates; it reads whatever
version it finds, which is why every column added later must be nullable.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 5000

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS media_items (
    path        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    show        TEXT,
    season      INTEGER,
    episode     INTEGER,
    year        INTEGER,
    lookup_key  TEXT NOT NULL,
    duration    REAL,
    seen_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_media_show ON media_items(show);
CREATE INDEX IF NOT EXISTS idx_media_kind ON media_items(kind);

CREATE TABLE IF NOT EXISTS subtitle_audit (
    path           TEXT PRIMARY KEY REFERENCES media_items(path) ON DELETE CASCADE,
    stream_count   INTEGER NOT NULL DEFAULT 0,
    codecs         TEXT,
    languages      TEXT,
    forced_streams INTEGER NOT NULL DEFAULT 0,
    sidecars       INTEGER NOT NULL DEFAULT 0,
    level1_status  TEXT,
    level2_status  TEXT,
    reason         TEXT,
    explanation    TEXT,
    provenance     TEXT,
    cue_count      INTEGER,
    coverage       REAL,
    usable         INTEGER NOT NULL DEFAULT 0,
    needs_whisper  INTEGER NOT NULL DEFAULT 0,
    est_minutes    REAL,
    error          TEXT,
    checked_at     TEXT NOT NULL
);

-- Caption sources added from the dashboard. Read as tier 0 of the acquisition
-- ladder, ahead of anything embedded in the file.
CREATE TABLE IF NOT EXISTS subtitle_overrides (
    path         TEXT PRIMARY KEY,
    srt_path     TEXT,
    stream_index INTEGER,
    force_whisper INTEGER NOT NULL DEFAULT 0,
    note         TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS series (
    series_id  TEXT PRIMARY KEY,
    source     TEXT NOT NULL,
    title      TEXT NOT NULL,
    part_total INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shorts (
    clip_id     TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    lookup_key  TEXT NOT NULL,
    title       TEXT,
    show        TEXT,
    quote       TEXT NOT NULL,
    category    TEXT,
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    duration    REAL NOT NULL,
    score       REAL NOT NULL,
    output      TEXT NOT NULL,
    thumbnail   TEXT,
    provenance  TEXT,
    series_id   TEXT REFERENCES series(series_id) ON DELETE SET NULL,
    part_index  INTEGER,
    part_total  INTEGER,
    created_at  TEXT NOT NULL,
    deleted_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_shorts_series  ON shorts(series_id);
CREATE INDEX IF NOT EXISTS idx_shorts_show    ON shorts(show);
CREATE INDEX IF NOT EXISTS idx_shorts_live    ON shorts(deleted_at);

-- Written by the dashboard, read by the worker. Nullable everywhere so a
-- dashboard that hasn't been updated yet simply leaves it empty.
CREATE TABLE IF NOT EXISTS review (
    clip_id   TEXT PRIMARY KEY REFERENCES shorts(clip_id) ON DELETE CASCADE,
    status    TEXT,
    note      TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    command    TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    items_seen INTEGER NOT NULL DEFAULT 0,
    produced   INTEGER NOT NULL DEFAULT 0,
    skipped    INTEGER NOT NULL DEFAULT 0,
    reasons    TEXT,
    error      TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT NOT NULL,
    payload     TEXT,
    status      TEXT NOT NULL DEFAULT 'queued',
    progress    REAL NOT NULL DEFAULT 0,
    detail      TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, job_id);

CREATE TABLE IF NOT EXISTS test_runs (
    run_id     TEXT PRIMARY KEY,
    suite      TEXT NOT NULL,
    passed     INTEGER NOT NULL DEFAULT 0,
    failed     INTEGER NOT NULL DEFAULT 0,
    duration   REAL,
    detail     TEXT,
    created_at TEXT NOT NULL
);

-- The viewer-activity heartbeat. The dashboard owns this table; the worker only
-- reads it. See yield_to_viewers.
CREATE TABLE IF NOT EXISTS lab_state (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    active_streams   INTEGER NOT NULL DEFAULT 0,
    transcodes       INTEGER NOT NULL DEFAULT 0,
    last_activity_at TEXT,
    updated_at       TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if not readonly:
        path.parent.mkdir(parents=True, exist_ok=True)

    uri = f"file:{path}?mode=ro" if readonly else f"file:{path}"
    connection = sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT_MS / 1000)
    connection.row_factory = sqlite3.Row

    if not readonly:
        # WAL is persistent once set, but setting it every time is harmless and
        # means a database created by hand still gets it.
        connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


class Database:
    """Writer-side access. The dashboard opens the same file read-only."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.connection = connect(self.path)
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection:
            yield self.connection

    def _migrate(self) -> None:
        with self.connection:
            self.connection.executescript(SCHEMA)
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    # -- media and audit ---------------------------------------------------

    def upsert_media(self, item, duration: float | None = None) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO media_items
                    (path, kind, title, show, season, episode, year,
                     lookup_key, duration, seen_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(path) DO UPDATE SET
                    kind=excluded.kind, title=excluded.title, show=excluded.show,
                    season=excluded.season, episode=excluded.episode,
                    year=excluded.year, lookup_key=excluded.lookup_key,
                    duration=COALESCE(excluded.duration, media_items.duration),
                    seen_at=excluded.seen_at
                """,
                (
                    str(item.path), item.kind, item.title, item.show,
                    item.season, item.episode, item.year, item.lookup_key,
                    duration, now(),
                ),
            )

    def upsert_audit(self, row) -> None:
        self.upsert_media(row.item, row.duration)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO subtitle_audit
                    (path, stream_count, codecs, languages, forced_streams,
                     sidecars, level1_status, level2_status, reason, explanation,
                     provenance, cue_count, coverage, usable, needs_whisper,
                     est_minutes, error, checked_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(path) DO UPDATE SET
                    stream_count=excluded.stream_count, codecs=excluded.codecs,
                    languages=excluded.languages,
                    forced_streams=excluded.forced_streams,
                    sidecars=excluded.sidecars,
                    level1_status=excluded.level1_status,
                    level2_status=excluded.level2_status,
                    reason=excluded.reason, explanation=excluded.explanation,
                    provenance=excluded.provenance, cue_count=excluded.cue_count,
                    coverage=excluded.coverage, usable=excluded.usable,
                    needs_whisper=excluded.needs_whisper,
                    est_minutes=excluded.est_minutes, error=excluded.error,
                    checked_at=excluded.checked_at
                """,
                (
                    str(row.item.path), row.stream_count, "|".join(row.codecs),
                    "|".join(row.languages), row.forced_flags, row.sidecars,
                    str(row.level1), str(row.level2) if row.level2 else None,
                    str(row.reason) if row.reason else None, row.explain(),
                    str(row.provenance) if row.provenance else None,
                    row.cue_count, row.coverage, int(row.usable),
                    int(row.needs_whisper), row.whisper_minutes or None,
                    row.error, now(),
                ),
            )

    # -- shorts ------------------------------------------------------------

    def record_short(self, record, item=None) -> None:
        duration = record.end - record.start
        with self.connection:
            if record.series and record.part_total:
                self.connection.execute(
                    "INSERT OR IGNORE INTO series"
                    " (series_id, source, title, part_total, created_at)"
                    " VALUES (?,?,?,?,?)",
                    (
                        record.series, record.source,
                        item.display_name if item else record.lookup_key,
                        record.part_total, now(),
                    ),
                )
            self.connection.execute(
                """
                INSERT INTO shorts
                    (clip_id, source, lookup_key, title, show, quote, category,
                     start, end, duration, score, output, thumbnail, provenance,
                     series_id, part_index, part_total, created_at, deleted_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)
                ON CONFLICT(clip_id) DO UPDATE SET
                    output=excluded.output, thumbnail=excluded.thumbnail,
                    score=excluded.score, start=excluded.start, end=excluded.end,
                    duration=excluded.duration, deleted_at=NULL
                """,
                (
                    record.clip_id, record.source, record.lookup_key,
                    item.display_name if item else None,
                    item.show if item else None,
                    record.quote, record.category, record.start, record.end,
                    duration, record.score, record.output, record.thumbnail,
                    record.provenance, record.series, record.part_index,
                    record.part_total, record.created_at,
                ),
            )

    def soft_delete_short(self, clip_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE shorts SET deleted_at=? WHERE clip_id=?", (now(), clip_id)
            )

    # -- runs --------------------------------------------------------------

    def start_run(self, run_id: str, command: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO runs (run_id, command, started_at)"
                " VALUES (?,?,?)",
                (run_id, command, now()),
            )

    def finish_run(
        self, run_id: str, *, items_seen: int, produced: int,
        skipped: int, reasons: dict[str, int] | None = None,
        error: str | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE runs SET finished_at=?, items_seen=?, produced=?,"
                " skipped=?, reasons=?, error=? WHERE run_id=?",
                (
                    now(), items_seen, produced, skipped,
                    json.dumps(reasons or {}), error, run_id,
                ),
            )

    # -- jobs --------------------------------------------------------------

    def enqueue(self, job_type: str, payload: dict[str, Any] | None = None) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO jobs (type, payload, created_at) VALUES (?,?,?)",
                (job_type, json.dumps(payload or {}), now()),
            )
        return int(cursor.lastrowid)

    def claim_job(self) -> sqlite3.Row | None:
        """
        Take the oldest queued job.

        The UPDATE..WHERE status='queued' is what makes this safe with more than
        one worker: only one can win the transition, so a job is never run twice.
        """
        with self.connection:
            row = self.connection.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY job_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None

            changed = self.connection.execute(
                "UPDATE jobs SET status='running', started_at=?"
                " WHERE job_id=? AND status='queued'",
                (now(), row["job_id"]),
            ).rowcount
            if changed == 0:
                return None

        return row

    def finish_job(
        self, job_id: int, *, status: str = "done",
        detail: str | None = None, error: str | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE jobs SET status=?, finished_at=?, detail=?, error=?,"
                " progress=1 WHERE job_id=?",
                (status, now(), detail, error, job_id),
            )

    def requeue_job(self, job_id: int) -> None:
        """Put a paused job back so it resumes when viewers are done."""
        with self.connection:
            self.connection.execute(
                "UPDATE jobs SET status='queued', started_at=NULL WHERE job_id=?",
                (job_id,),
            )

    def update_progress(self, job_id: int, progress: float, detail: str = "") -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE jobs SET progress=?, detail=? WHERE job_id=?",
                (max(0.0, min(1.0, progress)), detail, job_id),
            )

    # -- overrides and lab state ------------------------------------------

    def override_for(self, path: Path) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM subtitle_overrides WHERE path=?", (str(path),)
        ).fetchone()

    def set_override(
        self, path: Path, *, srt_path: Path | None = None,
        stream_index: int | None = None, force_whisper: bool = False,
        note: str | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO subtitle_overrides"
                " (path, srt_path, stream_index, force_whisper, note, created_at)"
                " VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(path) DO UPDATE SET srt_path=excluded.srt_path,"
                " stream_index=excluded.stream_index,"
                " force_whisper=excluded.force_whisper, note=excluded.note,"
                " created_at=excluded.created_at",
                (
                    str(path), str(srt_path) if srt_path else None,
                    stream_index, int(force_whisper), note, now(),
                ),
            )

    def lab_state(self) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM lab_state WHERE id=1"
        ).fetchone()

    def set_lab_state(
        self, *, active_streams: int, transcodes: int = 0,
        last_activity_at: str | None = None,
    ) -> None:
        """Normally the dashboard's job; exposed here for testing the gate."""
        with self.connection:
            self.connection.execute(
                "INSERT INTO lab_state"
                " (id, active_streams, transcodes, last_activity_at, updated_at)"
                " VALUES (1,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET"
                " active_streams=excluded.active_streams,"
                " transcodes=excluded.transcodes,"
                " last_activity_at=excluded.last_activity_at,"
                " updated_at=excluded.updated_at",
                (active_streams, transcodes, last_activity_at, now()),
            )
