"""
The job worker: drains the queue the dashboard writes to.

Why a queue rather than an HTTP endpoint the dashboard calls directly: the work
here is long (a Whisper pass can run for minutes) and must be interruptible. A
web request handler is the wrong place for either. The dashboard inserts a row
and returns immediately; this process does the work when the machine is free.

The viewer gate is checked before every job and, for renders, continuously
during one. Nothing here outranks someone watching a film.
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from pathlib import Path

from src.audit import audit_library
from src.config import Config
from src.db import Database
from src.gate import ViewerGate
from src.producer import produce
from src.quote_finder import QuoteStore, QuotesError
from src.sources.filesystem import FilesystemSource
from src.state import State

POLL_SECONDS = 5.0


class Worker:
    def __init__(self, config: Config, *, verbose: bool = True) -> None:
        if not config.db_path:
            raise ValueError(
                "worker needs database.path set — that is how the dashboard "
                "queues work."
            )
        self.config = config
        self.database = Database(config.db_path)
        self.gate = ViewerGate(config, self.database)
        self.verbose = verbose
        self.state = State.load(config.output_dir / "state.json")

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # -- main loop ---------------------------------------------------------

    def run_forever(self, *, poll: float = POLL_SECONDS) -> None:
        self.log("worker: waiting for jobs")
        last_status = ""
        while True:
            status = self.gate.status()
            if status.paused:
                if status.reason != last_status:
                    self.log(f"worker: {status.reason}")
                    last_status = status.reason
                time.sleep(poll)
                continue

            if last_status:
                self.log("worker: resuming")
                last_status = ""

            job = self.database.claim_job()
            if job is None:
                time.sleep(poll)
                continue

            self.run_job(job)

    def run_job(self, job) -> None:
        job_id = int(job["job_id"])
        job_type = job["type"]
        try:
            payload = json.loads(job["payload"] or "{}")
        except json.JSONDecodeError:
            payload = {}

        self.log(f"worker: job {job_id} ({job_type}) starting")

        handlers = {
            "render": self._job_render,
            "audit": self._job_audit,
            "transcribe": self._job_transcribe,
            "delete": self._job_delete,
        }
        handler = handlers.get(job_type)

        if handler is None:
            self.database.finish_job(
                job_id, status="failed", error=f"unknown job type {job_type!r}"
            )
            return

        try:
            detail = handler(job_id, payload)
        except PausedMidJob:
            # Not a failure. Put it back so it resumes when viewers are done.
            self.database.requeue_job(job_id)
            self.log(f"worker: job {job_id} paused, requeued")
            return
        except Exception as exc:
            self.database.finish_job(
                job_id, status="failed",
                error=f"{type(exc).__name__}: {exc}",
                detail=traceback.format_exc()[-2000:],
            )
            self.log(f"worker: job {job_id} failed — {exc}")
            return

        self.database.finish_job(job_id, detail=detail)
        self.log(f"worker: job {job_id} done — {detail}")

    # -- handlers ----------------------------------------------------------

    def _items(self, payload: dict):
        source = FilesystemSource(self.config.roots.movies, self.config.roots.tv)
        items = list(source.iter_items())

        if payload.get("path"):
            wanted = str(payload["path"])
            items = [item for item in items if str(item.path) == wanted]
        if payload.get("show"):
            wanted = str(payload["show"]).casefold()
            items = [i for i in items if i.show and wanted in i.show.casefold()]
        if payload.get("title"):
            wanted = str(payload["title"]).casefold()
            items = [i for i in items if wanted in i.title.casefold()]
        return items

    def _override_for(self, path: Path) -> tuple[Path | None, bool]:
        """
        Caption sources added from the dashboard — tier 0 of the ladder.

        An uploaded file is still validated like any other track; being chosen
        by a human doesn't exempt it from the gates.
        """
        row = self.database.override_for(path)
        if row is None:
            return None, False

        srt = row["srt_path"]
        return (Path(srt) if srt else None), bool(row["force_whisper"])

    def _job_render(self, job_id: int, payload: dict) -> str:
        items = self._items(payload)
        if not items:
            return "no matching media"

        try:
            store = QuoteStore.load(self.config.quotes.path)
        except QuotesError as exc:
            raise RuntimeError(str(exc)) from exc

        from src.pipeline import build_transcriber

        transcriber = build_transcriber(self.config, quiet=True)
        made = 0

        for position, item in enumerate(items, start=1):
            if self.gate.should_pause():
                raise PausedMidJob

            override, force_whisper = self._override_for(item.path)

            outcome = produce(
                item, self.config, store, self.state,
                override=override,
                force_whisper=force_whisper,
                category=payload.get("category"),
                multipart=bool(payload.get("multipart")),
                parts=payload.get("parts"),
                transcriber=transcriber,
                force=bool(payload.get("force")),
                gate=self.gate,
            )

            for record in outcome.produced:
                self.database.record_short(record, item)
                made += 1

            self.state.save()
            self.database.update_progress(
                job_id, position / len(items), f"{made} clip(s) so far"
            )

        return f"produced {made} clip(s) from {len(items)} item(s)"

    def _job_audit(self, job_id: int, payload: dict) -> str:
        items = self._items(payload)
        if not items:
            return "no matching media"

        def progress(position: int, total: int, _item) -> None:
            self.database.update_progress(job_id, position / total, f"{position}/{total}")

        rows = audit_library(
            items, self.config, deep=bool(payload.get("deep", True)),
            sample=payload.get("sample"), on_progress=progress,
        )
        for row in rows:
            self.database.upsert_audit(row)

        usable = sum(1 for row in rows if row.usable)
        return f"audited {len(rows)} item(s); {usable} usable"

    def _job_transcribe(self, job_id: int, payload: dict) -> str:
        items = self._items(payload)
        if not items:
            return "no matching media"

        from src.transcribe import Transcriber

        transcriber = Transcriber(
            self.config,
            should_pause=self.gate.should_pause,
            on_progress=lambda done, total: self.database.update_progress(
                job_id, done / total, f"chunk {done}/{total}"
            ),
        )

        done = 0
        for item in items:
            path, cached = transcriber(item.path)
            if path is None:
                # None means paused (checkpoint kept) rather than failed.
                raise PausedMidJob
            done += 0 if cached else 1

        return f"transcribed {done} item(s) ({len(items) - done} already cached)"

    def _job_delete(self, job_id: int, payload: dict) -> str:
        """
        Remove generated artifacts only.

        Scoped deliberately: clips, thumbnails and the .ass beside them. Nothing
        under the library roots is ever a delete target, and the paths are
        checked against the configured output directory before anything is
        unlinked.
        """
        clip_ids = payload.get("clip_ids") or []
        if isinstance(clip_ids, str):
            clip_ids = [clip_ids]

        removed = 0
        for clip_id in clip_ids:
            row = self.database.connection.execute(
                "SELECT output, thumbnail FROM shorts WHERE clip_id=?", (clip_id,)
            ).fetchone()
            if row is None:
                continue

            for candidate in (row["output"], row["thumbnail"]):
                if candidate and _inside(Path(candidate), self.config.output_dir):
                    Path(candidate).unlink(missing_ok=True)
                    removed += 1

            self.database.soft_delete_short(clip_id)
            self.state.forget(clip_id)

        self.state.save()
        return f"deleted {removed} file(s) for {len(clip_ids)} clip(s)"


class PausedMidJob(Exception):
    """A job stopped at a safe point because someone started watching."""


def _inside(path: Path, directory: Path) -> bool:
    """Refuse to unlink anything outside the output directory."""
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
