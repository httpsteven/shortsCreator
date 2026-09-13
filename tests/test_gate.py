"""
The viewer gate.

Fully testable here without Plex: the signal is a row in SQLite, so "someone
started watching" is a write, and process suspension is observable directly.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.config import Config
from src.db import Database
from src.gate import ViewerGate, run_pausable


def iso(offset_seconds: float = 0.0) -> str:
    moment = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return moment.isoformat(timespec="seconds")


@pytest.fixture
def database(tmp_path: Path) -> Database:
    with Database(tmp_path / "shorts.db") as db:
        yield db


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------


def test_runs_when_nobody_is_watching(config: Config, database: Database) -> None:
    database.set_lab_state(active_streams=0, last_activity_at=iso(-9999))
    gate = ViewerGate(config, database)

    assert not gate.should_pause()


def test_pauses_while_a_stream_is_active(config: Config, database: Database) -> None:
    database.set_lab_state(active_streams=1, last_activity_at=iso())
    gate = ViewerGate(config, database)

    status = gate.status()
    assert status.paused
    assert "1 stream" in status.reason


def test_hysteresis_delays_resume_after_the_last_viewer(
    config: Config, database: Database
) -> None:
    """
    Someone finishing an episode and starting the next shouldn't hear the box
    spin up in the gap between them.
    """
    database.set_lab_state(active_streams=0, last_activity_at=iso(-30))
    gate = ViewerGate(config, database)

    status = gate.status()
    assert status.paused
    assert "cooling down" in status.reason


def test_resumes_once_the_quiet_period_has_passed(
    config: Config, database: Database
) -> None:
    quiet_for = config.yield_to_viewers.resume_after_idle + 60
    database.set_lab_state(active_streams=0, last_activity_at=iso(-quiet_for))

    assert not ViewerGate(config, database).should_pause()


def test_stale_heartbeat_means_pause(config: Config, database: Database) -> None:
    """
    A dashboard that stopped updating tells us nothing about who is watching.
    The safe assumption is that someone is.
    """
    database.set_lab_state(active_streams=0, last_activity_at=iso(-9999))
    stale = iso(-(config.yield_to_viewers.heartbeat_stale_after + 120))
    with database.transaction() as connection:
        connection.execute("UPDATE lab_state SET updated_at=? WHERE id=1", (stale,))

    status = ViewerGate(config, database).status()
    assert status.paused
    assert "stale" in status.reason


def test_absent_heartbeat_means_run(config: Config, database: Database) -> None:
    """
    Never-written is not the same as stopped-updating. An optional integration
    that was never set up must not block the pipeline forever.
    """
    status = ViewerGate(config, database).status()

    assert not status.paused
    assert "no dashboard heartbeat" in status.reason


def test_disabled_gate_never_pauses(config: Config, database: Database) -> None:
    database.set_lab_state(active_streams=5, last_activity_at=iso())
    config.yield_to_viewers.enabled = False

    assert not ViewerGate(config, database).should_pause()


def test_force_open_overrides_active_streams(
    config: Config, database: Database
) -> None:
    database.set_lab_state(active_streams=3, last_activity_at=iso())
    gate = ViewerGate(config, database)
    assert gate.should_pause()

    gate.force_open()
    assert not gate.should_pause()


def test_transcode_only_mode_ignores_direct_play(
    config: Config, database: Database
) -> None:
    """Direct play costs almost nothing; transcoding is what hurts."""
    config.yield_to_viewers.pause_on_any_stream = False
    database.set_lab_state(active_streams=2, transcodes=0, last_activity_at=iso(-9999))

    assert not ViewerGate(config, database).should_pause()

    database.set_lab_state(active_streams=2, transcodes=1, last_activity_at=iso(-9999))
    assert ViewerGate(config, database).should_pause()


def test_database_error_does_not_stick_the_pipeline(config: Config) -> None:
    class Broken:
        def lab_state(self):
            raise RuntimeError("disk gone")

    status = ViewerGate(config, Broken()).status()
    assert not status.paused
    assert "unreadable" in status.reason


# --------------------------------------------------------------------------
# Actually suspending work in flight
# --------------------------------------------------------------------------


class ToggleGate:
    def __init__(self) -> None:
        self.paused = False

    def should_pause(self) -> bool:
        return self.paused


def test_running_process_is_suspended_and_resumed(tmp_path: Path) -> None:
    """
    SIGSTOP freezes the process where it is and SIGCONT continues it, so no
    work is lost — the alternative, killing and restarting, would throw away
    however much of the render had completed.
    """
    marker = tmp_path / "ticks"
    script = (
        "import time\n"
        f"path = {str(marker)!r}\n"
        "for _ in range(400):\n"
        "    open(path, 'a').write('x')\n"
        "    time.sleep(0.01)\n"
    )

    gate = ToggleGate()
    captured: dict[str, object] = {}

    def run() -> None:
        captured["result"] = run_pausable(
            [sys.executable, "-c", script], gate, poll=0.05
        )

    worker = threading.Thread(target=run)
    worker.start()

    # Let it get going, then pause.
    time.sleep(0.5)
    assert marker.exists() and marker.stat().st_size > 0
    gate.paused = True

    time.sleep(0.4)
    frozen_at = marker.stat().st_size
    time.sleep(0.5)

    # The defining assertion: no progress at all while suspended.
    assert marker.stat().st_size == frozen_at

    gate.paused = False
    worker.join(timeout=30)

    assert not worker.is_alive(), "process did not resume"
    assert captured["result"].returncode == 0
    assert marker.stat().st_size > frozen_at


def test_process_is_not_left_suspended_on_error(tmp_path: Path) -> None:
    """
    A stopped process with nothing left to resume it would hang forever,
    holding its file handles.
    """
    gate = ToggleGate()
    gate.paused = True

    with pytest.raises(Exception):
        run_pausable(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            gate, poll=0.05, timeout=0.6,
        )


def test_wait_until_clear_respects_a_timeout(config: Config, database: Database) -> None:
    """
    An overnight batch should wait for the house to go quiet; it should not be
    possible for it to wait forever with no way out.
    """
    database.set_lab_state(active_streams=1, last_activity_at=iso())
    gate = ViewerGate(config, database)

    started = time.monotonic()
    opened = gate.wait_until_clear(poll=0.05, timeout=0.3)

    assert opened is False
    assert time.monotonic() - started < 5


def test_wait_until_clear_returns_once_viewers_stop(
    config: Config, database: Database
) -> None:
    database.set_lab_state(active_streams=1, last_activity_at=iso())
    messages: list[str] = []

    # The real writer is the dashboard, in its own process. Simulated here by
    # flipping the state after a couple of polls rather than from a thread —
    # sqlite3 connections are single-thread by default, so a threaded writer
    # would be testing the test, not the gate.
    polls = 0

    class ClearsAfterTwoPolls:
        def lab_state(self):
            nonlocal polls
            polls += 1
            if polls > 2:
                database.set_lab_state(active_streams=0, last_activity_at=iso(-9999))
            return database.lab_state()

    gate = ViewerGate(config, ClearsAfterTwoPolls())

    opened = gate.wait_until_clear(
        poll=0.05, timeout=10, announce=messages.append, reannounce_every=0.01
    )

    assert opened is True
    # It keeps saying why it's waiting — a silent pause is indistinguishable
    # from a hung process.
    assert len(messages) >= 2
    assert "resuming" in messages[-1]
