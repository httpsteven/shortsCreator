"""
The viewer gate: yielding the machine back to whoever is watching.

This box is also the Plex server. A render pegging four cores, or a Whisper job
holding the GPU, degrades playback for real people — so pipeline work stops when
anyone is streaming and resumes when they're done.

The signal comes from the dashboard, which already tracks live activity over
Plex's WebSocket and Tautulli. It writes a heartbeat into `lab_state`; this
reads it. No second connection to Plex, no duplicated polling.

Two judgement calls worth stating:

* A STALE heartbeat means pause. If the dashboard has stopped updating, we have
  no idea whether anyone is watching, and the safe assumption is that they are.
* A heartbeat that has NEVER been written means run. That's not a dashboard
  that died, it's a dashboard that was never set up — and refusing to work
  because an optional integration is absent would be wrong.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from src.config import Config


@dataclass
class GateStatus:
    paused: bool
    reason: str

    def __str__(self) -> str:
        return self.reason


def _parse(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class ViewerGate:
    """Decides whether heavy work may run right now."""

    def __init__(self, config: Config, database=None) -> None:
        self.config = config
        self.database = database
        self._forced_open = False

    def force_open(self) -> None:
        """'Run anyway' — ignore the gate until the current queue drains."""
        self._forced_open = True

    def status(self) -> GateStatus:
        settings = self.config.yield_to_viewers

        if not settings.enabled or self._forced_open:
            return GateStatus(False, "gate disabled")
        if self.database is None:
            return GateStatus(False, "no database configured")

        try:
            row = self.database.lab_state()
        except Exception as exc:
            # A database problem must not become a stuck pipeline.
            return GateStatus(False, f"lab_state unreadable ({exc})")

        if row is None:
            return GateStatus(False, "no dashboard heartbeat yet")

        now = datetime.now(timezone.utc)

        updated = _parse(row["updated_at"])
        if updated is None:
            return GateStatus(True, "heartbeat has no timestamp — assuming viewers")

        age = (now - updated).total_seconds()
        if age > settings.heartbeat_stale_after:
            return GateStatus(
                True,
                f"dashboard heartbeat is {age:.0f}s stale — assuming viewers",
            )

        streams = int(row["active_streams"] or 0)
        transcodes = int(row["transcodes"] or 0)
        watching = streams > 0 if settings.pause_on_any_stream else transcodes > 0

        if watching:
            what = f"{streams} stream(s)" if settings.pause_on_any_stream else f"{transcodes} transcode(s)"
            return GateStatus(True, f"paused — {what} active")

        # Hysteresis. Someone finishing an episode and starting the next
        # shouldn't hear the box spin up in the gap between them.
        last_activity = _parse(row["last_activity_at"])
        if last_activity is not None:
            quiet_for = (now - last_activity).total_seconds()
            if quiet_for < settings.resume_after_idle:
                remaining = settings.resume_after_idle - quiet_for
                return GateStatus(
                    True, f"paused — cooling down ({remaining / 60:.1f}m left)"
                )

        return GateStatus(False, "idle — clear to run")

    def should_pause(self) -> bool:
        return self.status().paused

    def wait_until_clear(
        self,
        *,
        poll: float = 10.0,
        announce=None,
        timeout: float | None = None,
        reannounce_every: float = 60.0,
    ) -> bool:
        """
        Block until the gate opens. Used between jobs, never inside one.

        Re-announces periodically rather than printing once and going quiet: a
        legitimate pause and a hung process look identical from the outside, and
        this is the only thing that distinguishes them.

        Returns True if the gate opened, False if `timeout` elapsed first.
        """
        started = time.monotonic()
        last_announced = 0.0
        announced_any = False

        while True:
            status = self.status()
            if not status.paused:
                if announced_any and announce:
                    announce("lab is quiet — resuming")
                return True

            now = time.monotonic()
            if announce and (not announced_any or now - last_announced >= reannounce_every):
                waited = now - started
                suffix = f" (waiting {waited / 60:.0f}m)" if waited >= 60 else ""
                announce(f"{status}{suffix}")
                last_announced = now
                announced_any = True

            if timeout is not None and now - started > timeout:
                return False

            time.sleep(poll)


def run_pausable(
    command: list[str],
    gate: ViewerGate | None = None,
    *,
    poll: float = 2.0,
    timeout: float | None = None,
    nice: int | None = None,
    ionice_class: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run an ffmpeg command that can be suspended mid-flight.

    SIGSTOP/SIGCONT rather than kill-and-restart: the process freezes exactly
    where it is, releases the CPU immediately, and resumes with no work lost.
    That is the right trade for ffmpeg, whose memory footprint is small.

    It is deliberately NOT how Whisper pauses — a suspended process keeps its
    VRAM, and Plex may want that GPU. See `transcribe.Transcriber`.
    """
    def preexec() -> None:
        # Runs in the child between fork and exec. Both calls are best-effort:
        # this box is shared with Plex, so being a polite neighbour matters, but
        # failing to lower our own priority is never a reason to abandon a
        # render.
        if nice is not None:
            try:
                os.nice(nice)
            except OSError:
                pass
        if ionice_class is not None and hasattr(os, "setpriority"):
            try:
                # Linux only — there is no portable ionice syscall wrapper in
                # the stdlib, so shell out only where it exists.
                import shutil as _shutil

                if _shutil.which("ionice"):
                    os.system(f"ionice -c {ionice_class} -p {os.getpid()} >/dev/null 2>&1")
            except Exception:
                pass

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=preexec,
    )

    suspended = False
    started = time.monotonic()

    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=poll)
                return subprocess.CompletedProcess(
                    command, process.returncode, stdout, stderr
                )
            except subprocess.TimeoutExpired:
                pass

            if timeout is not None and time.monotonic() - started > timeout:
                process.kill()
                process.communicate()
                raise subprocess.TimeoutExpired(command, timeout)

            if gate is None:
                continue

            should_pause = gate.should_pause()
            if should_pause and not suspended:
                process.send_signal(signal.SIGSTOP)
                suspended = True
            elif not should_pause and suspended:
                process.send_signal(signal.SIGCONT)
                suspended = False
    except BaseException:
        # Never leave a stopped process behind — it would sit suspended forever,
        # holding its file handles, with nothing left to resume it.
        if suspended:
            try:
                process.send_signal(signal.SIGCONT)
            except ProcessLookupError:
                pass
        process.kill()
        process.wait()
        raise
