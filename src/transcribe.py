"""
Whisper transcription — the fallback when a file has no usable text subtitles.

This still honours the project's core invariant. Whisper produces a TIMED
transcript; the timestamps come from the audio, not from a model being asked
when something happened. It is the same kind of evidence as a subtitle track,
just generated rather than shipped.

Two properties matter as much as accuracy here:

* Cached. A two-hour film is transcribed once, and every short and every part
  of a multipart series reuses it. Without this, a 5-part movie split would pay
  the transcription cost five times.
* Interruptible. This box is also the Plex server, so a transcription has to be
  able to stop when someone starts watching. It runs in chunks, checkpointing
  after each one, and releases the GPU when it pauses — a suspended process
  would keep its VRAM, which Plex may want for its own transcoding.
"""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.config import Config

# Whisper models expect 16 kHz mono; feeding anything else means resampling
# inside the model anyway.
SAMPLE_RATE = 16_000


class TranscriptionPaused(Exception):
    """Raised when the viewer gate stops work at a chunk boundary."""


@dataclass
class Segment:
    start: float
    end: float
    text: str


def transcript_key(media: Path) -> str:
    """
    Cache identity: path plus size plus mtime.

    Including size and mtime means re-encoding or replacing a file invalidates
    its transcript, rather than silently reusing one that no longer matches.
    """
    try:
        stat = media.stat()
        fingerprint = f"{media.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    except OSError:
        fingerprint = str(media.resolve())
    return hashlib.sha1(fingerprint.encode()).hexdigest()[:16]


def _srt_timestamp(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(segments: list[Segment], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, segment in enumerate(segments, start=1):
        text = segment.text.strip()
        if not text:
            continue
        blocks.append(
            f"{index}\n"
            f"{_srt_timestamp(segment.start)} --> {_srt_timestamp(segment.end)}\n"
            f"{text}\n"
        )
    destination.write_text("\n".join(blocks), encoding="utf-8")
    return destination


def extract_audio_chunk(
    media: Path, destination: Path, start: float, duration: float
) -> bool:
    """Pull one span of audio as 16 kHz mono wav. -ss before -i, as everywhere."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(media), "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(destination),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=1800, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and destination.exists()


def media_duration(media: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(media)],
            capture_output=True, text=True, timeout=120, check=False,
        ).stdout.strip()
        value = float(out)
        return value if value > 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


class Transcriber:
    """
    Callable that turns a media path into a cached .srt.

    Returns (path, was_cached). Returns (None, False) when transcription could
    not complete — including when it was paused, which is not an error: the
    checkpoint is kept and the next attempt resumes from it.
    """

    def __init__(
        self,
        config: Config,
        *,
        should_pause: Callable[[], bool] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self.config = config
        self.should_pause = should_pause
        self.on_progress = on_progress
        self._model = None

    # -- paths -------------------------------------------------------------

    def transcript_path(self, media: Path) -> Path:
        return self.config.transcripts_dir / f"{transcript_key(media)}.srt"

    def checkpoint_path(self, media: Path) -> Path:
        return self.config.transcripts_dir / f"{transcript_key(media)}.partial.json"

    # -- model lifecycle ---------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed "
                "(pip install -r requirements-whisper.txt)"
            ) from exc

        settings = self.config.whisper
        device = settings.device
        compute_type = settings.compute_type

        if device == "cuda":
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() == 0:
                    device, compute_type = "cpu", settings.cpu_compute_type
            except Exception:
                device, compute_type = "cpu", settings.cpu_compute_type

        self._model = WhisperModel(
            settings.model, device=device, compute_type=compute_type
        )
        return self._model

    def release(self) -> None:
        """
        Drop the model and free its VRAM.

        Called when pausing. Suspending the process instead would keep the GPU
        memory allocated, and Plex may want that card for its own hardware
        transcoding — the very thing pausing is meant to avoid interfering with.
        """
        self._model = None
        gc.collect()

    # -- checkpointing -----------------------------------------------------

    def _read_checkpoint(self, media: Path) -> tuple[int, list[Segment]]:
        path = self.checkpoint_path(media)
        if not path.exists():
            return 0, []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
            return 0, []
        if raw.get("key") != transcript_key(media):
            return 0, []
        segments = [Segment(**segment) for segment in raw.get("segments", [])]
        return int(raw.get("chunks_done", 0)), segments

    def _write_checkpoint(
        self, media: Path, chunks_done: int, segments: list[Segment]
    ) -> None:
        path = self.checkpoint_path(media)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "key": transcript_key(media),
                "chunks_done": chunks_done,
                "segments": [vars(segment) for segment in segments],
            }),
            encoding="utf-8",
        )

    # -- the work ----------------------------------------------------------

    def __call__(self, media: Path) -> tuple[Path | None, bool]:
        destination = self.transcript_path(media)

        # Only a COMPLETE transcript is ever served from cache. A partial one
        # lives in the checkpoint file and is never mistaken for a finished
        # transcript.
        if destination.exists() and destination.stat().st_size > 0:
            return destination, True

        duration = media_duration(media)
        if not duration:
            return None, False

        chunk_seconds = max(30.0, self.config.whisper.chunk_seconds)
        total_chunks = max(1, int((duration + chunk_seconds - 1) // chunk_seconds))

        chunks_done, segments = self._read_checkpoint(media)

        try:
            for index in range(chunks_done, total_chunks):
                # The gate is checked BETWEEN chunks: a chunk in flight runs to
                # completion so its work isn't wasted, and the worst case is one
                # chunk of delay before the GPU is released.
                if self.should_pause and self.should_pause():
                    self._write_checkpoint(media, index, segments)
                    self.release()
                    raise TranscriptionPaused

                start = index * chunk_seconds
                length = min(chunk_seconds, duration - start)
                if length <= 0:
                    break

                segments += self._transcribe_chunk(media, start, length)
                self._write_checkpoint(media, index + 1, segments)

                if self.on_progress:
                    self.on_progress(index + 1, total_chunks)

        except TranscriptionPaused:
            return None, False
        except Exception:
            self.release()
            raise

        self.release()
        write_srt(segments, destination)
        self.checkpoint_path(media).unlink(missing_ok=True)
        return destination, False

    def _transcribe_chunk(
        self, media: Path, start: float, length: float
    ) -> list[Segment]:
        import tempfile

        model = self._load_model()
        settings = self.config.whisper

        with tempfile.TemporaryDirectory() as work_dir:
            wav = Path(work_dir) / "chunk.wav"
            if not extract_audio_chunk(media, wav, start, length):
                return []

            found, _info = model.transcribe(
                str(wav),
                beam_size=settings.beam_size,
                vad_filter=settings.vad_filter,
            )

            # Offsets are added back here: the model saw a chunk starting at
            # zero, but every consumer needs times relative to the whole file.
            return [
                Segment(
                    start=segment.start + start,
                    end=segment.end + start,
                    text=segment.text.strip(),
                )
                for segment in found
                if segment.text.strip()
            ]
