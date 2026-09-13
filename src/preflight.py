"""
Environment checks that run before any real work.

Each of these has a failure mode that is confusing if it surfaces later:
missing libass produces video with no captions and no error; a library root
that doesn't resolve produces an empty run indistinguishable from "nothing to
do"; an encoder the local ffmpeg wasn't built with fails deep inside a render.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.config import Config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = False

    @property
    def symbol(self) -> str:
        if self.ok:
            return "ok  "
        return "FAIL" if self.fatal else "warn"


def _ffmpeg_banner() -> str:
    try:
        return subprocess.run(
            ["ffmpeg", "-hide_banner", "-version"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _ffmpeg_lists(kind: str) -> str:
    try:
        return subprocess.run(
            ["ffmpeg", "-hide_banner", f"-{kind}"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def run_checks(config: Config) -> list[Check]:
    checks: list[Check] = []

    # -- ffmpeg / ffprobe --------------------------------------------------
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        checks.append(
            Check(tool, bool(path), path or f"{tool} not found on PATH", fatal=True)
        )

    banner = _ffmpeg_banner()

    # libass is what burns captions in. Without it the ass filter is missing and
    # renders silently produce uncaptioned video — the worst kind of failure,
    # because the output looks fine until you watch it.
    has_libass = "--enable-libass" in banner or "ass" in _ffmpeg_lists("filters")
    checks.append(
        Check(
            "libass",
            has_libass,
            "caption burn-in available"
            if has_libass
            else "ffmpeg has no libass — captions would be silently omitted",
            fatal=True,
        )
    )

    # -- encoder -----------------------------------------------------------
    encoders = _ffmpeg_lists("encoders")
    wanted = config.video.encoder
    has_encoder = wanted in encoders
    checks.append(
        Check(
            f"encoder:{wanted}",
            has_encoder,
            "available" if has_encoder else f"ffmpeg cannot encode with {wanted}",
            fatal=True,
        )
    )

    if "h264_nvenc" in encoders and wanted != "h264_nvenc":
        checks.append(
            Check(
                "encoder:h264_nvenc",
                True,
                "available but unused — set video.encoder to use the GPU",
            )
        )

    # -- GPU ---------------------------------------------------------------
    checks.append(_nvidia_check())
    checks.append(_whisper_check(config))

    # -- library roots -----------------------------------------------------
    for label, root in (("roots.movies", config.roots.movies), ("roots.tv", config.roots.tv)):
        if root is None:
            continue
        exists = root.is_dir()
        checks.append(
            Check(
                label,
                exists,
                str(root) if exists else f"{root} does not exist or is not a directory",
                fatal=True,
            )
        )

    # -- writable output ---------------------------------------------------
    checks.append(_writable("output.dir", config.output_dir))
    checks.append(_writable("cache.dir", config.cache_dir))

    if config.db_path:
        checks.append(_writable("database.path", config.db_path.parent))

    return checks


def _writable(label: str, directory: Path) -> Check:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write-test"
        probe.write_text("")
        probe.unlink()
        return Check(label, True, str(directory))
    except OSError as exc:
        return Check(label, False, f"{directory} is not writable: {exc}", fatal=True)


def _nvidia_check() -> Check:
    if shutil.which("nvidia-smi") is None:
        return Check("gpu", False, "no nvidia-smi — CPU only (fine, just slower)")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return Check("gpu", False, "nvidia-smi present but did not respond")

    if result.returncode != 0:
        return Check("gpu", False, "nvidia-smi failed — driver problem?")

    return Check("gpu", True, result.stdout.strip().replace("\n", "; "))


def _whisper_check(config: Config) -> Check:
    if not config.whisper.enabled:
        return Check("whisper", True, "disabled in config")

    try:
        import ctranslate2  # noqa: F401
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        return Check(
            "whisper",
            False,
            "faster-whisper not installed — items without text subtitles will "
            "be skipped (pip install -r requirements-whisper.txt)",
        )

    import ctranslate2

    try:
        cuda_devices = ctranslate2.get_cuda_device_count()
    except Exception:
        cuda_devices = 0

    if config.whisper.device == "cuda" and cuda_devices == 0:
        return Check(
            "whisper",
            False,
            "configured for CUDA but ctranslate2 sees no GPU — will fall back "
            "to CPU (check cuDNN/cuBLAS are installed)",
        )

    detail = f"{config.whisper.model} on {config.whisper.device}"
    if config.whisper.device == "cuda":
        detail += f" ({cuda_devices} GPU) compute_type={config.whisper.compute_type}"
        if config.whisper.compute_type == "float16":
            return Check(
                "whisper", False,
                detail + " — float16 is a poor choice on Pascal (GTX 10xx), "
                "which runs fp16 at 1/64 rate; use int8_float32",
            )
    return Check("whisper", True, detail)


def report(checks: list[Check]) -> bool:
    """Print the results. Returns False when any fatal check failed."""
    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"  [{check.symbol}] {check.name:<{width}}  {check.detail}")

    failures = [check for check in checks if not check.ok and check.fatal]
    if failures:
        print()
        print(f"  {len(failures)} fatal problem(s) — fix these before running.")
    return not failures
