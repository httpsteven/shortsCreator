"""
ASS subtitle generation and the vertical 9:16 render.

Two things here are easy to get wrong and hard to notice:

1. Commas in dialogue text must NOT be escaped. The ASS `Dialogue:` line splits
   on commas only for its fixed leading fields — the Text field is "everything
   after the last one". Escaping commas as "\\," renders a literal backslash on
   screen, and nothing catches it except looking at a frame.

2. Cue times must already be relative to the clip, not the source file. That
   shift happens in `subtitle_utils.cues_in_window`; this module assumes it has
   been done and renders whatever it is handed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.clip_extractor import ClipWindow, build_input_args, encoder_args
from src.config import CaptionStyle, Config, PartBadgeStyle
from src.gate import run_pausable
from src.subtitle_utils import Cue

# ASS uses -1 for true. Not 1 — that is a different value and renders unbolded.
_ASS_TRUE = -1
_ASS_FALSE = 0

_STYLE_FORMAT = (
    "Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
    "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
    "MarginV, Encoding"
)

_EVENT_FORMAT = (
    "Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
)


def ass_time(seconds: float) -> str:
    """ASS timestamps are H:MM:SS.cc — one hour digit, centiseconds."""
    seconds = max(0.0, seconds)
    centiseconds = int(round(seconds * 100))
    hours, centiseconds = divmod(centiseconds, 360_000)
    minutes, centiseconds = divmod(centiseconds, 6_000)
    secs, centiseconds = divmod(centiseconds, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def ass_text(text: str) -> str:
    """
    Prepare dialogue text for the Text field.

    Newlines become "\\N", the ASS hard line break, preserving the break the
    subtitle author chose. libass still soft-wraps anything too wide for the
    frame, so this only adds breaks, never removes the ability to wrap.

    Commas are left ALONE on purpose; see the module docstring.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return r"\N".join(lines)


def _style_line(name: str, style: CaptionStyle | PartBadgeStyle) -> str:
    bold = _ASS_TRUE if style.bold else _ASS_FALSE
    return (
        f"Style: {name},{style.font},{style.font_size},"
        f"{style.primary_colour},{style.primary_colour},"
        f"{style.outline_colour},{style.back_colour},"
        f"{bold},0,0,0,100,100,0,0,"
        f"{style.border_style},{style.outline},{style.shadow},"
        f"{style.alignment},{style.margin_l},{style.margin_r},{style.margin_v},1"
    )


def build_ass(
    cues: list[Cue],
    config: Config,
    *,
    part_label: str | None = None,
    clip_duration: float | None = None,
) -> str:
    """
    Build a complete .ass file for one clip.

    PlayResX/Y are set to the OUTPUT resolution so font sizes and margins in
    config are in real output pixels. Without that, libass assumes 384x288 and
    every size is wrong by a factor of three.
    """
    video = config.video

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {video.width}",
        f"PlayResY: {video.height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: None",
        "",
        "[V4+ Styles]",
        f"Format: {_STYLE_FORMAT}",
        _style_line("Caption", config.captions),
    ]

    if part_label and config.part_badge.enabled:
        lines.append(_style_line("PartBadge", config.part_badge))

    lines += ["", "[Events]", f"Format: {_EVENT_FORMAT}"]

    for cue in cues:
        text = ass_text(cue.text)
        if not text:
            continue
        lines.append(
            f"Dialogue: 0,{ass_time(cue.start)},{ass_time(cue.end)},"
            f"Caption,,0,0,0,,{text}"
        )

    # The badge is one dialogue line spanning the whole clip — no second ffmpeg
    # pass, no concat, no extra encode.
    if part_label and config.part_badge.enabled:
        end = clip_duration if clip_duration else (cues[-1].end if cues else 60.0)
        lines.append(
            f"Dialogue: 1,{ass_time(0)},{ass_time(end)},"
            f"PartBadge,,0,0,0,,{ass_text(part_label)}"
        )

    return "\n".join(lines) + "\n"


def escape_filter_path(path: Path) -> str:
    """
    Escape a path for use inside an ffmpeg filter string.

    Backslashes become forward slashes and colons are escaped, because ffmpeg's
    filter parser treats ":" as an argument separator — an unescaped Windows
    drive letter or a colon in a show name silently breaks the whole chain.
    """
    text = str(path).replace("\\", "/")
    text = text.replace(":", r"\:")
    text = text.replace("'", r"\'")
    return text


def build_filter_chain(ass_path: Path, config: Config) -> str:
    """
    Blur-fill vertical composition, then burned-in captions.

    The source is 16:9 and the output is 9:16, so the frame is composed twice:
    once scaled UP and cropped to fill the canvas and then blurred, as a
    background; once scaled DOWN to fit the width, as the sharp foreground. The
    alternative — pillarboxing onto flat black — wastes two thirds of a phone
    screen.

    `scale=W:-2` rather than `-1`: the -2 forces an EVEN height. Odd dimensions
    are rejected by libx264, and the failure appears as an encoder error with no
    obvious connection to the scale filter.
    """
    width, height = config.video.width, config.video.height
    blur = config.video.blur_sigma
    escaped = escape_filter_path(ass_path)

    return (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma={blur}[bg];"
        f"[0:v]scale={width}:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base];"
        f"[base]ass='{escaped}'[out]"
    )


def render(
    source: Path,
    window: ClipWindow,
    cues: list[Cue],
    destination: Path,
    config: Config,
    *,
    part_label: str | None = None,
    ass_path: Path | None = None,
    gate=None,
) -> tuple[bool, str]:
    """
    Cut, compose to 9:16 and burn in captions — in a single ffmpeg pass.

    One pass rather than cut-then-render: every extra pass is another full
    re-encode, and generational loss on a 20-60 second clip is avoidable.

    Returns (ok, message); the message carries ffmpeg's own error on failure so
    a skip is diagnosable.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    ass_path = ass_path or destination.with_suffix(".ass")
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(
        build_ass(
            cues, config, part_label=part_label, clip_duration=window.duration
        ),
        encoding="utf-8",
    )

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *build_input_args(source, window),
        "-filter_complex", build_filter_chain(ass_path, config),
        "-map", "[out]",
        "-map", "0:a?",          # "?" — clips from silent sources must not fail
        *encoder_args(config),
        "-pix_fmt", "yuv420p",   # required for playback on phones and browsers
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(destination),
    ]

    try:
        # Routed through run_pausable so a render in flight can be suspended the
        # moment someone starts watching, and resumed with no work lost.
        result = run_pausable(
            command, gate, timeout=1800,
            nice=config.worker.nice, ionice_class=config.worker.ionice_class
        )
    except FileNotFoundError:
        return False, "ffmpeg not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out while rendering"

    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        return False, detail[-1] if detail else f"ffmpeg exited {result.returncode}"

    if not destination.exists() or destination.stat().st_size == 0:
        return False, "ffmpeg reported success but produced no output"

    return True, "ok"


def extract_thumbnail(
    clip: Path, destination: Path, *, at: float = 1.0
) -> bool:
    """
    Grab a poster frame for the dashboard grid.

    Taken a second in rather than at zero: the first frame is often a fade from
    black, which makes every thumbnail look identical.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{at:.3f}", "-i", str(clip),
        "-frames:v", "1", "-q:v", "3",
        str(destination),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and destination.exists()
