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


def foreground_size(config: Config) -> tuple[int, int] | None:
    """
    Pixel size of the sharp centre panel, or None to fit the full width.

    Height is forced even — libx264 rejects odd dimensions, and the failure
    surfaces as an encoder error with no obvious connection to the aspect
    ratio you typed.
    """
    raw = (config.video.foreground_aspect or "").strip()
    if not raw:
        return None

    try:
        left, right = raw.replace("/", ":").split(":")
        ratio_w, ratio_h = float(left), float(right)
        if ratio_w <= 0 or ratio_h <= 0:
            return None
    except ValueError:
        return None

    width = config.video.width
    height = int(round(width * ratio_h / ratio_w))
    height -= height % 2
    # A panel taller than the canvas is just a full-frame crop.
    return width, min(height, config.video.height)


def composite_chain(index: int, config: Config, out_label: str) -> str:
    """
    Blur-fill composition for one input: blurred cover behind, sharp panel in
    front, centred.

    The background is scaled UP and cropped to fill the whole canvas, then
    blurred. The foreground is the part you actually watch — cropped to
    `foreground_aspect` when one is set, otherwise scaled to fit the full width.
    """
    width, height = config.video.width, config.video.height
    blur = config.video.blur_sigma
    panel = foreground_size(config)

    if panel:
        panel_w, panel_h = panel
        foreground = (
            f"[{index}:v]scale={panel_w}:{panel_h}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={panel_w}:{panel_h},setsar=1[fg{index}];"
        )
    else:
        # scale=W:-2, not -1: the -2 forces an EVEN height. Odd dimensions are
        # rejected by libx264 with an error that points nowhere near here.
        foreground = f"[{index}:v]scale={width}:-2,setsar=1[fg{index}];"

    return (
        f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma={blur}[bg{index}];"
        f"{foreground}"
        f"[bg{index}][fg{index}]overlay=(W-w)/2:(H-h)/2,setsar=1[{out_label}];"
    )


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
    return (
        composite_chain(0, config, "base")
        + f"[base]ass='{escape_filter_path(ass_path)}'[out]"
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


# --------------------------------------------------------------------------
# Recaps — several moments stitched into one clip
# --------------------------------------------------------------------------


def has_audio_stream(path: Path) -> bool:
    """Whether a file carries audio, so the concat graph knows what to join."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return bool(result.stdout.strip())


def build_recap_filter(
    segment_count: int, ass_path: Path, config: Config, *, with_audio: bool
) -> str:
    """
    Compose N segments into one vertical clip, then burn captions over the lot.

    Each segment gets the same blur-fill treatment a standalone clip does, then
    they are concatenated and the caption track is applied ONCE, across the
    joined timeline. Applying it per segment instead would mean N subtitle
    files and N chances for the offsets to drift apart.

    setsar=1 on every segment matters: concat refuses inputs whose sample
    aspect ratios disagree, and seeking into different parts of a file can
    report them differently.
    """
    parts: list[str] = []
    for index in range(segment_count):
        parts.append(composite_chain(index, config, f"v{index}"))
        if with_audio:
            # Normalised so concat never sees a format change mid-stream.
            parts.append(
                f"[{index}:a]aformat=sample_fmts=fltp:sample_rates=48000:"
                f"channel_layouts=stereo[a{index}];"
            )

    if with_audio:
        joined = "".join(f"[v{i}][a{i}]" for i in range(segment_count))
        parts.append(f"{joined}concat=n={segment_count}:v=1:a=1[cv][ca];")
    else:
        joined = "".join(f"[v{i}]" for i in range(segment_count))
        parts.append(f"{joined}concat=n={segment_count}:v=1:a=0[cv];")

    parts.append(f"[cv]ass='{escape_filter_path(ass_path)}'[out]")
    return "".join(parts)


def render_recap(
    segments,
    cues_by_segment: list[list[Cue]],
    destination: Path,
    config: Config,
    *,
    ass_path: Path | None = None,
    gate=None,
) -> tuple[bool, str]:
    """
    Render a recap or compilation: one file, several moments, one caption track.

    Each segment carries its own source, so the moments can come from one
    episode (a rundown) or from twenty different ones (a compilation) with no
    difference here.

    `cues_by_segment` holds each segment's cues already rebased to that
    segment's own start. They are shifted again here by the segment's offset in
    the finished clip, which is the only place that knows the output timeline.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    shifted: list[Cue] = []
    for segment, cues in zip(segments, cues_by_segment):
        for cue in cues:
            shifted.append(
                Cue(
                    index=len(shifted),
                    start=cue.start + segment.offset,
                    end=min(cue.end + segment.offset,
                            segment.offset + segment.duration),
                    text=cue.text,
                )
            )

    total = sum(segment.duration for segment in segments)

    ass_path = ass_path or destination.with_suffix(".ass")
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(
        build_ass(shifted, config, clip_duration=total), encoding="utf-8"
    )

    # Every source must have audio or the concat graph can't join them — one
    # silent episode in a compilation would otherwise fail the whole render
    # with an unhelpful filter error. Dropping audio for all of them is the
    # graceful answer.
    sources = {segment.source for segment in segments}
    with_audio = all(has_audio_stream(path) for path in sources)

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for segment in segments:
        command += [
            "-ss", f"{segment.window.start:.3f}",
            "-t", f"{segment.duration:.3f}",
            "-i", str(segment.source),
        ]

    command += [
        "-filter_complex",
        build_recap_filter(len(segments), ass_path, config, with_audio=with_audio),
        "-map", "[out]",
    ]
    if with_audio:
        command += ["-map", "[ca]", "-c:a", "aac", "-b:a", "160k"]

    command += [
        *encoder_args(config),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(destination),
    ]

    try:
        result = run_pausable(
            command, gate, timeout=3600,
            nice=config.worker.nice, ionice_class=config.worker.ionice_class,
        )
    except FileNotFoundError:
        return False, "ffmpeg not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out while rendering the recap"

    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        return False, detail[-1] if detail else f"ffmpeg exited {result.returncode}"

    if not destination.exists() or destination.stat().st_size == 0:
        return False, "ffmpeg reported success but produced no output"

    return True, "ok"
