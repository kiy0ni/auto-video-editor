"""Thin wrappers around the ffmpeg / ffprobe executables."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from collections import deque
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from typing import Callable, List, Optional, Sequence

from .reporting import Cancelled

# Hide console windows spawned from the GUI on Windows.
_CREATION_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


class FFmpegError(RuntimeError):
    """ffmpeg/ffprobe returned an error."""


def ffmpeg_bin() -> str:
    return os.environ.get("AVE_FFMPEG", "ffmpeg")


def ffprobe_bin() -> str:
    return os.environ.get("AVE_FFPROBE", "ffprobe")


def missing_tools() -> List[str]:
    return [name for name in (ffmpeg_bin(), ffprobe_bin()) if shutil.which(name) is None]


def require_tools() -> None:
    missing = missing_tools()
    if missing:
        raise FFmpegError(
            f"Required tool(s) not found: {', '.join(missing)}. "
            "Install FFmpeg (https://ffmpeg.org/download.html) and make sure it is on your PATH."
        )


# -- probing ---------------------------------------------------------------------------

@dataclass
class MediaInfo:
    path: str
    duration: float
    has_video: bool
    has_audio: bool
    width: int = 0
    height: int = 0
    fps: float = 30.0
    audio_channels: int = 0
    sample_rate: int = 0
    video_codec: str = ""
    audio_codec: str = ""
    start_timecode: str = ""  # e.g. "01:00:00:00" when the file carries a timecode
    title: str = ""           # title/description metadata, used to recognise the game

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 16 / 9

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MediaInfo":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _parse_rate(value: str) -> float:
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den) if float(den) else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def probe(path: str) -> MediaInfo:
    cmd = [
        ffprobe_bin(), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", creationflags=_CREATION_FLAGS)
    except FileNotFoundError:
        require_tools()
        raise
    if proc.returncode != 0:
        raise FFmpegError(f"Cannot read media file: {proc.stderr.strip() or path}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _parse_rate(fmt.get("duration", "0"))
    if duration <= 0:
        durations = [_parse_rate(s.get("duration", "0")) for s in streams]
        duration = max(durations or [0.0])
    if duration <= 0:
        raise FFmpegError("Could not determine the media duration.")

    info = MediaInfo(path=path, duration=duration, has_video=video is not None, has_audio=audio is not None)
    if video:
        info.width = int(video.get("width") or 0)
        info.height = int(video.get("height") or 0)
        info.video_codec = video.get("codec_name", "")
        fps = _parse_rate(video.get("avg_frame_rate", "0"))
        if not 1 <= fps <= 240:
            fps = _parse_rate(video.get("r_frame_rate", "0"))
        info.fps = fps if 1 <= fps <= 240 else 30.0
        rotation = _rotation(video)
        if rotation in (90, 270):  # ffmpeg auto-rotates, so swap the dimensions
            info.width, info.height = info.height, info.width
        info.start_timecode = (video.get("tags", {}) or {}).get("timecode", "")
    if audio:
        info.audio_channels = int(audio.get("channels") or 0)
        info.sample_rate = int(audio.get("sample_rate") or 0)
        info.audio_codec = audio.get("codec_name", "")
    format_tags = fmt.get("tags", {}) or {}
    if not info.start_timecode:
        info.start_timecode = format_tags.get("timecode", "")
    info.title = " ".join(str(v) for k, v in format_tags.items()
                          if k.lower() in ("title", "comment", "description", "synopsis", "album", "show"))[:400]
    return info


def _rotation(stream: dict) -> int:
    try:
        rotate = int(float((stream.get("tags") or {}).get("rotate", 0)))
    except ValueError:
        rotate = 0
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                rotate = int(float(side["rotation"]))
            except (TypeError, ValueError):
                pass
    return abs(rotate) % 360


# -- running ffmpeg ----------------------------------------------------------------------

def run_ffmpeg(
    args: Sequence[str],
    *,
    duration: Optional[float] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    cwd: Optional[str] = None,
    loglevel: str = "error",
) -> str:
    """Run ffmpeg with progress reporting and cancellation. Returns stderr output."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-loglevel", loglevel,
           "-progress", "pipe:1", "-nostats", *args]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd,
            text=True, encoding="utf-8", errors="replace", creationflags=_CREATION_FLAGS,
        )
    except FileNotFoundError:
        require_tools()
        raise

    stderr_lines: deque = deque(maxlen=400)
    reader = threading.Thread(target=lambda: stderr_lines.extend(proc.stderr), daemon=True)
    reader.start()

    try:
        for line in proc.stdout:
            if cancel_event is not None and cancel_event.is_set():
                _terminate(proc)
                raise Cancelled("Cancelled by user")
            if on_progress and duration and line.startswith("out_time_us="):
                try:
                    micros = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                on_progress(min(1.0, max(0.0, micros / 1e6 / duration)))
        proc.wait()
    except BaseException:
        _terminate(proc)
        raise
    reader.join(timeout=5)

    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled("Cancelled by user")
    stderr = "".join(stderr_lines)
    if proc.returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        raise FFmpegError(f"ffmpeg failed (exit code {proc.returncode}):\n{tail}")
    if on_progress:
        on_progress(1.0)
    return stderr


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def pcm_reader_command(path: str, sample_rate: int, start: Optional[float] = None,
                       duration: Optional[float] = None) -> List[str]:
    """ffmpeg command that writes mono float32 PCM of the first audio stream to stdout."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", path, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]
    return cmd


def popen_binary(cmd: Sequence[str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=_CREATION_FLAGS)


# -- encoders --------------------------------------------------------------------------------

_QUALITY_INDEX = {"draft": 0, "standard": 1, "high": 2}


def video_codec_args(encoder: str, quality: str = "high") -> List[str]:
    q = _QUALITY_INDEX.get(quality, 2)
    if encoder == "libx264":
        return ["-c:v", "libx264", "-preset", ("veryfast", "fast", "medium")[q],
                "-crf", ("23", "20", "18")[q], "-profile:v", "high", "-pix_fmt", "yuv420p"]
    if encoder == "libx265":
        return ["-c:v", "libx265", "-preset", ("veryfast", "fast", "medium")[q],
                "-crf", ("28", "24", "21")[q], "-pix_fmt", "yuv420p", "-tag:v", "hvc1"]
    if encoder in ("h264_videotoolbox", "hevc_videotoolbox"):
        args = ["-c:v", encoder, "-q:v", ("50", "62", "72")[q], "-pix_fmt", "yuv420p"]
        return args + (["-tag:v", "hvc1"] if encoder.startswith("hevc") else [])
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        args = ["-c:v", encoder, "-preset", "p5", "-rc", "vbr", "-cq", ("28", "23", "19")[q],
                "-b:v", "0", "-pix_fmt", "yuv420p"]
        return args + (["-tag:v", "hvc1"] if encoder.startswith("hevc") else [])
    if encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", ("28", "23", "19")[q], "-pix_fmt", "nv12"]
    if encoder == "h264_amf":
        qp = ("28", "23", "19")[q]
        return ["-c:v", "h264_amf", "-rc", "cqp", "-qp_i", qp, "-qp_p", qp, "-pix_fmt", "yuv420p"]
    raise ValueError(f"Unsupported encoder: {encoder}")


def audio_codec_args(bitrate: int = 192) -> List[str]:
    return ["-c:a", "aac", "-b:a", f"{int(bitrate)}k", "-ar", "48000", "-ac", "2"]


AUDIO_CODEC_ARGS = audio_codec_args()


@lru_cache(maxsize=None)
def encoder_works(encoder: str) -> bool:
    """Encode a few frames to check that an encoder (and its hardware) really works."""
    try:
        args = video_codec_args(encoder, "standard")
    except ValueError:
        return False
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error",
           "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30:d=0.2", *args, "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30, creationflags=_CREATION_FLAGS)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def pick_encoder(preference: str = "auto") -> str:
    """Resolve the ``auto`` encoder to the fastest one that works on this machine."""
    if preference and preference != "auto":
        if encoder_works(preference):
            return preference
        return "libx264"
    if sys.platform == "darwin":
        candidates = ["h264_videotoolbox"]
    else:
        candidates = ["h264_nvenc", "h264_qsv", "h264_amf"]
    for name in candidates:
        if encoder_works(name):
            return name
    return "libx264"
