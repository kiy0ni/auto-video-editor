"""Automatic decisions: what the video looks like, what kind of content it is, and the settings that fit.

Everything here is cheap compared to transcription: a few dozen single-frame seeks for the picture,
the loudness envelope and the transcript that the analysis computes anyway.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from . import vision
from .ffmpeg import _CREATION_FLAGS, MediaInfo, ffmpeg_bin
from .glossary import detect_game
from .reporting import Cancelled

INSIGHTS_VERSION = 2  # 2: neural face detector, corner pass, full-screen camera shots
Box = Tuple[float, float, float, float]  # x, y, width, height as fractions of the frame


@dataclass
class VideoInsights:
    version: int = INSIGHTS_VERSION
    motion: float = 0.0                 # median change between frames 0.5 s apart (0..1)
    static_ratio: float = 0.0           # share of samples where the picture barely moves
    faces_checked: bool = False
    face_ratio: float = 0.0             # share of sampled frames showing a face
    face_size: float = 0.0              # median width of the main face (fraction of the frame)
    big_face_ratio: float = 0.0         # share of frames with a full-screen camera shot
    detector: str = ""                  # face detector used: yunet or haar
    face_center: Optional[List[float]] = None
    facecam: Optional[List[float]] = None  # x, y, w, h of a webcam overlay
    title_game: Optional[str] = None
    # Filled by the speech preview during the analysis.
    language: str = ""
    sample_text: str = ""
    sample_model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VideoInsights":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


# -- cache ----------------------------------------------------------------------------------------

def insights_cache_path(media_path: str) -> Path:
    from .audio import media_fingerprint
    from .settings import cache_dir

    return cache_dir() / "insights" / f"{media_fingerprint(media_path)}-v{INSIGHTS_VERSION}.json"


def load_insights(media_path: str) -> Optional[VideoInsights]:
    try:
        data = json.loads(insights_cache_path(media_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("version") != INSIGHTS_VERSION:
        return None
    return VideoInsights.from_dict(data)


def save_insights(media_path: str, insights: VideoInsights) -> None:
    try:
        path = insights_cache_path(media_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(insights.to_dict(), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


# -- picture scan ----------------------------------------------------------------------------------

def grab_frames(path: str, t: float, width: int, height: int, count: int = 1, rate: Optional[float] = None,
                keyframe: bool = False, color: bool = False) -> List[np.ndarray]:
    """Frames at time ``t`` (``count`` of them, ``rate`` per second): grayscale, or BGR with ``color``."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error"]
    if keyframe:
        cmd += ["-skip_frame", "nokey"]
    video_filter = f"scale={width}:{height},format={'bgr24' if color else 'gray'}"
    if rate:
        video_filter = f"fps={rate}," + video_filter
    cmd += ["-ss", f"{max(0.0, t):.3f}", "-i", path, "-map", "0:v:0", "-an", "-vf", video_filter,
            "-frames:v", str(count), "-f", "rawvideo", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=90, creationflags=_CREATION_FLAGS)
    except (OSError, subprocess.TimeoutExpired):
        return []
    channels = 3 if color else 1
    size = width * height * channels
    data = proc.stdout
    shape = (height, width, 3) if color else (height, width)
    return [np.frombuffer(data, dtype=np.uint8, count=size, offset=i * size).reshape(shape)
            for i in range(len(data) // size)]


def _spread(duration: float, count: int) -> List[float]:
    return [duration * (0.05 + 0.9 * (i + 0.5) / count) for i in range(count)]


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def scan_video(info: MediaInfo, on_progress: Optional[Callable[[float], None]] = None,
               cancel_event=None, motion_samples: int = 24, face_samples: int = 48) -> VideoInsights:
    insights = VideoInsights()
    insights.title_game = detect_game([info.title, Path(info.path).stem, Path(info.path).parent.name],
                                      min_score=5)[0]
    if not info.has_video or info.duration <= 1 or not info.width or not info.height:
        return insights

    use_faces = vision.available()
    jobs = motion_samples + (face_samples if use_faces else 0)
    done = [0]

    def tick() -> None:
        done[0] += 1
        if on_progress:
            on_progress(done[0] / jobs)

    def check_cancel(pool) -> None:
        if cancel_event is not None and cancel_event.is_set():
            pool.shutdown(wait=False, cancel_futures=True)
            raise Cancelled("Cancelled by user")

    small_h = _even(160 * info.height / info.width)
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        futures = [pool.submit(grab_frames, info.path, t, 160, small_h, 2, 2.0)
                   for t in _spread(info.duration, motion_samples)]
        diffs = []
        for future in futures:
            check_cancel(pool)
            frames = future.result()
            tick()
            if len(frames) == 2:
                diffs.append(float(np.mean(np.abs(frames[0].astype(np.int16) - frames[1]))) / 255.0)
        if diffs:
            insights.motion = round(float(np.median(diffs)), 4)
            insights.static_ratio = round(float(np.mean(np.asarray(diffs) < 0.01)), 3)

        if use_faces:
            insights.detector = vision.backend_name()
            big_h = _even(960 * info.height / info.width)
            futures = [pool.submit(grab_frames, info.path, t, 960, big_h, 1, None, True, True)
                       for t in _spread(info.duration, face_samples)]
            boxes: List[List[Box]] = []
            small: List[np.ndarray] = []  # half-size grayscale copies, to find the overlay borders
            for future in futures:
                check_cancel(pool)
                frames = future.result()
                tick()
                if frames:
                    boxes.append(vision.detect_faces(frames[0]))
                    small.append(frames[0][::2, ::2].mean(axis=2).astype(np.uint8))
            analyse_faces(insights, boxes, info.width, info.height, small)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return insights


def analyse_faces(insights: VideoInsights, boxes: Sequence[Sequence[Box]], width: int, height: int,
                  gray_frames: Optional[Sequence[np.ndarray]] = None) -> None:
    insights.faces_checked = bool(boxes)
    if not boxes:
        return
    with_faces = [b for b in boxes if b]
    insights.face_ratio = round(len(with_faces) / len(boxes), 3)
    if with_faces:
        main = [max(b, key=lambda r: r[2] * r[3]) for b in with_faces]
        insights.face_size = round(float(np.median([r[2] for r in main])), 3)
        insights.big_face_ratio = round(sum(1 for r in main if r[2] >= vision.BIG_FACE) / len(boxes), 3)
        insights.face_center = [round(float(np.median([r[0] + r[2] / 2 for r in main])), 3),
                                round(float(np.median([r[1] + r[3] / 2 for r in main])), 3)]
    insights.facecam = detect_facecam(boxes, width, height, gray_frames)


def detect_facecam(frames: Sequence[Sequence[Box]], width: int, height: int,
                   gray_frames: Optional[Sequence[np.ndarray]] = None) -> Optional[List[float]]:
    """A webcam overlay shows up as a small face that stays at the same off-center place.

    Returns the overlay rectangle: snapped to the real overlay borders when ``gray_frames`` (the
    sampled frames, same order as ``frames``) are given, otherwise estimated from the face size.
    """
    n = len(frames)
    if n < 6:
        return None
    # Only small faces can be an overlay; full-screen camera shots must not hide it.
    frames = [[b for b in boxes if b[2] <= 0.2] for boxes in frames]
    faces = [(x + w / 2, y + h / 2, w, h) for boxes in frames for (x, y, w, h) in boxes]
    if not faces:
        return None

    def near(cx, cy, box) -> bool:
        x, y, w, h = box
        return abs(x + w / 2 - cx) < 0.05 and abs(y + h / 2 - cy) < 0.07

    best = max(((sum(1 for boxes in frames if any(near(cx, cy, b) for b in boxes)), cx, cy)
                for cx, cy, _w, _h in faces))
    support, cx, cy = best
    if support < max(3, 0.2 * n):  # the overlay is hidden while the camera is full screen
        return None
    cluster = [b for boxes in frames for b in boxes if near(cx, cy, b)]
    mcx = float(np.median([b[0] + b[2] / 2 for b in cluster]))
    mcy = float(np.median([b[1] + b[3] / 2 for b in cluster]))
    fw = float(np.median([b[2] for b in cluster]))
    fh = float(np.median([b[3] for b in cluster]))
    if fw > 0.22:
        return None  # a big face is the main shot, not an overlay
    if abs(mcx - 0.5) < 0.18 and abs(mcy - 0.5) < 0.2:
        return None  # centered face: talking head

    cam_h = min(0.6, fh * 3.0)
    cam_w = min(0.6, max(fw * 3.0, cam_h * height * 16 / 9 / max(1, width)))
    x0 = min(max(0.0, mcx - cam_w / 2), 1.0 - cam_w)
    y0 = min(max(0.0, mcy - cam_h * 0.42), 1.0 - cam_h)
    if x0 < 0.05:
        x0 = 0.0
    if x0 + cam_w > 0.95:
        x0 = 1.0 - cam_w
    if y0 < 0.05:
        y0 = 0.0
    if y0 + cam_h > 0.95:
        y0 = 1.0 - cam_h
    guess = [round(x0, 3), round(y0, 3), round(cam_w, 3), round(cam_h, 3)]
    if gray_frames is not None and len(gray_frames) == len(frames):
        with_overlay = [g for g, boxes in zip(gray_frames, frames) if any(near(cx, cy, b) for b in boxes)]
        return refine_overlay(with_overlay, (mcx - fw / 2, mcy - fh / 2, fw, fh), guess)
    return guess


def refine_overlay(gray_frames: Sequence[np.ndarray], face: Box, guess: Sequence[float]) -> List[float]:
    """Snap an estimated webcam rectangle to the real overlay borders.

    Frames sampled minutes apart: the webcam picture (a room) barely changes while the game around it
    changes a lot, so each border is where the temporal variation jumps, searched outward from the face.
    Falls back to ``guess`` for the sides where no clear border is found.
    """
    guess = [float(v) for v in guess]
    if len(gray_frames) < 4:
        return [round(v, 3) for v in guess]
    height, width = gray_frames[0].shape
    std = np.stack(gray_frames).astype(np.float32).std(axis=0)
    fx, fy, fw, fh = face[0] * width, face[1] * height, face[2] * width, face[3] * height
    k = max(2, int(round(width / 240)))
    rows = std[int(max(0, fy - fh)):int(min(height, fy + 2 * fh))]
    cols = std[:, int(max(0, fx - fw)):int(min(width, fx + 2 * fw))]
    column_profile = rows.mean(axis=0) if rows.size else np.zeros(width)
    row_profile = cols.mean(axis=1) if cols.size else np.zeros(height)

    def border(profile: np.ndarray, start: float, stop: float, outward: int) -> Optional[int]:
        threshold = max(6.0, 0.25 * float(np.median(profile)))
        best, best_step = None, threshold
        for i in range(max(k, int(start)), min(len(profile) - k, int(stop)) + 1):
            before, after = float(profile[i - k:i].mean()), float(profile[i:i + k].mean())
            step = after - before if outward > 0 else before - after
            if step > best_step:
                best, best_step = i, step
        return best

    right = border(column_profile, fx + 1.3 * fw, fx + 5 * fw, +1)
    left = border(column_profile, fx - 4 * fw, fx - 0.3 * fw, -1)
    bottom = border(row_profile, fy + 1.3 * fh, fy + 5.5 * fh, +1)
    top = border(row_profile, fy - 3 * fh, fy - 0.3 * fh, -1)

    gx, gy, gw, gh = guess
    x0 = left / width if left is not None else gx
    x1 = right / width if right is not None else gx + gw
    y0 = top / height if top is not None else gy
    y1 = bottom / height if bottom is not None else gy + gh
    x0, y0 = (0.0 if x0 < 0.02 else x0), (0.0 if y0 < 0.02 else y0)
    x1, y1 = (1.0 if x1 > 0.98 else x1), (1.0 if y1 > 0.98 else y1)
    w, h = x1 - x0, y1 - y0
    inside = x0 <= face[0] and y0 <= face[1] and face[0] + face[2] <= x1 and face[1] + face[3] <= y1
    if not inside or w < face[2] * 1.5 or h < face[3] * 1.5 or w > 0.6 or h > 0.7:
        return [round(v, 3) for v in guess]
    return [round(x0, 3), round(y0, 3), round(w, 3), round(h, 3)]


def describe_position(rect: Sequence[float]) -> str:
    x, y, w, h = rect
    cx, cy = x + w / 2, y + h / 2
    vertical = "top" if cy < 0.4 else "bottom" if cy > 0.6 else "middle"
    horizontal = "left" if cx < 0.4 else "right" if cx > 0.6 else "center"
    return "center" if vertical == "middle" and horizontal == "center" else f"{vertical}-{horizontal}"


# -- content ------------------------------------------------------------------------------------------

def speech_stats(transcript, env, duration: float) -> dict:
    n = max(1, int(math.ceil(duration)))
    loud, _ = env.per_second(n)
    active = loud > env.silence_threshold()
    stats = {"active_ratio": float(active.mean()), "speech_ratio": None, "background_ratio": None}
    if transcript is not None:
        speech = np.zeros(n, dtype=bool)
        for seg in transcript.segments:
            speech[max(0, int(seg.start)):min(n, int(math.ceil(seg.end)))] = True
        stats["speech_ratio"] = float(speech.mean())
        stats["background_ratio"] = float((active & ~speech).mean())
    return stats


def classify_content(insights: VideoInsights, stats: dict, speakers: int, duration: float,
                     game: Optional[str]) -> Tuple[str, str]:
    """(content type, human readable reason)."""
    if game:
        return "gaming", f"{game} detected"
    if insights.facecam:
        return "gaming", "webcam overlay on screen"
    speech, background = stats.get("speech_ratio"), stats.get("background_ratio")
    if speech is not None:
        if speech >= 0.5 and background < 0.2 and insights.motion < 0.025:
            return "talk", "mostly speech over a steady picture"
        if speakers >= 2 and speech >= 0.45 and background < 0.25:
            return "talk", f"{speakers} people talking most of the time"
    if insights.faces_checked and insights.face_ratio >= 0.5 and insights.face_size >= 0.1 \
            and insights.motion >= 0.025:
        return "vlog", "a person on camera, moving picture"
    if background is not None and background >= 0.3:
        return "gaming", "continuous game or music audio"
    if background is None and stats.get("active_ratio", 0) >= 0.8 and insights.motion >= 0.03:
        return "gaming", "constant sound and action"
    if duration >= 45 * 60:
        return "gaming", "long live recording"
    return "vlog", "short recording with varied content"


def auto_trim(transcript, duration: float) -> Tuple[float, float]:
    """Seconds to skip at the start and at the end: waiting screens and outros where nobody talks."""
    if transcript is None or not transcript.segments or duration < 300:
        return 0.0, 0.0
    n = int(math.ceil(duration))
    counts = np.zeros(n)
    for word in transcript.words():
        if 0 <= int(word.start) < n:
            counts[int(word.start)] += 1
    window = 60
    if n <= window:
        return 0.0, 0.0
    density = np.convolve(counts, np.ones(window), mode="valid")  # words in [i, i + 60 s)
    talking = np.flatnonzero(density >= 15)
    if talking.size == 0:
        return 0.0, 0.0
    first, last_end = int(talking[0]), int(talking[-1]) + window
    start = float(first - 10) if 90 <= first <= duration * 0.35 else 0.0
    tail = duration - last_end
    end = float(tail - 15) if 90 <= tail <= duration * 0.35 else 0.0
    return max(0.0, start), max(0.0, end)


# -- settings decided from the analysis ------------------------------------------------------------------

@lru_cache(maxsize=1)
def has_cuda() -> bool:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is not None:
            import torch

            return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass
    return False


def fast_cpu() -> bool:
    return (sys.platform == "darwin" and platform.machine() == "arm64") or (os.cpu_count() or 2) >= 8


def auto_model(duration: float) -> Tuple[str, str]:
    """Speech model that gives the best accuracy in a reasonable time on this computer."""
    if has_cuda():
        return "large-v3", "NVIDIA GPU"
    hours = duration / 3600
    if fast_cpu():
        return ("small", "fast processor") if hours <= 4 else ("base", "very long recording")
    return ("base", "standard processor") if hours <= 2 else ("tiny", "long recording on a standard processor")


_REALTIME = {"tiny": 0.03, "base": 0.05, "small": 0.12, "medium": 0.35, "large-v3": 0.7, "turbo": 0.3}


def estimate_transcription(duration: float, model: str) -> float:
    factor = _REALTIME.get(model, 0.12)
    if has_cuda():
        factor *= 0.12
    elif not fast_cpu():
        factor *= 1.8
    return duration * factor


def estimate_render(highlight_seconds: float, shorts_seconds: float, hardware: bool) -> float:
    factor = 0.12 if hardware else 0.4
    return highlight_seconds * factor + shorts_seconds * factor * 1.5 + 10


def auto_target(candidates, duration: float, min_score: float) -> float:
    """Reel length that follows how much great content the recording has."""
    strong = sum(c.duration for c in candidates if c.score >= max(min_score, 35.0)) * 0.9
    ratio = 0.3 if duration < 30 * 60 else 0.2 if duration < 2 * 3600 else 0.12
    upper = min(20 * 60.0, duration * ratio)
    lower = min(60.0, duration * 0.25, upper)
    return float(min(max(strong, lower), upper))


def auto_shorts_count(candidates, duration: float) -> int:
    good = sum(1 for c in candidates if c.score >= 55)
    cap = 3 if duration < 20 * 60 else 6 if duration < 2 * 3600 else 10
    return max(1, min(good, cap))


def auto_layout(insights: VideoInsights, content_type: str) -> Tuple[str, str]:
    if insights.facecam:
        return "split", f"webcam overlay {describe_position(insights.facecam)}"
    if insights.faces_checked and (insights.big_face_ratio >= 0.4
                                   or (insights.face_ratio >= 0.4 and insights.face_size >= 0.07)):
        return "smart", "a face is on screen most of the time"
    if content_type in ("talk", "vlog") and vision.available():
        return "smart", "talking content"
    return "blur", "keeps the whole picture visible"
