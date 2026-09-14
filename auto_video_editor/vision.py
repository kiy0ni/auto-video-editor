"""Face detection (OpenCV): aims the vertical crop, finds webcam overlays and full-screen camera shots.

Uses the YuNet neural face detector (a 230 KB model downloaded once, robust to headsets, three-quarter
views and dim webcams) and falls back to Haar cascades when the model cannot be obtained.
"""

from __future__ import annotations

import hashlib
import importlib.util
import threading
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .ffmpeg import MediaInfo, ffmpeg_bin, popen_binary

Box = Tuple[float, float, float, float]  # x, y, width, height as fractions of the frame

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
BIG_FACE = 0.12  # a face wider than 12% of the frame is a camera shot, not a webcam overlay
_detect_lock = threading.Lock()


@lru_cache(maxsize=1)
def available() -> bool:
    """OpenCV with a usable face detector (OpenCV 4.x; 5.x moved Haar cascades out of the main package)."""
    try:
        if importlib.util.find_spec("cv2") is None:
            return False
        import cv2

        return hasattr(cv2, "FaceDetectorYN") or (hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"))
    except Exception:  # noqa: BLE001
        return False


def _model_path() -> Path:
    from .settings import cache_dir

    return cache_dir() / "models" / "face_detection_yunet_2023mar.onnx"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _yunet_model() -> Optional[str]:
    path = _model_path()
    try:
        if path.exists() and _sha256(path) == YUNET_SHA256:
            return str(path)
        with urllib.request.urlopen(YUNET_URL, timeout=20) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != YUNET_SHA256:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return str(path)
    except Exception:  # noqa: BLE001 - offline: the Haar fallback is used
        return None


@lru_cache(maxsize=1)
def _detector():
    import cv2

    if hasattr(cv2, "FaceDetectorYN"):
        model = _yunet_model()
        if model:
            try:
                return "yunet", cv2.FaceDetectorYN.create(model, "", (320, 320), 0.6, 0.3, 100)
            except Exception:  # noqa: BLE001
                pass
    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    return "haar", (frontal, profile)


def backend_name() -> str:
    return _detector()[0] if available() else "none"


def _detect_pixels(image: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
    """Faces as (x, y, w, h, confidence) in pixels."""
    import cv2

    kind, detector = _detector()
    height, width = image.shape[:2]
    if kind == "yunet":
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        with _detect_lock:
            detector.setInputSize((width, height))
            _, faces = detector.detect(np.ascontiguousarray(bgr))
        if faces is None:
            return []
        return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])) for f in faces]

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frontal, profile = detector
    size = max(18, int(height * 0.04))
    boxes = [(float(x), float(y), float(w), float(h), 0.7)
             for x, y, w, h in frontal.detectMultiScale(gray, 1.1, 5, minSize=(size, size))]
    if not profile.empty():
        boxes += [(float(x), float(y), float(w), float(h), 0.6)
                  for x, y, w, h in profile.detectMultiScale(gray, 1.1, 5, minSize=(size, size))]
        flipped = cv2.flip(gray, 1)
        boxes += [(float(width - x - w), float(y), float(w), float(h), 0.6)
                  for x, y, w, h in profile.detectMultiScale(flipped, 1.1, 5, minSize=(size, size))]
    return boxes


def _iou(a, b) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect_faces(image: np.ndarray, corners: bool = True) -> List[Box]:
    """Faces on a frame (BGR or grayscale) as fractions of the frame.

    With ``corners`` each corner is also analysed at twice the size, which finds the small face of a
    webcam overlay that a full-frame pass misses.
    """
    if not available():
        return []
    import cv2

    height, width = image.shape[:2]
    found = [(x / width, y / height, w / width, h / height, score) for x, y, w, h, score in _detect_pixels(image)]
    if corners:
        cw, ch = int(width * 0.45), int(height * 0.5)
        for x0, y0 in ((0, 0), (width - cw, 0), (0, height - ch), (width - cw, height - ch)):
            crop = cv2.resize(image[y0:y0 + ch, x0:x0 + cw], (cw * 2, ch * 2), interpolation=cv2.INTER_CUBIC)
            for x, y, w, h, score in _detect_pixels(crop):
                found.append(((x0 + x / 2) / width, (y0 + y / 2) / height, w / 2 / width, h / 2 / height, score))
    kept: List[Tuple[float, float, float, float, float]] = []
    for box in sorted(found, key=lambda b: -b[4]):
        if all(_iou(box, other) < 0.3 for other in kept):
            kept.append(box)
    return [box[:4] for box in kept]


def _sample_color_frames(info: MediaInfo, start: float, duration: float, samples: int,
                         width: int = 640) -> List[np.ndarray]:
    height = max(2, int(round(info.height * width / info.width / 2)) * 2)
    rate = max(0.2, samples / duration)
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", info.path, "-map", "0:v:0", "-an",
           "-vf", f"fps={rate:.4f},scale={width}:{height},format=bgr24", "-f", "rawvideo", "-"]
    proc = popen_binary(cmd)
    try:
        data, _ = proc.communicate(timeout=300)
    except Exception:  # noqa: BLE001
        proc.kill()
        return []
    size = width * height * 3
    return [np.frombuffer(data, dtype=np.uint8, count=size, offset=i * size).reshape(height, width, 3)
            for i in range(len(data) // size)]


def analyse_window(info: MediaInfo, start: float, duration: float, samples: int = 10) -> Tuple[Optional[float], float]:
    """Over a time window: (horizontal position 0..1 of the main face or None, share of frames
    showing a big face, i.e. a full-screen camera shot)."""
    if not available() or not info.has_video or info.width <= 0 or duration <= 0:
        return None, 0.0
    frames = _sample_color_frames(info, start, duration, samples)
    if not frames:
        return None, 0.0
    centers, weights, with_faces, big = [], [], 0, 0
    for frame in frames:
        faces = detect_faces(frame, corners=False)
        if not faces:
            continue
        with_faces += 1
        main = max(faces, key=lambda b: b[2] * b[3])
        if main[2] >= BIG_FACE:
            big += 1
        centers.append(main[0] + main[2] / 2)
        weights.append(main[2] * main[3])
    big_share = big / len(frames)
    if with_faces < max(2, len(frames) // 4):
        return None, big_share
    order = np.argsort(centers)
    cumulative = np.cumsum(np.asarray(weights)[order])
    median_index = order[int(np.searchsorted(cumulative, cumulative[-1] / 2))]
    return float(centers[median_index]), big_share


def detect_subject_x(info: MediaInfo, start: float, duration: float, samples: int = 16) -> Optional[float]:
    """Horizontal position (0..1) of the main face in a time window, or ``None``."""
    return analyse_window(info, start, duration, samples)[0]
