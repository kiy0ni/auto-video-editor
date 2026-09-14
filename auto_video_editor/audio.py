"""Streaming audio analysis: a loudness envelope computed in a single ffmpeg pass."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .ffmpeg import FFmpegError, MediaInfo, pcm_reader_command, popen_binary
from .reporting import Cancelled

SAMPLE_RATE = 16000
HOP_SECONDS = 0.05
_HOP = int(SAMPLE_RATE * HOP_SECONDS)
_FLOOR_DB = -90.0


@dataclass
class AudioEnvelope:
    """Per-hop (50 ms) RMS and peak levels in dBFS."""

    hop: float
    rms_db: np.ndarray
    peak_db: np.ndarray

    @property
    def duration(self) -> float:
        return len(self.rms_db) * self.hop

    def frame(self, t: float) -> int:
        return int(np.clip(int(t / self.hop), 0, max(0, len(self.rms_db) - 1)))

    def window(self, start: float, end: float) -> np.ndarray:
        return self.rms_db[self.frame(start): max(self.frame(start) + 1, self.frame(end))]

    def per_second(self, seconds: int) -> tuple:
        """Return ``(loudness_db, max_frame_db)`` arrays with one value per second."""
        per = int(round(1.0 / self.hop))
        total = seconds * per
        rms = self.rms_db[:total]
        if len(rms) < total:
            rms = np.concatenate([rms, np.full(total - len(rms), _FLOOR_DB, dtype=np.float32)])
        blocks = rms.reshape(seconds, per)
        power = np.mean(10.0 ** (blocks / 10.0), axis=1)
        loudness = 10.0 * np.log10(np.maximum(power, 1e-9))
        return loudness.astype(np.float32), blocks.max(axis=1).astype(np.float32)

    def silence_threshold(self) -> float:
        """Adaptive silence threshold between the noise floor and typical loud levels."""
        active = self.rms_db[self.rms_db > _FLOOR_DB + 1]
        if active.size < 20:
            return -45.0
        low, high = np.percentile(active, [10, 90])
        return float(np.clip(low + 0.3 * (high - low), -55.0, -22.0))

    def quietest_point(self, center: float, radius: float) -> float:
        """Time of the quietest 50 ms frame within ``center ± radius``."""
        a, b = self.frame(center - radius), self.frame(center + radius)
        if b <= a:
            return center
        seg = self.rms_db[a:b + 1]
        # Prefer the point closest to the center among near-minimal frames.
        min_val = seg.min()
        candidates = np.flatnonzero(seg <= min_val + 1.5)
        mid = (b - a) / 2
        best = candidates[np.argmin(np.abs(candidates - mid))]
        return (a + int(best)) * self.hop + self.hop / 2

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, hop=self.hop, rms_db=self.rms_db, peak_db=self.peak_db)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "AudioEnvelope":
        with np.load(path) as data:
            return cls(float(data["hop"]), data["rms_db"].astype(np.float32), data["peak_db"].astype(np.float32))


def silent_envelope(duration: float) -> AudioEnvelope:
    frames = max(1, int(np.ceil(duration / HOP_SECONDS)))
    floor = np.full(frames, _FLOOR_DB, dtype=np.float32)
    return AudioEnvelope(HOP_SECONDS, floor, floor.copy())


def analyze_audio(
    info: MediaInfo,
    on_progress: Optional[Callable[[float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> AudioEnvelope:
    """Decode the audio once (16 kHz mono) and compute the loudness envelope."""
    if not info.has_audio:
        return silent_envelope(info.duration)

    proc = popen_binary(pcm_reader_command(info.path, SAMPLE_RATE))
    stderr_chunks: list = []
    reader = threading.Thread(target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True)
    reader.start()

    rms_parts, peak_parts = [], []
    leftover = np.zeros(0, dtype=np.float32)
    block_bytes = _HOP * 4 * 400  # 20 seconds of audio per read
    decoded = 0
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled("Cancelled by user")
            raw = proc.stdout.read(block_bytes)
            if not raw:
                break
            samples = np.frombuffer(raw[: len(raw) - len(raw) % 4], dtype=np.float32)
            if leftover.size:
                samples = np.concatenate([leftover, samples])
            usable = samples.size - samples.size % _HOP
            frames = samples[:usable].reshape(-1, _HOP)
            leftover = samples[usable:].copy()
            if frames.size:
                rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
                peak = np.max(np.abs(frames), axis=1)
                rms_parts.append((20 * np.log10(np.maximum(rms, 1e-9))).astype(np.float32))
                peak_parts.append((20 * np.log10(np.maximum(peak, 1e-9))).astype(np.float32))
                decoded += frames.shape[0]
                if on_progress and info.duration:
                    on_progress(decoded * HOP_SECONDS / info.duration)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    reader.join(timeout=5)
    if proc.returncode != 0:
        message = b"".join(stderr_chunks).decode("utf-8", "replace").strip()
        raise FFmpegError(f"Audio analysis failed: {message[-800:]}")

    if not rms_parts:
        return silent_envelope(info.duration)
    rms_db = np.maximum(np.concatenate(rms_parts), _FLOOR_DB)
    peak_db = np.maximum(np.concatenate(peak_parts), _FLOOR_DB)
    return AudioEnvelope(HOP_SECONDS, rms_db, peak_db)


def media_fingerprint(path: str) -> str:
    """Cheap identity for cache keys: absolute path, size and modification time."""
    stat = os.stat(path)
    key = f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
