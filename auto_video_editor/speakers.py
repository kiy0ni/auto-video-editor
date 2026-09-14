"""Built-in speaker detection (no extra dependency).

Every transcript sentence gets a small voice signature: the average timbre (MFCC statistics)
and the pitch of the voice. The signatures are then clustered, one cluster per speaker.
It works well for two or three clearly different voices; overlapping speech is attributed to
the dominant voice of the sentence.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Sequence

import numpy as np

SR = 16000
N_FFT = 512
WIN = 400   # 25 ms
HOP = 160   # 10 ms
N_MELS = 26
N_CEPS = 13
PITCH_MIN, PITCH_MAX = 60, 400
FEATURE_SIZE = 2 * (N_CEPS - 1) + 2


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


@lru_cache(maxsize=1)
def _mel_filterbank() -> np.ndarray:
    edges = _mel_to_hz(np.linspace(_hz_to_mel(50), _hz_to_mel(SR / 2), N_MELS + 2))
    bins = np.floor((N_FFT + 1) * edges / SR).astype(int)
    bank = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(N_MELS):
        a, b, c = bins[i], max(bins[i + 1], bins[i] + 1), max(bins[i + 2], bins[i + 1] + 2)
        bank[i, a:b] = (np.arange(a, b) - a) / (b - a)
        bank[i, b:c] = (c - np.arange(b, c)) / (c - b)
    return bank


@lru_cache(maxsize=1)
def _dct_matrix() -> np.ndarray:
    n = np.arange(N_MELS)
    k = np.arange(N_CEPS)[:, None]
    matrix = np.cos(np.pi * k * (2 * n + 1) / (2 * N_MELS)) * np.sqrt(2.0 / N_MELS)
    matrix[0] /= np.sqrt(2.0)
    return matrix.astype(np.float32)


def _pitch(frames: np.ndarray) -> np.ndarray:
    """Fundamental frequency (Hz) of the voiced frames, by normalized autocorrelation."""
    spectrum = np.fft.rfft(frames, 1024)
    ac = np.fft.irfft(np.abs(spectrum) ** 2, 1024)[:, :WIN]
    ac0 = ac[:, :1].copy()
    ac0[ac0 <= 0] = 1.0
    ac /= ac0
    lo, hi = SR // PITCH_MAX, SR // PITCH_MIN
    window = ac[:, lo:hi]
    idx = window.argmax(axis=1)
    peak = window[np.arange(len(idx)), idx]
    voiced = peak > 0.55
    if not voiced.any():
        return np.zeros(0)
    return SR / (idx[voiced] + lo)


def voice_features(audio: np.ndarray, start: float, end: float) -> Optional[List[Optional[float]]]:
    """Voice signature of ``audio[start:end]`` (16 kHz mono), or ``None`` when too short."""
    a, b = max(0, int(start * SR)), min(audio.size, int(end * SR))
    seg = audio[a:b].astype(np.float32)
    if seg.size < int(0.4 * SR):
        return None
    seg = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])  # pre-emphasis
    n_frames = 1 + (seg.size - WIN) // HOP
    if n_frames < 8:
        return None
    stride = seg.strides[0]
    frames = np.lib.stride_tricks.as_strided(seg, shape=(n_frames, WIN), strides=(stride * HOP, stride))
    frames = frames * np.hamming(WIN).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames, N_FFT)) ** 2
    energy = spectrum.sum(axis=1)
    active = energy > np.percentile(energy, 40)
    if active.sum() < 5:
        active = np.ones(n_frames, dtype=bool)
    mel = np.log(spectrum[active] @ _mel_filterbank().T + 1e-8)
    mfcc = (mel @ _dct_matrix().T)[:, 1:]  # drop c0: it only measures loudness
    features = [round(float(v), 4) for v in np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])]
    f0 = _pitch(frames[active])
    if f0.size >= 3:
        log_f0 = np.log2(f0)
        features += [round(float(np.median(log_f0)), 4), round(float(np.std(log_f0)), 4)]
    else:
        features += [None, None]
    return features


# -- clustering -------------------------------------------------------------------------------------

def _kmeans(points: np.ndarray, k: int, seed: int, iterations: int = 60):
    rng = np.random.default_rng(seed)
    centers = [points[rng.integers(len(points))]]
    for _ in range(1, k):
        dist = np.min(((points[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        total = dist.sum()
        centers.append(points[rng.choice(len(points), p=dist / total if total > 0 else None)])
    centers = np.asarray(centers)
    labels = np.zeros(len(points), dtype=int)
    for _ in range(iterations):
        labels = ((points[:, None, :] - centers[None]) ** 2).sum(-1).argmin(axis=1)
        updated = np.array([points[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
                            for j in range(k)])
        if np.allclose(updated, centers):
            break
        centers = updated
    inertia = float(((points - centers[labels]) ** 2).sum())
    return labels, inertia


def _silhouette(points: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    if len(points) > 1200:
        pick = np.random.default_rng(seed).choice(len(points), 1200, replace=False)
        points, labels = points[pick], labels[pick]
    dist = np.sqrt(((points[:, None, :] - points[None]) ** 2).sum(-1))
    scores = []
    unique = np.unique(labels)
    for i in range(len(points)):
        own = labels == labels[i]
        own[i] = False
        if not own.any():
            continue
        a = dist[i, own].mean()
        b = min(dist[i, labels == other].mean() for other in unique if other != labels[i])
        scores.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(scores)) if scores else 0.0


def assign_speakers(segments: Sequence, count: int = 0, max_speakers: int = 6, min_silhouette: float = 0.12) -> int:
    """Set ``segment.speaker`` on every segment and return the number of speakers found.

    ``count`` fixes the number of speakers; 0 picks it automatically.
    """
    for seg in segments:
        seg.speaker = 0
    usable = [i for i, seg in enumerate(segments) if getattr(seg, "embedding", None)]
    if count == 1 or len(usable) < 8:
        return 1 if segments else 0

    raw = np.array([[np.nan if v is None else v for v in segments[i].embedding] for i in usable], dtype=np.float64)
    means = np.nanmean(raw, axis=0)
    raw = np.where(np.isnan(raw), means, raw)
    std = raw.std(axis=0) + 1e-6
    points = (raw - raw.mean(axis=0)) / std
    points[:, -2] *= 2.0  # the pitch of a voice separates speakers better than any timbre coefficient

    options = [count] if count > 1 else list(range(2, min(max_speakers, len(usable) // 8) + 1))
    best_labels, best_score = np.zeros(len(usable), dtype=int), -1.0
    for k in options:
        labels, _inertia = min((_kmeans(points, k, seed) for seed in range(8)), key=lambda r: r[1])
        score = _silhouette(points, labels)
        if score > best_score:
            best_labels, best_score = labels, score
    if count == 0 and best_score < min_silhouette:
        return 1

    for i, label in zip(usable, best_labels):
        segments[i].speaker = int(label)
    # Very short sentences carry too little voice: give them the speaker of the previous sentence.
    previous = 0
    for seg in segments:
        if seg.end - seg.start < 0.6 or not getattr(seg, "embedding", None):
            seg.speaker = previous
        previous = seg.speaker
    # Speaker 0 = the person who talks the most.
    talk_time = {}
    for seg in segments:
        talk_time[seg.speaker] = talk_time.get(seg.speaker, 0.0) + (seg.end - seg.start)
    order = {old: new for new, old in enumerate(sorted(talk_time, key=lambda s: -talk_time[s]))}
    for seg in segments:
        seg.speaker = order[seg.speaker]
    return len(order)
