"""Publishing kit: titles, descriptions, hashtags and thumbnails for the reel and every short."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .ffmpeg import MediaInfo, run_ffmpeg
from .glossary import normalize
from .insights import grab_frames
from .render import vertical_filter
from .timeline import format_clock

_SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]*")
_SPEAKER_RE = re.compile(r"(?:(?<=\s)|^)[A-Z]:\s+")


def clean_text(text: str) -> str:
    return _SPEAKER_RE.sub("", " ".join(text.split()))


def suggest_title(text: str, keywords: Iterable[str] = (), max_length: int = 70) -> str:
    """The punchiest sentence of a moment: hype phrases, exclamations and a readable length win."""
    text = clean_text(text)
    sentences = [s.strip(" ,;:-") for s in _SENTENCE_RE.findall(text)]
    sentences = [s for s in sentences if len(s) >= 4 and len(normalize(s).split()) >= 2]
    if not sentences:
        return ""
    phrases = [normalize(k) for k in keywords if normalize(k)]

    def score(sentence: str) -> float:
        low = f" {normalize(sentence)} "
        value = 3.0 * any(f" {p} " in low for p in phrases)
        value += 2.0 * ("!" in sentence) + 1.0 * ("?" in sentence)
        value += 1.0 if 12 <= len(sentence) <= max_length else 0.0
        return value - abs(len(sentence) - 40) / 100.0

    best = max(sentences, key=score)
    if len(best) > max_length:
        best = best[:max_length].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return best[:1].upper() + best[1:]


def describe(text: str, max_length: int = 220) -> str:
    text = clean_text(text)
    out = ""
    for sentence in _SENTENCE_RE.findall(text):
        if len(out) + len(sentence) > max_length:
            break
        out += sentence
    return (out or text[:max_length]).strip()


def short_kit_text(title: str, text: str, tags: Sequence[str]) -> str:
    return (f"TITLE\n{title}\n\nDESCRIPTION\n{describe(text)}\n\n{' '.join(tags)}\n\n"
            f"TRANSCRIPT\n{clean_text(text)}\n")


def reel_kit_text(title: str, chapters: Sequence[Tuple[float, str]], tags: Sequence[str]) -> str:
    lines = [f"TITLE\n{title}\n", "DESCRIPTION", "The best moments of the stream.", "", "CHAPTERS"]
    lines += [f"{format_clock(offset)} {name}" for offset, name in chapters]
    lines += ["", " ".join(tags), ""]
    return "\n".join(lines)


# -- thumbnails -------------------------------------------------------------------------------------

def sharpness(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    laplacian = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(laplacian.var())


def best_frame_time(info: MediaInfo, start: float, end: float, peak: float) -> float:
    """Sharpest, well exposed frame around the peak of a moment (no motion blur, no black frame)."""
    low, high = start + 0.2, max(start + 0.3, end - 0.2)
    peak = peak if start <= peak <= end else (start + end) / 2
    width = 320
    height = max(2, int(round(width * (info.height or 9) / (info.width or 16) / 2)) * 2)
    best_t, best_score = min(max(peak, low), high), -1.0
    for delta in (-1.5, -0.75, 0.0, 0.75, 1.5):
        t = min(max(peak + delta, low), high)
        frames = grab_frames(info.path, t, width, height)
        if not frames:
            continue
        brightness = float(frames[0].mean())
        value = sharpness(frames[0]) * (0.3 if brightness < 35 or brightness > 220 else 1.0)
        if value > best_score:
            best_t, best_score = t, value
    return best_t


def write_thumbnail(info: MediaInfo, t: float, output: Path, layout: Optional[str] = None,
                    subject_x: Optional[float] = None, facecam=None, blur: int = 10, cancel_event=None) -> None:
    args = ["-ss", f"{max(0.0, t):.3f}", "-i", info.path]
    if layout:
        args += ["-filter_complex", vertical_filter("[0:v:0]", "[out]", info, layout, subject_x, facecam, blur),
                 "-map", "[out]"]
    else:
        args += ["-map", "0:v:0", "-vf", "scale=1280:-2"]
    args += ["-frames:v", "1", "-q:v", "2", "-update", "1", str(output)]
    run_ffmpeg(args, cancel_event=cancel_event)
