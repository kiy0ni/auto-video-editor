"""The editing brain: score every second, find candidate moments and pick the best ones."""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio import AudioEnvelope
from .glossary import is_laughter
from .timeline import Range, merge_ranges, subtract_ranges, total_duration
from .transcribe import Transcript

LEAD_SECONDS = 6.0   # build-up kept before a peak
TAIL_SECONDS = 3.0   # reaction kept after a peak


@dataclass
class Moment:
    start: float
    end: float
    score: float = 0.0       # 0..100, relative to the best moment of the recording
    peak: float = 0.0
    text: str = ""
    reasons: List[str] = field(default_factory=list)
    selected: bool = True
    keep: List[Range] = field(default_factory=list)   # source ranges left after jump cuts
    words: List[list] = field(default_factory=list)   # [[start, end, text, speaker], ...] in source time
    hook: List[Range] = field(default_factory=list)   # teaser played before a short (its best seconds)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def ranges(self, jump_cuts: bool) -> List[Range]:
        if jump_cuts and self.keep:
            return list(self.keep)
        return [(self.start, self.end)]

    def edited_duration(self, jump_cuts: bool) -> float:
        return total_duration(self.ranges(jump_cuts))

    def title(self, max_words: int = 7) -> str:
        words = re.sub(r"\s+", " ", self.text).strip().split(" ")
        words = [w for w in words if w]
        if not words:
            return ""
        title = " ".join(words[:max_words]).strip(" ,.;:")
        return title + ("..." if len(words) > max_words else "")

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3), "end": round(self.end, 3), "score": round(self.score, 1),
            "peak": round(self.peak, 2), "text": self.text, "reasons": self.reasons,
            "selected": self.selected, "keep": [[round(a, 3), round(b, 3)] for a, b in self.keep],
            "words": self.words, "hook": [[round(a, 3), round(b, 3)] for a, b in self.hook],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Moment":
        return cls(
            start=float(data["start"]), end=float(data["end"]), score=float(data.get("score", 0)),
            peak=float(data.get("peak", data["start"])), text=data.get("text", ""),
            reasons=list(data.get("reasons", [])), selected=bool(data.get("selected", True)),
            keep=[(float(a), float(b)) for a, b in data.get("keep", [])],
            words=[list(w) for w in data.get("words", [])],
            hook=[(float(a), float(b)) for a, b in data.get("hook", [])],
        )


# -- scoring --------------------------------------------------------------------------------

@dataclass
class ScoreCurve:
    score: np.ndarray        # smoothed per-second score, 0..1
    raw: np.ndarray          # unsmoothed per-second score
    excitement: np.ndarray
    burst: np.ndarray
    speech: np.ndarray
    keywords: np.ndarray
    keyword_hits: Dict[int, str]
    has_speech: bool


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.astype(np.float32)
    kernel = np.bartlett(window + 2)[1:-1]
    kernel /= kernel.sum()
    padded = np.pad(values, window // 2 + 1, mode="edge")
    out = np.convolve(padded, kernel, mode="same")
    return out[window // 2 + 1: window // 2 + 1 + values.size].astype(np.float32)


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    if values.size == 0:
        return values
    padded = np.pad(values, half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    # Subsample long windows to keep this fast on multi-hour recordings.
    step = max(1, view.shape[1] // 31)
    return np.median(view[:, ::step], axis=1).astype(np.float32)


_TOKEN_RE = re.compile(r"[^\w']+", re.UNICODE)


def normalize_text(text: str) -> str:
    return _TOKEN_RE.sub(" ", text.lower().replace("’", "'")).strip()


def build_curve(env: AudioEnvelope, transcript: Optional[Transcript], duration: float,
                keywords: Sequence[str] = (), weights: Optional[Dict[str, float]] = None,
                skip: Tuple[float, float] = (0.0, 0.0)) -> ScoreCurve:
    """Per-second score. ``weights`` multiplies each signal: loudness, spikes, speech, keywords, exclamations.
    ``skip`` = seconds ignored at the start and at the end of the recording."""
    gain = {"loudness": 1.0, "spikes": 1.0, "speech": 1.0, "keywords": 1.0, "exclamations": 1.0}
    gain.update(weights or {})
    n = max(1, int(math.ceil(duration)))
    loud, loudest_frame = env.per_second(n)
    silence = env.silence_threshold()

    # Loudness relative to a baseline: the "normal" level of that part of the stream (5 minute
    # rolling median), but never below the typical level of the whole recording. Otherwise a calm
    # intro or outro where someone just talks looks "exciting" compared to the silence around it.
    local = _rolling_median(loud, 301)
    audible = loud[loud > -85.0]
    global_level = float(np.percentile(audible, 40)) if audible.size else float(local.min())
    baseline = np.maximum(local, global_level)
    deviation = loud - baseline
    spread = float(np.median(np.abs(deviation - np.median(deviation))) * 1.4826)
    spread = max(spread, 1.5)
    excitement = np.clip(deviation / spread, 0, 4) / 4
    burst = np.clip((loudest_frame - baseline) / spread - 1.5, 0, 4) / 4
    active = (loud > silence).astype(np.float32)

    speech = np.zeros(n, dtype=np.float32)
    keyword_score = np.zeros(n, dtype=np.float32)
    exclaim = np.zeros(n, dtype=np.float32)
    hits: Dict[int, str] = {}
    words = list(transcript.words()) if transcript else []

    if words:
        for w in words:
            i = int((w.start + w.end) / 2)
            if 0 <= i < n:
                speech[i] += 1
        speech = _smooth(speech, 3)
        nonzero = speech[speech > 0]
        reference = float(np.percentile(nonzero, 95)) if nonzero.size else 1.0
        speech = np.clip(speech / max(reference, 1e-6), 0, 1)

        tokens = [(normalize_text(w.text), w.start, w.end) for w in words]
        tokens = [t for t in tokens if t[0]]
        token_text = [t[0] for t in tokens]
        for phrase in keywords:
            parts = normalize_text(phrase).split()
            if not parts:
                continue
            size = len(parts)
            for i in range(len(token_text) - size + 1):
                if token_text[i:i + size] == parts:
                    t0, t1 = tokens[i][1], tokens[i + size - 1][2]
                    a, b = max(0, int(t0) - 1), min(n, int(t1) + 3)
                    keyword_score[a:b] = 1.0
                    hits.setdefault(int(t0), phrase)

        for token, t0, t1 in tokens:
            if is_laughter(token):
                a, b = max(0, int(t0) - 1), min(n, int(t1) + 3)
                keyword_score[a:b] = 1.0
                hits.setdefault(int(t0), "laughter")

        for seg in transcript.segments:
            marks = seg.text.count("!")
            if marks:
                a, b = max(0, int(seg.start)), min(n, int(seg.end) + 1)
                exclaim[a:b] = np.maximum(exclaim[a:b], min(1.0, 0.5 * marks))

        raw = (0.35 * gain["loudness"] * excitement + 0.20 * gain["spikes"] * burst
               + 0.15 * gain["speech"] * speech + 0.20 * gain["keywords"] * keyword_score
               + 0.10 * gain["exclamations"] * exclaim)
    else:
        raw = 0.60 * gain["loudness"] * excitement + 0.40 * gain["spikes"] * burst

    raw = (raw * (0.4 + 0.6 * active)).astype(np.float32)
    skip_start, skip_end = skip
    if skip_start > 0:
        raw[:min(n, int(skip_start))] = 0.0
    if skip_end > 0:
        raw[max(0, n - int(skip_end)):] = 0.0
    return ScoreCurve(_smooth(raw, 5), raw, excitement, burst, speech, keyword_score, hits, bool(words))


# -- boundary snapping ------------------------------------------------------------------------

class _SpeechIndex:
    """Fast lookups of the sentence/word containing a timestamp."""

    def __init__(self, transcript: Optional[Transcript]) -> None:
        segments = transcript.segments if transcript else []
        self.seg_starts = [s.start for s in segments]
        self.seg_ends = [s.end for s in segments]
        words = list(transcript.words()) if transcript else []
        self.word_starts = [w.start for w in words]
        self.word_ends = [w.end for w in words]
        self.words = words

    def __bool__(self) -> bool:
        return bool(self.seg_starts)

    @staticmethod
    def _containing(starts, ends, t):
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and starts[i] < t < ends[i]:
            return starts[i], ends[i]
        return None

    def segment_at(self, t):
        return self._containing(self.seg_starts, self.seg_ends, t)

    def word_at(self, t):
        return self._containing(self.word_starts, self.word_ends, t)


def _snap(start: float, end: float, speech: _SpeechIndex, env: AudioEnvelope,
          lower: float, upper: float, max_shift: float) -> tuple:
    """Move cut points so they never land in the middle of a sentence (or a word)."""
    if speech:
        seg = speech.segment_at(start)
        if seg and start - seg[0] <= max_shift and seg[0] >= lower:
            start = seg[0]
        else:
            word = speech.word_at(start)
            if word and word[0] >= lower:
                start = word[0]
        seg = speech.segment_at(end)
        if seg and seg[1] - end <= max_shift and seg[1] <= upper:
            end = seg[1]
        else:
            word = speech.word_at(end)
            if word and word[1] <= upper:
                end = word[1]
    else:
        start = min(start, max(lower, env.quietest_point(start, 1.5)))
        end = max(end, min(upper, env.quietest_point(end, 1.5)))
    return max(lower, start), min(upper, end)


# -- candidates ---------------------------------------------------------------------------------

def find_candidates(curve: ScoreCurve, env: AudioEnvelope, transcript: Optional[Transcript],
                    duration: float, min_len: float, max_len: float, limit: int,
                    lead: float = LEAD_SECONDS, tail: float = TAIL_SECONDS) -> List[Moment]:
    """Detect up to ``limit`` non-overlapping moments, best first.

    ``lead``/``tail`` are the seconds of build-up and reaction kept around each peak.
    """
    score = curve.score
    n = score.size
    if n == 0 or float(score.max()) <= 1e-4:
        return []
    speech = _SpeechIndex(transcript)
    floor = float(score.max()) * 0.08
    blocked = np.zeros(n, dtype=bool)
    intervals: List[Range] = []
    moments: List[Moment] = []

    for p in np.argsort(-score, kind="stable"):
        if len(moments) >= limit:
            break
        p = int(p)
        peak_value = float(score[p])
        if peak_value <= floor:
            break
        if blocked[p]:
            continue

        # Grow the region while the score stays above half of the peak.
        reach = int(max_len * 0.6)
        a = p
        while a > 0 and p - a < reach and score[a - 1] >= 0.5 * peak_value and not blocked[a - 1]:
            a -= 1
        b = p
        while b < n - 1 and b - p < reach and score[b + 1] >= 0.5 * peak_value and not blocked[b + 1]:
            b += 1

        lower = max((e for s, e in intervals if e <= p), default=0.0)
        upper = min((s for s, e in intervals if s >= p + 1), default=duration)
        start, end = a - lead, b + 1 + tail
        if end - start < min_len:
            deficit = min_len - (end - start)
            start, end = start - deficit * 0.6, end + deficit * 0.4
        if end - start > max_len:
            start = max(start, p + 0.5 - max_len * 0.6)
            end = start + max_len
        start, end = max(start, lower), min(end, upper)

        snapped = _snap(start, end, speech, env, lower, upper, max_shift=6.0)
        if snapped[1] - snapped[0] > max_len * 1.15:
            snapped = _snap(start, end, speech, env, lower, upper, max_shift=0.0)
        start, end = snapped

        guard_a, guard_b = max(0, int(start) - 2), min(n, int(math.ceil(end)) + 2)
        if end - start < min(min_len * 0.5, 3.0):
            blocked[max(0, p - 2):p + 3] = True
            continue
        blocked[guard_a:guard_b] = True
        intervals.append((start, end))

        a_i, b_i = int(start), max(int(start) + 1, int(math.ceil(end)))
        value = 0.6 * peak_value + 0.4 * float(curve.raw[a_i:b_i].mean())
        moments.append(Moment(start, end, value, float(p) + 0.5, reasons=_reasons(curve, a_i, b_i)))

    moments = _merge_close(moments, max_len, gap=2.0)
    best = max((m.score for m in moments), default=0.0) or 1.0
    for m in moments:
        m.score = round(100.0 * m.score / best, 1)
        if transcript:
            m.text = transcript.text_between(m.start, m.end)
    moments.sort(key=lambda m: m.score, reverse=True)
    return moments


def _reasons(curve: ScoreCurve, a: int, b: int) -> List[str]:
    reasons = []
    if curve.excitement[a:b].max(initial=0) >= 0.5:
        reasons.append("loud")
    if curve.burst[a:b].max(initial=0) >= 0.5:
        reasons.append("spike")
    if curve.has_speech and curve.speech[a:b].mean() >= 0.6:
        reasons.append("fast speech")
    phrases = sorted({phrase for t, phrase in curve.keyword_hits.items() if a <= t < b})
    reasons.extend(f'"{p}"' for p in phrases[:3])
    return reasons


def _merge_close(moments: List[Moment], max_len: float, gap: float) -> List[Moment]:
    merged: List[Moment] = []
    for m in sorted(moments, key=lambda m: m.start):
        if merged and m.start - merged[-1].end <= gap and m.end - merged[-1].start <= max_len:
            prev = merged[-1]
            best = prev if prev.score >= m.score else m
            reasons = prev.reasons + [r for r in m.reasons if r not in prev.reasons]
            merged[-1] = Moment(prev.start, max(prev.end, m.end), max(prev.score, m.score), best.peak,
                                reasons=reasons)
        else:
            merged.append(m)
    return merged


# -- selection ----------------------------------------------------------------------------------

def select_highlight(candidates: Sequence[Moment], target: float, duration: float,
                     pad_before: float = 0.0, pad_after: float = 0.0,
                     jump_cuts: bool = False, diversity: float = 0.2) -> List[Moment]:
    """Pick the best moments up to the target length, spread over the whole recording.

    With ``jump_cuts`` the length of a moment is measured after dead air removal.
    ``diversity`` (0..1) penalizes moments close to already selected ones.
    """
    pool = sorted(candidates, key=lambda m: m.score, reverse=True)
    chosen: List[Moment] = []
    total = 0.0
    window = max(120.0, duration * 0.04)
    while pool and total < target:
        def effective(m: Moment) -> float:
            nearby = sum(1 for c in chosen if abs((c.start + c.end) - (m.start + m.end)) / 2 < window)
            return m.score * ((1.0 - diversity) ** nearby)

        index = max(range(len(pool)), key=lambda i: effective(pool[i]))
        moment = pool.pop(index)
        length = moment.edited_duration(jump_cuts)
        if chosen and total + length > target * 1.1:
            continue
        chosen.append(moment)
        total += length

    chosen.sort(key=lambda m: m.start)
    return [_copy(m) for m in pad_moments(chosen, duration, pad_before, pad_after)]


def pad_moments(moments: List[Moment], duration: float, before: float, after: float) -> List[Moment]:
    """Add breathing room around chronologically sorted moments without creating overlaps."""
    for i, m in enumerate(moments):
        prev_end = moments[i - 1].end if i else 0.0
        next_start = moments[i + 1].start if i + 1 < len(moments) else duration
        m.start = max(prev_end, 0.0, m.start - before)
        m.end = min(next_start, duration, m.end + after)
    return moments


def plan_shorts(candidates: Sequence[Moment], curve: ScoreCurve, transcript: Optional[Transcript],
                env: AudioEnvelope, duration: float, count: int, min_len: float, max_len: float,
                pad_before: float = 0.0, pad_after: float = 0.0) -> List[Moment]:
    """Turn the best moments into standalone vertical shorts of ``min_len``..``max_len`` seconds."""
    speech = _SpeechIndex(transcript)
    shorts: List[Moment] = []
    for c in sorted(candidates, key=lambda m: m.score, reverse=True):
        if len(shorts) >= count:
            break
        start, end = c.start, c.end
        if end - start > max_len - pad_before - pad_after:
            start, end = _best_window(curve.raw, start, end, max_len * 0.85)
            start, end = _snap(start, end, speech, env, c.start, c.end, max_shift=4.0)
        if end - start < min_len:
            deficit = min_len - (end - start)
            wide_start = max(0.0, start - deficit * 0.4)
            wide_end = min(duration, end + deficit * 0.6)
            start, end = _snap(wide_start, wide_end, speech, env, max(0.0, wide_start - 8),
                               min(duration, wide_end + 8), max_shift=8.0)
        start, end = max(0.0, start - pad_before), min(duration, end + pad_after)
        if end - start > max_len:
            end = start + max_len
        if end - start < min(min_len, duration) * 0.75:
            continue
        if any(start < s.end and end > s.start for s in shorts):
            continue
        short = Moment(start, end, c.score, c.peak, reasons=list(c.reasons))
        if transcript:
            short.text = transcript.text_between(start, end)
        shorts.append(short)
    return shorts


def _best_window(raw: np.ndarray, start: float, end: float, length: float) -> tuple:
    a, b = int(start), int(math.ceil(end))
    width = max(1, int(length))
    values = raw[a:b]
    if values.size <= width:
        return start, min(end, start + length)
    sums = np.convolve(values, np.ones(width), mode="valid")
    offset = int(np.argmax(sums))
    return float(a + offset), float(a + offset + width)


def _copy(m: Moment) -> Moment:
    return Moment.from_dict(m.to_dict())


def compute_hook(start: float, end: float, peak: float, words: Sequence, length: float = 3.0) -> List[Range]:
    """The best ~3 seconds of a short, used as a teaser at its very beginning.

    Empty when the short is too short or already starts on its peak.
    """
    if end - start < 12 or not start <= peak <= end or peak - start < 5:
        return []
    a = max(start, peak - length * 0.4)
    b = min(end, a + length)
    for w in words:
        ws, we = (w[0], w[1]) if isinstance(w, (list, tuple)) else (w.start, w.end)
        if ws < a < we:
            a = ws
        if ws < b < we:
            b = we
    if b - a < 1.5:
        return []
    return [(round(a, 3), round(min(b, a + length + 1.5), 3))]


# -- jump cuts --------------------------------------------------------------------------------

def compute_keep_ranges(start: float, end: float, env: AudioEnvelope, words: Sequence,
                        min_silence: float, threshold: Optional[float] = None,
                        margin: float = 0.15) -> List[Range]:
    """Remove dead air (quiet AND wordless) longer than ``min_silence`` inside a clip.

    Loud passages without speech (explosions, music, gameplay) are always kept.
    """
    threshold = env.silence_threshold() if threshold is None else threshold
    a, b = env.frame(start), env.frame(end)
    if b - a < 2:
        return [(start, end)]
    quiet = env.rms_db[a:b] < threshold
    for w in words:
        ws, we = (w[0], w[1]) if isinstance(w, (list, tuple)) else (w.start, w.end)
        if we < start or ws > end:
            continue
        quiet[max(0, env.frame(ws - 0.1) - a): max(0, env.frame(we + 0.1) - a + 1)] = False

    cuts: List[Range] = []
    padded = np.concatenate([[False], quiet, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for run_start, run_end in zip(edges[::2], edges[1::2]):
        t0 = start + run_start * env.hop
        t1 = min(end, start + run_end * env.hop)
        if t1 - t0 >= min_silence:
            cut_start = t0 + (margin if t0 > start + env.hop else 0.0)
            cut_end = t1 - (margin if t1 < end - env.hop else 0.0)
            if cut_end - cut_start > 0.1:
                cuts.append((cut_start, cut_end))

    keep = [r for r in subtract_ranges((start, end), cuts) if r[1] - r[0] >= 0.25]
    return merge_ranges(keep) or [(start, end)]
