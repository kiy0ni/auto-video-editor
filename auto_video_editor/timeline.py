"""Time range helpers: merging, subtracting, remapping and formatting."""

from __future__ import annotations

import bisect
import re
from typing import Iterable, List, Optional, Sequence, Tuple

Range = Tuple[float, float]


def merge_ranges(ranges: Iterable[Range], gap: float = 0.0) -> List[Range]:
    """Merge overlapping ranges (and ranges closer than ``gap`` seconds)."""
    result: List[Range] = []
    for start, end in sorted((float(a), float(b)) for a, b in ranges if b > a):
        if result and start <= result[-1][1] + gap:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def subtract_ranges(base: Range, cuts: Iterable[Range]) -> List[Range]:
    """Return the parts of ``base`` that are not covered by ``cuts``."""
    start, end = base
    keep: List[Range] = []
    cursor = start
    for cut_start, cut_end in merge_ranges(cuts):
        if cut_end <= cursor or cut_start >= end:
            continue
        if cut_start > cursor:
            keep.append((cursor, min(cut_start, end)))
        cursor = max(cursor, cut_end)
    if cursor < end:
        keep.append((cursor, end))
    return keep


def total_duration(ranges: Iterable[Range]) -> float:
    return sum(max(0.0, b - a) for a, b in ranges)


def overlap(a: Range, b: Range) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


class TimeMap:
    """Maps source timestamps onto an edited timeline made of kept source ranges."""

    def __init__(self, ranges: Sequence[Range]) -> None:
        self.ranges: List[Range] = [(float(a), float(b)) for a, b in ranges if b > a]
        self._starts = [a for a, _ in self.ranges]
        self._offsets: List[float] = []
        cursor = 0.0
        for a, b in self.ranges:
            self._offsets.append(cursor)
            cursor += b - a
        self.duration = cursor

    def _index(self, t: float) -> Optional[int]:
        i = bisect.bisect_right(self._starts, t) - 1
        if i >= 0 and self.ranges[i][0] <= t <= self.ranges[i][1]:
            return i
        return None

    def to_output(self, t: float) -> Optional[float]:
        """Output time for source time ``t``, or ``None`` if ``t`` was cut."""
        i = self._index(t)
        if i is None:
            return None
        return self._offsets[i] + (t - self.ranges[i][0])

    def map_interval(self, start: float, end: float) -> Optional[Range]:
        """Map a source interval (e.g. a word) onto the output timeline.

        The interval is kept if its midpoint survived the edit; its edges are
        clamped to the kept range containing the midpoint.
        """
        mid = (start + end) / 2.0
        i = self._index(mid)
        if i is None:
            return None
        a, b = self.ranges[i]
        s, e = max(start, a), min(end, b)
        return self._offsets[i] + (s - a), self._offsets[i] + (e - a)

    def output_segments(self) -> List[Tuple[float, float, float]]:
        """List of ``(source_start, source_end, output_start)``."""
        return [(a, b, off) for (a, b), off in zip(self.ranges, self._offsets)]


# -- formatting ----------------------------------------------------------------------

def format_clock(seconds: float, always_hours: bool = False) -> str:
    """``3725.4`` -> ``1:02:05`` (or ``02:05`` when under an hour)."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h or always_hours:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_ass_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def format_tag(seconds: float) -> str:
    """Filesystem friendly timestamp: ``1h02m05s`` / ``02m05s``."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


_DURATION_UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0}


def parse_duration(text: str) -> float:
    """Parse ``"90"``, ``"90s"``, ``"10m"``, ``"1h30m"``, ``"1:30"`` or ``"1:02:03"``."""
    text = str(text).strip().lower().replace(" ", "")
    if not text:
        raise ValueError("empty duration")
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        if len(parts) > 3:
            raise ValueError(f"invalid duration: {text!r}")
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        return seconds
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    matches = re.findall(r"(\d+(?:\.\d+)?)([hms])", text)
    if not matches or "".join(n + u for n, u in matches) != text:
        raise ValueError(f"invalid duration: {text!r}")
    return sum(float(n) * _DURATION_UNITS[u] for n, u in matches)


def slugify(text: str, max_length: int = 40) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:max_length].strip("-")
