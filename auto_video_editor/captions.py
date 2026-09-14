"""Subtitle generation: burned-in ASS captions for shorts and SRT sidecar files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .timeline import TimeMap, format_ass_time, format_srt_time


SPEAKER_COLORS = ["#FFFFFF", "#7DD3FC", "#F9A8D4", "#86EFAC", "#FDBA74", "#C4B5FD"]


@dataclass
class TimedWord:
    start: float
    end: float
    text: str
    speaker: int = 0


def remap_words(words: Iterable, time_map: TimeMap) -> List[TimedWord]:
    """Move source-timed words onto an edited timeline, dropping words that were cut."""
    result: List[TimedWord] = []
    for w in words:
        if hasattr(w, "start"):
            start, end, text, speaker = w.start, w.end, w.text, getattr(w, "speaker", 0)
        else:
            start, end, text = w[0], w[1], w[2]
            speaker = int(w[3]) if len(w) > 3 else 0
        text = text.strip()
        if not text:
            continue
        mapped = time_map.map_interval(start, end)
        if mapped is None:
            continue
        s, e = mapped
        if result and s < result[-1].end:
            s = result[-1].end
        result.append(TimedWord(s, max(e, s + 0.05), text, speaker))
    return result


def remap_words_sequence(words: Iterable, ranges: Sequence) -> List[TimedWord]:
    """Like :func:`remap_words` for ranges played in the given order, which may repeat or go back in
    time (a teaser hook played before the short)."""
    result: List[TimedWord] = []
    offset = 0.0
    for a, b in ranges:
        for word in remap_words(words, TimeMap([(a, b)])):
            word.start += offset
            word.end += offset
            result.append(word)
        offset += b - a
    return result


def group_words(words: Sequence[TimedWord], max_words: int = 3, max_chars: int = 18,
                max_gap: float = 0.6, max_duration: float = 1.8) -> List[List[TimedWord]]:
    """Group words into short caption lines, breaking on pauses, punctuation and speaker changes."""
    groups: List[List[TimedWord]] = []
    current: List[TimedWord] = []
    for word in words:
        if current:
            chars = sum(len(w.text) + 1 for w in current) + len(word.text)
            if (len(current) >= max_words or chars > max_chars or word.speaker != current[-1].speaker
                    or word.start - current[-1].end > max_gap
                    or word.end - current[0].start > max_duration
                    or current[-1].text[-1:] in ".!?,;:"):
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def ass_color(hex_color: str) -> str:
    """``#RRGGBB`` -> ASS ``&HBBGGRR&``."""
    value = hex_color.strip().lstrip("#")
    return f"&H{value[4:6]}{value[2:4]}{value[0:2]}&".upper()


def build_ass(words: Sequence[TimedWord], width: int = 1080, height: int = 1920,
              style: str = "karaoke", margin_v: int = 0, font: str = "Arial", size_scale: float = 1.0,
              uppercase: bool = True, highlight: str = "#FFE600", max_words: int = 3,
              position: str = "auto", speaker_colors: bool = False) -> str:
    """Build an ASS subtitle file with bold, centered captions.

    ``karaoke`` highlights the active word (``highlight`` color) with a small pop animation;
    ``simple`` shows plain white lines. ``position`` is auto (``margin_v`` from the bottom),
    top, middle or bottom. With ``speaker_colors`` each detected speaker gets its own text color.
    """
    multi_speaker = speaker_colors and len({w.speaker for w in words}) > 1
    font_size = int(height * 0.047 * size_scale)
    outline = max(3, int(font_size * 0.09))
    alignment = 2
    if position == "top":
        alignment, margin_v = 8, int(height * 0.14)
    elif position == "middle":
        alignment, margin_v = 5, 0
    elif position == "bottom":
        margin_v = int(height * 0.14)
    else:
        margin_v = margin_v or int(height * 0.30)
    highlight_tag = "{\\c" + ass_color(highlight) + "\\fscx112\\fscy112\\t(0,90,\\fscx100\\fscy100)}"
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,{font},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,1,0,1,{outline},2,{alignment},60,60,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for group in group_words(words, max_words=max_words, max_chars=max(12, 6 * max_words)):
        texts = [_ass_escape(w.text.upper() if uppercase else w.text) for w in group]
        group_end = group[-1].end
        prefix, reset = "", "{\\r}"
        if multi_speaker:
            color = ass_color(SPEAKER_COLORS[group[0].speaker % len(SPEAKER_COLORS)])
            prefix, reset = "{\\c" + color + "}", "{\\r\\c" + color + "}"
        if style != "karaoke":
            lines.append(_dialogue(group[0].start, group_end, prefix + " ".join(texts)))
            continue
        for i, word in enumerate(group):
            start = word.start
            end = group[i + 1].start if i + 1 < len(group) else group_end
            if end <= start:
                continue
            parts = []
            for j, text in enumerate(texts):
                if j == i:
                    parts.append(highlight_tag + text + reset)
                else:
                    parts.append(text)
            lines.append(_dialogue(start, end, prefix + " ".join(parts)))
    return header + "".join(lines)


def _dialogue(start: float, end: float, text: str) -> str:
    return f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},Caption,,0,0,0,,{text}\n"


def build_srt(words: Sequence[TimedWord], max_chars: int = 42, max_duration: float = 5.0,
              max_gap: float = 1.2) -> str:
    """Build readable SRT cues (up to two lines of ``max_chars``) from timed words."""
    cues: List[List[TimedWord]] = []
    current: List[TimedWord] = []
    for word in words:
        if current:
            length = sum(len(w.text) + 1 for w in current) + len(word.text)
            if (length > max_chars * 2 or word.start - current[-1].end > max_gap
                    or word.speaker != current[-1].speaker
                    or word.end - current[0].start > max_duration
                    or (current[-1].text[-1:] in ".!?" and length > max_chars * 0.6)):
                cues.append(current)
                current = []
        current.append(word)
    if current:
        cues.append(current)

    out = []
    for index, cue in enumerate(cues, 1):
        text = " ".join(w.text for w in cue)
        if len(text) > max_chars:
            text = _balance_lines(text, max_chars)
        end = max(cue[-1].end, cue[0].start + 0.5)
        if index < len(cues):
            end = min(end, cues[index][0].start)
        out.append(f"{index}\n{format_srt_time(cue[0].start)} --> {format_srt_time(end)}\n{text}\n")
    return "\n".join(out)


def _balance_lines(text: str, max_chars: int) -> str:
    words = text.split(" ")
    best, best_score = text, float("inf")
    for i in range(1, len(words)):
        first, second = " ".join(words[:i]), " ".join(words[i:])
        score = abs(len(first) - len(second)) + (100 if max(len(first), len(second)) > max_chars else 0)
        if score < best_score:
            best, best_score = f"{first}\n{second}", score
    return best
