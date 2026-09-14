"""Hand-off files for editors: EDL, Final Cut Pro 7 XML, chapters and a moments CSV."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Sequence, Tuple
from xml.sax.saxutils import escape

from .ffmpeg import MediaInfo
from .timeline import Range, format_clock

Event = Tuple[float, float]  # (source_in, source_out) in seconds


def timebase(fps: float) -> Tuple[int, bool]:
    """Integer timebase and NTSC flag, e.g. 29.97 -> (30, True)."""
    rounded = int(round(fps))
    ntsc = abs(fps - rounded * 1000 / 1001) < 0.01 and abs(fps - rounded) > 0.001
    return max(1, rounded), ntsc


def to_frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def frames_to_timecode(frames: int, base: int) -> str:
    frames = max(0, int(frames))
    ff = frames % base
    total_seconds = frames // base
    return f"{total_seconds // 3600:02d}:{(total_seconds // 60) % 60:02d}:{total_seconds % 60:02d}:{ff:02d}"


def timecode_to_frames(timecode: str, base: int) -> int:
    match = re.fullmatch(r"(\d+)[:;](\d+)[:;](\d+)[:;.](\d+)", timecode.strip()) if timecode else None
    if not match:
        return 0
    h, m, s, f = (int(v) for v in match.groups())
    return ((h * 60 + m) * 60 + s) * base + f


def write_edl(path: Path, title: str, media: MediaInfo, events: Sequence[Event]) -> None:
    """CMX 3600 EDL (imports into Premiere Pro, DaVinci Resolve, Avid...)."""
    base, _ntsc = timebase(media.fps)
    source_offset = timecode_to_frames(media.start_timecode, base)
    clip_name = Path(media.path).name
    lines = [f"TITLE: {title[:60]}", "FCM: NON-DROP FRAME", ""]
    record = 0
    for index, (src_in, src_out) in enumerate(events, 1):
        f_in, f_out = to_frames(src_in, media.fps), to_frames(src_out, media.fps)
        if f_out <= f_in:
            continue
        length = f_out - f_in
        lines.append(
            f"{index:03d}  AX       AA/V  C        "
            f"{frames_to_timecode(f_in + source_offset, base)} {frames_to_timecode(f_out + source_offset, base)} "
            f"{frames_to_timecode(record, base)} {frames_to_timecode(record + length, base)}"
        )
        lines.append(f"* FROM CLIP NAME: {clip_name}")
        lines.append("")
        record += length
    path.write_text("\n".join(lines), encoding="utf-8")


def write_fcp_xml(path: Path, title: str, media: MediaInfo, events: Sequence[Event]) -> None:
    """Final Cut Pro 7 XML (xmeml v5), importable in Premiere Pro and DaVinci Resolve."""
    base, ntsc = timebase(media.fps)
    rate = f"<rate><timebase>{base}</timebase><ntsc>{'TRUE' if ntsc else 'FALSE'}</ntsc></rate>"
    src_frames = to_frames(media.duration, media.fps)
    file_url = Path(media.path).resolve().as_uri()
    name = escape(Path(media.path).name)
    width, height = media.width or 1920, media.height or 1080
    channels = max(1, min(2, media.audio_channels or 2))

    file_full = (
        f'<file id="file-1"><name>{name}</name><pathurl>{escape(file_url)}</pathurl>{rate}'
        f"<duration>{src_frames}</duration>"
        f"<timecode>{rate}<string>{escape(media.start_timecode or '00:00:00:00')}</string>"
        f"<displayformat>NDF</displayformat></timecode>"
        f"<media><video><samplecharacteristics>{rate}<width>{width}</width><height>{height}</height>"
        f"<pixelaspectratio>square</pixelaspectratio></samplecharacteristics></video>"
        + (f"<audio><samplecharacteristics><depth>16</depth><samplerate>{media.sample_rate or 48000}"
           f"</samplerate></samplecharacteristics><channelcount>{channels}</channelcount></audio>"
           if media.has_audio else "")
        + "</media></file>"
    )

    clips: List[Tuple[int, int, int, int]] = []  # (rec_start, rec_end, src_in, src_out)
    record = 0
    for src_in, src_out in events:
        f_in, f_out = to_frames(src_in, media.fps), to_frames(src_out, media.fps)
        if f_out <= f_in:
            continue
        clips.append((record, record + f_out - f_in, f_in, f_out))
        record += f_out - f_in

    def clipitem(kind: str, index: int, clip, track_index: int = 1) -> str:
        rec_start, rec_end, f_in, f_out = clip
        file_ref = file_full if (kind == "v" and index == 0) else '<file id="file-1"/>'
        source_track = (f"<sourcetrack><mediatype>audio</mediatype><trackindex>{track_index}</trackindex></sourcetrack>"
                        if kind == "a" else "")
        return (
            f'<clipitem id="clipitem-{kind}{track_index}-{index + 1}"><name>{name}</name>'
            f"<duration>{src_frames}</duration>{rate}<start>{rec_start}</start><end>{rec_end}</end>"
            f"<in>{f_in}</in><out>{f_out}</out>{file_ref}{source_track}</clipitem>"
        )

    video_track = "<track>" + "".join(clipitem("v", i, c) for i, c in enumerate(clips)) + "</track>"
    audio_tracks = ""
    if media.has_audio:
        audio_tracks = "".join(
            "<track>" + "".join(clipitem("a", i, c, ch) for i, c in enumerate(clips)) + "</track>"
            for ch in range(1, channels + 1)
        )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="5">'
        f'<sequence id="sequence-1"><name>{escape(title)}</name><duration>{record}</duration>{rate}'
        f"<media><video><format><samplecharacteristics>{rate}<width>{width}</width><height>{height}</height>"
        f"<pixelaspectratio>square</pixelaspectratio></samplecharacteristics></format>{video_track}</video>"
        f"<audio>{audio_tracks}</audio></media></sequence></xmeml>\n"
    )
    path.write_text(xml, encoding="utf-8")


def write_chapters(path: Path, entries: Sequence[Tuple[float, str]]) -> None:
    """YouTube-style chapter list: ``00:00 Title`` per line."""
    lines = []
    for output_time, title in entries:
        lines.append(f"{format_clock(output_time)} {title}".rstrip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_moments_csv(path: Path, rows: Sequence[dict]) -> None:
    columns = ["kind", "index", "output_file", "output_start", "source_start", "source_end",
               "duration", "score", "reasons", "text"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def events_from_ranges(ranges_per_moment: Sequence[Sequence[Range]]) -> List[Event]:
    return [(a, b) for ranges in ranges_per_moment for a, b in ranges]
