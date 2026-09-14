"""Building ffmpeg filter graphs and rendering clips, reels and vertical shorts."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from .ffmpeg import MediaInfo, audio_codec_args, run_ffmpeg, video_codec_args
from .timeline import Range, total_duration

VERTICAL_W, VERTICAL_H = 1080, 1920
SPLIT_TOP_H = 640
MICRO_FADE = 0.02  # seconds; removes clicks at every audio cut


def fps_expression(fps: float) -> str:
    for base in (24, 30, 60, 120):
        if abs(fps - base * 1000 / 1001) < 0.005:
            return f"{base * 1000}/1001"
    return f"{fps:.3f}".rstrip("0").rstrip(".")


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


# -- vertical layouts ---------------------------------------------------------------------------

def vertical_filter(src: str, dst: str, info: MediaInfo, layout: str,
                    subject_x: Optional[float] = None,
                    facecam: Optional[Tuple[float, float, float, float]] = None, blur: int = 10) -> str:
    """Filter graph that turns ``src`` into a 1080x1920 frame labelled ``dst``."""
    W, H = VERTICAL_W, VERTICAL_H
    if layout == "blur":
        radius = max(1, min(60, int(blur)))
        return (
            f"{src}split=2[bgsrc][fgsrc];"
            f"[bgsrc]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
            f"boxblur={radius}:2,scale={W}:{H},eq=brightness=-0.06:saturation=1.15[bg];"
            f"[fgsrc]scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{dst}"
        )
    if layout == "split" and facecam:
        fx, fy, fw, fh = facecam
        cw, ch = _even(fw * info.width), _even(fh * info.height)
        cx = min(max(0, int(fx * info.width)), max(0, info.width - cw))
        cy = min(max(0, int(fy * info.height)), max(0, info.height - ch))
        bottom = H - SPLIT_TOP_H
        return (
            f"{src}split=2[camsrc][gamesrc];"
            f"[camsrc]crop={cw}:{ch}:{cx}:{cy},scale={W}:{SPLIT_TOP_H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{SPLIT_TOP_H}[cam];"
            f"[gamesrc]scale={W}:{bottom}:force_original_aspect_ratio=increase,crop={W}:{bottom}[game];"
            f"[cam][game]vstack=inputs=2,setsar=1{dst}"
        )
    # crop / smart crop
    src_w, src_h = info.width or 1920, info.height or 1080
    if src_w / src_h > W / H:
        cw, ch = min(src_w, _even(src_h * W / H)), src_h
        center = (subject_x if subject_x is not None else 0.5) * src_w
        x, y = int(min(max(0.0, center - cw / 2), src_w - cw)), 0
    else:
        cw, ch = src_w, min(src_h, _even(src_w * H / W))
        x, y = 0, int((src_h - ch) / 2)
    return f"{src}crop={cw}:{ch}:{x}:{y},scale={W}:{H}:flags=lanczos,setsar=1{dst}"


def caption_margin(layout: str, info: MediaInfo) -> int:
    """Vertical margin (from the bottom) that keeps captions off the busy parts of the frame."""
    if layout == "split":
        return VERTICAL_H - SPLIT_TOP_H - 190
    if layout == "blur":
        video_h = VERTICAL_W / (info.aspect or 16 / 9)
        return max(260, int((VERTICAL_H - video_h) / 2 - 190))
    return int(VERTICAL_H * 0.28)


# -- loudness ---------------------------------------------------------------------------------------

def loudnorm_measure_filter(target_lufs: float) -> str:
    return f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json"


def parse_loudnorm(stderr: str) -> Optional[dict]:
    matches = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr)
    if not matches:
        return None
    try:
        data = json.loads(matches[-1])
        float(data["input_i"])
        if data["input_i"] in ("-inf", "inf"):
            return None
        return data
    except (ValueError, KeyError):
        return None


def loudnorm_apply_filter(measured: Optional[dict], target_lufs: float) -> str:
    if not measured:
        return f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
    return (
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:measured_I={measured['input_i']}"
        f":measured_TP={measured['input_tp']}:measured_LRA={measured['input_lra']}"
        f":measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:linear=true"
    )


# -- renderer -----------------------------------------------------------------------------------------

class Renderer:
    def __init__(self, info: MediaInfo, encoder: str, quality: str,
                 cancel_event: Optional[threading.Event] = None, audio_bitrate: int = 192,
                 blur: int = 10) -> None:
        self.info = info
        self.encoder = encoder
        self.quality = quality
        self.cancel_event = cancel_event
        self.audio_args = audio_codec_args(audio_bitrate)
        self.blur = blur

    def build_segment_command(
        self,
        ranges: Sequence[Range],
        output: str,
        *,
        layout: Optional[str] = None,
        subject_x: Optional[float] = None,
        facecam: Optional[Tuple[float, float, float, float]] = None,
        captions_file: Optional[str] = None,
        fade: float = 0.0,
        audio_filter: Optional[str] = None,
    ) -> List[str]:
        """ffmpeg arguments that cut ``ranges`` out of the source into one file."""
        info = self.info
        ranges = [(a, b) for a, b in ranges if b - a > 0.04]
        if not ranges:
            raise ValueError("Nothing to render: empty time range")
        out_duration = total_duration(ranges)

        # Chronological ranges share one input. A range that goes back in time (a teaser hook) opens a
        # new input, so ffmpeg never has to keep minutes of decoded frames in memory.
        runs: List[List[Range]] = []
        for rg in ranges:
            if runs and rg[0] >= runs[-1][-1][1] - 1e-3:
                runs[-1].append(rg)
            else:
                runs.append([rg])
        args: List[str] = []
        graph: List[str] = []
        vsrc: List[str] = []
        asrc: List[str] = []
        local: List[Range] = []
        for index, run in enumerate(runs):
            origin, finish = run[0][0], run[-1][1]
            args += ["-ss", f"{origin:.3f}", "-t", f"{finish - origin + 0.1:.3f}", "-i", info.path]
            local += [(a - origin, b - origin) for a, b in run]
            if len(run) == 1:
                vsrc.append(f"[{index}:v:0]")
                asrc.append(f"[{index}:a:0]")
                continue
            vlabels = [f"[vs{index}_{j}]" for j in range(len(run))]
            alabels = [f"[as{index}_{j}]" for j in range(len(run))]
            graph.append(f"[{index}:v:0]split={len(run)}{''.join(vlabels)}")
            if info.has_audio:
                graph.append(f"[{index}:a:0]asplit={len(run)}{''.join(alabels)}")
            vsrc += vlabels
            asrc += alabels
        silent_input = len(runs)
        if not info.has_audio:
            args += ["-f", "lavfi", "-t", f"{out_duration:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]

        count = len(local)
        if count == 1:
            a, b = local[0]
            graph.append(f"{vsrc[0]}trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[vcat]")
            if info.has_audio:
                graph.append(f"{asrc[0]}atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[acat]")
        else:
            for i, (a, b) in enumerate(local):
                graph.append(f"{vsrc[i]}trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]")
                if info.has_audio:
                    graph.append(
                        f"{asrc[i]}atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,"
                        f"afade=t=in:d={MICRO_FADE},afade=t=out:st={max(0.0, b - a - MICRO_FADE):.3f}:d={MICRO_FADE}[a{i}]"
                    )
            if info.has_audio:
                pairs = "".join(f"[v{i}][a{i}]" for i in range(count))
                graph.append(f"{pairs}concat=n={count}:v=1:a=1[vcat][acat]")
            else:
                graph.append("".join(f"[v{i}]" for i in range(count)) + f"concat=n={count}:v=1:a=0[vcat]")

        # Video post-processing.
        current = "[vcat]"
        video_steps = [f"fps={fps_expression(info.fps)}"]
        if info.width % 2 or info.height % 2:
            video_steps.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
        graph.append(f"{current}{','.join(video_steps)}[vfps]")
        current = "[vfps]"
        if layout:
            graph.append(vertical_filter(current, "[vlay]", info, layout, subject_x, facecam, self.blur))
            current = "[vlay]"
        tail = []
        if fade > 0:
            tail.append(f"fade=t=in:st=0:d={fade:.3f},fade=t=out:st={max(0.0, out_duration - fade):.3f}:d={fade:.3f}")
        if captions_file:
            tail.append(f"ass={captions_file}")
        tail.append("format=yuv420p")
        graph.append(f"{current}{','.join(tail)}[vout]")

        # Audio post-processing.
        edge_fade = max(fade, MICRO_FADE)
        audio_steps = [f"afade=t=in:d={edge_fade:.3f}",
                       f"afade=t=out:st={max(0.0, out_duration - edge_fade):.3f}:d={edge_fade:.3f}"]
        if audio_filter:
            audio_steps.append(audio_filter)
        audio_steps.append("aresample=48000")
        audio_src = "[acat]" if info.has_audio else f"[{silent_input}:a]"
        graph.append(f"{audio_src}{','.join(audio_steps)}[aout]")

        args += [
            "-filter_complex", ";".join(graph),
            "-map", "[vout]", "-map", "[aout]",
            *video_codec_args(self.encoder, self.quality),
            *self.audio_args,
            "-t", f"{out_duration:.3f}",
            "-movflags", "+faststart",
            output,
        ]
        return args

    def render_segment(self, ranges: Sequence[Range], output: Path,
                       on_progress: Optional[Callable[[float], None]] = None,
                       cwd: Optional[str] = None, **options) -> None:
        output = Path(output)
        tmp = output.with_name(f".{output.stem}.partial{output.suffix}")
        args = self.build_segment_command(ranges, str(tmp.resolve()), **options)
        try:
            run_ffmpeg(args, duration=total_duration(ranges), on_progress=on_progress,
                       cancel_event=self.cancel_event, cwd=cwd)
            os.replace(tmp, output)
        finally:
            if tmp.exists():
                tmp.unlink()

    # -- loudness ----------------------------------------------------------------------------------
    def measure_segment_loudness(self, ranges: Sequence[Range], target_lufs: float) -> Optional[dict]:
        if not self.info.has_audio or not ranges:
            return None
        origin, finish = min(a for a, _ in ranges), max(b for _, b in ranges)
        args = ["-ss", f"{origin:.3f}", "-t", f"{finish - origin:.3f}", "-i", self.info.path,
                "-map", "0:a:0", "-vn", "-af", loudnorm_measure_filter(target_lufs), "-f", "null", "-"]
        stderr = run_ffmpeg(args, cancel_event=self.cancel_event, loglevel="info")
        return parse_loudnorm(stderr)

    # -- concatenation -------------------------------------------------------------------------------
    def concat(self, files: Sequence[Path], output: Path, work_dir: Path, total: float,
               normalize: bool, target_lufs: float,
               on_progress: Optional[Callable[[float], None]] = None) -> None:
        list_file = work_dir / "concat.txt"
        with list_file.open("w", encoding="utf-8") as handle:
            for f in files:
                escaped = str(Path(f).resolve()).replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
        source = ["-f", "concat", "-safe", "0", "-i", str(list_file)]
        tmp = output.with_name(f".{output.stem}.partial{output.suffix}")
        try:
            if normalize:
                measure = run_ffmpeg(
                    [*source, "-vn", "-af", loudnorm_measure_filter(target_lufs), "-f", "null", "-"],
                    duration=total, cancel_event=self.cancel_event, loglevel="info",
                    on_progress=(lambda p: on_progress(p * 0.4)) if on_progress else None,
                )
                apply = loudnorm_apply_filter(parse_loudnorm(measure), target_lufs)
                args = [*source, "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
                        "-af", f"{apply},aresample=48000", *self.audio_args,
                        "-movflags", "+faststart", str(tmp)]
                progress = (lambda p: on_progress(0.4 + p * 0.6)) if on_progress else None
            else:
                args = [*source, "-c", "copy", "-movflags", "+faststart", str(tmp)]
                progress = on_progress
            run_ffmpeg(args, duration=total, on_progress=progress, cancel_event=self.cancel_event)
            os.replace(tmp, output)
        finally:
            if tmp.exists():
                tmp.unlink()
            if list_file.exists():
                list_file.unlink()
