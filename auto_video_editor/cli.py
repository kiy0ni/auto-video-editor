"""Command line interface. Run without arguments to open the graphical interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __app_name__, __version__
from .reporting import Cancelled, Reporter
from .settings import (CAPTION_STYLES, CONTENT_TYPES, ENCODERS, PROFILES, QUALITIES, SHORTS_LAYOUTS,
                       TRANSCRIBERS, TRANSITIONS, WHISPER_MODELS, Settings)
from .timeline import parse_duration


def _duration(value: str) -> float:
    try:
        return parse_duration(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="auto-video-editor",
        description="Turn long recordings (streams, VODs, podcasts) into a highlight reel and vertical shorts.",
        epilog="Examples:\n"
               "  auto-video-editor stream.mp4\n"
               "  auto-video-editor stream.mp4 --profile short --shorts 8 --layout smart\n"
               "  auto-video-editor stream.mp4 --analyze-only   # review/edit project.json, then:\n"
               "  auto-video-editor --project stream_edit/project.json\n"
               "  auto-video-editor podcast.mp4 --preset talk --set caption_size=130 --set diversity=0.5\n"
               "Run without arguments to open the graphical interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", nargs="*", help="video file(s) to edit; several files are processed one after another")
    p.add_argument("-o", "--output", help="output folder (default: <input>_edit next to the input)")
    p.add_argument("--project", help="render an existing (optionally hand-edited) project.json")
    p.add_argument("--analyze-only", action="store_true", help="analyze and write project.json without rendering")
    p.add_argument("--gui", action="store_true", help="open the graphical interface")
    p.add_argument("--preset", choices=CONTENT_TYPES,
                   help="auto (default: detected per video), gaming, talk or vlog")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", default=[],
                   help="change any setting, e.g. --set caption_color=#00FF88 (repeatable)")
    p.add_argument("--list-settings", action="store_true", help="list every setting with its default value")
    p.add_argument("--version", action="version", version=f"{__app_name__} {__version__}")

    g = p.add_argument_group("highlight reel")
    g.add_argument("--profile", choices=["auto"] + list(PROFILES),
                   help="highlight length (default: auto, follows the number of great moments)")
    g.add_argument("--target", type=_duration, help="exact highlight length, e.g. 12m, 1h, 90s")
    g.add_argument("--min-clip", type=_duration, help="shortest moment (default: 8s)")
    g.add_argument("--max-clip", type=_duration, help="longest moment (default: 90s)")
    g.add_argument("--jump-cuts", action=argparse.BooleanOptionalAction, default=None,
                   help="remove dead air inside highlight clips")
    g.add_argument("--min-silence", type=float, help="shortest pause removed by jump cuts, seconds (default: 0.8)")
    g.add_argument("--transition", choices=TRANSITIONS, help="cut (default) or short fade between moments")
    g.add_argument("--no-highlight", action="store_true", help="do not render the highlight reel")
    g.add_argument("--no-clips", action="store_true", help="do not keep the individual clip files")

    g = p.add_argument_group("vertical shorts")
    g.add_argument("--shorts", metavar="N|auto", help="number of shorts, auto (default) or 0 to disable")
    g.add_argument("--shorts-min", type=_duration, help="shortest short (default: 20s)")
    g.add_argument("--shorts-max", type=_duration, help="longest short (default: 59s)")
    g.add_argument("--layout", choices=SHORTS_LAYOUTS,
                   help="auto (default), blur, crop, smart (face tracking), split (facecam on top)")
    g.add_argument("--hook", action=argparse.BooleanOptionalAction, default=None,
                   help="open each short with its best 3 seconds")
    g.add_argument("--facecam", metavar="X,Y,W,H", help="facecam area for the split layout, fractions 0-1")
    g.add_argument("--captions", action=argparse.BooleanOptionalAction, default=None, help="burn captions")
    g.add_argument("--caption-style", choices=CAPTION_STYLES)
    g.add_argument("--shorts-jump-cuts", action=argparse.BooleanOptionalAction, default=None,
                   help="remove dead air inside shorts (default: on)")

    g = p.add_argument_group("analysis")
    g.add_argument("--transcriber", choices=TRANSCRIBERS, help="speech-to-text engine (default: auto)")
    g.add_argument("--model", choices=WHISPER_MODELS, help="Whisper model (default: auto, fits this computer)")
    g.add_argument("--auto-vocabulary", action=argparse.BooleanOptionalAction, default=None,
                   help="add the vocabulary of the detected game (default: on)")
    g.add_argument("--auto-trim", action=argparse.BooleanOptionalAction, default=None,
                   help="skip intro/outro where nobody talks (default: on)")
    g.add_argument("--language", help="spoken language code, e.g. en, fr (default: auto-detect)")
    g.add_argument("--keywords", help="comma separated hype phrases (replaces the default list)")
    g.add_argument("--vocabulary", help="comma separated names/jargon the transcriber should know (e.g. Minecraft,mob)")
    g.add_argument("--censor", action=argparse.BooleanOptionalAction, default=None,
                   help="mask swear words in captions and texts (p*tain)")
    g.add_argument("--speakers", type=int, metavar="N", help="number of speakers (0 = auto, 1 = single voice)")
    g.add_argument("--skip-start", type=_duration, help="ignore the start of the recording, e.g. 2m")
    g.add_argument("--skip-end", type=_duration, help="ignore the end of the recording, e.g. 90s")
    g.add_argument("--no-cache", action="store_true", help="ignore cached audio analysis and transcripts")

    g = p.add_argument_group("output")
    g.add_argument("--encoder", choices=ENCODERS, help="video encoder (default: auto, uses hardware if available)")
    g.add_argument("--quality", choices=QUALITIES, help="encoding quality (default: high)")
    g.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None,
                   help="normalize loudness (default: on)")
    g.add_argument("--lufs", type=float, help="loudness target (default: -14 LUFS)")
    g.add_argument("--srt", action=argparse.BooleanOptionalAction, default=None, help="export subtitles")
    g.add_argument("--edl", action=argparse.BooleanOptionalAction, default=None, help="export an EDL")
    g.add_argument("--xml", action=argparse.BooleanOptionalAction, default=None, help="export FCP7 XML")
    g.add_argument("--publish-kit", action=argparse.BooleanOptionalAction, default=None,
                   help="titles, descriptions, hashtags and thumbnails (default: on)")
    return p


def settings_from_args(args: argparse.Namespace, base: Optional[Settings] = None) -> Settings:
    s = Settings.from_dict(base.to_dict()) if base else Settings()
    if getattr(args, "preset", None):
        s.apply_content_preset(args.preset)
    mapping = {
        "profile": "profile", "target": "target_duration", "min_clip": "min_clip", "max_clip": "max_clip",
        "jump_cuts": "jump_cuts", "min_silence": "min_silence", "transition": "transition",
        "shorts_min": "shorts_min", "shorts_max": "shorts_max", "layout": "shorts_layout",
        "facecam": "facecam", "captions": "captions", "caption_style": "caption_style",
        "shorts_jump_cuts": "shorts_jump_cuts", "transcriber": "transcriber", "model": "whisper_model",
        "language": "language", "encoder": "encoder", "quality": "quality", "normalize": "normalize_audio",
        "lufs": "target_lufs", "srt": "export_srt", "edl": "export_edl", "xml": "export_xml",
        "censor": "censor_profanity", "speakers": "speaker_count", "skip_start": "skip_start", "skip_end": "skip_end",
        "hook": "shorts_hook", "publish_kit": "publish_kit", "auto_trim": "auto_trim",
        "auto_vocabulary": "auto_vocabulary",
    }
    for arg_name, field_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(s, field_name, value)
    if args.shorts is not None:
        if str(args.shorts).strip().lower() == "auto":
            s.shorts_auto, s.make_shorts = True, True
        else:
            try:
                count = int(args.shorts)
            except ValueError:
                raise ValueError("--shorts expects a number or 'auto'")
            s.shorts_count, s.shorts_auto, s.make_shorts = max(0, count), False, count > 0
    if args.no_highlight:
        s.make_highlight = False
    if args.no_clips:
        s.export_clips = False
    if args.keywords is not None:
        s.keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    if getattr(args, "vocabulary", None) is not None:
        s.vocabulary = [k.strip() for k in args.vocabulary.split(",") if k.strip()]
    if getattr(args, "speakers", None) is not None:
        s.diarize = args.speakers != 1
    if args.no_cache:
        s.use_cache = False
    if s.facecam and args.layout is None and args.facecam is not None:
        s.shorts_layout = "split"
    for item in getattr(args, "set", None) or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        s = s.with_value(key.strip(), value.strip())
    return s


class ConsoleProgress:
    def __init__(self) -> None:
        self.interactive = sys.stderr.isatty()
        self._line = False
        self._last_label = ""

    def log(self, message: str) -> None:
        self._clear()
        print(message, flush=True)

    def progress(self, value: float, label: str) -> None:
        if not self.interactive:
            if label != self._last_label:
                self._last_label = label
                print(f"[{label}]", file=sys.stderr, flush=True)
            return
        width = 30
        filled = int(width * value)
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(f"\r[{bar}] {value * 100:5.1f}%  {label[:40]:<40}")
        sys.stderr.flush()
        self._line = True

    def _clear(self) -> None:
        if self._line:
            sys.stderr.write("\r" + " " * 90 + "\r")
            sys.stderr.flush()
            self._line = False

    def finish(self) -> None:
        self._clear()


def _launch_gui(path: Optional[str] = None) -> int:
    try:
        from .gui import main as gui_main
    except ImportError as exc:
        print(f"The desktop app needs extra packages ({exc}).\n"
              "Install them with: pip install customtkinter pillow tkinterdnd2", file=sys.stderr)
        return 1
    return gui_main(path)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv == ["--gui"]:
        return _launch_gui()

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gui:
        return _launch_gui(args.input[0] if args.input else None)
    if args.list_settings:
        for key, value in Settings().to_dict().items():
            shown = ",".join(value) if isinstance(value, list) else value
            print(f"{key} = {shown}")
        return 0

    from .pipeline import Pipeline, Project

    console = ConsoleProgress()
    reporter = Reporter(console.log, console.progress)
    try:
        if args.project:
            project = Project.load(Path(args.project))
            if args.output:
                project.output_dir = str(Path(args.output).expanduser().resolve())
            settings = settings_from_args(args, project.settings)
            pipeline = Pipeline(project.input, settings, project.output_dir, reporter)
            pipeline.render(project)
            return 0

        if not args.input:
            parser.error("an input video (or --project) is required")
        settings = settings_from_args(args)
        inputs = list(args.input)
        failures = 0
        for number, video in enumerate(inputs, 1):
            if len(inputs) > 1:
                console.log(f"\n=== [{number}/{len(inputs)}] {Path(video).name} ===")
            output = args.output
            if output and len(inputs) > 1:
                output = str(Path(output) / f"{Path(video).stem}_edit")
            try:
                pipeline = Pipeline(video, settings, output, reporter)
                project = pipeline.analyze()
                if args.analyze_only:
                    console.finish()
                    print(f"\nReview or edit {project.path} (set \"selected\": false to drop a moment), then run:\n"
                          f"  auto-video-editor --project \"{project.path}\"")
                    continue
                pipeline.render(project)
            except (Cancelled, KeyboardInterrupt):
                raise
            except Exception as exc:  # keep going with the other files of a batch
                if len(inputs) == 1:
                    raise
                failures += 1
                console.finish()
                print(f"ERROR on {video}: {exc}", file=sys.stderr)
        return 1 if failures else 0
    except KeyboardInterrupt:
        reporter.cancel()
        console.finish()
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Cancelled:
        console.finish()
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:  # report cleanly instead of a traceback
        console.finish()
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        console.finish()
