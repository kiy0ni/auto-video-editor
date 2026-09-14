"""Desktop interface (CustomTkinter).

Simple mode: drop a video, pick what you want, click Create.
Advanced mode: every setting of the editing engine is exposed in the sidebar pages.
"""

from __future__ import annotations

import io
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageTk

try:  # drag & drop is optional
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:  # pragma: no cover - depends on the platform build
    DND_FILES = TkinterDnD = None

from . import __app_name__, __version__, icons, vision
from .ffmpeg import _CREATION_FLAGS, MediaInfo, ffmpeg_bin, missing_tools, probe
from .glossary import (Pack, all_categories, auto_packs, get_pack, load_packs, pack_errors, resolve_packs,
                       write_pack_template)
from .insights import (VideoInsights, auto_layout, auto_model, describe_position, estimate_render,
                       estimate_transcription, has_cuda, load_insights, save_insights, scan_video)
from .reporting import Cancelled, Reporter
from .settings import DEFAULT_KEYWORDS, PROFILES, Settings, config_dir, settings_path
from .timeline import format_clock, parse_duration
from .transcribe import available_backends

# -- design tokens: (light, dark) -----------------------------------------------------------------
BG = ("#f3f4f8", "#0d0f14")
SIDEBAR = ("#ffffff", "#12151c")
CARD = ("#ffffff", "#171a22")
CARD_DIM = ("#f7f8fa", "#13161d")
SUBTLE = ("#eef0f5", "#222733")
BORDER = ("#e3e6ec", "#272c38")
TEXT = ("#111318", "#eceff5")
MUTED = ("#667085", "#8a93a6")
FAINT = ("#98a2b3", "#5b6475")
ACCENT = ("#6a5cff", "#7b6dff")
ACCENT_HOVER = ("#5747f5", "#6a5cff")
ACCENT_SOFT = ("#eeecff", "#241f48")
GREEN = ("#12a150", "#34d399")
AMBER = ("#d97706", "#fbbf24")
RED = ("#dc2626", "#f87171")
WHITE = ("#ffffff", "#ffffff")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".wmv", ".ts", ".m4v", ".mts"}
VIDEO_TYPES = [("Video files", " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))), ("All files", "*.*")]

PAGES = [
    ("home", "Create", "Drop a recording, pick what you want, get your edit.", "film"),
    ("highlight", "Highlight reel", "Length, moment boundaries, selection and pacing of the 16:9 reel.", "reel"),
    ("shorts", "Shorts & captions", "Vertical 1080×1920 clips: layout, pacing and caption styling.", "phone"),
    ("detection", "Detection", "Speech recognition and how each second of the recording is scored.", "wave"),
    ("export", "Export", "Encoding, loudness and hand-off files for your editing software.", "export"),
    ("review", "Review", "Preview, keep or drop each moment before rendering.", "star"),
]
SIMPLE_PAGES = ("home", "review")
NAV_GROUPS = {"home": "CREATE", "highlight": "SETTINGS", "review": "RESULT"}

GOALS = [
    ("both", "layers", "Reel + shorts", "A 16:9 highlight reel and vertical clips"),
    ("highlight", "reel", "Highlight reel", "The best moments assembled in 16:9"),
    ("shorts", "phone", "Shorts only", "TikTok, YouTube Shorts and Reels"),
]
CONTENT_TYPES = [
    ("auto", "spark", "Auto-detect", "We watch and listen, then pick the style"),
    ("gaming", "gamepad", "Gaming stream", "Reactions, hype and loud moments"),
    ("talk", "mic", "Podcast & talk", "Strong statements, tight pacing"),
    ("vlog", "camera", "Vlog & IRL", "Dynamic moments, smooth fades"),
]
LAYOUTS = [
    ("auto", "Automatic", "Picked from what is on screen"),
    ("blur", "Blur fill", "Whole frame over a blurred background"),
    ("crop", "Center crop", "Fills the screen, the sides are cut"),
    ("smart", "Face tracking", "The crop follows the detected face"),
    ("split", "Facecam split", "Facecam on top, gameplay below"),
]
LAYOUT_SHORT = [("Auto", "auto"), ("Blur", "blur"), ("Crop", "crop"), ("Face", "smart"), ("Split", "split")]
LAYOUT_NAMES = {key: title for key, title, _ in LAYOUTS}
PROFILE_OPTIONS = [("Auto", "auto"), ("Short", "short"), ("Medium", "medium"), ("Long", "long"), ("Custom", "custom")]

LANGUAGES = [
    ("Auto-detect", ""), ("English", "en"), ("French", "fr"), ("Spanish", "es"), ("German", "de"),
    ("Italian", "it"), ("Portuguese", "pt"), ("Dutch", "nl"), ("Polish", "pl"), ("Turkish", "tr"),
    ("Russian", "ru"), ("Arabic", "ar"), ("Japanese", "ja"), ("Korean", "ko"), ("Chinese", "zh"),
]
ENCODER_LABELS = [
    ("Automatic (hardware if available)", "auto"), ("x264 · CPU, best compatibility", "libx264"),
    ("x265 / HEVC · CPU", "libx265"), ("Apple VideoToolbox H.264", "h264_videotoolbox"),
    ("Apple VideoToolbox HEVC", "hevc_videotoolbox"), ("NVIDIA NVENC H.264", "h264_nvenc"),
    ("NVIDIA NVENC HEVC", "hevc_nvenc"), ("Intel Quick Sync H.264", "h264_qsv"), ("AMD AMF H.264", "h264_amf"),
]
TRANSCRIBER_LABELS = [
    ("Automatic", "auto"), ("faster-whisper (recommended)", "faster-whisper"),
    ("OpenAI Whisper", "openai-whisper"), ("Off · audio only", "none"),
]
MODEL_LABELS = [
    ("Automatic · best for this computer", "auto"), ("Tiny · fastest", "tiny"), ("Base · fast", "base"), ("Small · balanced", "small"),
    ("Medium · accurate", "medium"), ("Large v3 · best", "large-v3"), ("Turbo · best on GPU", "turbo"),
]
FONT_LABELS = [(name, name) for name in ("Arial", "Arial Black", "Helvetica", "Impact", "Verdana",
                                         "Trebuchet MS", "Georgia", "Courier New")]
CAPTION_COLORS = ["#FFE600", "#39FF88", "#22D3EE", "#FF4FD8", "#FF8A00", "#FFFFFF"]


# -- helpers ----------------------------------------------------------------------------------------

def _tuple(color) -> Tuple[str, str]:
    return (color, color) if isinstance(color, str) else color


def themed_icon(name: str, size: int = 18, color=TEXT) -> ctk.CTkImage:
    light, dark = _tuple(color)
    return ctk.CTkImage(light_image=icons.icon(name, light, size * 2),
                        dark_image=icons.icon(name, dark, size * 2), size=(size, size))


def score_color(score: float):
    return GREEN if score >= 70 else AMBER if score >= 40 else MUTED


def extract_frame(path: str, t: float, width: int = 480) -> Optional[Image.Image]:
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t):.2f}", "-i", path,
           "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "png", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30, creationflags=_CREATION_FLAGS)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        return Image.open(io.BytesIO(proc.stdout)).convert("RGB")
    except OSError:
        return None


def open_path(path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        pass


def _ui_prefs_path() -> Path:
    return config_dir() / "ui.json"


def load_ui_prefs() -> dict:
    try:
        return json.loads(_ui_prefs_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_ui_prefs(prefs: dict) -> None:
    try:
        _ui_prefs_path().parent.mkdir(parents=True, exist_ok=True)
        _ui_prefs_path().write_text(json.dumps(prefs), encoding="utf-8")
    except OSError:
        pass


def _file_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    return f"{size / 1e9:.2f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"


class WheelScroll:
    """Mouse wheel and trackpad scrolling for CTkScrollableFrame pages.

    CustomTkinter only handles ``<MouseWheel>`` with Tk 8.6 deltas; Tk 9 reports wheel deltas in
    multiples of 120 and sends trackpad gestures as ``<TouchpadScroll>``, so pages did not scroll.
    """

    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        self.canvases: List[tk.Canvas] = []
        patch = str(root.tk.call("info", "patchlevel"))
        self.tk9 = int(patch.split(".")[0]) >= 9
        self.rebind()

    def rebind(self) -> None:
        """Install the handlers. Call again after creating a CTkScrollableFrame: CustomTkinter adds its
        own global wheel handler every time one is created."""
        root = self.root
        for sequence in ("<MouseWheel>", "<Shift-MouseWheel>", "<Button-4>", "<Button-5>", "<TouchpadScroll>"):
            try:
                root.unbind_all(sequence)
            except tk.TclError:
                pass
        root.bind_all("<MouseWheel>", self._on_wheel, add="+")
        root.bind_all("<Button-4>", lambda e: self._scroll(e.widget, -60), add="+")
        root.bind_all("<Button-5>", lambda e: self._scroll(e.widget, 60), add="+")
        try:
            root.bind_all("<TouchpadScroll>", self._on_touchpad, add="+")
        except tk.TclError:  # Tk 8.6 has no touchpad event
            pass

    def register(self, frame: ctk.CTkScrollableFrame) -> None:
        canvas = frame._parent_canvas
        canvas.configure(yscrollincrement=1)  # scroll by pixels
        self.canvases.append(canvas)

    def unregister(self, frame: ctk.CTkScrollableFrame) -> None:
        canvas = getattr(frame, "_parent_canvas", None)
        if canvas in self.canvases:
            self.canvases.remove(canvas)

    def _on_wheel(self, event) -> None:
        if self.tk9 or sys.platform != "darwin":
            pixels = -event.delta / 4.0          # 120 per notch -> 30 px
        else:
            pixels = -event.delta * 12.0         # Tk 8.6 on macOS: small raw deltas
        self._scroll(event.widget, pixels)

    def _on_touchpad(self, event) -> None:
        try:
            _dx, dy = (int(v) for v in self.root.tk.splitlist(
                self.root.tk.call("tk::PreciseScrollDeltas", event.delta)))
        except (tk.TclError, ValueError):
            return
        self._scroll(event.widget, -dy)

    def _scroll(self, widget, pixels: float) -> None:
        if isinstance(widget, str):
            try:
                widget = self.root.nametowidget(widget)
            except (KeyError, tk.TclError):
                return
        try:
            if widget.winfo_class() == "Text":  # text boxes scroll themselves
                return
        except (AttributeError, tk.TclError):
            return
        path = str(widget)
        for canvas in list(self.canvases):
            base = str(canvas)
            if not (path == base or path.startswith(base + ".")):
                continue
            try:
                if not canvas.winfo_ismapped():
                    continue
                if canvas.yview() == (0.0, 1.0) or not pixels:
                    return
                step = int(round(pixels)) or (1 if pixels > 0 else -1)
                canvas.yview_scroll(step, "units")
            except tk.TclError:  # window closed
                self.canvases.remove(canvas)
            return


# ====================================================================================================
# Application
# ====================================================================================================

class App:
    def __init__(self, root: ctk.CTk, initial_input: Optional[str] = None) -> None:
        self.root = root
        self.settings = Settings.load(settings_path())
        self.prefs = load_ui_prefs()
        self.mode = self.prefs.get("mode", "simple")
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.reporter: Optional[Reporter] = None
        self.project = None
        self.media: Optional[MediaInfo] = None
        self.media_frame: Optional[Image.Image] = None
        self.insights: Optional[VideoInsights] = None
        self.busy = False
        self.current_page = "home"
        self.vars: Dict[str, tk.Variable] = {}
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.silence_auto = tk.BooleanVar(value=True)
        self.cards: Dict[tuple, SimpleNamespace] = {}
        self._thumbs: Dict[tuple, Image.Image] = {}
        self._thumb_generation = 0
        self._build_queue: List[tuple] = []
        self._phase_started = time.monotonic()
        self._estimate_labels: List[ctk.CTkLabel] = []
        self._custom_lines: List[ctk.CTkFrame] = []
        self._facecam_widgets: List[Tuple[ctk.CTkBaseClass, dict]] = []
        self._layout_cards: Dict[str, SimpleNamespace] = {}
        self._tile_refreshers: List[Callable[[], None]] = []
        self._swatches: Dict[str, ctk.CTkButton] = {}
        self._shorts_count_widgets: List[ctk.CTkBaseClass] = []
        self._mode_widgets: List[Tuple[ctk.CTkBaseClass, str, dict]] = []
        self.review_kind = "h"

        self.f = SimpleNamespace(
            display=ctk.CTkFont(size=28, weight="bold"),
            title=ctk.CTkFont(size=20, weight="bold"),
            h2=ctk.CTkFont(size=16, weight="bold"),
            body=ctk.CTkFont(size=13),
            body_bold=ctk.CTkFont(size=13, weight="bold"),
            small=ctk.CTkFont(size=12),
            small_bold=ctk.CTkFont(size=12, weight="bold"),
            tiny=ctk.CTkFont(size=11),
            tiny_bold=ctk.CTkFont(size=11, weight="bold"),
            score=ctk.CTkFont(size=26, weight="bold"),
            cta=ctk.CTkFont(size=16, weight="bold"),
            mono=ctk.CTkFont(family="Menlo" if sys.platform == "darwin" else "Consolas", size=12),
        )
        self._placeholder = ctk.CTkImage(
            light_image=icons.placeholder((352, 198), SUBTLE[0], FAINT[0]),
            dark_image=icons.placeholder((352, 198), SUBTLE[1], FAINT[1]), size=(176, 99))
        self._source_placeholder = ctk.CTkImage(
            light_image=icons.placeholder((512, 288), SUBTLE[0], FAINT[0], 24),
            dark_image=icons.placeholder((512, 288), SUBTLE[1], FAINT[1], 24), size=(256, 144))

        root.title(__app_name__)
        root.configure(fg_color=BG)
        width, height = 1320, 900
        x = max(0, (root.winfo_screenwidth() - width) // 2)
        y = max(0, (root.winfo_screenheight() - height) // 3)
        root.geometry(f"{width}x{height}+{x}+{y}")
        root.minsize(1120, 760)
        self._set_window_icon()

        self._build()
        self.vars["facecam"].trace_add("write", lambda *_: self._update_detected())
        self.vars["vocabulary_packs"].trace_add("write", lambda *_: self._update_detected())
        self.scroll = WheelScroll(root)
        for page in self.pages.values():
            self.scroll.register(page)
        self._load_settings()
        self._apply_mode()
        self.show_page("home")
        self._setup_drag_and_drop()
        self._bind_shortcuts()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(60, self._poll_events)
        if initial_input:
            self.set_input(initial_input)

    # ================================================================================================
    # Skeleton
    # ================================================================================================
    def _set_window_icon(self) -> None:
        try:
            self._icon_photo = ImageTk.PhotoImage(icons.logo(256))
            self.root.iconphoto(True, self._icon_photo)
        except tk.TclError:
            pass

    def _build(self) -> None:
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self._build_sidebar()

        main = ctk.CTkFrame(self.root, fg_color=BG, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)
        self.main = main

        header = ctk.CTkFrame(main, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=40, pady=(30, 14))
        self.page_title = ctk.CTkLabel(header, text="", font=self.f.display, text_color=TEXT, anchor="w")
        self.page_title.pack(anchor="w")
        self.page_subtitle = ctk.CTkLabel(header, text="", font=self.f.body, text_color=MUTED, anchor="w")
        self.page_subtitle.pack(anchor="w")

        host = ctk.CTkFrame(main, fg_color="transparent")
        host.grid(row=1, column=0, sticky="nsew", padx=(28, 22))
        self.pages: Dict[str, ctk.CTkScrollableFrame] = {}
        for key, *_ in PAGES:
            self.pages[key] = ctk.CTkScrollableFrame(
                host, fg_color="transparent", scrollbar_button_color=SUBTLE,
                scrollbar_button_hover_color=BORDER)

        self._build_home_page(self.pages["home"])
        self._build_highlight_page(self.pages["highlight"])
        self._build_shorts_page(self.pages["shorts"])
        self._build_detection_page(self.pages["detection"])
        self._build_export_page(self.pages["export"])
        self._build_review_page(self.pages["review"])
        self._build_action_bar(main)
        self._build_log(main)

    def _build_sidebar(self) -> None:
        bar = ctk.CTkFrame(self.root, width=252, corner_radius=0, fg_color=SIDEBAR)
        bar.grid(row=0, column=0, sticky="nsw")
        bar.grid_propagate(False)
        bar.pack_propagate(False)
        ctk.CTkFrame(self.root, width=1, corner_radius=0, fg_color=BORDER).grid(row=0, column=0, sticky="nse")

        brand = ctk.CTkFrame(bar, fg_color="transparent")
        brand.pack(fill="x", padx=22, pady=(28, 20))
        self._logo = ctk.CTkImage(icons.logo(96), size=(42, 42))
        ctk.CTkLabel(brand, image=self._logo, text="").pack(side="left")
        names = ctk.CTkFrame(brand, fg_color="transparent")
        names.pack(side="left", padx=12)
        ctk.CTkLabel(names, text="Auto Video Editor", font=self.f.h2, text_color=TEXT, anchor="w",
                     height=20).pack(anchor="w")
        ctk.CTkLabel(names, text=f"Highlights & shorts · v{__version__}", font=self.f.tiny, text_color=MUTED,
                     anchor="w", height=16).pack(anchor="w")

        self.mode_switch = ctk.CTkSegmentedButton(
            bar, values=["Simple", "Advanced"], command=self._on_mode_click, height=34, corner_radius=10,
            fg_color=SUBTLE, selected_color=ACCENT, selected_hover_color=ACCENT_HOVER, unselected_color=SUBTLE,
            unselected_hover_color=BORDER, text_color=TEXT, font=self.f.small_bold, dynamic_resizing=False)
        self.mode_switch.pack(fill="x", padx=16, pady=(0, 8))
        self.mode_switch.set("Advanced" if self.mode == "advanced" else "Simple")

        self.nav_frame = ctk.CTkFrame(bar, fg_color="transparent")
        self.nav_frame.pack(fill="x")
        self.nav: Dict[str, SimpleNamespace] = {}
        self.nav_captions: Dict[str, ctk.CTkLabel] = {}
        for key, title, _subtitle, icon_name in PAGES:
            if key in NAV_GROUPS:
                self.nav_captions[key] = ctk.CTkLabel(self.nav_frame, text=NAV_GROUPS[key], font=self.f.tiny_bold,
                                                      text_color=FAINT, anchor="w", height=18)
            normal, active = themed_icon(icon_name, 18, MUTED), themed_icon(icon_name, 18, ACCENT)
            button = ctk.CTkButton(
                self.nav_frame, text=f"  {title}", image=normal, compound="left", anchor="w", height=42,
                corner_radius=12, fg_color="transparent", hover_color=SUBTLE, text_color=MUTED,
                font=self.f.body_bold, command=lambda k=key: self.show_page(k))
            self.nav[key] = SimpleNamespace(button=button, normal=normal, active=active, title=title)

        appearance = ctk.CTkSegmentedButton(
            bar, values=["Light", "Dark", "System"], command=self._set_appearance, height=32, corner_radius=10,
            fg_color=SUBTLE, selected_color=CARD, selected_hover_color=CARD, unselected_color=SUBTLE,
            unselected_hover_color=BORDER, text_color=TEXT, font=self.f.small_bold)
        appearance.pack(side="bottom", fill="x", padx=16, pady=(0, 20))
        appearance.set(self.prefs.get("appearance", "Dark"))

        status = ctk.CTkFrame(bar, fg_color=SUBTLE, corner_radius=14)
        status.pack(side="bottom", fill="x", padx=16, pady=(0, 12))
        ctk.CTkLabel(status, text="ENGINES", font=self.f.tiny_bold, text_color=MUTED, anchor="w").pack(
            anchor="w", padx=14, pady=(10, 2))
        backends = available_backends()
        dnd = getattr(self.root, "dnd_ok", False)
        for name, value, ok in (
            ("Speech", backends[0] if backends else "not installed", bool(backends)),
            ("Face tracking", "OpenCV" if vision.available() else "not installed", vision.available()),
            ("Drag & drop", "on" if dnd else "off", dnd),
        ):
            row = ctk.CTkFrame(status, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=1)
            ctk.CTkLabel(row, text="●", font=self.f.tiny, text_color=GREEN if ok else AMBER, width=12).pack(side="left")
            ctk.CTkLabel(row, text=name, font=self.f.small, text_color=TEXT, height=22).pack(side="left", padx=(6, 0))
            ctk.CTkLabel(row, text=value, font=self.f.tiny, text_color=MUTED, height=22).pack(side="right")
        ctk.CTkFrame(status, height=8, fg_color="transparent").pack()

    def _layout_nav(self) -> None:
        for widget in self.nav_frame.winfo_children():
            widget.pack_forget()
        for key, *_ in PAGES:
            if self.mode != "advanced" and key not in SIMPLE_PAGES:
                continue
            if key in self.nav_captions and (self.mode == "advanced" or key != "highlight"):
                self.nav_captions[key].pack(anchor="w", padx=28, pady=(14, 4))
            self.nav[key].button.pack(fill="x", padx=14, pady=2)

    def show_page(self, key: str) -> None:
        if self.mode != "advanced" and key not in SIMPLE_PAGES:
            key = "home"
        for name, page in self.pages.items():
            if name != key:
                page.pack_forget()
        self.pages[key].pack(fill="both", expand=True)
        for name, item in self.nav.items():
            selected = name == key
            item.button.configure(fg_color=ACCENT_SOFT if selected else "transparent",
                                  text_color=ACCENT if selected else MUTED,
                                  image=item.active if selected else item.normal)
        _, title, subtitle, _ = next(p for p in PAGES if p[0] == key)
        self.page_title.configure(text=title)
        self.page_subtitle.configure(text=subtitle)
        self.current_page = key

    def _on_mode_click(self, label: str) -> None:
        self.mode = "advanced" if label == "Advanced" else "simple"
        self.prefs["mode"] = self.mode
        save_ui_prefs(self.prefs)
        self._apply_mode()
        self.show_page(self.current_page)
        self.toast("Advanced mode: every setting is in the sidebar" if self.mode == "advanced"
                   else "Simple mode: presets take care of the details", "info")

    def _apply_mode(self) -> None:
        advanced = self.mode == "advanced"
        self._layout_nav()
        self.analyze_button.configure(text="Analyze" if advanced else "Review first")
        self.render_button.configure(text="Render" if advanced else "Create edit")
        for widget, when, pack_options in self._mode_widgets:
            if (when == "advanced") == advanced:
                widget.pack(**pack_options)
            else:
                widget.pack_forget()

    def _only_in(self, mode: str, widget, **pack_options) -> None:
        self._mode_widgets.append((widget, mode, pack_options))

    def _set_appearance(self, mode: str) -> None:
        ctk.set_appearance_mode(mode.lower())
        self.prefs["appearance"] = mode
        save_ui_prefs(self.prefs)

    # ================================================================================================
    # Building blocks
    # ================================================================================================
    def _var(self, name: str, factory) -> tk.Variable:
        if name not in self.vars:
            self.vars[name] = factory()
        return self.vars[name]

    def _section(self, parent, title: str, subtitle: str = "", icon_name: Optional[str] = None,
                 step: Optional[str] = None, pack: bool = True) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=18, border_width=1, border_color=BORDER)
        if pack:
            card.pack(fill="x", padx=12, pady=(0, 18))
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=24, pady=(20, 6))
        if step:
            ctk.CTkLabel(head, text=step, font=self.f.body_bold, text_color=WHITE, width=30, height=30,
                         fg_color=ACCENT, corner_radius=15).pack(side="left", padx=(0, 12))
        elif icon_name:
            ctk.CTkLabel(head, text="", image=themed_icon(icon_name, 18, ACCENT), width=36, height=36,
                         fg_color=ACCENT_SOFT, corner_radius=10).pack(side="left", padx=(0, 12))
        texts = ctk.CTkFrame(head, fg_color="transparent")
        texts.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(texts, text=title, font=self.f.h2, text_color=TEXT, anchor="w", height=22).pack(anchor="w")
        if subtitle:
            ctk.CTkLabel(texts, text=subtitle, font=self.f.small, text_color=MUTED, anchor="w",
                         height=18).pack(anchor="w")
        card.head = head
        card.row_count = 0
        return card

    def _row(self, card, title: str, description: str = "") -> ctk.CTkFrame:
        if card.row_count:
            ctk.CTkFrame(card, height=1, fg_color=BORDER).pack(fill="x", padx=24)
        card.row_count += 1
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=14)
        row.grid_columnconfigure(0, weight=1)
        texts = ctk.CTkFrame(row, fg_color="transparent")
        texts.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(texts, text=title, font=self.f.body_bold, text_color=TEXT, anchor="w", height=20).pack(anchor="w")
        if description:
            ctk.CTkLabel(texts, text=description, font=self.f.small, text_color=MUTED, anchor="w",
                         height=18).pack(anchor="w")
        controls = ctk.CTkFrame(row, fg_color="transparent")
        controls.grid(row=0, column=1, sticky="e", padx=(20, 0))
        controls.texts = texts
        return controls

    @staticmethod
    def _end(card) -> None:
        ctk.CTkFrame(card, height=8, fg_color="transparent").pack()

    def _switch(self, parent, name: str, command: Optional[Callable] = None, text: str = "",
                variable: Optional[tk.BooleanVar] = None) -> ctk.CTkSwitch:
        var = variable if variable is not None else self._var(name, tk.BooleanVar)
        switch = ctk.CTkSwitch(
            parent, text=text, variable=var, onvalue=True, offvalue=False, switch_width=46, switch_height=24,
            progress_color=ACCENT, fg_color=BORDER, button_color="#ffffff", button_hover_color="#f4f4f8",
            font=self.f.small, text_color=TEXT, command=command)
        switch.pack(side="right")
        return switch

    def _slider(self, parent, name: str, low: float, high: float, step: float, fmt: str,
                width: int = 250, integer: bool = False) -> ctk.CTkSlider:
        var = self._var(name, tk.DoubleVar)
        value_label = ctk.CTkLabel(parent, text="", font=self.f.body_bold, text_color=TEXT, width=76, anchor="e")
        slider = ctk.CTkSlider(
            parent, from_=low, to=high, number_of_steps=max(1, int(round((high - low) / step))), variable=var,
            width=width, height=18, progress_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            fg_color=SUBTLE)

        def refresh(*_):
            try:
                value = var.get()
            except tk.TclError:
                return
            value_label.configure(text=fmt.format(round(value) if integer else value))

        var.trace_add("write", refresh)
        refresh()
        slider.pack(side="left")
        value_label.pack(side="left", padx=(10, 0))
        return slider

    def _stepper(self, parent, name: str, low: int, high: int, suffix: str = "") -> ctk.CTkFrame:
        var = self._var(name, tk.DoubleVar)
        frame = ctk.CTkFrame(parent, fg_color=SUBTLE, corner_radius=11)

        def change(delta: int) -> None:
            try:
                current = int(round(var.get()))
            except tk.TclError:
                current = low
            var.set(max(low, min(high, current + delta)))

        button = dict(width=36, height=36, corner_radius=10, fg_color="transparent", hover_color=BORDER,
                      text_color=TEXT, font=self.f.h2)
        ctk.CTkButton(frame, text="−", command=lambda: change(-1), **button).pack(side="left", padx=2, pady=2)
        label = ctk.CTkLabel(frame, text="", width=80, font=self.f.body_bold, text_color=TEXT)
        label.pack(side="left")
        ctk.CTkButton(frame, text="+", command=lambda: change(1), **button).pack(side="left", padx=2, pady=2)

        def refresh(*_):
            try:
                value = int(round(var.get()))
            except tk.TclError:
                return
            label.configure(text=f"{value}{suffix}")

        var.trace_add("write", refresh)
        refresh()
        frame.pack(side="right")
        return frame

    def _choice_vars(self, name: str, options: Sequence[Tuple[str, str]]):
        value_var = self._var(name, tk.StringVar)
        to_value = dict(options)
        to_label = {v: k for k, v in options}
        display = tk.StringVar(value=to_label.get(value_var.get(), options[0][0]))
        value_var.trace_add("write", lambda *_: display.set(to_label.get(value_var.get(), display.get())))
        return value_var, display, to_value

    def _segmented(self, parent, name: str, options: Sequence[Tuple[str, str]],
                   command: Optional[Callable] = None, width: Optional[int] = None) -> ctk.CTkSegmentedButton:
        value_var, display, to_value = self._choice_vars(name, options)

        def on_click(label):
            value_var.set(to_value[label])
            if command:
                command()

        widget = ctk.CTkSegmentedButton(
            parent, values=[label for label, _ in options], variable=display, command=on_click, height=34,
            corner_radius=10, fg_color=SUBTLE, selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
            unselected_color=SUBTLE, unselected_hover_color=BORDER, text_color=TEXT, font=self.f.small_bold,
            dynamic_resizing=False, width=width or 90 * len(options))
        widget.pack(side="right")
        return widget

    def _option(self, parent, name: str, options: Sequence[Tuple[str, str]], width: int = 260) -> ctk.CTkOptionMenu:
        value_var, display, to_value = self._choice_vars(name, options)
        widget = ctk.CTkOptionMenu(
            parent, values=[label for label, _ in options], variable=display,
            command=lambda label: value_var.set(to_value[label]), width=width, height=36, corner_radius=10,
            fg_color=SUBTLE, button_color=SUBTLE, button_hover_color=BORDER, text_color=TEXT, font=self.f.small_bold,
            dropdown_fg_color=CARD, dropdown_hover_color=SUBTLE, dropdown_text_color=TEXT,
            dropdown_font=self.f.small, dynamic_resizing=False)
        widget.pack(side="right")
        return widget

    def _entry(self, parent, var: tk.StringVar, width: int = 220) -> ctk.CTkEntry:
        return ctk.CTkEntry(parent, textvariable=var, width=width, height=36, corner_radius=10, border_width=1,
                            fg_color=SUBTLE, border_color=BORDER, text_color=TEXT, font=self.f.body)

    def _button(self, parent, text: str, command, icon_name: Optional[str] = None, kind: str = "subtle",
                width: int = 120, height: int = 36, font=None) -> ctk.CTkButton:
        styles = {
            "accent": dict(fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=WHITE, icon=WHITE),
            "subtle": dict(fg_color=SUBTLE, hover_color=BORDER, text_color=TEXT, icon=TEXT),
            "ghost": dict(fg_color="transparent", hover_color=SUBTLE, text_color=TEXT, icon=TEXT),
            "danger": dict(fg_color="transparent", hover_color=SUBTLE, text_color=RED, icon=RED, border_color=RED,
                           border_width=1),
        }[kind]
        icon_color = styles.pop("icon")
        return ctk.CTkButton(
            parent, text=text, command=command, width=width, height=height, corner_radius=11,
            image=themed_icon(icon_name, 15, icon_color) if icon_name else None, compound="left",
            font=font or self.f.body_bold, **styles)

    def _chip(self, parent, text: str, color=SUBTLE, text_color=TEXT) -> ctk.CTkLabel:
        return ctk.CTkLabel(parent, text=f"  {text}  ", font=self.f.tiny_bold, fg_color=color, text_color=text_color,
                            corner_radius=8, height=24)

    @staticmethod
    def _bind_click(widget, callback) -> None:
        """Make a whole card clickable. CustomTkinter widgets wrap inner Tk widgets, so the same click can
        reach several bindings: a short debounce makes sure the callback runs once."""
        last = [0.0]

        def fire(_event=None):
            now = time.monotonic()
            if now - last[0] > 0.4:
                last[0] = now
                callback()

        def bind(w):
            if isinstance(w, (ctk.CTkButton, ctk.CTkSwitch, ctk.CTkSegmentedButton)):
                return
            try:
                w.bind("<Button-1>", fire, add="+")
            except (TypeError, ValueError, NotImplementedError):
                w.bind("<Button-1>", fire)
            for child in w.winfo_children():
                bind(child)

        bind(widget)

    def _tiles(self, parent, options, is_selected: Callable[[str], bool], on_pick: Callable[[str], None]) -> None:
        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.pack(fill="x", padx=18, pady=(10, 20))
        tiles = {}
        for index, (key, icon_name, title, description) in enumerate(options):
            grid.grid_columnconfigure(index, weight=1, uniform="tiles")
            tile = ctk.CTkFrame(grid, fg_color=SUBTLE, corner_radius=16, border_width=2, border_color=SUBTLE)
            tile.grid(row=0, column=index, sticky="nsew", padx=6)
            icon_label = ctk.CTkLabel(tile, text="", image=themed_icon(icon_name, 22, ACCENT), width=46, height=46,
                                      fg_color=ACCENT_SOFT, corner_radius=13)
            icon_label.pack(anchor="w", padx=16, pady=(16, 10))
            title_label = ctk.CTkLabel(tile, text=title, font=self.f.body_bold, text_color=TEXT, anchor="w", height=20)
            title_label.pack(anchor="w", padx=16)
            ctk.CTkLabel(tile, text=description, font=self.f.small, text_color=MUTED, anchor="w", justify="left",
                         wraplength=170).pack(anchor="w", padx=16, pady=(2, 16))
            self._bind_click(tile, lambda k=key: on_pick(k))
            tiles[key] = SimpleNamespace(tile=tile, title=title_label, icon=icon_label)

        def refresh():
            for key, item in tiles.items():
                selected = is_selected(key)
                item.tile.configure(border_color=ACCENT if selected else SUBTLE,
                                    fg_color=ACCENT_SOFT if selected else SUBTLE)
                item.icon.configure(fg_color=CARD if selected else ACCENT_SOFT)
                item.title.configure(text_color=ACCENT if selected else TEXT)

        self._tile_refreshers.append(refresh)

    def _refresh_tiles(self) -> None:
        for refresh in self._tile_refreshers:
            refresh()

    def _profile_control(self, controls) -> None:
        self._segmented(controls, "profile", PROFILE_OPTIONS, command=self._on_profile_change, width=420).pack(
            side="top", anchor="e")
        line = ctk.CTkFrame(controls, fg_color="transparent")
        ctk.CTkLabel(line, text="Exact length", font=self.f.small, text_color=MUTED).pack(side="left")
        self._entry(line, self._var("target_duration", tk.StringVar), width=110).pack(side="left", padx=(10, 0))
        self._custom_lines.append(line)
        estimate = ctk.CTkLabel(controls.texts, text="", font=self.f.small, text_color=ACCENT, anchor="w",
                                image=themed_icon("clock", 14, ACCENT), compound="left", height=20)
        estimate.pack(anchor="w", pady=(4, 0))
        self._estimate_labels.append(estimate)

    # ================================================================================================
    # Home page (simple & advanced)
    # ================================================================================================
    def _build_home_page(self, page) -> None:
        self._build_source_card(page)

        self.found_card = self._section(page, "What we found", "The automatic choices are based on this.", "spark",
                                        pack=False)
        self.found_chips = ctk.CTkFrame(self.found_card, fg_color="transparent")
        self.found_chips.pack(fill="x", padx=24, pady=(8, 4))
        self.found_eta = ctk.CTkLabel(self.found_card, text="", font=self.f.small, text_color=MUTED, anchor="w",
                                      image=themed_icon("clock", 14, MUTED), compound="left")
        self.found_eta.pack(anchor="w", padx=24, pady=(4, 18))
        self.found_webcam_button = self._button(self.found_card, "Mark my webcam", self._open_facecam_picker, "crop",
                                                "subtle", width=170, height=32)

        goal = self._section(page, "What do you want to create?", step="1")
        self._tiles(goal, GOALS, self._goal_selected, self._pick_goal)

        content = self._section(page, "What kind of video is it?",
                                "Tunes moment detection, pacing and the shorts layout for you.", step="2")
        self._tiles(content, CONTENT_TYPES, lambda k: self.vars["content_type"].get() == k, self._apply_content)
        self._var("content_type", tk.StringVar)

        options = self._section(page, "Adjust the essentials", step="3")
        self._profile_control(self._row(options, "Reel length"))
        controls = self._row(options, "Number of shorts", "Auto makes one short per great moment.")
        self._shorts_count_widgets.append(self._stepper(controls, "shorts_count", 1, 20, " shorts"))
        self._switch(controls, "shorts_auto", command=self._refresh_auto_widgets, text="Auto  ").pack(
            side="right", padx=(0, 14))
        controls = self._row(options, "Shorts format", "How the video fits the vertical screen.")
        self._segmented(controls, "shorts_layout", LAYOUT_SHORT, width=380).pack(side="top", anchor="e")
        self.layout_hint = ctk.CTkLabel(controls, text="", font=self.f.tiny, text_color=ACCENT, height=16)
        self.layout_hint.pack(side="top", anchor="e", pady=(6, 0))
        facecam_button = self._button(controls, "Draw facecam area", self._open_facecam_picker, "crop", "subtle",
                                      width=190, height=32)
        self._facecam_widgets.append((facecam_button, dict(side="top", anchor="e", pady=(8, 0))))
        self._switch(self._row(options, "Animated captions", "Word-by-word subtitles burned into the shorts."),
                     "captions")
        self._option(self._row(options, "Spoken language", "Setting it avoids wrong detections."),
                     "language", LANGUAGES, width=200)
        controls = self._row(options, "Vocabulary", "Packs of games and topics, plus your own words "
                                                    "(names, jargon), comma separated.")
        self._entry(controls, self._var("vocabulary", tk.StringVar), width=240).pack(side="right")
        self._button(controls, "Packs…", self._open_vocab_picker, "sliders", "subtle", width=100).pack(
            side="right", padx=(0, 8))
        self._var("vocabulary_packs", tk.StringVar)
        self.vocab_hint = ctk.CTkLabel(controls.texts, text="", font=self.f.small, text_color=ACCENT, anchor="w",
                                       height=18)
        self.vocab_hint.pack(anchor="w")
        self._switch(self._row(options, "Censor swear words", "They are transcribed, then masked: p*tain, sh*t."),
                     "censor_profanity")
        self._end(options)

        cta = ctk.CTkFrame(page, fg_color=ACCENT_SOFT, corner_radius=18)
        cta.pack(fill="x", padx=12, pady=(0, 18))
        inner = ctk.CTkFrame(cta, fg_color="transparent")
        inner.pack(fill="x", padx=24, pady=20)
        texts = ctk.CTkFrame(inner, fg_color="transparent")
        texts.pack(side="left")
        ctk.CTkLabel(texts, text="Ready when you are", font=self.f.h2, text_color=TEXT, anchor="w",
                     height=22).pack(anchor="w")
        self.cta_detail = ctk.CTkLabel(texts, text="Create does everything in one go. Review first lets you "
                                                   "preview and drop moments before rendering.",
                                       font=self.f.small, text_color=MUTED, anchor="w", height=18)
        self.cta_detail.pack(anchor="w")
        self._button(inner, "Create my edit", self._render, "play", "accent", width=190, height=48,
                     font=self.f.cta).pack(side="right")
        self._button(inner, "Review first", self._analyze, "star", "subtle", width=140, height=48).pack(
            side="right", padx=10)

        hint_simple = ctk.CTkFrame(page, fg_color="transparent")
        ctk.CTkLabel(hint_simple, text="", image=themed_icon("sliders", 16, MUTED)).pack(side="left", padx=(16, 8))
        ctk.CTkLabel(hint_simple, text="Want full control over detection, captions and encoding?",
                     font=self.f.small, text_color=MUTED).pack(side="left")
        ctk.CTkButton(hint_simple, text="Switch to Advanced", command=lambda: self._switch_mode("Advanced"),
                      fg_color="transparent", hover_color=SUBTLE, text_color=ACCENT, font=self.f.small_bold,
                      width=150, height=28).pack(side="left", padx=6)
        self._only_in("simple", hint_simple, fill="x", padx=12, pady=(0, 24))

    def _switch_mode(self, label: str) -> None:
        self.mode_switch.set(label)
        self._on_mode_click(label)

    def _build_source_card(self, page) -> None:
        self.drop_card = ctk.CTkFrame(page, fg_color=CARD, corner_radius=20, border_width=2, border_color=BORDER)
        self.drop_card.pack(fill="x", padx=12, pady=(0, 18))

        empty = ctk.CTkFrame(self.drop_card, fg_color="transparent")
        self.source_empty = empty
        ctk.CTkLabel(empty, text="", image=themed_icon("export", 30, ACCENT), width=76, height=76,
                     fg_color=ACCENT_SOFT, corner_radius=22).pack(pady=(34, 14))
        dnd = getattr(self.root, "dnd_ok", False)
        ctk.CTkLabel(empty, text="Drop a recording here" if dnd else "Choose a recording", font=self.f.title,
                     text_color=TEXT).pack()
        ctk.CTkLabel(empty, text="Streams, VODs, podcasts, gameplay · MP4, MKV, MOV, FLV, WEBM",
                     font=self.f.body, text_color=MUTED).pack(pady=(4, 16))
        self._button(empty, "Browse files", self._browse_input, "folder", "accent", width=160, height=40).pack(
            pady=(0, 34))
        self._bind_click(empty, self._browse_input)

        source = ctk.CTkFrame(self.drop_card, fg_color="transparent")
        self.source_view = source
        source.grid_columnconfigure(1, weight=1)
        self.source_thumb = ctk.CTkLabel(source, text="", image=self._source_placeholder)
        self.source_thumb.grid(row=0, column=0, rowspan=4, padx=(22, 22), pady=22, sticky="w")
        self.source_name = ctk.CTkLabel(source, text="", font=self.f.title, text_color=TEXT, anchor="w",
                                        justify="left", wraplength=560)
        self.source_name.grid(row=0, column=1, sticky="sw", pady=(24, 0))
        self.source_chips = ctk.CTkFrame(source, fg_color="transparent")
        self.source_chips.grid(row=1, column=1, sticky="w", pady=(8, 0))
        out = ctk.CTkFrame(source, fg_color="transparent")
        out.grid(row=2, column=1, sticky="ew", pady=(12, 0), padx=(0, 22))
        out.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(out, text="", image=themed_icon("folder", 15, MUTED)).grid(row=0, column=0, padx=(0, 8))
        self._entry(out, self.output_var).grid(row=0, column=1, sticky="ew")
        self._button(out, "Change", self._browse_output, None, "ghost", width=80, height=36).grid(
            row=0, column=2, padx=(6, 0))
        buttons = ctk.CTkFrame(source, fg_color="transparent")
        buttons.grid(row=3, column=1, sticky="nw", pady=(12, 22))
        self._button(buttons, "Change video", self._browse_input, "film", "subtle", width=140).pack(side="left")
        self._button(buttons, "Preview", self._preview_source, "play", "ghost", width=110).pack(side="left", padx=6)
        self._button(buttons, "Open folder", self._open_output, "folder", "ghost", width=130).pack(side="left")
        self.source_empty.pack(fill="x")

    def _goal_selected(self, key: str) -> bool:
        highlight, shorts = bool(self.vars["make_highlight"].get()), bool(self.vars["make_shorts"].get())
        return {"both": highlight and shorts, "highlight": highlight and not shorts,
                "shorts": shorts and not highlight}[key]

    def _pick_goal(self, key: str) -> None:
        self.vars["make_highlight"].set(key in ("both", "highlight"))
        self.vars["make_shorts"].set(key in ("both", "shorts"))
        self._refresh_tiles()

    def _apply_content(self, key: str) -> None:
        current = self._collect_settings(quiet=True, strict=False)
        current.apply_content_preset(key)
        self.settings = current
        self._load_settings()
        if key == "auto":
            self.toast("The style will be picked after listening to the video", "success")
        else:
            title = next(t for k, _i, t, _d in CONTENT_TYPES if k == key)
            self.toast(f"{title} preset applied", "success")

    # ================================================================================================
    # Advanced pages
    # ================================================================================================
    def _build_highlight_page(self, page) -> None:
        self._switch(self._section(page, "Highlight reel", "The 16:9 edit made of the best moments.", "reel").head,
                     "make_highlight", command=self._refresh_tiles)

        length = self._section(page, "Length", "How long the final reel should be.", "clock")
        self._profile_control(self._row(length, "Preset", "Short ≈ 5% of the source, Medium ≈ 10%, Long ≈ 20%."))
        self._end(length)

        moments = self._section(page, "Moment boundaries", "Cuts always snap to the start and end of sentences.",
                                "crop")
        self._slider(self._row(moments, "Shortest moment", "Shorter moments are extended."),
                     "min_clip", 3, 30, 1, "{:.0f} s")
        self._slider(self._row(moments, "Longest moment", "Longer moments are tightened around their peak."),
                     "max_clip", 15, 180, 5, "{:.0f} s")
        self._slider(self._row(moments, "Build-up before the peak", "Context that sets up the moment."),
                     "context_before", 0, 20, 0.5, "{:.1f} s")
        self._slider(self._row(moments, "Reaction after the peak", "Keeps the laugh, the scream, the reply."),
                     "context_after", 0, 20, 0.5, "{:.1f} s")
        self._slider(self._row(moments, "Lead-in padding", "Breathing room added before each cut."),
                     "pad_before", 0, 2, 0.1, "{:.1f} s")
        self._slider(self._row(moments, "Tail padding", "Breathing room added after each cut."),
                     "pad_after", 0, 3, 0.1, "{:.1f} s")
        self._end(moments)

        selection = self._section(page, "Selection", "Which moments make it into the reel.", "star")
        self._slider(self._row(selection, "Spread over the recording",
                               "0 = only the highest scores, 1 = cover the whole recording."),
                     "diversity", 0, 1, 0.05, "{:.2f}")
        self._slider(self._row(selection, "Minimum score", "Moments below this score are never used, even if "
                                                            "the reel ends up shorter than the target."),
                     "min_score", 0, 90, 5, "{:.0f}")
        self._switch(self._row(selection, "Auto-skip intro and outro",
                               "Finds the waiting screen and the goodbyes: long parts where nobody talks."),
                     "auto_trim")
        self._slider(self._row(selection, "Skip the start", "Waiting screen, greetings, setup."),
                     "skip_start", 0, 900, 15, "{:.0f} s")
        self._slider(self._row(selection, "Skip the end", "Outro, raid, goodbyes."),
                     "skip_end", 0, 900, 15, "{:.0f} s")
        self._end(selection)

        editing = self._section(page, "Pacing & transitions", "How moments flow into each other.", "sliders")
        self._segmented(self._row(editing, "Transition", "Hard cuts, or a short dip to black between moments."),
                        "transition", [("Cut", "cut"), ("Fade", "fade")], width=180)
        self._slider(self._row(editing, "Fade duration", "Only used by the Fade transition."),
                     "fade_duration", 0.1, 1.5, 0.05, "{:.2f} s")
        self._switch(self._row(editing, "Jump cuts",
                               "Remove pauses inside moments. Loud action without speech is always kept."),
                     "jump_cuts")
        self._slider(self._row(editing, "Shortest pause removed", "Pauses shorter than this stay in the edit."),
                     "min_silence", 0.3, 3, 0.1, "{:.1f} s")
        controls = self._row(editing, "Silence level", "What counts as silence for jump cuts.")
        self.silence_slider = self._slider(controls, "silence_threshold", -70, -15, 1, "{:.0f} dB", width=180)
        self._switch(controls, "", command=self._refresh_silence, text="Auto  ", variable=self.silence_auto).pack(
            side="left", padx=(16, 0))
        self._switch(self._row(editing, "Save each moment as a clip", "Writes clips/clip_001_….mp4 for re-use."),
                     "export_clips")
        self._end(editing)

    def _build_shorts_page(self, page) -> None:
        shorts = self._section(page, "Shorts", "Standalone vertical clips built from the best moments.", "phone")
        self._switch(shorts.head, "make_shorts", command=self._refresh_tiles)
        controls = self._row(shorts, "Number of shorts", "Auto makes one short per great moment.")
        self._shorts_count_widgets.append(self._slider(controls, "shorts_count", 1, 20, 1, "{}", integer=True))
        self._switch(controls, "shorts_auto", command=self._refresh_auto_widgets, text="Auto  ").pack(
            side="left", padx=(16, 0))
        self._slider(self._row(shorts, "Minimum length", "Short moments get more context around them."),
                     "shorts_min", 5, 60, 1, "{:.0f} s")
        self._slider(self._row(shorts, "Maximum length", "59 s is the safe limit for every platform."),
                     "shorts_max", 10, 180, 1, "{:.0f} s")
        self._switch(self._row(shorts, "Jump cuts in shorts", "Tighter pacing, never below the minimum length."),
                     "shorts_jump_cuts")
        self._switch(self._row(shorts, "Hook", "Open with the best 3 seconds, then play the short from the start."),
                     "shorts_hook")
        self._end(shorts)

        layout = self._section(page, "Layout", "How the 16:9 video is placed in the 9:16 frame.", "crop")
        grid = ctk.CTkFrame(layout, fg_color="transparent")
        grid.pack(fill="x", padx=18, pady=(10, 14))
        var = self._var("shorts_layout", tk.StringVar)
        for index, (key, title, description) in enumerate(LAYOUTS):
            grid.grid_columnconfigure(index, weight=1, uniform="layout")
            card = ctk.CTkFrame(grid, fg_color=SUBTLE, corner_radius=16, border_width=2, border_color=SUBTLE)
            card.grid(row=0, column=index, sticky="nsew", padx=6)
            image = ctk.CTkImage(icons.layout_preview(key, 128, 224), size=(64, 112))
            ctk.CTkLabel(card, text="", image=image).pack(pady=(18, 10))
            name = ctk.CTkLabel(card, text=title, font=self.f.body_bold, text_color=TEXT)
            name.pack()
            ctk.CTkLabel(card, text=description, font=self.f.tiny, text_color=MUTED, wraplength=120,
                         justify="center").pack(padx=8, pady=(2, 16))
            self._bind_click(card, lambda k=key: var.set(k))
            self._layout_cards[key] = SimpleNamespace(card=card, name=name)
        var.trace_add("write", lambda *_: self._refresh_layout())

        facecam = ctk.CTkFrame(layout, fg_color=ACCENT_SOFT, corner_radius=14)
        inner = ctk.CTkFrame(facecam, fg_color="transparent")
        inner.pack(fill="x", padx=18, pady=14)
        inner.grid_columnconfigure(0, weight=1)
        texts = ctk.CTkFrame(inner, fg_color="transparent")
        texts.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(texts, text="Facecam area", font=self.f.body_bold, text_color=TEXT, anchor="w",
                     height=20).pack(anchor="w")
        ctk.CTkLabel(texts, text="Draw it on a frame of your video, or type x, y, width, height (0–1).",
                     font=self.f.small, text_color=MUTED, anchor="w", height=18).pack(anchor="w")
        self._entry(inner, self._var("facecam", tk.StringVar), width=190).grid(row=0, column=1, padx=10)
        self._button(inner, "Draw on video", self._open_facecam_picker, "crop", "accent", width=150).grid(
            row=0, column=2)
        self._facecam_widgets.append((facecam, dict(fill="x", padx=24, pady=(0, 14))))
        self._slider(self._row(layout, "Background blur", "Blur fill layout only."), "blur_strength", 2, 30, 1,
                     "{:.0f}", integer=True)
        self._end(layout)

        captions = self._section(page, "Captions", "Big, readable, word-by-word subtitles.", "captions")
        self._switch(captions.head, "captions")
        self._segmented(self._row(captions, "Style", "Karaoke highlights the word being spoken."),
                        "caption_style", [("Karaoke", "karaoke"), ("Simple", "simple")], width=200)
        self._option(self._row(captions, "Font", "Must be installed on this computer."), "caption_font",
                     FONT_LABELS, width=200)
        self._slider(self._row(captions, "Size"), "caption_size", 50, 200, 5, "{:.0f} %", integer=True)
        self._segmented(self._row(captions, "Position", "Auto keeps captions away from the video's busy parts."),
                        "caption_position",
                        [("Auto", "auto"), ("Top", "top"), ("Middle", "middle"), ("Bottom", "bottom")], width=320)
        self._slider(self._row(captions, "Words per line"), "caption_words", 1, 6, 1, "{}", integer=True)
        self._switch(self._row(captions, "Uppercase"), "caption_uppercase")
        self._switch(self._row(captions, "Color by speaker", "White, blue, pink... one color per detected voice."),
                     "caption_speaker_colors")
        controls = self._row(captions, "Highlight color", "Color of the word being spoken.")
        color_var = self._var("caption_color", tk.StringVar)
        self._entry(controls, color_var, width=100).pack(side="right", padx=(10, 0))
        for color in reversed(CAPTION_COLORS):
            swatch = ctk.CTkButton(controls, text="", width=28, height=28, corner_radius=14, fg_color=color,
                                   hover_color=color, border_width=2, border_color=SUBTLE,
                                   command=lambda c=color: color_var.set(c))
            swatch.pack(side="right", padx=3)
            self._swatches[color] = swatch
        color_var.trace_add("write", lambda *_: self._refresh_swatches())
        self._end(captions)

    def _build_detection_page(self, page) -> None:
        speech = self._section(page, "Speech recognition",
                               "Used for sentence-aware cuts, hype phrases and captions.", "wave")
        self._option(self._row(speech, "Engine", "faster-whisper is ~4x faster than OpenAI Whisper."),
                     "transcriber", TRANSCRIBER_LABELS)
        self._option(self._row(speech, "Model", "Bigger models are more accurate but slower on CPU."),
                     "whisper_model", MODEL_LABELS)
        self._option(self._row(speech, "Spoken language", "Setting it avoids wrong detections on noisy audio."),
                     "language", LANGUAGES)
        self._slider(self._row(speech, "Beam size", "Higher = slightly more accurate, slower."), "beam_size",
                     1, 10, 1, "{}", integer=True)
        self._switch(self._row(speech, "Reuse previous analysis",
                               "Re-running with other settings takes seconds instead of minutes."), "use_cache")
        self._end(speech)

        context = self._section(page, "Vocabulary & swear words",
                                "What the transcriber is told before it listens.", "captions")
        self._entry(self._row(context, "Vocabulary hints", "Game names, jargon, anglicisms, nicknames. "
                                                            "Comma separated."),
                    self._var("vocabulary", tk.StringVar), width=320).pack(side="right")
        controls = self._row(context, "Vocabulary packs", "Names and jargon of games and topics.")
        self._button(controls, "My packs folder", self._open_packs_folder, "folder", "ghost", width=150).pack(
            side="right")
        self._button(controls, "Choose packs…", self._open_vocab_picker, "sliders", "accent", width=150).pack(
            side="right", padx=(0, 8))
        self.packs_detail = ctk.CTkLabel(controls.texts, text="", font=self.f.small, text_color=ACCENT, anchor="w",
                                         height=18)
        self.packs_detail.pack(anchor="w")
        self._switch(self._row(context, "Recognise packs automatically",
                               "From the title, the metadata and a speech sample: one game and one topic."),
                     "auto_vocabulary", command=self._update_detected)
        self._switch(self._row(context, "Transcribe swear words",
                               "Off, the model tends to skip them silently."), "profanity_prompt")
        self._switch(self._row(context, "Censor swear words", "Masked everywhere they appear: p*tain, sh*t."),
                     "censor_profanity")
        self._entry(self._row(context, "Extra words to censor", "Added to the built-in French/English list."),
                    self._var("censor_words", tk.StringVar), width=320).pack(side="right")
        self._end(context)

        speakers = self._section(page, "Speakers", "Who is talking: colors the captions and labels the "
                                                   "transcript (A:, B:). Built-in, works best with distinct voices.",
                                 "mic")
        self._switch(self._row(speakers, "Detect speakers"), "diarize")
        self._segmented(self._row(speakers, "Number of speakers", "Auto picks the most likely count."),
                        "speaker_count", [("Auto", "0"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5")],
                        width=300)
        self._end(speakers)

        weights = self._section(page, "Scoring", "How much each signal counts when rating a second of video.",
                                "sliders")
        for name, title, description in (
            ("weight_loudness", "Loudness", "Louder than the usual level of that part of the recording."),
            ("weight_spikes", "Sudden spikes", "Shouts, explosions, claps."),
            ("weight_speech", "Speech rate", "People talking fast and a lot."),
            ("weight_keywords", "Hype phrases", "One of the phrases below is said."),
            ("weight_exclamations", "Exclamations", "Sentences transcribed with an exclamation mark."),
        ):
            self._slider(self._row(weights, title, description), name, 0, 3, 0.1, "×{:.1f}")
        self._end(weights)

        hype = self._section(page, "Hype phrases",
                             "Moments where one of these is said get a boost. One phrase per line.", "spark")
        self._switch(self._row(hype, "Built-in phrases for the detected language",
                               "\"no way\", \"c'est chaud\", laughter... Your own phrases below are added."),
                     "auto_keywords")
        self.keywords_box = ctk.CTkTextbox(hype, height=200, corner_radius=12, border_width=0, fg_color=SUBTLE,
                                           text_color=TEXT, font=self.f.body, wrap="word")
        self.keywords_box.pack(fill="x", padx=24, pady=(10, 12))
        row = ctk.CTkFrame(hype, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=(0, 20))
        self._button(row, "Clear", self._reset_keywords, None, "subtle", width=100).pack(side="left")

    def _build_export_page(self, page) -> None:
        video = self._section(page, "Video", "Hardware encoding is much faster on long reels.", "film")
        self._option(self._row(video, "Encoder"), "encoder", ENCODER_LABELS, width=290)
        self._segmented(self._row(video, "Quality", "Draft is great for a quick check."), "quality",
                        [("Draft", "draft"), ("Standard", "standard"), ("High", "high")], width=270)
        self._end(video)

        audio = self._section(page, "Audio", "Consistent volume across moments and platforms.", "wave")
        self._switch(self._row(audio, "Normalize loudness", "Two-pass EBU R128 normalization."), "normalize_audio")
        self._slider(self._row(audio, "Target loudness", "-14 LUFS matches YouTube, TikTok and Spotify."),
                     "target_lufs", -24, -9, 1, "{:.0f} LUFS")
        self._segmented(self._row(audio, "Audio bitrate"), "audio_bitrate",
                        [("128k", "128"), ("192k", "192"), ("256k", "256"), ("320k", "320")], width=280)
        self._end(audio)

        files = self._section(page, "Files for your editor",
                              "Fine-tune the edit on the original recording in your NLE.", "export")
        self._switch(self._row(files, "Subtitles (.srt)", "For the reel and every short."), "export_srt")
        self._switch(self._row(files, "EDL", "CMX 3600 · Premiere Pro, DaVinci Resolve, Avid."), "export_edl")
        self._switch(self._row(files, "Final Cut Pro 7 XML", "Premiere Pro and DaVinci Resolve timelines."),
                     "export_xml")
        self._switch(self._row(files, "Publishing kit", "Titles, descriptions, hashtags and thumbnails for the "
                                                        "reel and every short."), "publish_kit")
        self._end(files)

        reset = self._section(page, "Defaults", "Start again from the recommended settings.", "x")
        row = ctk.CTkFrame(reset, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=(8, 22))
        self._button(row, "Reset all settings", self._reset_all, None, "danger", width=170).pack(side="left")

    def _build_review_page(self, page) -> None:
        self.review_empty = ctk.CTkFrame(page, fg_color=CARD, corner_radius=20, border_width=1, border_color=BORDER)
        ctk.CTkLabel(self.review_empty, text="", image=themed_icon("star", 30, ACCENT), width=76, height=76,
                     fg_color=ACCENT_SOFT, corner_radius=22).pack(pady=(46, 16))
        ctk.CTkLabel(self.review_empty, text="No moments yet", font=self.f.title, text_color=TEXT).pack()
        ctk.CTkLabel(self.review_empty, text="Analyze a recording to see its best moments here, with a score "
                                             "and the reason each one was picked.",
                     font=self.f.body, text_color=MUTED, wraplength=460).pack(pady=(6, 18))
        self._button(self.review_empty, "Find the moments", self._analyze, "star", "accent", width=180,
                     height=40).pack(pady=(0, 46))
        self.review_empty.pack(fill="x", padx=12)

        head = ctk.CTkFrame(page, fg_color=CARD, corner_radius=18, border_width=1, border_color=BORDER)
        self.review_head = head
        top = ctk.CTkFrame(head, fg_color="transparent")
        top.pack(fill="x", padx=22, pady=(18, 8))
        self.review_switch = ctk.CTkSegmentedButton(
            top, values=["Highlight reel", "Shorts"], command=self._on_review_kind, height=36, corner_radius=10,
            fg_color=SUBTLE, selected_color=ACCENT, selected_hover_color=ACCENT_HOVER, unselected_color=SUBTLE,
            unselected_hover_color=BORDER, text_color=TEXT, font=self.f.small_bold, width=280,
            dynamic_resizing=False)
        self.review_switch.pack(side="left")
        self.review_switch.set("Highlight reel")
        self._button(top, "Render these", self._render, "play", "accent", width=140).pack(side="right")
        self._button(top, "None", lambda: self._set_all(False), None, "ghost", width=70).pack(side="right", padx=6)
        self._button(top, "Keep all", lambda: self._set_all(True), "check", "subtle", width=110).pack(side="right")
        stats = ctk.CTkFrame(head, fg_color="transparent")
        stats.pack(fill="x", padx=22, pady=(4, 18))
        self.review_stats = ctk.CTkLabel(stats, text="", font=self.f.body_bold, text_color=TEXT, anchor="w")
        self.review_stats.pack(anchor="w")
        self.review_meter = ctk.CTkProgressBar(stats, height=8, corner_radius=4, progress_color=ACCENT,
                                               fg_color=SUBTLE)
        self.review_meter.pack(fill="x", pady=(8, 0))
        self.review_auto = ctk.CTkFrame(head, fg_color="transparent")
        self.review_auto.pack(fill="x", padx=22, pady=(0, 16))
        self.review_lists = {"h": ctk.CTkFrame(page, fg_color="transparent"),
                             "s": ctk.CTkFrame(page, fg_color="transparent")}

    def _build_action_bar(self, main) -> None:
        bar = ctk.CTkFrame(main, fg_color=CARD, corner_radius=18, border_width=1, border_color=BORDER)
        bar.grid(row=2, column=0, sticky="ew", padx=40, pady=(10, 22))
        bar.grid_columnconfigure(0, weight=1)

        left = ctk.CTkFrame(bar, fg_color="transparent")
        left.grid(row=0, column=0, sticky="ew", padx=(22, 16), pady=16)
        top = ctk.CTkFrame(left, fg_color="transparent")
        top.pack(fill="x")
        self.status_dot = ctk.CTkLabel(top, text="●", font=self.f.small, text_color=FAINT, width=14)
        self.status_dot.pack(side="left")
        self.status_label = ctk.CTkLabel(top, text="Ready", font=self.f.body_bold, text_color=TEXT)
        self.status_label.pack(side="left", padx=(6, 0))
        self.status_detail = ctk.CTkLabel(top, text="Pick a video to get started", font=self.f.small,
                                          text_color=MUTED)
        self.status_detail.pack(side="left", padx=(12, 0))
        self.percent_label = ctk.CTkLabel(top, text="", font=self.f.body_bold, text_color=ACCENT)
        self.percent_label.pack(side="right")
        self.progress = ctk.CTkProgressBar(left, height=8, corner_radius=4, progress_color=ACCENT, fg_color=SUBTLE)
        self.progress.set(0)
        self.progress.pack(fill="x", pady=(10, 0))

        right = ctk.CTkFrame(bar, fg_color="transparent")
        right.grid(row=0, column=1, padx=(0, 18))
        self.log_button = ctk.CTkButton(right, text="", image=themed_icon("terminal", 18, MUTED), width=42, height=42,
                                        corner_radius=11, fg_color="transparent", hover_color=SUBTLE,
                                        command=self.toggle_log)
        self.log_button.pack(side="left", padx=(0, 6))
        self.analyze_button = self._button(right, "Analyze", self._analyze, "star", "subtle", width=140, height=44)
        self.render_button = self._button(right, "Render", self._render, "play", "accent", width=150, height=44)
        self.cancel_button = self._button(right, "Cancel", self._cancel, "stop", "danger", width=130, height=44)
        self.analyze_button.pack(side="left", padx=6)
        self.render_button.pack(side="left", padx=(6, 0))

    def _build_log(self, main) -> None:
        frame = ctk.CTkFrame(main, fg_color=CARD, corner_radius=18, border_width=1, border_color=BORDER)
        self.log_frame = frame
        head = ctk.CTkFrame(frame, fg_color="transparent")
        head.pack(fill="x", padx=18, pady=(12, 0))
        ctk.CTkLabel(head, text="Activity log", font=self.f.body_bold, text_color=TEXT).pack(side="left")
        self._button(head, "Copy", self._copy_log, None, "ghost", width=64, height=28).pack(side="right")
        self.log_box = ctk.CTkTextbox(frame, height=180, font=self.f.mono, fg_color="transparent", text_color=TEXT,
                                      wrap="word", border_width=0)
        self.log_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_box.configure(state="disabled")
        self.log_visible = False

    # ================================================================================================
    # Settings <-> widgets
    # ================================================================================================
    def _load_settings(self) -> None:
        s = self.settings
        for name, var in self.vars.items():
            if name == "profile":
                var.set("custom" if s.target_duration > 0 else s.profile)
            elif name == "target_duration":
                var.set(format_clock(s.target_duration) if s.target_duration > 0 else "")
            elif name == "silence_threshold":
                var.set(s.silence_threshold if s.silence_threshold < 0 else -40)
            elif isinstance(getattr(s, name, None), list):
                var.set(", ".join(getattr(s, name)))
            elif hasattr(s, name):
                var.set(getattr(s, name))
        self.silence_auto.set(s.silence_threshold == 0)
        self.keywords_box.delete("1.0", "end")
        self.keywords_box.insert("1.0", "\n".join(s.keywords))
        self._on_profile_change()
        self._refresh_layout()
        self._refresh_silence()
        self._refresh_swatches()
        self._refresh_tiles()
        self._refresh_auto_widgets()

    def _collect_settings(self, quiet: bool = False, strict: bool = True) -> Optional[Settings]:
        """Read the widgets. ``strict`` validates (and saves); otherwise returns a best-effort copy."""
        data = self.settings.to_dict()
        defaults = Settings()
        problems = []
        for name, var in self.vars.items():
            if name in ("profile", "target_duration", "silence_threshold"):
                continue
            try:
                value = var.get()
                default = getattr(defaults, name)
                if isinstance(default, bool):
                    data[name] = bool(value)
                elif isinstance(default, int):
                    data[name] = int(round(float(value)))
                elif isinstance(default, float):
                    data[name] = round(float(value), 2)
                elif isinstance(default, list):
                    data[name] = [part.strip() for part in str(value).split(",") if part.strip()]
                else:
                    data[name] = str(value).strip()
            except (ValueError, tk.TclError):
                problems.append(f"{name}: invalid value")
        profile = self.vars["profile"].get()
        if profile == "custom":
            try:
                data["target_duration"] = parse_duration(self.vars["target_duration"].get())
                data["profile"] = self.settings.profile if self.settings.profile in PROFILES else "medium"
            except ValueError:
                problems.append("Enter an exact reel length for the Custom preset (e.g. 12m, 1h30m or 8:30).")
        else:
            data["profile"], data["target_duration"] = profile, 0.0
        try:
            data["silence_threshold"] = 0.0 if self.silence_auto.get() else round(
                float(self.vars["silence_threshold"].get()), 1)
        except (ValueError, tk.TclError):
            problems.append("silence level: invalid value")
        data["keywords"] = [k.strip() for k in self.keywords_box.get("1.0", "end").splitlines() if k.strip()]
        settings = Settings.from_dict(data)
        if not strict:
            return settings
        errors = problems + settings.validate()
        if errors:
            if not quiet:
                messagebox.showerror("Please check the settings", "\n".join(f"• {e}" for e in errors))
            return None
        self.settings = settings
        try:
            settings.save(settings_path())
        except OSError:
            pass
        return settings

    def _reset_keywords(self) -> None:
        self.keywords_box.delete("1.0", "end")
        self.keywords_box.insert("1.0", "\n".join(DEFAULT_KEYWORDS))

    def _reset_all(self) -> None:
        if messagebox.askyesno("Reset all settings", "Restore every setting to its recommended value?"):
            self.settings = Settings()
            self._load_settings()
            self.toast("Settings restored", "success")

    def _on_profile_change(self) -> None:
        custom = self.vars.get("profile") is not None and self.vars["profile"].get() == "custom"
        for line in self._custom_lines:
            if custom:
                line.pack(side="top", anchor="e", pady=(8, 0))
            else:
                line.pack_forget()
        self._update_estimates()

    def _refresh_auto_widgets(self) -> None:
        auto_count = bool(self.vars["shorts_auto"].get()) if "shorts_auto" in self.vars else False
        for widget in self._shorts_count_widgets:
            if isinstance(widget, ctk.CTkSlider):
                widget.configure(state="disabled" if auto_count else "normal",
                                 button_color=FAINT if auto_count else ACCENT,
                                 progress_color=FAINT if auto_count else ACCENT)
                continue
            for child in widget.winfo_children():
                if isinstance(child, ctk.CTkButton):
                    child.configure(state="disabled" if auto_count else "normal")
                elif isinstance(child, ctk.CTkLabel):
                    child.configure(text_color=FAINT if auto_count else TEXT)
        self._update_detected()

    def _selected_pack_ids(self) -> List[str]:
        raw = self.vars["vocabulary_packs"].get() if "vocabulary_packs" in self.vars else ""
        return [key.strip() for key in str(raw).split(",") if key.strip()]

    def _refresh_vocab_hints(self) -> None:
        if not hasattr(self, "vocab_hint"):
            return
        names = [pack.name for pack in resolve_packs(self._selected_pack_ids())]
        auto = bool(self.vars["auto_vocabulary"].get()) if "auto_vocabulary" in self.vars else False
        if auto and self.media is not None:
            for pack in auto_packs([self.media.title, Path(self.media.path).stem]):
                if pack.name not in names:
                    names.append(f"{pack.name} (auto)")
        if names:
            text = "Packs: " + ", ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "")
        elif auto:
            text = "Packs: recognised automatically from the title and the speech"
        else:
            text = "No vocabulary pack"
        self.vocab_hint.configure(text=text)
        if hasattr(self, "packs_detail"):
            self.packs_detail.configure(text=f"{text}  ·  {len(load_packs())} available")

    def _open_vocab_picker(self) -> None:
        picker = getattr(self, "vocab_picker", None)
        if picker is not None and picker.win.winfo_exists():
            picker.win.lift()
            return
        self.vocab_picker = VocabularyPicker(self, self._selected_pack_ids(), self._set_packs)

    def _set_packs(self, ids: List[str]) -> None:
        self.vars["vocabulary_packs"].set(", ".join(ids))
        self.toast(f"{len(ids)} vocabulary pack{'s' if len(ids) != 1 else ''} selected", "success")

    def _open_packs_folder(self) -> None:
        folder = write_pack_template()
        open_path(folder)
        self.toast("Add your own packs as .json files, then reopen the pack list", "info")

    def _update_detected(self) -> None:
        """Refresh everything that explains the automatic choices for the current video."""
        if not hasattr(self, "found_card") or "shorts_layout" not in self.vars:
            return
        ins, media = self.insights, self.media
        layout = self.vars["shorts_layout"].get()
        manual_cam = self.vars["facecam"].get().strip() if "facecam" in self.vars else ""
        if layout == "auto" and manual_cam:
            self.layout_hint.configure(text="Auto → Facecam split (webcam area set by you)")
        elif layout == "auto":
            guess, why = auto_layout(ins or VideoInsights(), "gaming")
            self.layout_hint.configure(text=f"Auto → {LAYOUT_NAMES[guess]} ({why})" if ins else
                                       "Auto: decided from what is on screen")
        else:
            self.layout_hint.configure(text="")
        game = ins.title_game if ins else None
        self._refresh_vocab_hints()
        if media is None:
            self.found_card.pack_forget()
            return
        self.found_card.pack(fill="x", padx=12, pady=(0, 18), after=self.drop_card)
        for child in self.found_chips.winfo_children():
            child.destroy()
        chips = []
        if ins is None:
            chips.append("Scanning the picture…")
        else:
            if game:
                chips.append(f"Game: {game}")
            if manual_cam:
                chips.append("Webcam area set by you")
            elif ins.facecam:
                chips.append(f"Webcam overlay {describe_position(ins.facecam)}")
            if ins.faces_checked:
                if ins.big_face_ratio >= 0.6:
                    chips.append("Camera shot")
                elif ins.big_face_ratio >= 0.02:  # a single sampled frame is enough to mention it
                    chips.append("Full-screen camera at times")
                if ins.face_ratio >= 0.4:
                    chips.append("Face on screen")
                elif ins.face_ratio >= 0.1:
                    chips.append("Face on screen at times")
                else:
                    chips.append("No face detected")
            chips.append("Static picture" if ins.motion < 0.025 else "Lots of motion" if ins.motion > 0.06
                         else "Some motion")
            if not media.has_audio:
                chips.append("No audio track")
        for text in chips:
            self._chip(self.found_chips, text, ACCENT_SOFT, ACCENT).pack(side="left", padx=(0, 6))
        if ins is not None and not ins.facecam and not manual_cam:
            self.found_webcam_button.pack(anchor="w", padx=24, pady=(0, 18))
        else:
            self.found_webcam_button.pack_forget()

        model = self.vars["whisper_model"].get() if "whisper_model" in self.vars else "auto"
        if model == "auto":
            model, _why = auto_model(media.duration)
        speech = estimate_transcription(media.duration, model) if available_backends() and media.has_audio else 0
        reel = Settings(profile="medium").target_for(media.duration)
        shorts = 5 * 40
        hardware = sys.platform == "darwin" or has_cuda()
        total = speech + estimate_render(reel, shorts, hardware)
        text = f"  About {format_clock(total, True)} on this computer"
        if speech:
            text += f" (speech model {model}: ~{format_clock(speech, True)}, rendering: ~{format_clock(total - speech)})"
        text += ". Rough estimate; re-runs with other settings take seconds."
        self.found_eta.configure(text=text)

    def _update_estimates(self) -> None:
        profile = self.vars["profile"].get() if "profile" in self.vars else "medium"
        settings = Settings(profile=profile if profile in PROFILES else "medium")
        valid = True
        if profile == "custom":
            try:
                settings.target_duration = parse_duration(self.vars["target_duration"].get())
            except ValueError:
                valid = False
        if not valid:
            text = "Type a length such as 12m, 1h30m or 8:30"
        elif profile == "auto":
            text = ("Auto: the more great moments, the longer the reel" if self.media is None else
                    f"Auto: from 1 min to {format_clock(Settings(profile='medium').target_for(self.media.duration))}"
                    ", depending on the great moments found")
        elif self.media is None:
            text = ("Exactly " + format_clock(settings.target_duration) if profile == "custom"
                    else {"short": "About 5% of the video (1–10 min)", "medium": "About 10% of the video (2–20 min)",
                          "long": "About 20% of the video (4–40 min)"}.get(profile, ""))
        else:
            text = f"≈ {format_clock(settings.target_for(self.media.duration))} from " \
                   f"{format_clock(self.media.duration, True)} of video"
        for label in self._estimate_labels:
            label.configure(text=f"  {text}")

    def _refresh_layout(self) -> None:
        selected = self.vars["shorts_layout"].get() if "shorts_layout" in self.vars else "blur"
        for key, item in self._layout_cards.items():
            active = key == selected
            item.card.configure(border_color=ACCENT if active else SUBTLE, fg_color=ACCENT_SOFT if active else SUBTLE)
            item.name.configure(text_color=ACCENT if active else TEXT)
        for widget, pack_options in self._facecam_widgets:
            if selected == "split":
                widget.pack(**pack_options)
            else:
                widget.pack_forget()
        self._update_detected()

    def _refresh_silence(self) -> None:
        self.silence_slider.configure(state="disabled" if self.silence_auto.get() else "normal",
                                      button_color=FAINT if self.silence_auto.get() else ACCENT,
                                      progress_color=FAINT if self.silence_auto.get() else ACCENT)

    def _refresh_swatches(self) -> None:
        current = self.vars["caption_color"].get().strip().upper() if "caption_color" in self.vars else ""
        for color, swatch in self._swatches.items():
            swatch.configure(border_color=TEXT if color.upper() == current else SUBTLE)

    # ================================================================================================
    # Source video
    # ================================================================================================
    def _browse_input(self) -> None:
        if self.busy:
            return
        filename = filedialog.askopenfilename(title="Select a recording", filetypes=VIDEO_TYPES)
        if filename:
            self.set_input(filename)

    def _browse_output(self) -> None:
        folder = filedialog.askdirectory(title="Select the output folder")
        if folder:
            self.output_var.set(folder)

    def set_input(self, path: str) -> None:
        from .pipeline import default_output_dir

        path = str(Path(path).expanduser())
        self.input_var.set(path)
        self.output_var.set(str(default_output_dir(path)))
        self.media, self.media_frame, self.insights = None, None, None
        self.source_empty.pack_forget()
        self.source_view.pack(fill="x")
        self.drop_card.configure(border_color=BORDER)
        self.source_name.configure(text=Path(path).name)
        self.source_thumb.configure(image=self._source_placeholder)
        self._set_chips(["Reading video…"])
        self.show_page("home")
        self._set_status("Reading video…", "", FAINT)
        self._update_estimates()

        if self.project is not None and Path(self.project.input).resolve() != Path(path).resolve():
            self.project = None
            self._populate_review()

        def job():
            try:
                info = probe(path)
            except Exception as exc:  # noqa: BLE001 - shown to the user
                self.events.put(("probe_failed", path, str(exc)))
                return
            frame = extract_frame(path, min(max(1.0, info.duration * 0.15), max(0.0, info.duration - 0.5)), 640)
            self.events.put(("probed", path, info, frame))

        threading.Thread(target=job, daemon=True).start()

    def _on_probed(self, path: str, info: MediaInfo, frame: Optional[Image.Image]) -> None:
        if path != self.input_var.get():
            return
        self.media, self.media_frame = info, frame
        if frame is not None:
            image = icons.rounded_image(frame, (512, 288), 26)
            self._source_image = ctk.CTkImage(image, size=(256, 144))
            self.source_thumb.configure(image=self._source_image)
        chips = [format_clock(info.duration, True), f"{info.width}×{info.height}", f"{info.fps:.0f} fps"]
        size = _file_size(Path(path))
        if size:
            chips.append(size)
        chips.append("audio" if info.has_audio else "no audio")
        self._set_chips(chips)
        self._update_estimates()
        self._set_status("Ready", "Create my edit does everything, Review first lets you choose the moments.",
                         GREEN)
        self._update_detected()
        self._try_load_previous_project(path)
        use_cache = bool(self.vars["use_cache"].get()) if "use_cache" in self.vars else True

        def scan():
            try:
                insights = load_insights(path) if use_cache else None
                if insights is None:
                    insights = scan_video(info)
                    save_insights(path, insights)
            except Exception as exc:  # noqa: BLE001 - the scan only improves the automatic choices
                self.events.put(("log", f"Video scan skipped: {exc}"))
                return
            self.events.put(("scanned", path, insights))

        threading.Thread(target=scan, daemon=True).start()

    def _set_chips(self, chips: Sequence[str]) -> None:
        for child in self.source_chips.winfo_children():
            child.destroy()
        for text in chips:
            self._chip(self.source_chips, text).pack(side="left", padx=(0, 6))

    def _try_load_previous_project(self, path: str) -> None:
        from .pipeline import Project

        project_file = Path(self.output_var.get()) / "project.json"
        if self.project is not None or not project_file.exists():
            return
        try:
            project = Project.load(project_file)
        except Exception:  # noqa: BLE001 - an unreadable old project is simply ignored
            return
        if Path(project.input).resolve() == Path(path).resolve():
            self.project = project
            self._populate_review()
            self.toast("Previous analysis loaded, see Review", "success")

    def _preview_source(self) -> None:
        path = self.input_var.get()
        if path:
            self._play(path, 0.0, None, Path(path).name)

    # ================================================================================================
    # Facecam picker
    # ================================================================================================
    def _open_facecam_picker(self) -> None:
        if not self.media:
            self.show_page("home")
            self.toast("Pick a video first", "error")
            return
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        frame = extract_frame(self.media.path, min(max(1.0, self.media.duration * 0.15), self.media.duration - 0.5),
                              1280) or self.media_frame
        self.root.configure(cursor="")
        if frame is None:
            messagebox.showerror("Facecam area", "Could not read a frame from the video.")
            return
        try:
            initial = Settings(facecam=self.vars["facecam"].get()).facecam_rect()
        except ValueError:
            initial = None
        if initial is None and self.insights is not None and self.insights.facecam:
            initial = tuple(self.insights.facecam)  # pre-filled with the detected webcam overlay
        self.facecam_picker = FacecamPicker(self, frame, initial, lambda rect: self.vars["facecam"].set(
            ",".join(f"{v:.3f}" for v in rect)))

    # ================================================================================================
    # Jobs
    # ================================================================================================
    def _validate_input(self) -> Optional[str]:
        path = self.input_var.get().strip()
        if not path:
            self.show_page("home")
            self.toast("Pick a video first", "error")
            return None
        if not os.path.isfile(path):
            messagebox.showerror("File not found", f"The file does not exist:\n{path}")
            return None
        return path

    def _analyze(self, then_render: bool = False) -> None:
        if self.busy:
            return
        path = self._validate_input()
        settings = self._collect_settings() if path else None
        if not path or not settings:
            return
        from .pipeline import Pipeline

        pipeline = Pipeline(path, settings, self.output_var.get().strip() or None, self._new_reporter())

        def job():
            project = pipeline.analyze()
            self.events.put(("analyzed", project, then_render))
            if then_render:
                self.events.put(("render_started",))
                self.events.put(("rendered", pipeline.render(project)))

        self._start(job, "Finding the best moments")

    def _render(self) -> None:
        if self.busy:
            return
        path = self._validate_input()
        settings = self._collect_settings() if path else None
        if not path or not settings:
            return
        project = self.project
        if project is None or Path(project.input).resolve() != Path(path).resolve():
            self._analyze(then_render=True)
            return
        if project.needs_reanalysis(settings):
            if messagebox.askyesno(
                    "Settings changed",
                    "Some settings that decide which moments are picked changed since the last analysis.\n\n"
                    "Analyze again before rendering? Your keep/drop choices will be reset."):
                self._analyze(then_render=True)
                return
        output = self.output_var.get().strip()
        if output:
            project.output_dir = str(Path(output).expanduser().resolve())
        from .pipeline import Pipeline

        pipeline = Pipeline(path, settings, project.output_dir, self._new_reporter())
        self._start(lambda: self.events.put(("rendered", pipeline.render(project))), "Rendering")

    def _new_reporter(self) -> Reporter:
        self.reporter = Reporter(
            on_log=lambda message: self.events.put(("log", message)),
            on_progress=lambda value, label: self.events.put(("progress", value, label)),
        )
        return self.reporter

    def _start(self, job: Callable[[], None], label: str) -> None:
        if self.worker and self.worker.is_alive():
            return
        self._clear_log()
        self._set_busy(True)
        self._phase_started = time.monotonic()
        self.progress.set(0)
        self._set_status(label, "Starting…", ACCENT)

        def run():
            try:
                job()
            except Cancelled:
                self.events.put(("cancelled",))
            except Exception as exc:  # noqa: BLE001 - reported in the UI
                self.events.put(("log", traceback.format_exc()))
                self.events.put(("error", str(exc)))
            finally:
                self.events.put(("finished",))

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _cancel(self) -> None:
        if self.reporter:
            self.reporter.cancel()
            self._set_status("Cancelling…", "Stopping ffmpeg and cleaning up", AMBER)

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.analyze_button.pack_forget()
        self.render_button.pack_forget()
        self.cancel_button.pack_forget()
        if busy:
            self.cancel_button.pack(side="left", padx=(6, 0))
            self.percent_label.configure(text="0%")
        else:
            self.analyze_button.pack(side="left", padx=6)
            self.render_button.pack(side="left", padx=(6, 0))
            self.percent_label.configure(text="")

    def _set_status(self, title: str, detail: str, color=FAINT) -> None:
        self.status_label.configure(text=title)
        self.status_detail.configure(text=detail)
        self.status_dot.configure(text_color=color)

    # ================================================================================================
    # Event loop
    # ================================================================================================
    def _poll_events(self) -> None:
        handled = 0
        try:
            while handled < 200:
                self._handle(self.events.get_nowait())
                handled += 1
        except queue.Empty:
            pass
        self.root.after(60, self._poll_events)

    def _handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "log":
            self._append_log(event[1])
        elif kind == "progress":
            self._on_progress(event[1], event[2])
        elif kind == "probed":
            self._on_probed(*event[1:])
        elif kind == "scanned":
            if event[1] == self.input_var.get():
                self.insights = event[2]
                self._update_detected()
        elif kind == "probe_failed":
            if event[1] == self.input_var.get():
                self._set_chips(["Unreadable file"])
                self._set_status("Cannot read this file", event[2][:120], RED)
        elif kind == "thumb":
            _, generation, key, image = event
            if generation == self._thumb_generation:
                self._thumbs[key] = image
                self._apply_thumb(key)
        elif kind == "analyzed":
            project, then_render = event[1], event[2]
            self.project = project
            self._populate_review()
            count = len(project.highlight) + len(project.shorts)
            if not then_render:
                self.show_page("review")
                self.toast(f"{count} moments found", "success")
        elif kind == "render_started":
            self._phase_started = time.monotonic()
            self.progress.set(0)
        elif kind == "rendered":
            self._set_status("Render complete", str(event[1].output_dir), GREEN)
            self.progress.set(1)
            self._show_result(event[1])
        elif kind == "cancelled":
            self._append_log("Cancelled.")
            self._set_status("Cancelled", "Nothing was left half-written.", AMBER)
            self.progress.set(0)
        elif kind == "error":
            self._set_status("Something went wrong", event[1].splitlines()[0][:140] if event[1] else "", RED)
            if not self.log_visible:
                self.toggle_log()
            messagebox.showerror("Error", event[1])
        elif kind == "finished":
            self._set_busy(False)

    def _on_progress(self, value: float, label: str) -> None:
        self.progress.set(value)
        self.percent_label.configure(text=f"{value * 100:.0f}%")
        elapsed = time.monotonic() - self._phase_started
        detail = f"{format_clock(elapsed)} elapsed"
        if 0.03 < value < 1 and elapsed > 4:
            detail += f"  ·  about {format_clock(elapsed * (1 - value) / value)} left"
        self._set_status(label, detail, ACCENT)

    # ================================================================================================
    # Review
    # ================================================================================================
    def _populate_review(self) -> None:
        for frame in self.review_lists.values():
            for child in frame.winfo_children():
                child.destroy()
            frame.pack_forget()
        self.cards.clear()
        self._thumbs.clear()
        self._thumb_generation += 1
        if not self.project:
            self.review_head.pack_forget()
            self.review_empty.pack(fill="x", padx=12)
            self._update_review_nav()
            return
        self.review_empty.pack_forget()
        self.review_head.pack(fill="x", padx=12, pady=(0, 16))
        for child in self.review_auto.winfo_children():
            child.destroy()
        choices = list(getattr(self.project, "auto", []))
        if choices:
            ctk.CTkLabel(self.review_auto, text="AUTO CHOICES", font=self.f.tiny_bold, text_color=FAINT,
                         anchor="w", height=16).pack(anchor="w", pady=(0, 6))
            for start in range(0, len(choices), 5):
                line = ctk.CTkFrame(self.review_auto, fg_color="transparent")
                line.pack(anchor="w", pady=(0, 6))
                for label, value in choices[start:start + 5]:
                    self._chip(line, f"{label}: {value}", ACCENT_SOFT, ACCENT).pack(side="left", padx=(0, 6))
        self._build_queue = [("h", i, m) for i, m in enumerate(self.project.highlight)] + \
                            [("s", i, m) for i, m in enumerate(self.project.shorts)]
        label = "Highlight reel" if self.project.highlight else "Shorts"
        self.review_switch.set(label)
        self._on_review_kind(label)
        self._build_cards_batch()
        self._load_thumbnails()

    def _build_cards_batch(self) -> None:
        for _ in range(8):
            if not self._build_queue:
                self._update_review_summary()
                return
            kind, index, moment = self._build_queue.pop(0)
            self._build_moment_card(kind, index, moment)
        self.root.after(1, self._build_cards_batch)

    def _build_moment_card(self, kind: str, index: int, m) -> None:
        card = ctk.CTkFrame(self.review_lists[kind], fg_color=CARD, corner_radius=18, border_width=1,
                            border_color=BORDER)
        card.pack(fill="x", pady=(0, 12))
        card.grid_columnconfigure(1, weight=1)

        thumb = ctk.CTkLabel(card, text="", image=self._placeholder)
        thumb.grid(row=0, column=0, rowspan=2, padx=(16, 18), pady=16, sticky="w")

        info = ctk.CTkFrame(card, fg_color="transparent")
        info.grid(row=0, column=1, sticky="nsew", pady=(18, 0))
        top = ctk.CTkFrame(info, fg_color="transparent")
        top.pack(fill="x")
        self._chip(top, f"#{index + 1}", ACCENT_SOFT, ACCENT).pack(side="left")
        time_label = ctk.CTkLabel(top, text=f"{format_clock(m.start, True)}  →  {format_clock(m.end, True)}",
                                  font=self.f.body_bold, text_color=TEXT)
        time_label.pack(side="left", padx=(10, 8))
        ctk.CTkLabel(top, text=f"{m.duration:.1f} s", font=self.f.small, text_color=MUTED).pack(side="left")
        text = " ".join(m.text.split()) or "No speech in this moment."
        if len(text) > 210:
            text = text[:207].rsplit(" ", 1)[0] + "…"
        text_label = ctk.CTkLabel(info, text=f"“{text}”" if m.text.strip() else text, font=self.f.small,
                                  text_color=MUTED, anchor="w", justify="left", wraplength=520)
        text_label.pack(anchor="w", pady=(8, 0))

        chips = ctk.CTkFrame(card, fg_color="transparent")
        chips.grid(row=1, column=1, sticky="sw", pady=(8, 18))
        for reason in m.reasons[:5]:
            self._chip(chips, reason.strip('"')).pack(side="left", padx=(0, 6))

        score = ctk.CTkFrame(card, fg_color="transparent")
        score.grid(row=0, column=2, rowspan=2, padx=18)
        color = score_color(m.score)
        ctk.CTkLabel(score, text=f"{m.score:.0f}", font=self.f.score, text_color=color, height=30).pack()
        ctk.CTkLabel(score, text="SCORE", font=self.f.tiny_bold, text_color=FAINT, height=14).pack()
        meter = ctk.CTkProgressBar(score, width=66, height=5, corner_radius=3, progress_color=color, fg_color=SUBTLE)
        meter.set(max(0.0, min(1.0, m.score / 100)))
        meter.pack(pady=(8, 0))

        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.grid(row=0, column=3, rowspan=2, padx=(4, 20))
        var = tk.BooleanVar(value=m.selected)
        ctk.CTkSwitch(actions, text="Keep", variable=var, onvalue=True, offvalue=False, switch_width=44,
                      switch_height=22, progress_color=ACCENT, fg_color=BORDER, button_color="#ffffff",
                      button_hover_color="#f4f4f8", font=self.f.small_bold, text_color=TEXT,
                      command=lambda: self._on_keep(kind, index, var)).pack(anchor="e", pady=(0, 12))
        self._button(actions, "Preview", lambda: self._play(self.project.media.path, m.start, m.duration,
                                                            f"Moment #{index + 1}"),
                     "play", "subtle", width=110, height=34).pack(anchor="e")

        self.cards[(kind, index)] = SimpleNamespace(card=card, thumb=thumb, var=var, text=text_label,
                                                    time=time_label)
        self._style_card(kind, index)
        self._apply_thumb((kind, index))

    def _moments(self, kind: str):
        return self.project.highlight if kind == "h" else self.project.shorts

    def _style_card(self, kind: str, index: int) -> None:
        item = self.cards.get((kind, index))
        if not item:
            return
        keep = self._moments(kind)[index].selected
        item.card.configure(fg_color=CARD if keep else CARD_DIM)
        item.time.configure(text_color=TEXT if keep else FAINT)
        item.text.configure(text_color=MUTED if keep else FAINT)

    def _on_keep(self, kind: str, index: int, var: tk.BooleanVar) -> None:
        self._moments(kind)[index].selected = bool(var.get())
        self._style_card(kind, index)
        self._update_review_summary()
        self._save_project_quietly()

    def _set_all(self, keep: bool) -> None:
        if not self.project:
            return
        for index, moment in enumerate(self._moments(self.review_kind)):
            moment.selected = keep
            item = self.cards.get((self.review_kind, index))
            if item:
                item.var.set(keep)
                self._style_card(self.review_kind, index)
        self._update_review_summary()
        self._save_project_quietly()

    def _on_review_kind(self, label: str) -> None:
        self.review_kind = "h" if label.startswith("Highlight") else "s"
        for key, frame in self.review_lists.items():
            if key == self.review_kind:
                frame.pack(fill="x", padx=12)
            else:
                frame.pack_forget()
        self._update_review_summary()

    def _update_review_summary(self) -> None:
        self._update_review_nav()
        if not self.project:
            return
        p = self.project
        if self.review_kind == "h":
            kept = [m for m in p.highlight if m.selected]
            jump_cuts = bool(p.resolved.get("jump_cuts", p.settings.jump_cuts))
            total = sum(m.edited_duration(jump_cuts) for m in kept)
            self.review_stats.configure(
                text=f"{len(kept)} of {len(p.highlight)} moments kept  ·  {format_clock(total)} of "
                     f"{format_clock(p.target_duration)} target")
            self.review_meter.set(min(1.0, total / p.target_duration) if p.target_duration else 0)
        else:
            kept = [m for m in p.shorts if m.selected]
            total = sum(m.duration for m in kept)
            self.review_stats.configure(
                text=f"{len(kept)} of {len(p.shorts)} shorts kept  ·  {format_clock(total)} of vertical video")
            self.review_meter.set(len(kept) / len(p.shorts) if p.shorts else 0)

    def _update_review_nav(self) -> None:
        count = 0
        if self.project:
            count = sum(m.selected for m in self.project.highlight) + sum(m.selected for m in self.project.shorts)
        self.nav["review"].button.configure(text=f"  Review   ·  {count}" if count else "  Review")

    def _save_project_quietly(self) -> None:
        try:
            self.project.save()
        except OSError:
            pass

    def _load_thumbnails(self) -> None:
        generation = self._thumb_generation
        project = self.project
        items = [("h", i, m) for i, m in enumerate(project.highlight)] + \
                [("s", i, m) for i, m in enumerate(project.shorts)]

        def run():
            for kind, index, m in items:
                if generation != self._thumb_generation:
                    return
                t = m.peak if m.start <= m.peak <= m.end else (m.start + m.end) / 2
                frame = extract_frame(project.media.path, t, 352)
                if frame is not None:
                    self.events.put(("thumb", generation, (kind, index), icons.rounded_image(frame, (352, 198), 20)))

        threading.Thread(target=run, daemon=True).start()

    def _apply_thumb(self, key: tuple) -> None:
        item, image = self.cards.get(key), self._thumbs.get(key)
        if item is not None and image is not None:
            item.thumb_image = ctk.CTkImage(image, size=(176, 99))
            item.thumb.configure(image=item.thumb_image)

    # ================================================================================================
    # Results, toasts, log
    # ================================================================================================
    def _show_result(self, result) -> None:
        win = ctk.CTkToplevel(self.root)
        win.title("Render complete")
        win.configure(fg_color=CARD)
        win.resizable(False, False)
        win.transient(self.root)
        ctk.CTkLabel(win, text="", image=themed_icon("check", 30, WHITE), width=78, height=78, fg_color=GREEN,
                     corner_radius=39).pack(pady=(32, 14))
        ctk.CTkLabel(win, text="Your edit is ready", font=self.f.title, text_color=TEXT).pack()
        ctk.CTkLabel(win, text=str(result.output_dir), font=self.f.small, text_color=MUTED, wraplength=470).pack(
            padx=30, pady=(4, 20))

        rows = ctk.CTkFrame(win, fg_color=SUBTLE, corner_radius=14)
        rows.pack(fill="x", padx=30)
        entries = []
        if result.highlight:
            try:
                duration = format_clock(probe(str(result.highlight)).duration)
            except Exception:  # noqa: BLE001
                duration = ""
            entries.append(("reel", "Highlight reel", " · ".join(filter(None, [duration, _file_size(result.highlight)]))))
        if result.clips:
            entries.append(("film", "Individual clips", f"{len(result.clips)} files"))
        if result.shorts:
            entries.append(("phone", "Vertical shorts", f"{len(result.shorts)} files"))
        if result.extras:
            entries.append(("export", "Editor, subtitle & report files", f"{len(result.extras)} files"))
        for i, (icon_name, title, value) in enumerate(entries):
            row = ctk.CTkFrame(rows, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=(12 if i == 0 else 4, 12 if i == len(entries) - 1 else 4))
            ctk.CTkLabel(row, text="", image=themed_icon(icon_name, 16, ACCENT)).pack(side="left")
            ctk.CTkLabel(row, text=title, font=self.f.body_bold, text_color=TEXT).pack(side="left", padx=10)
            ctk.CTkLabel(row, text=value, font=self.f.small, text_color=MUTED).pack(side="right")

        buttons = ctk.CTkFrame(win, fg_color="transparent")
        buttons.pack(fill="x", padx=30, pady=24)
        self._button(buttons, "Close", win.destroy, None, "ghost", width=90).pack(side="left")
        if result.highlight:
            self._button(buttons, "Play reel", lambda: open_path(result.highlight), "play", "accent",
                         width=130).pack(side="right")
        target = result.shorts[0].parent if result.shorts and not result.highlight else result.output_dir
        self._button(buttons, "Open folder", lambda: open_path(target), "folder", "subtle",
                     width=140).pack(side="right", padx=8)
        win.update_idletasks()
        w, h = 540, win.winfo_reqheight()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - w) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - h) // 3
        win.geometry(f"{w}x{h}+{x}+{y}")
        win.after(100, win.lift)
        self.result_window = win

    def toast(self, text: str, kind: str = "info") -> None:
        color = {"info": ACCENT, "success": GREEN, "error": RED}.get(kind, ACCENT)
        toast = ctk.CTkFrame(self.main, fg_color=CARD, corner_radius=14, border_width=1, border_color=BORDER)
        ctk.CTkLabel(toast, text="●", text_color=color, font=self.f.small).pack(side="left", padx=(16, 8), pady=12)
        ctk.CTkLabel(toast, text=text, font=self.f.body_bold, text_color=TEXT).pack(side="left", padx=(0, 18))
        toast.place(relx=1.0, x=-40, y=30, anchor="ne")
        toast.lift()
        self.root.after(3200, toast.destroy)

    def toggle_log(self) -> None:
        self.log_visible = not self.log_visible
        if self.log_visible:
            self.log_frame.grid(row=3, column=0, sticky="nsew", padx=40, pady=(0, 22))
            self.log_button.configure(fg_color=ACCENT_SOFT)
        else:
            self.log_frame.grid_forget()
            self.log_button.configure(fg_color="transparent")

    def _append_log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _copy_log(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log_box.get("1.0", "end"))
        self.toast("Log copied", "success")

    # ================================================================================================
    # Misc
    # ================================================================================================
    def _play(self, path: str, start: float, duration: Optional[float], title: str) -> None:
        player = shutil.which("ffplay")
        if not player:
            open_path(path)
            return
        cmd = [player, "-hide_banner", "-loglevel", "error", "-autoexit", "-x", "960", "-y", "540",
               "-window_title", title, "-ss", f"{start:.2f}"]
        if duration:
            cmd += ["-t", f"{duration:.2f}"]
        subprocess.Popen(cmd + [path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=_CREATION_FLAGS)

    def _open_output(self) -> None:
        folder = Path(self.output_var.get()) if self.output_var.get().strip() else None
        if not folder or not folder.exists():
            self.toast("Nothing rendered yet: the folder will be created on render", "info")
            return
        open_path(folder)

    def _setup_drag_and_drop(self) -> None:
        if not getattr(self.root, "dnd_ok", False):
            return
        try:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<DropEnter>>", self._on_drag_enter)
            self.root.dnd_bind("<<DropLeave>>", self._on_drag_leave)
            self.root.dnd_bind("<<Drop>>", self._on_drop)
        except tk.TclError:
            pass

    def _on_drag_enter(self, event):
        self.show_page("home")
        self.drop_card.configure(border_color=ACCENT)
        return event.action

    def _on_drag_leave(self, event):
        self.drop_card.configure(border_color=BORDER)
        return event.action

    def _on_drop(self, event):
        self.drop_card.configure(border_color=BORDER)
        paths = [p for p in self.root.tk.splitlist(event.data) if os.path.isfile(p)]
        videos = [p for p in paths if Path(p).suffix.lower() in VIDEO_EXTENSIONS] or paths
        if videos and not self.busy:
            self.set_input(videos[0])
        return event.action

    def _bind_shortcuts(self) -> None:
        mod = "Command" if sys.platform == "darwin" else "Control"
        self.root.bind(f"<{mod}-o>", lambda _e: self._browse_input())
        self.root.bind(f"<{mod}-Return>", lambda _e: self._render())
        self.root.bind(f"<{mod}-l>", lambda _e: self.toggle_log())
        for number, (key, *_rest) in enumerate(PAGES, 1):
            self.root.bind(f"<{mod}-Key-{number}>", lambda _e, k=key: self.show_page(k))

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Quit", "A job is running. Cancel it and quit?"):
                return
            if self.reporter:
                self.reporter.cancel()
        self._collect_settings(quiet=True)
        self.root.destroy()


# ====================================================================================================
# Facecam picker window
# ====================================================================================================

class VocabularyPicker:
    """Choose vocabulary packs: search, filter by category and preview the terms of each pack."""

    def __init__(self, app: App, selected: Sequence[str], on_done: Callable[[List[str]], None]) -> None:
        self.app, self.on_done = app, on_done
        order = {name: index for index, name in enumerate(all_categories())}
        self.packs: List[Pack] = sorted(load_packs(refresh=True).values(),
                                        key=lambda p: (order.get(p.category, 99), p.name.lower()))
        known = {pack.id for pack in self.packs}
        self.selected: List[str] = []
        for key in selected:
            pack = get_pack(key)
            if pack is not None and pack.id in known and pack.id not in self.selected:
                self.selected.append(pack.id)
        self.rows: List[SimpleNamespace] = []
        self.current: Optional[Pack] = None

        win = ctk.CTkToplevel(app.root)
        self.win = win
        win.title("Vocabulary packs")
        win.configure(fg_color=CARD)
        win.geometry("1060x720")
        win.minsize(920, 600)
        win.transient(app.root)
        win.grid_columnconfigure(0, weight=3)
        win.grid_columnconfigure(1, weight=2)
        win.grid_rowconfigure(2, weight=1)

        head = ctk.CTkFrame(win, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(20, 4))
        ctk.CTkLabel(head, text="Vocabulary packs", font=app.f.title, text_color=TEXT, anchor="w").pack(anchor="w")
        ctk.CTkLabel(head, text=f"{len(self.packs)} games and topics. Their names and jargon help the transcriber "
                                "write them correctly, and become hashtags.",
                     font=app.f.small, text_color=MUTED, anchor="w").pack(anchor="w")

        filters = ctk.CTkFrame(win, fg_color="transparent")
        filters.grid(row=1, column=0, columnspan=2, sticky="ew", padx=24, pady=(8, 10))
        self.query = tk.StringVar()
        self.search = app._entry(filters, self.query, width=320)
        self.search.pack(side="left")
        ctk.CTkLabel(filters, text="", image=themed_icon("sliders", 16, MUTED)).pack(side="left", padx=(14, 6))
        self.category = tk.StringVar(value="All categories")
        ctk.CTkOptionMenu(filters, values=["All categories", *all_categories(), "Selected"], variable=self.category,
                          command=lambda _value: self._filter(), width=220, height=36, corner_radius=10,
                          fg_color=SUBTLE, button_color=SUBTLE, button_hover_color=BORDER, text_color=TEXT,
                          font=app.f.small_bold, dropdown_fg_color=CARD, dropdown_hover_color=SUBTLE,
                          dropdown_text_color=TEXT, dynamic_resizing=False).pack(side="left")
        self.count = ctk.CTkLabel(filters, text="", font=app.f.small_bold, text_color=ACCENT)
        self.count.pack(side="right")
        self.query.trace_add("write", lambda *_: self._filter())

        self.list = ctk.CTkScrollableFrame(win, fg_color=SUBTLE, corner_radius=14, scrollbar_button_color=BORDER,
                                           scrollbar_button_hover_color=FAINT)
        self.list.grid(row=2, column=0, sticky="nsew", padx=(24, 8))
        app.scroll.rebind()
        app.scroll.register(self.list)

        preview = ctk.CTkFrame(win, fg_color=SUBTLE, corner_radius=14)
        preview.grid(row=2, column=1, sticky="nsew", padx=(8, 24))
        self.preview_name = ctk.CTkLabel(preview, text="", font=app.f.h2, text_color=TEXT, anchor="w", justify="left",
                                         wraplength=340)
        self.preview_name.pack(anchor="w", padx=18, pady=(16, 0))
        self.preview_meta = ctk.CTkLabel(preview, text="", font=app.f.small, text_color=MUTED, anchor="w")
        self.preview_meta.pack(anchor="w", padx=18)
        self.preview_aliases = ctk.CTkLabel(preview, text="", font=app.f.small, text_color=TEXT, anchor="w",
                                            justify="left", wraplength=340)
        self.preview_aliases.pack(anchor="w", padx=18, pady=(8, 6))
        self.preview_toggle = app._button(preview, "Add to selection", self._toggle_current, "check", "accent",
                                          width=190, height=34)
        self.preview_toggle.pack(anchor="w", padx=18, pady=(2, 8))
        self.preview_terms = ctk.CTkTextbox(preview, font=app.f.small, fg_color=CARD, text_color=TEXT, wrap="word",
                                            border_width=0, corner_radius=10)
        self.preview_terms.pack(fill="both", expand=True, padx=18, pady=(4, 18))

        footer = ctk.CTkFrame(win, fg_color="transparent")
        footer.grid(row=3, column=0, columnspan=2, sticky="ew", padx=24, pady=16)
        app._button(footer, "My packs folder", app._open_packs_folder, "folder", "ghost", width=160).pack(side="left")
        app._button(footer, "Clear selection", self._clear, None, "ghost", width=140).pack(side="left", padx=6)
        errors = pack_errors()
        if errors:
            ctk.CTkLabel(footer, text=f"{len(errors)} pack file(s) skipped: {errors[0][:70]}", font=app.f.tiny,
                         text_color=AMBER).pack(side="left", padx=8)
        app._button(footer, "Apply", self._apply, "check", "accent", width=130).pack(side="right")
        app._button(footer, "Cancel", win.destroy, None, "ghost", width=100).pack(side="right", padx=8)

        win.bind("<Destroy>", self._on_destroy, add="+")
        first = get_pack(self.selected[0]) if self.selected else (self.packs[0] if self.packs else None)
        if first is not None:
            self._show(first)
        self._build_rows(0)
        win.after(150, self._focus)

    def _focus(self) -> None:
        try:
            self.win.lift()
            self.search.focus_set()
        except tk.TclError:
            pass

    def _build_rows(self, start: int) -> None:
        if not self.win.winfo_exists():
            return
        for pack in self.packs[start:start + 25]:
            row = ctk.CTkFrame(self.list, fg_color=CARD, corner_radius=10)
            var = tk.BooleanVar(value=pack.id in self.selected)
            ctk.CTkCheckBox(row, text=pack.name, variable=var, onvalue=True, offvalue=False,
                            command=lambda p=pack, v=var: self._set(p, bool(v.get())), font=self.app.f.body_bold,
                            text_color=TEXT, fg_color=ACCENT, hover_color=ACCENT_HOVER, border_color=FAINT,
                            checkbox_width=20, checkbox_height=20, corner_radius=6).pack(side="left", padx=12, pady=9)
            detail = f"{pack.genre or pack.category}  ·  {len(pack.terms)} terms" + ("  ·  yours" if pack.custom else "")
            ctk.CTkLabel(row, text=detail, font=self.app.f.tiny, text_color=MUTED).pack(side="right", padx=12)
            App._bind_click(row, lambda p=pack: self._show(p))
            self.rows.append(SimpleNamespace(pack=pack, row=row, var=var))
        self._filter()
        if start + 25 < len(self.packs):
            self.win.after(1, lambda: self._build_rows(start + 25))

    def _filter(self) -> None:
        query, category = self.query.get(), self.category.get()
        shown = 0
        for item in self.rows:
            item.row.pack_forget()
        for item in self.rows:
            pack = item.pack
            if category == "Selected" and pack.id not in self.selected:
                continue
            if category not in ("All categories", "Selected") and pack.category != category:
                continue
            if not pack.matches(query):
                continue
            item.row.pack(fill="x", padx=8, pady=3)
            shown += 1
        self.count.configure(text=f"{shown} shown  ·  {len(self.selected)} selected")

    def _set(self, pack: Pack, on: bool) -> None:
        if on and pack.id not in self.selected:
            self.selected.append(pack.id)
        elif not on and pack.id in self.selected:
            self.selected.remove(pack.id)
        for item in self.rows:
            if item.pack.id == pack.id:
                item.var.set(on)
        self._show(pack)
        if self.category.get() == "Selected":
            self._filter()
        else:
            shown = sum(1 for item in self.rows if item.row.winfo_manager())
            self.count.configure(text=f"{shown} shown  ·  {len(self.selected)} selected")

    def _show(self, pack: Pack) -> None:
        self.current = pack
        self.preview_name.configure(text=pack.name)
        self.preview_meta.configure(text="  ·  ".join(filter(None, [pack.category, pack.genre,
                                                                    f"{len(pack.terms)} terms"])))
        self.preview_aliases.configure(text="Recognised in titles: " + ", ".join(pack.aliases[:8])
                                       if pack.aliases else "Only used when you select it")
        self.preview_toggle.configure(text="Remove from selection" if pack.id in self.selected else "Add to selection")
        self.preview_terms.configure(state="normal")
        self.preview_terms.delete("1.0", "end")
        self.preview_terms.insert("1.0", "  ·  ".join(pack.terms) + "\n\nHashtags: " + " ".join(f"#{t}" for t in pack.tags))
        self.preview_terms.configure(state="disabled")

    def _toggle_current(self) -> None:
        if self.current is not None:
            self._set(self.current, self.current.id not in self.selected)

    def _clear(self) -> None:
        self.selected.clear()
        for item in self.rows:
            item.var.set(False)
        if self.current is not None:
            self._show(self.current)
        self._filter()

    def _apply(self) -> None:
        self.on_done(list(self.selected))
        self.win.destroy()

    def _on_destroy(self, event) -> None:
        if event.widget is self.win:
            self.app.scroll.unregister(self.list)


class FacecamPicker:
    def __init__(self, app: App, frame: Image.Image, initial, on_done: Callable) -> None:
        self.app, self.on_done = app, on_done
        win = ctk.CTkToplevel(app.root)
        self.win = win
        win.title("Facecam area")
        win.configure(fg_color=CARD)
        win.resizable(False, False)
        win.transient(app.root)

        scale = min(1.0, 880 / frame.width, 520 / frame.height)
        self.w, self.h = int(frame.width * scale), int(frame.height * scale)
        self.photo = ImageTk.PhotoImage(frame.resize((self.w, self.h), Image.LANCZOS))

        ctk.CTkLabel(win, text="Select the facecam", font=app.f.title, text_color=TEXT, anchor="w").pack(
            anchor="w", padx=24, pady=(22, 0))
        ctk.CTkLabel(win, text="Drag a rectangle around your camera. It will sit on top of the short, "
                               "with the gameplay below.", font=app.f.small, text_color=MUTED, anchor="w").pack(
            anchor="w", padx=24, pady=(2, 14))
        self.canvas = tk.Canvas(win, width=self.w, height=self.h, highlightthickness=0, bd=0, bg="#000000",
                                cursor="crosshair")
        self.canvas.pack(padx=24)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.rect_id = None
        self.handles: List[int] = []
        self.start: Optional[Tuple[int, int]] = None
        self.rect: Optional[Tuple[float, float, float, float]] = None
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)

        bottom = ctk.CTkFrame(win, fg_color="transparent")
        bottom.pack(fill="x", padx=24, pady=18)
        self.coords = ctk.CTkLabel(bottom, text="No area selected yet", font=app.f.small, text_color=MUTED)
        self.coords.pack(side="left")
        self.use_button = app._button(bottom, "Use this area", self._done, "check", "accent", width=150)
        self.use_button.pack(side="right")
        app._button(bottom, "Cancel", win.destroy, None, "ghost", width=90).pack(side="right", padx=8)
        self.use_button.configure(state="disabled")

        if initial:
            x, y, w, h = initial
            self._draw(x * self.w, y * self.h, (x + w) * self.w, (y + h) * self.h)
        win.after(150, self._grab)

    def _grab(self) -> None:
        try:
            self.win.lift()
            self.win.grab_set()
        except tk.TclError:
            pass

    def _press(self, event) -> None:
        self.start = (event.x, event.y)

    def _drag(self, event) -> None:
        if self.start:
            self._draw(self.start[0], self.start[1], event.x, event.y)

    def _release(self, event) -> None:
        self._drag(event)
        self.start = None

    def _draw(self, x0, y0, x1, y1) -> None:
        x0, x1 = sorted((max(0, min(self.w, x0)), max(0, min(self.w, x1))))
        y0, y1 = sorted((max(0, min(self.h, y0)), max(0, min(self.h, y1))))
        for item in [self.rect_id, *self.handles]:
            if item:
                self.canvas.delete(item)
        self.handles = []
        accent = ACCENT[1]
        self.rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline=accent, width=3)
        for hx, hy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            self.handles.append(self.canvas.create_rectangle(hx - 5, hy - 5, hx + 5, hy + 5, fill="#ffffff",
                                                             outline=accent, width=2))
        if x1 - x0 < 8 or y1 - y0 < 8:
            self.rect = None
            self.use_button.configure(state="disabled")
            return
        self.rect = (x0 / self.w, y0 / self.h, (x1 - x0) / self.w, (y1 - y0) / self.h)
        self.coords.configure(text="x {:.2f}  ·  y {:.2f}  ·  w {:.2f}  ·  h {:.2f}".format(*self.rect))
        self.use_button.configure(state="normal")

    def _done(self) -> None:
        if self.rect:
            self.on_done(self.rect)
            self.app.toast("Facecam area saved", "success")
        self.win.destroy()


# ====================================================================================================
# Entry point
# ====================================================================================================

def _make_root() -> ctk.CTk:
    if TkinterDnD is not None:
        class DnDRoot(ctk.CTk, TkinterDnD.DnDWrapper):
            def __init__(self):
                super().__init__()
                self.dnd_ok = False
                try:
                    require = getattr(TkinterDnD, "_require", None) or getattr(TkinterDnD, "require")
                    self.TkdndVersion = require(self)
                    self.dnd_ok = True
                except Exception:  # noqa: BLE001 - tkdnd binary missing for this platform
                    pass

        return DnDRoot()
    root = ctk.CTk()
    root.dnd_ok = False
    return root


def main(initial_input: Optional[str] = None) -> int:
    prefs = load_ui_prefs()
    ctk.set_appearance_mode(prefs.get("appearance", "Dark").lower())
    root = _make_root()
    if missing_tools():
        root.withdraw()
        messagebox.showerror(
            "FFmpeg not found",
            "FFmpeg and FFprobe are required but were not found on your PATH.\n\n"
            "Install FFmpeg (https://ffmpeg.org/download.html), then restart the application.")
        root.destroy()
        return 1
    App(root, initial_input)
    root.mainloop()
    return 0
