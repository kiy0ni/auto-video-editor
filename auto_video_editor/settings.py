"""User settings, presets and application directories."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Highlight length presets: (fraction of the source, minimum seconds, maximum seconds).
PROFILES = {
    "short": (0.05, 60.0, 10 * 60.0),
    "medium": (0.10, 2 * 60.0, 20 * 60.0),
    "long": (0.20, 4 * 60.0, 40 * 60.0),
}

PROFILE_CHOICES = ("auto",) + tuple(PROFILES)
SHORTS_LAYOUTS = ("auto", "blur", "crop", "smart", "split")
TRANSCRIBERS = ("auto", "faster-whisper", "openai-whisper", "none")
WHISPER_MODELS = ("auto", "tiny", "base", "small", "medium", "large-v3", "turbo")
TRANSITIONS = ("cut", "fade")
QUALITIES = ("draft", "standard", "high")
CAPTION_STYLES = ("karaoke", "simple")
CAPTION_POSITIONS = ("auto", "top", "middle", "bottom")
CONTENT_TYPES = ("auto", "gaming", "talk", "vlog")
CONTENT_LABELS = {"auto": "Auto-detect", "gaming": "Gaming stream", "talk": "Podcast & talk", "vlog": "Vlog & IRL"}
# Settings whose "auto" value is decided per video during the analysis.
AUTO_CHOICE_FIELDS = ("content_type", "profile", "shorts_layout", "whisper_model")
ENCODERS = (
    "auto",
    "libx264",
    "libx265",
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "h264_nvenc",
    "hevc_nvenc",
    "h264_qsv",
    "h264_amf",
)
WEIGHT_FIELDS = ("weight_loudness", "weight_spikes", "weight_speech", "weight_keywords", "weight_exclamations")

# Your own hype phrases. With auto_keywords the built-in phrases of the detected language are added
# (see glossary.py), so the default list only needs what is specific to you.
DEFAULT_KEYWORDS: List[str] = []

SETTINGS_VERSION = 2
# Phrase list of version 1, removed from saved settings on upgrade (now built into glossary.py).
_V1_KEYWORDS = [
    "no way", "let's go", "lets go", "oh my god", "oh my gosh", "what the", "holy",
    "insane", "crazy", "unbelievable", "clip that", "clip it", "wow", "haha", "laughs",
    "laughter", "yes yes", "no no no", "gg",
    "incroyable", "c'est pas possible", "trop fort", "mais non", "oh non", "allez",
]

# One-click presets for the simple mode: they tune detection and pacing for a kind of content.
CONTENT_PRESETS: Dict[str, Dict[str, Any]] = {
    "gaming": dict(
        jump_cuts=False, shorts_jump_cuts=True, min_clip=8.0, max_clip=90.0, context_before=6.0,
        context_after=3.0, min_silence=0.8, transition="cut", shorts_layout="blur", min_score=25.0,
        weight_loudness=1.2, weight_spikes=1.2, weight_speech=0.8, weight_keywords=1.0, weight_exclamations=1.0,
    ),
    "talk": dict(
        jump_cuts=True, shorts_jump_cuts=True, min_clip=15.0, max_clip=120.0, context_before=4.0,
        context_after=2.0, min_silence=0.6, transition="cut", shorts_layout="smart", min_score=15.0,
        weight_loudness=0.7, weight_spikes=0.5, weight_speech=1.4, weight_keywords=1.3, weight_exclamations=1.3,
    ),
    "vlog": dict(
        jump_cuts=True, shorts_jump_cuts=True, min_clip=6.0, max_clip=60.0, context_before=4.0,
        context_after=2.5, min_silence=0.8, transition="fade", shorts_layout="smart", min_score=20.0,
        weight_loudness=1.0, weight_spikes=1.0, weight_speech=1.0, weight_keywords=1.0, weight_exclamations=1.0,
    ),
}


@dataclass
class Settings:
    content_type: str = "auto"

    # Highlight reel ----------------------------------------------------------------
    make_highlight: bool = True
    profile: str = "auto"         # auto = follows how many great moments the recording has
    target_duration: float = 0.0  # seconds; 0 = derived from the profile
    min_clip: float = 8.0
    max_clip: float = 90.0
    pad_before: float = 0.3
    pad_after: float = 0.5
    context_before: float = 6.0   # build-up kept before the peak of a moment
    context_after: float = 3.0    # reaction kept after the peak
    diversity: float = 0.2        # 0 = best scores only, 1 = spread over the whole recording
    min_score: float = 20.0       # drop moments scoring below this (0-100): a shorter reel beats a boring one
    skip_start: float = 0.0       # seconds ignored at the start (waiting screen, greetings)
    skip_end: float = 0.0         # seconds ignored at the end (outro, raid)
    auto_trim: bool = True        # detect waiting screens / outros without speech
    jump_cuts: bool = False
    min_silence: float = 0.8
    silence_threshold: float = 0.0  # dBFS used by jump cuts; 0 = automatic
    transition: str = "cut"
    fade_duration: float = 0.25
    export_clips: bool = True

    # Vertical shorts ---------------------------------------------------------------
    make_shorts: bool = True
    shorts_count: int = 5
    shorts_auto: bool = True       # number of shorts = number of great moments
    shorts_min: float = 20.0
    shorts_max: float = 59.0
    shorts_layout: str = "auto"
    shorts_hook: bool = False      # open each short with its best 3 seconds, then play it from the start
    facecam: str = ""  # "x,y,w,h" as fractions of the frame, used by the split layout
    blur_strength: int = 10
    shorts_jump_cuts: bool = True
    captions: bool = True
    caption_style: str = "karaoke"
    caption_font: str = "Arial"
    caption_size: int = 100        # percent of the default size
    caption_position: str = "auto"
    caption_uppercase: bool = True
    caption_color: str = "#FFE600"  # color of the word being spoken
    caption_words: int = 3          # maximum words per caption line
    caption_speaker_colors: bool = True  # one text color per detected speaker

    # Analysis ----------------------------------------------------------------------
    transcriber: str = "auto"
    whisper_model: str = "auto"
    language: str = ""  # empty = auto-detect
    beam_size: int = 5
    vocabulary: List[str] = field(default_factory=list)  # names/jargon the transcriber should know
    auto_vocabulary: bool = True    # add the vocabulary of the detected game
    profanity_prompt: bool = True   # tell the model swear words exist so it writes them down
    censor_profanity: bool = False  # mask them in captions and texts: p*tain
    censor_words: List[str] = field(default_factory=list)  # extra words to mask
    diarize: bool = True            # detect who is speaking
    speaker_count: int = 0          # 0 = automatic
    keywords: List[str] = field(default_factory=lambda: list(DEFAULT_KEYWORDS))
    auto_keywords: bool = True      # add the built-in hype phrases of the detected language
    use_cache: bool = True
    weight_loudness: float = 1.0
    weight_spikes: float = 1.0
    weight_speech: float = 1.0
    weight_keywords: float = 1.0
    weight_exclamations: float = 1.0

    # Audio / encoding --------------------------------------------------------------
    normalize_audio: bool = True
    target_lufs: float = -14.0
    audio_bitrate: int = 192
    encoder: str = "auto"
    quality: str = "high"

    # Extra exports -----------------------------------------------------------------
    export_srt: bool = True
    export_edl: bool = True
    export_xml: bool = True
    publish_kit: bool = True        # titles, descriptions, hashtags and thumbnails

    # -------------------------------------------------------------------------------
    def target_for(self, source_duration: float) -> float:
        """Target highlight duration in seconds for a source of the given length."""
        if self.target_duration and self.target_duration > 0:
            return min(self.target_duration, source_duration)
        # "auto" is decided after the analysis; before that the medium preset is a fair estimate.
        ratio, low, high = PROFILES.get(self.profile, PROFILES["medium"])
        target = min(max(source_duration * ratio, low), high)
        # Never ask for more than half of the source.
        return min(target, source_duration * 0.5)

    def score_weights(self) -> Dict[str, float]:
        return {
            "loudness": self.weight_loudness, "spikes": self.weight_spikes, "speech": self.weight_speech,
            "keywords": self.weight_keywords, "exclamations": self.weight_exclamations,
        }

    def apply_content_preset(self, name: str) -> None:
        if name == "auto":
            self.content_type = "auto"
            return
        if name not in CONTENT_PRESETS:
            raise ValueError(f"unknown preset '{name}' (choose from auto, {', '.join(CONTENT_PRESETS)})")
        # An automatic layout stays automatic (it takes the content type into account), and a configured
        # facecam split is kept.
        keep_layout = self.shorts_layout == "auto" or (self.shorts_layout == "split" and bool(self.facecam.strip()))
        for key, value in CONTENT_PRESETS[name].items():
            if key == "shorts_layout" and keep_layout:
                continue
            setattr(self, key, value)
        self.content_type = name

    def fill_from_preset(self, name: str) -> None:
        """Apply a content preset only to the settings still at their default value."""
        defaults = Settings()
        for key, value in CONTENT_PRESETS.get(name, {}).items():
            if key == "shorts_layout":
                continue  # decided by the automatic layout
            if getattr(self, key) == getattr(defaults, key):
                setattr(self, key, value)

    @classmethod
    def merge_auto(cls, user: "Settings", resolved: Optional[dict]) -> "Settings":
        """Settings to render with: what the user set explicitly, and the automatic decisions of the
        analysis for everything left on auto or at its default value."""
        if not resolved:
            return cls.from_dict(user.to_dict())
        defaults = cls().to_dict()
        data = user.to_dict()
        for key, value in data.items():
            if key not in resolved:
                continue
            left_to_auto = value == defaults.get(key) or (key in AUTO_CHOICE_FIELDS and value == "auto")
            if key in ("shorts_count",) and user.shorts_auto:
                left_to_auto = True
            if key == "target_duration" and user.profile == "auto":
                left_to_auto = True
            if left_to_auto:
                data[key] = resolved[key]
        return cls.from_dict(data)

    def facecam_rect(self) -> Optional[Tuple[float, float, float, float]]:
        if not self.facecam.strip():
            return None
        try:
            x, y, w, h = (float(v) for v in self.facecam.replace(" ", "").split(","))
        except ValueError:
            raise ValueError("facecam must be 'x,y,w,h' with values between 0 and 1")
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1.0001 and y + h <= 1.0001):
            raise ValueError("facecam must be 'x,y,w,h' with values between 0 and 1")
        return x, y, w, h

    def validate(self) -> List[str]:
        errors = []

        def check_choice(name, choices):
            if getattr(self, name) not in choices:
                errors.append(f"{name} must be one of: {', '.join(choices)}")

        def check_range(name, low, high):
            value = getattr(self, name)
            if not low <= value <= high:
                errors.append(f"{name} must be between {low:g} and {high:g}")

        check_choice("profile", PROFILE_CHOICES)
        check_choice("shorts_layout", SHORTS_LAYOUTS)
        check_choice("transcriber", TRANSCRIBERS)
        check_choice("transition", TRANSITIONS)
        check_choice("quality", QUALITIES)
        check_choice("caption_style", CAPTION_STYLES)
        check_choice("caption_position", CAPTION_POSITIONS)
        check_choice("content_type", CONTENT_TYPES)
        check_choice("encoder", ENCODERS)
        if self.min_clip <= 0 or self.max_clip < self.min_clip:
            errors.append("clip length: 0 < min_clip <= max_clip is required")
        if self.shorts_min <= 0 or self.shorts_max < self.shorts_min:
            errors.append("shorts length: 0 < shorts_min <= shorts_max is required")
        if self.shorts_count < 0:
            errors.append("shorts_count cannot be negative")
        if self.min_silence < 0.2:
            errors.append("min_silence must be at least 0.2 seconds")
        if not -30 <= self.target_lufs <= -5:
            errors.append("target_lufs must be between -30 and -5")
        if self.pad_before < 0 or self.pad_after < 0:
            errors.append("padding cannot be negative")
        for name in WEIGHT_FIELDS:
            check_range(name, 0, 3)
        if all(getattr(self, name) == 0 for name in WEIGHT_FIELDS):
            errors.append("at least one scoring weight must be above 0")
        check_range("context_before", 0, 30)
        check_range("context_after", 0, 30)
        check_range("diversity", 0, 1)
        check_range("min_score", 0, 100)
        check_range("skip_start", 0, 7200)
        check_range("skip_end", 0, 7200)
        check_range("speaker_count", 0, 10)
        if self.silence_threshold != 0:
            check_range("silence_threshold", -80, -10)
        check_range("fade_duration", 0.05, 2)
        check_range("blur_strength", 1, 40)
        check_range("caption_size", 40, 250)
        check_range("caption_words", 1, 8)
        check_range("beam_size", 1, 10)
        check_range("audio_bitrate", 64, 320)
        if not re.fullmatch(r"#?[0-9a-fA-F]{6}", self.caption_color.strip()):
            errors.append("caption_color must be a hex color such as #FFE600")
        if self.make_shorts and self.shorts_layout == "split":
            try:
                if self.facecam_rect() is None:
                    errors.append("the split layout needs a facecam area (x,y,w,h)")
            except ValueError as exc:
                errors.append(str(exc))
        if not self.make_highlight and not self.make_shorts:
            errors.append("nothing to do: enable the highlight reel and/or shorts")
        return errors

    # -- serialization -----------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        known = {f.name for f in fields(cls)}
        kwargs = {}
        defaults = cls()
        for key, value in (data or {}).items():
            if key not in known:
                continue
            try:
                kwargs[key] = _coerce(getattr(defaults, key), value)
            except (TypeError, ValueError):
                continue
        return cls(**kwargs)

    def with_value(self, key: str, raw: Any) -> "Settings":
        """Copy with one setting changed from a (possibly textual) value. Raises ValueError."""
        if key not in {f.name for f in fields(self)}:
            raise ValueError(f"unknown setting '{key}' (run with --list-settings to see them all)")
        try:
            value = _coerce(getattr(Settings(), key), raw)
        except (TypeError, ValueError):
            raise ValueError(f"invalid value for '{key}': {raw!r}")
        data = self.to_dict()
        data[key] = value
        return Settings.from_dict(data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(self.to_dict(), settings_version=SETTINGS_VERSION)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Settings":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if int(data.get("settings_version", 1)) < SETTINGS_VERSION:
            data = _migrate_v1(data)
        return cls.from_dict(data)


def _migrate_v1(data: dict) -> dict:
    """Version 1 had no automatic mode: switch the choices it forced to "auto"."""
    data = dict(data)
    data["content_type"] = "auto"
    if not float(data.get("target_duration", 0) or 0):
        data["profile"] = "auto"
    if not (data.get("shorts_layout") == "split" and str(data.get("facecam", "")).strip()):
        data["shorts_layout"] = "auto"
    data["whisper_model"] = "auto"
    data["shorts_auto"] = True
    data["keywords"] = [k for k in data.get("keywords", []) if k not in _V1_KEYWORDS]
    return data


def _coerce(default: Any, value: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("1", "true", "yes", "on"):
                return True
            if text in ("0", "false", "no", "off"):
                return False
            raise ValueError(value)
        return bool(value)
    if isinstance(default, int):
        return int(round(float(value)))
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [str(v) for v in value]
    return str(value)


# -- application directories --------------------------------------------------------

APP_DIR_NAME = "auto-video-editor"


def cache_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / APP_DIR_NAME


def config_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_DIR_NAME


def settings_path() -> Path:
    return config_dir() / "settings.json"
