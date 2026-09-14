"""End-to-end orchestration: analyze a recording into an editable Project, then render it."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from . import __version__, publish, vision
from .audio import AudioEnvelope, analyze_audio, media_fingerprint
from .captions import build_ass, build_srt, remap_words, remap_words_sequence
from .censor import censor_text, profanity_set
from .exporters import write_chapters, write_edl, write_fcp_xml, write_moments_csv
from .ffmpeg import MediaInfo, pick_encoder, probe, require_tools
from .glossary import detect_game, game_vocabulary, hashtags, hype_phrases
from .insights import (VideoInsights, auto_layout, auto_model, auto_shorts_count, auto_target, auto_trim,
                       classify_content, describe_position, estimate_transcription, load_insights, save_insights,
                       scan_video, speech_stats)
from .moments import (Moment, build_curve, compute_hook, compute_keep_ranges, find_candidates, plan_shorts,
                      select_highlight)
from .render import Renderer, caption_margin, loudnorm_apply_filter
from .reporting import Reporter
from .settings import CONTENT_LABELS, Settings, cache_dir
from .speakers import assign_speakers
from .timeline import TimeMap, format_clock, format_tag, slugify, total_duration
from .transcribe import Transcriber, Transcript, load_cached, resolve_backend, save_cached

PROJECT_FILE = "project.json"
PROJECT_VERSION = 2
TRANSCRIPT_CACHE_VERSION = "v4"  # bump when transcription parameters change

# Settings that change which moments are picked. Changing them requires a new analysis.
ANALYSIS_KEYS = (
    "content_type", "profile", "target_duration", "min_clip", "max_clip", "pad_before", "pad_after",
    "shorts_count", "shorts_auto", "shorts_min", "shorts_max", "transcriber", "whisper_model", "language",
    "keywords", "auto_keywords", "jump_cuts", "min_silence", "silence_threshold", "context_before", "context_after",
    "diversity", "min_score", "beam_size", "weight_loudness", "weight_spikes", "weight_speech", "weight_keywords",
    "weight_exclamations", "skip_start", "skip_end", "auto_trim", "vocabulary", "auto_vocabulary",
    "profanity_prompt", "censor_profanity", "censor_words", "diarize", "speaker_count",
)


@dataclass
class Project:
    input: str
    output_dir: str
    media: MediaInfo
    settings: Settings            # what the user asked for, "auto" values included
    target_duration: float
    highlight: List[Moment] = field(default_factory=list)
    shorts: List[Moment] = field(default_factory=list)
    language: str = ""
    transcribed: bool = False
    created: str = ""
    resolved: dict = field(default_factory=dict)          # settings actually used by the analysis
    auto: List[List[str]] = field(default_factory=list)   # [label, value] for every automatic decision
    game: str = ""
    speakers: int = 0

    @property
    def path(self) -> Path:
        return Path(self.output_dir) / PROJECT_FILE

    @property
    def content_type(self) -> str:
        return self.resolved.get("content_type", self.settings.content_type)

    def needs_reanalysis(self, settings: Settings) -> bool:
        old, new = self.settings.to_dict(), settings.to_dict()
        return any(old.get(k) != new.get(k) for k in ANALYSIS_KEYS)

    def effective_settings(self, settings: Settings) -> Settings:
        return Settings.merge_auto(settings, self.resolved)

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": PROJECT_VERSION,
            "app_version": __version__,
            "created": self.created,
            "input": self.input,
            "output_dir": self.output_dir,
            "target_duration": round(self.target_duration, 2),
            "language": self.language,
            "transcribed": self.transcribed,
            "game": self.game,
            "speakers": self.speakers,
            "auto": self.auto,
            "media": self.media.to_dict(),
            "settings": self.settings.to_dict(),
            "resolved": self.resolved,
            "highlight": [m.to_dict() for m in self.highlight],
            "shorts": [m.to_dict() for m in self.shorts],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "Project":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != PROJECT_VERSION:
            raise ValueError(f"Unsupported project file version: {data.get('version')}")
        return cls(
            input=data["input"],
            output_dir=data.get("output_dir") or str(Path(path).parent),
            media=MediaInfo.from_dict(data["media"]),
            settings=Settings.from_dict(data.get("settings", {})),
            target_duration=float(data.get("target_duration", 0)),
            highlight=[Moment.from_dict(m) for m in data.get("highlight", [])],
            shorts=[Moment.from_dict(m) for m in data.get("shorts", [])],
            language=data.get("language", ""),
            transcribed=bool(data.get("transcribed", False)),
            created=data.get("created", ""),
            resolved=dict(data.get("resolved", {})),
            auto=[list(item) for item in data.get("auto", [])],
            game=data.get("game", ""),
            speakers=int(data.get("speakers", 0)),
        )


@dataclass
class RenderResult:
    output_dir: Path
    highlight: Optional[Path] = None
    clips: List[Path] = field(default_factory=list)
    shorts: List[Path] = field(default_factory=list)
    extras: List[Path] = field(default_factory=list)


def default_output_dir(input_path: str) -> Path:
    path = Path(input_path)
    return path.with_name(f"{path.stem}_edit")


class Pipeline:
    def __init__(self, input_path: str, settings: Settings, output_dir: Optional[str] = None,
                 reporter: Optional[Reporter] = None) -> None:
        # absolute() and not resolve(): a symlink keeps its own name, which may tell the game being played.
        self.input_path = str(Path(input_path).expanduser().absolute())
        self.settings = settings
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else default_output_dir(self.input_path)
        self.reporter = reporter or Reporter()
        self._envelope: Optional[AudioEnvelope] = None

    @property
    def log(self):
        return self.reporter.log

    # =============================================================================================
    # Analysis
    # =============================================================================================
    def analyze(self) -> Project:
        errors = self.settings.validate()
        if errors:
            raise ValueError("Invalid settings:\n- " + "\n- ".join(errors))
        require_tools()
        if not os.path.isfile(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        user = self.settings
        s = Settings.from_dict(user.to_dict())  # effective settings, auto values get decided below
        self.settings = s
        try:
            return self._analyze(user, s)
        finally:
            self.settings = user

    def _analyze(self, user: Settings, s: Settings) -> Project:
        r = self.reporter
        backend = resolve_backend(s.transcriber)
        r.plan([("probe", 1), ("scan", 4), ("audio", 8), ("preview", 4 if backend else 0),
                ("transcribe", 70 if backend else 0), ("moments", 3)])
        auto: List[List[str]] = []

        r.stage("probe", "Reading media")
        info = probe(self.input_path)
        if not info.has_video:
            raise ValueError("The input file has no video stream.")
        self.log(f"Input: {Path(info.path).name} | {format_clock(info.duration, True)} | "
                 f"{info.width}x{info.height} @ {info.fps:.2f} fps | audio: {'yes' if info.has_audio else 'no'}")
        if not info.has_audio:
            self.log("WARNING: no audio track. Moments cannot be detected reliably without sound.")

        insights = self._load_insights(info)
        env = self._load_envelope(info)

        # -- speech --------------------------------------------------------------------------------
        transcript: Optional[Transcript] = None
        game = insights.title_game
        if game:
            self.log(f"Game from the title: {game}.")
        speakers = 0
        if backend and info.has_audio:
            if s.whisper_model == "auto":
                s.whisper_model, why = auto_model(info.duration)
                eta = estimate_transcription(info.duration, s.whisper_model)
                self.log(f"Speech model: {s.whisper_model} (auto: {why}; first transcription ≈ {format_clock(eta)}).")
                auto.append(["Speech model", s.whisper_model])
            transcriber = Transcriber(backend, s.whisper_model, s.language, self.log, r.cancel_event,
                                      beam_size=s.beam_size, profanity=s.profanity_prompt)
            if not s.language or (s.auto_vocabulary and not game):
                language, sample = self._preview(info, transcriber, insights)
                if not s.language and language:
                    s.language = language
                    auto.append(["Language", language])
                if s.auto_vocabulary and not game:
                    game = detect_game([sample], min_score=3)[0]
                    if game:
                        self.log(f"Game recognised from what is said: {game}.")
            if s.auto_vocabulary and game:
                known = {v.lower() for v in s.vocabulary}
                extra = [t for t in game_vocabulary(game) if t.lower() not in known]
                s.vocabulary = list(s.vocabulary) + extra[:max(0, 30 - len(s.vocabulary))]
                self.log(f"Vocabulary: {len(extra)} {game} terms added automatically.")
            if game:
                auto.append(["Game", game])
            transcript = self._load_transcript(info, backend, s, transcriber)
            transcript = self._drop_silent_segments(transcript, env)
            if s.diarize:
                speakers = assign_speakers(transcript.segments, s.speaker_count)
                for seg in transcript.segments:
                    seg.set_speaker(seg.speaker)
                self.log(f"Speakers: {speakers} voice{'s' if speakers != 1 else ''}"
                         f"{' detected' if s.speaker_count == 0 else ' (as requested)'}.")
                if s.speaker_count == 0:
                    auto.append(["Speakers", str(speakers)])
            else:
                speakers = 1
        r.check_cancel()

        # -- content, trimming, phrases -------------------------------------------------------------------
        r.stage("moments", "Finding the best moments")
        language = s.language or (transcript.language if transcript else "")
        if s.auto_keywords:
            phrases = list(s.keywords) + [p for p in hype_phrases(language) if p not in s.keywords]
            s.keywords = phrases
        stats = speech_stats(transcript, env, info.duration)
        if s.content_type == "auto":
            kind, why = classify_content(insights, stats, speakers, info.duration, game)
            s.fill_from_preset(kind)
            s.content_type = kind
            self.log(f"Content: {CONTENT_LABELS[kind]} (auto: {why}).")
            auto.append(["Content", CONTENT_LABELS[kind]])
        if s.auto_trim and transcript is not None:
            start, end = auto_trim(transcript, info.duration)
            trimmed = []
            if start and not user.skip_start:
                s.skip_start = round(start)
                trimmed.append(f"first {format_clock(start)}")
            if end and not user.skip_end:
                s.skip_end = round(end)
                trimmed.append(f"last {format_clock(end)}")
            if trimmed:
                self.log(f"Intro/outro without speech skipped: {' and '.join(trimmed)}.")
                auto.append(["Skipped", " · ".join(trimmed)])
        if s.skip_start or s.skip_end:
            self.log(f"Ignoring the first {s.skip_start:.0f} s and the last {s.skip_end:.0f} s.")

        curve = build_curve(env, transcript, info.duration, s.keywords, s.score_weights(),
                            skip=(s.skip_start, s.skip_end))
        if transcript is not None and s.censor_profanity:
            transcript = self._censor(transcript)

        # -- moments -------------------------------------------------------------------------------------
        min_clip = min(s.min_clip, max(2.0, info.duration / 4))
        max_clip = max(min_clip, s.max_clip)
        limit = max(40, int(s.target_for(info.duration) / min_clip * 3) + 30)
        candidates = find_candidates(curve, env, transcript, info.duration, min_clip, max_clip, limit,
                                     lead=s.context_before, tail=s.context_after)
        if not candidates:
            raise RuntimeError("No usable moments were found. Is the audio track silent?")
        weak = [c for c in candidates if c.score < s.min_score]
        if weak and len(weak) < len(candidates):
            candidates = [c for c in candidates if c.score >= s.min_score]
            self.log(f"Found {len(candidates)} strong moments ({len(weak)} weak ones below score "
                     f"{s.min_score:.0f} ignored).")
        else:
            self.log(f"Found {len(candidates)} candidate moments.")

        if s.profile == "auto" and s.target_duration <= 0:
            s.target_duration = round(auto_target(candidates, info.duration, s.min_score), 1)
            self.log(f"Reel length: {format_clock(s.target_duration)} (auto: follows the number of great moments).")
            auto.append(["Reel length", format_clock(s.target_duration)])
        target = s.target_for(info.duration)
        if s.shorts_auto:
            s.shorts_count = auto_shorts_count(candidates, info.duration)
            self.log(f"Shorts: {s.shorts_count} (auto: one per great moment).")
            auto.append(["Shorts", str(s.shorts_count)])
        if s.shorts_layout == "auto":
            if s.facecam.strip():
                s.shorts_layout, why = "split", "webcam area set by you"
            else:
                s.shorts_layout, why = auto_layout(insights, s.content_type)
            self.log(f"Shorts layout: {s.shorts_layout} (auto: {why}).")
            auto.append(["Layout", {"blur": "Blur fill", "crop": "Center crop", "smart": "Face tracking",
                                    "split": "Facecam split"}[s.shorts_layout]])
        if s.shorts_layout == "split" and not s.facecam.strip():
            if insights.facecam:
                s.facecam = ",".join(f"{v:.3f}" for v in insights.facecam)
                auto.append(["Facecam", describe_position(insights.facecam)])
            else:
                self.log("The split layout needs a facecam area and none was found: using the blur layout.")
                s.shorts_layout = "blur"

        if s.jump_cuts:
            # Pick moments by their length after dead air is removed, so the reel hits the target.
            for candidate in candidates:
                self._attach_details(candidate, env, transcript)
        highlight = select_highlight(candidates, target, info.duration, s.pad_before, s.pad_after,
                                     jump_cuts=s.jump_cuts, diversity=s.diversity)
        shorts = []
        if s.shorts_count > 0:
            shorts = plan_shorts(candidates, curve, transcript, env, info.duration, s.shorts_count,
                                 s.shorts_min, s.shorts_max, s.pad_before, s.pad_after)
        for moment in highlight:
            self._attach_details(moment, env, transcript)
        for short in shorts:
            self._attach_details(short, env, transcript, is_short=True)
            short.hook = compute_hook(short.start, short.end, short.peak, short.words)

        project = Project(
            input=self.input_path, output_dir=str(self.output_dir), media=info,
            settings=Settings.from_dict(user.to_dict()), target_duration=target,
            highlight=highlight, shorts=shorts, language=language, transcribed=transcript is not None,
            created=datetime.now().isoformat(timespec="seconds"), resolved=s.to_dict(), auto=auto,
            game=game or "", speakers=speakers,
        )
        path = project.save()
        total = sum(m.edited_duration(s.jump_cuts) for m in highlight)
        self.log(f"Highlight plan: {len(highlight)} moments, {format_clock(total)} "
                 f"(target {format_clock(target)}).")
        if total < target * 0.7 and highlight:
            self.log("The reel is shorter than the target: only strong moments are kept. Lower the minimum "
                     "score in the settings to fill it up anyway.")
        self.log(f"Shorts plan: {len(shorts)} vertical clips.")
        if auto:
            self.log("Auto choices: " + " | ".join(f"{label}: {value}" for label, value in auto))
        self.log(f"Project saved: {path}")
        r.done("Analysis complete")
        return project

    # -- analysis helpers ------------------------------------------------------------------------------
    def _load_insights(self, info: MediaInfo) -> VideoInsights:
        if self.settings.use_cache:
            cached = load_insights(info.path)
            if cached is not None:
                self.log("Video scan: loaded from cache.")
                self._log_insights(cached)
                return cached
        self.log("Scanning the picture (motion, faces, webcam overlay)...")
        insights = scan_video(info, self.reporter.stage("scan", "Scanning the video"), self.reporter.cancel_event)
        save_insights(info.path, insights)
        self._log_insights(insights)
        return insights

    def _log_insights(self, insights: VideoInsights) -> None:
        parts = [f"motion {insights.motion:.3f}"]
        if insights.faces_checked:
            parts.append(f"faces in {insights.face_ratio:.0%} of frames")
        if insights.facecam:
            parts.append(f"webcam overlay {describe_position(insights.facecam)}")
        self.log("Video scan: " + ", ".join(parts) + ".")

    def _preview(self, info: MediaInfo, transcriber: Transcriber, insights: VideoInsights) -> Tuple[Optional[str], str]:
        if self.settings.use_cache and insights.sample_model:
            return insights.language or None, insights.sample_text
        self.log("Listening to a few samples (language, topic)...")
        language, text = transcriber.preview(info.path, info.duration,
                                             on_progress=self.reporter.stage("preview", "Listening to samples"))
        if language:
            self.log(f"Detected language: {language}.")
        insights.language, insights.sample_text, insights.sample_model = language or "", text, transcriber.model_name
        save_insights(info.path, insights)
        return language, text

    def _censor(self, transcript: Transcript) -> Transcript:
        """Mask swear words (p*tain) in every text that ends up on screen or in the exports."""
        words = profanity_set(self.settings.censor_words)
        count = 0
        for seg in transcript.segments:
            for w in seg.words:
                masked = censor_text(w.text, words)
                count += masked != w.text
                w.text = masked
            seg.text = censor_text(seg.text, words)
        self.log(f"Censored {count} swear word{'s' if count != 1 else ''}.")
        return transcript

    def _drop_silent_segments(self, transcript: Transcript, env: AudioEnvelope) -> Transcript:
        """Remove text Whisper "heard" in parts of the recording that are actually silent."""
        threshold = env.silence_threshold()
        kept, dropped_words = [], 0
        for seg in transcript.segments:
            if float(env.window(seg.start, seg.end).max(initial=-90.0)) < threshold:
                continue
            if seg.words:
                audible = [w for w in seg.words
                           if float(env.window(w.start - 0.1, w.end + 0.1).max(initial=-90.0)) >= threshold]
                dropped_words += len(seg.words) - len(audible)
                if not audible:
                    continue
                if len(audible) != len(seg.words):
                    seg.words = audible
                    seg.text = "".join(w.text for w in audible).strip()
            kept.append(seg)
        dropped = len(transcript.segments) - len(kept)
        if dropped or dropped_words:
            self.log(f"Ignored {dropped} line(s) and {dropped_words} word(s) transcribed over silence.")
        return Transcript(kept, transcript.language, transcript.backend, transcript.model)

    def _attach_details(self, moment: Moment, env: AudioEnvelope, transcript: Optional[Transcript],
                        is_short: bool = False) -> None:
        if transcript:
            words = transcript.words_between(moment.start, moment.end)
            moment.words = [[round(w.start, 3), round(w.end, 3), w.text, w.speaker] for w in words]
            moment.text = transcript.text_between(moment.start, moment.end)
        moment.keep = self._jump_cut_ranges(moment, env, is_short)

    def _jump_cut_ranges(self, moment: Moment, env: AudioEnvelope, is_short: bool) -> list:
        threshold = self.settings.silence_threshold if self.settings.silence_threshold < 0 else None
        keep = compute_keep_ranges(moment.start, moment.end, env, moment.words, self.settings.min_silence,
                                   threshold)
        if is_short and total_duration(keep) < min(self.settings.shorts_min, moment.duration):
            return []  # jump cuts must not shrink a short below its minimum length
        return keep

    def _cache_file(self, kind: str, name: str) -> Path:
        return cache_dir() / kind / name

    def _load_envelope(self, info: MediaInfo) -> AudioEnvelope:
        s, r = self.settings, self.reporter
        path = self._cache_file("audio", f"{media_fingerprint(info.path)}-v1.npz")
        if s.use_cache and path.exists():
            try:
                self._envelope = AudioEnvelope.load(path)
                self.log("Audio analysis: loaded from cache.")
                return self._envelope
            except (OSError, ValueError, KeyError):
                pass
        self.log("Analyzing audio levels...")
        env = analyze_audio(info, r.stage("audio", "Analyzing audio"), r.cancel_event)
        try:
            env.save(path)
        except OSError:
            pass
        self._envelope = env
        return env

    def _load_transcript(self, info: MediaInfo, backend: str, s: Settings,
                         transcriber: Transcriber) -> Transcript:
        r = self.reporter
        language = s.language.strip().lower() or "auto"
        context = hashlib.sha1(json.dumps([sorted(s.vocabulary), s.profanity_prompt]).encode("utf-8")).hexdigest()[:8]
        path = self._cache_file("transcripts", f"{media_fingerprint(info.path)}-{backend}-{s.whisper_model}-"
                                               f"{language}{'' if s.beam_size == 5 else f'-b{s.beam_size}'}-"
                                               f"{context}-{TRANSCRIPT_CACHE_VERSION}.json")
        if s.use_cache and path.exists():
            cached = load_cached(path)
            if cached is not None:
                self.log(f"Transcript: loaded from cache ({len(cached.segments)} sentences).")
                return cached
        self.log(f"Transcribing speech with {backend} ({s.whisper_model}). This is the longest step.")
        if s.vocabulary:
            self.log(f"Vocabulary hints: {', '.join(s.vocabulary[:12])}{'…' if len(s.vocabulary) > 12 else ''}")
        transcriber.language = s.language.strip() or None
        transcriber.vocabulary = list(s.vocabulary)
        transcript = transcriber.transcribe(info.path, info.duration, r.stage("transcribe", "Transcribing speech"))
        words = sum(1 for _ in transcript.words())
        self.log(f"Transcript: {len(transcript.segments)} sentences, {words} words.")
        try:  # always refresh the cache, "use_cache" only controls reading it
            save_cached(path, transcript)
        except OSError:
            pass
        return transcript

    # =============================================================================================
    # Rendering
    # =============================================================================================
    def render(self, project: Project) -> RenderResult:
        user = self.settings
        self.settings = project.effective_settings(user)
        if self.settings.shorts_layout == "auto":  # project from an older version
            self.settings.shorts_layout = "blur"
        try:
            return self._render(project)
        finally:
            self.settings = user

    def _render(self, project: Project) -> RenderResult:
        require_tools()
        s, r = self.settings, self.reporter
        info = project.media
        if not os.path.isfile(info.path):
            raise FileNotFoundError(f"Source video not found: {info.path}")
        errors = [e for e in s.validate() if "nothing to do" not in e]
        if errors:
            raise ValueError("Invalid settings:\n- " + "\n- ".join(errors))

        out = Path(project.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        work = out / ".work"
        work.mkdir(exist_ok=True)

        highlight = [m for m in project.highlight if m.selected] if s.make_highlight else []
        shorts = [m for m in project.shorts if m.selected] if s.make_shorts else []
        if not highlight and not shorts:
            raise ValueError("Nothing to render: no moment is selected.")

        env = self._envelope or self._cached_envelope(info)
        if env is not None:
            for m in highlight:
                m.keep = self._jump_cut_ranges(m, env, is_short=False)
            for m in shorts:
                m.keep = self._jump_cut_ranges(m, env, is_short=True)

        encoder = pick_encoder(s.encoder)
        self.log(f"Encoder: {encoder} ({s.quality} quality)")
        renderer = Renderer(info, encoder, s.quality, r.cancel_event, audio_bitrate=s.audio_bitrate,
                            blur=s.blur_strength)

        highlight_time = sum(m.edited_duration(s.jump_cuts) for m in highlight)
        shorts_time = sum(m.edited_duration(s.shorts_jump_cuts) for m in shorts)
        r.plan([("clips", highlight_time), ("concat", highlight_time * 0.12),
                ("shorts", shorts_time * 2.5), ("exports", 0.01 * (highlight_time + shorts_time) + 1)])

        result = RenderResult(out)
        stem = slugify(Path(info.path).stem, 60) or "video"
        rows: List[dict] = []
        try:
            if highlight:
                self._render_highlight(project, highlight, renderer, out, work, stem, result, rows)
            if shorts:
                self._render_shorts(project, shorts, renderer, out, work, result, rows)
            r.stage("exports", "Writing reports")
            csv_path = out / f"{stem}_moments.csv"
            write_moments_csv(csv_path, rows)
            result.extras.append(csv_path)
            project.save()
        finally:
            shutil.rmtree(work, ignore_errors=True)

        self.log("")
        self.log("=" * 60)
        self.log("RENDER COMPLETE")
        if result.highlight:
            self.log(f"  Highlight reel : {result.highlight}")
        if result.clips:
            self.log(f"  Clips          : {len(result.clips)} files in {out / 'clips'}")
        if result.shorts:
            self.log(f"  Shorts         : {len(result.shorts)} files in {out / 'shorts'}")
        self.log(f"  Output folder  : {out}")
        self.log("=" * 60)
        r.done("Render complete")
        return result

    def _cached_envelope(self, info: MediaInfo) -> Optional[AudioEnvelope]:
        try:
            path = self._cache_file("audio", f"{media_fingerprint(info.path)}-v1.npz")
            if path.exists():
                self._envelope = AudioEnvelope.load(path)
        except (OSError, ValueError, KeyError):
            self._envelope = None
        return self._envelope

    def _moment_title(self, moment: Moment, fallback: str, max_length: int = 70) -> str:
        return publish.suggest_title(moment.text, self.settings.keywords, max_length) or moment.title() or fallback

    def _render_highlight(self, project: Project, moments: List[Moment], renderer: Renderer, out: Path,
                          work: Path, stem: str, result: RenderResult, rows: List[dict]) -> None:
        s, r, info = self.settings, self.reporter, project.media
        clips_dir = out / "clips" if s.export_clips else work / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        for old in clips_dir.glob("clip_[0-9][0-9][0-9]_*.mp4"):
            old.unlink()

        total = sum(m.edited_duration(s.jump_cuts) for m in moments) or 1.0
        progress = r.stage("clips", "Rendering highlight clips")
        fade = s.fade_duration if s.transition == "fade" else 0.0
        files: List[Path] = []
        done = 0.0
        for i, m in enumerate(moments, 1):
            r.check_cancel()
            ranges = m.ranges(s.jump_cuts)
            length = total_duration(ranges)
            path = clips_dir / f"clip_{i:03d}_{format_tag(m.start)}.mp4"
            cut_note = f", {len(ranges) - 1} jump cuts" if len(ranges) > 1 else ""
            self.log(f"Clip {i}/{len(moments)}: {format_clock(m.start, True)} -> {format_clock(m.end, True)} "
                     f"({length:.1f}s{cut_note}, score {m.score:.0f})")
            renderer.render_segment(
                ranges, path, fade=fade,
                on_progress=lambda p, base=done, length=length: progress((base + p * length) / total),
            )
            files.append(path)
            done += length

        highlight_path = out / f"{stem}_highlight.mp4"
        self.log("Assembling the highlight reel" + (" and normalizing loudness..." if s.normalize_audio else "..."))
        renderer.concat(files, highlight_path, work, total, s.normalize_audio, s.target_lufs,
                        r.stage("concat", "Assembling highlight reel"))
        result.highlight = highlight_path
        if s.export_clips:
            result.clips = files

        # Output timeline positions, from the real clip durations (avoids drift on long reels).
        offsets, cursor = [], 0.0
        for path, m in zip(files, moments):
            offsets.append(cursor)
            try:
                cursor += probe(str(path)).duration
            except Exception:
                cursor += m.edited_duration(s.jump_cuts)

        all_ranges = [rg for m in moments for rg in m.ranges(s.jump_cuts)]
        title = f"{Path(info.path).stem} highlight"
        if s.export_srt and any(m.words for m in moments):
            words = []
            for m, offset in zip(moments, offsets):
                for w in remap_words(m.words, TimeMap(m.ranges(s.jump_cuts))):
                    w.start += offset
                    w.end += offset
                    words.append(w)
            srt_path = highlight_path.with_suffix(".srt")
            srt_path.write_text(build_srt(words), encoding="utf-8")
            result.extras.append(srt_path)
        if s.export_edl:
            edl_path = highlight_path.with_suffix(".edl")
            write_edl(edl_path, title, info, all_ranges)
            result.extras.append(edl_path)
        if s.export_xml:
            xml_path = highlight_path.with_suffix(".xml")
            write_fcp_xml(xml_path, title, info, all_ranges)
            result.extras.append(xml_path)

        chapters = [(offset, self._moment_title(m, f"Moment {i}", 50))
                    for i, (m, offset) in enumerate(zip(moments, offsets), 1)]
        chapters_path = out / f"{stem}_chapters.txt"
        write_chapters(chapters_path, chapters)
        result.extras.append(chapters_path)

        if s.publish_kit:
            best = max(moments, key=lambda m: m.score)
            try:
                thumb = out / f"{stem}_thumbnail.jpg"
                t = publish.best_frame_time(info, best.start, best.end, best.peak)
                publish.write_thumbnail(info, t, thumb, cancel_event=r.cancel_event)
                result.extras.append(thumb)
            except Exception as exc:  # noqa: BLE001 - the kit is a bonus, never fatal
                self.log(f"Thumbnail skipped ({exc}).")
            reel_title = self._moment_title(best, f"{Path(info.path).stem}: best moments")
            tags = [t for t in hashtags(project.game, project.content_type, project.language) if t != "#shorts"]
            kit = out / f"{stem}_publish.txt"
            kit.write_text(publish.reel_kit_text(reel_title, chapters, tags), encoding="utf-8")
            result.extras.append(kit)

        for i, (m, offset) in enumerate(zip(moments, offsets), 1):
            rows.append({
                "kind": "highlight", "index": i,
                "output_file": files[i - 1].name if s.export_clips else highlight_path.name,
                "output_start": format_clock(offset, True), "source_start": format_clock(m.start, True),
                "source_end": format_clock(m.end, True), "duration": round(m.edited_duration(s.jump_cuts), 2),
                "score": m.score, "reasons": ", ".join(m.reasons), "text": m.text,
            })

    def _render_shorts(self, project: Project, moments: List[Moment], renderer: Renderer, out: Path,
                       work: Path, result: RenderResult, rows: List[dict]) -> None:
        s, r, info = self.settings, self.reporter, project.media
        shorts_dir = out / "shorts"
        shorts_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("mp4", "srt", "jpg", "txt"):
            for old in shorts_dir.glob(f"short_[0-9][0-9]_*.{pattern}"):
                old.unlink()

        layout = s.shorts_layout
        facecam = s.facecam_rect() if layout == "split" else None
        if layout == "smart" and not vision.available():
            self.log("NOTE: smart crop needs OpenCV 4 (pip install \"opencv-python-headless<5\"). "
                     "Using a centered crop.")
        tags = hashtags(project.game, project.content_type, project.language)
        # With an automatic split layout, a moment filmed with the camera full screen gets face tracking.
        adaptive = project.settings.shorts_layout == "auto"

        def play_ranges(m: Moment) -> list:
            ranges = m.ranges(s.shorts_jump_cuts)
            return list(m.hook) + ranges if s.shorts_hook and m.hook else ranges

        total = sum(total_duration(play_ranges(m)) for m in moments) or 1.0
        progress = r.stage("shorts", "Rendering vertical shorts")
        done = 0.0
        for i, m in enumerate(moments, 1):
            r.check_cancel()
            ranges = play_ranges(m)
            length = total_duration(ranges)
            title = self._moment_title(m, f"Short {i}")
            name = slugify(title, 40) or format_tag(m.start)
            path = shorts_dir / f"short_{i:02d}_{name}.mp4"
            hook_note = ", with hook" if s.shorts_hook and m.hook else ""
            self.log(f"Short {i}/{len(moments)}: {format_clock(m.start, True)} ({length:.1f}s{hook_note}, "
                     f"score {m.score:.0f}) \"{title}\"")

            short_layout, subject_x = layout, None
            if vision.available() and (layout == "smart" or (layout == "split" and adaptive)):
                try:
                    subject_x, big_faces = vision.analyse_window(info, m.start, m.duration)
                except Exception as exc:  # face detection is a nice-to-have, never fatal
                    subject_x, big_faces = None, 0.0
                    self.log(f"  face detection failed ({exc})")
                if layout == "split" and big_faces >= 0.5:
                    short_layout = "smart"
                    self.log("  full-screen camera in this moment: face tracking instead of the split layout")
                if short_layout == "smart":
                    self.log("  face detected, crop follows the speaker" if subject_x is not None
                             else "  no face detected, using a centered crop")

            captions_file = None
            timed_words = remap_words_sequence(m.words, ranges) if m.words else []
            if s.captions and timed_words:
                captions_file = f"short_{i:02d}.ass"
                ass = build_ass(timed_words, style=s.caption_style, margin_v=caption_margin(short_layout, info),
                                font=s.caption_font, size_scale=s.caption_size / 100.0,
                                uppercase=s.caption_uppercase, highlight=s.caption_color,
                                max_words=s.caption_words, position=s.caption_position,
                                speaker_colors=s.caption_speaker_colors)
                (work / captions_file).write_text(ass, encoding="utf-8")

            audio_filter = None
            if s.normalize_audio:
                measured = renderer.measure_segment_loudness(ranges, s.target_lufs)
                audio_filter = loudnorm_apply_filter(measured, s.target_lufs)

            renderer.render_segment(
                ranges, path, layout=short_layout, subject_x=subject_x, facecam=facecam,
                captions_file=captions_file, audio_filter=audio_filter, cwd=str(work),
                on_progress=lambda p, base=done, length=length: progress((base + p * length) / total),
            )
            result.shorts.append(path)
            if s.export_srt and timed_words:
                path.with_suffix(".srt").write_text(build_srt(timed_words), encoding="utf-8")
            if s.publish_kit:
                try:
                    t = publish.best_frame_time(info, m.start, m.end, m.peak)
                    publish.write_thumbnail(info, t, path.with_suffix(".jpg"), layout=short_layout, subject_x=subject_x,
                                            facecam=facecam, blur=s.blur_strength, cancel_event=r.cancel_event)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"  cover image skipped ({exc})")
                path.with_suffix(".txt").write_text(publish.short_kit_text(title, m.text, tags), encoding="utf-8")
            done += length
            rows.append({
                "kind": "short", "index": i, "output_file": path.name, "output_start": "0:00:00",
                "source_start": format_clock(m.start, True), "source_end": format_clock(m.end, True),
                "duration": round(length, 2), "score": m.score, "reasons": ", ".join(m.reasons), "text": m.text,
            })
