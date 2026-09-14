"""Speech transcription with word timestamps (faster-whisper or openai-whisper)."""

from __future__ import annotations

import importlib.util
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .audio import SAMPLE_RATE
from .ffmpeg import FFmpegError, pcm_reader_command, popen_binary
from .reporting import Cancelled
from .speakers import voice_features

CHUNK_SECONDS = 600.0   # transcribe long recordings in 10 minute windows
OVERLAP_SECONDS = 30.0  # extra audio so sentences at a window edge are not cut


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0
    speaker: int = 0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    speaker: int = 0
    embedding: Optional[List[Optional[float]]] = None  # voice signature, see speakers.py

    def set_speaker(self, speaker: int) -> None:
        self.speaker = speaker
        for word in self.words:
            word.speaker = speaker


@dataclass
class Transcript:
    segments: List[Segment]
    language: str = ""
    backend: str = ""
    model: str = ""

    def words(self) -> Iterator[Word]:
        for segment in self.segments:
            yield from segment.words

    def words_between(self, start: float, end: float) -> List[Word]:
        return [w for w in self.words() if w.end > start and w.start < end]

    @property
    def speaker_count(self) -> int:
        return max((s.speaker for s in self.segments), default=0) + 1 if self.segments else 0

    def text_between(self, start: float, end: float) -> str:
        """Text spoken between two times; speakers are labelled (A:, B:) when there are several."""
        words = self.words_between(start, end)
        if not words:
            return " ".join(s.text.strip() for s in self.segments if s.end > start and s.start < end)
        if self.speaker_count <= 1:
            return _join_words(words)
        parts, current, run = [], None, []
        for w in words:
            if w.speaker != current and run:
                parts.append(f"{speaker_label(current)}: {_join_words(run)}")
                run = []
            current = w.speaker
            run.append(w)
        parts.append(f"{speaker_label(current)}: {_join_words(run)}")
        return "  ".join(parts)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "backend": self.backend,
            "model": self.model,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker, "embedding": s.embedding,
                 "words": [[round(w.start, 3), round(w.end, 3), w.text, round(w.probability, 3)] for w in s.words]}
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        segments = []
        for s in data.get("segments", []):
            segment = Segment(s["start"], s["end"], s["text"], [Word(*w[:4]) for w in s.get("words", [])],
                              embedding=s.get("embedding"))
            segment.set_speaker(int(s.get("speaker", 0)))
            segments.append(segment)
        return cls(segments, data.get("language", ""), data.get("backend", ""), data.get("model", ""))


def speaker_label(speaker: int) -> str:
    """0 -> A, 1 -> B, ..."""
    return chr(ord("A") + speaker) if 0 <= speaker < 26 else f"S{speaker + 1}"


def _join_words(words: List[Word]) -> str:
    # Whisper word tokens carry their own leading spaces.
    return "".join(w.text for w in words).strip()


# -- prompt -----------------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "fr": "On est en live, on discute et on réagit en direct.",
    "en": "We're live, chatting and reacting in real time.",
    "es": "Estamos en directo, charlando y reaccionando.",
    "de": "Wir sind live, wir quatschen und reagieren.",
    "it": "Siamo in diretta, chiacchieriamo e reagiamo.",
    "pt": "Estamos ao vivo, conversando e reagindo.",
}
VOCABULARY_LABELS = {"fr": "Vocabulaire", "es": "Vocabulario", "de": "Wortschatz", "it": "Vocabolario",
                     "pt": "Vocabulário"}


def build_prompt(language: Optional[str], vocabulary: Sequence[str] = (), profanity: bool = True) -> str:
    """Text the speech model is conditioned on: it nudges spelling towards the given vocabulary
    (game names, jargon, foreign words) and stops the model from silently skipping swear words."""
    from .censor import PROMPT_HINTS

    lang = (language or "en").lower()
    parts = [PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES["en"])]
    words = [w.strip() for w in vocabulary if w.strip()]
    if words:
        parts.append(f"{VOCABULARY_LABELS.get(lang, 'Vocabulary')}: {', '.join(words)}.")
    if profanity:
        parts.append(PROMPT_HINTS.get(lang, PROMPT_HINTS["en"]))
    return " ".join(parts)


# -- backends ---------------------------------------------------------------------------------

def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def available_backends() -> List[str]:
    backends = []
    if _has_module("faster_whisper"):
        backends.append("faster-whisper")
    if _has_module("whisper"):
        backends.append("openai-whisper")
    return backends


def resolve_backend(preference: str) -> Optional[str]:
    """Return the backend to use, or ``None`` when transcription is disabled/unavailable."""
    if preference == "none":
        return None
    available = available_backends()
    if preference == "auto":
        return available[0] if available else None
    if preference not in available:
        package = "faster-whisper" if preference == "faster-whisper" else "openai-whisper"
        raise RuntimeError(f"The '{preference}' transcriber is not installed. Run: pip install {package}")
    return preference


class Transcriber:
    def __init__(self, backend: str, model: str = "base", language: str = "",
                 log: Callable[[str], None] = print,
                 cancel_event: Optional[threading.Event] = None, beam_size: int = 5,
                 vocabulary: Sequence[str] = (), profanity: bool = True) -> None:
        self.beam_size = max(1, int(beam_size))
        self.vocabulary = list(vocabulary)
        self.profanity = profanity
        self.backend = backend
        self.model_name = model
        self.language = language.strip() or None
        self.log = log
        self.cancel_event = cancel_event or threading.Event()
        self._model = None

    # -- model loading --------------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        self.log(f"Loading the '{self.model_name}' speech model ({self.backend}). "
                 "The first run downloads it, please be patient...")
        if self.backend == "faster-whisper":
            from faster_whisper import WhisperModel

            device, compute_type = "cpu", "int8"
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    device, compute_type = "cuda", "float16"
            except Exception:
                pass
            threads = max(1, (os.cpu_count() or 2) - 1)
            self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type,
                                       cpu_threads=threads)
        else:
            import whisper

            name = "large-v3" if self.model_name == "large-v3" else self.model_name
            self._model = whisper.load_model(name)

    # -- transcription --------------------------------------------------------------------
    def transcribe(self, path: str, duration: float,
                   on_progress: Optional[Callable[[float], None]] = None) -> Transcript:
        self._load()
        segments: List[Segment] = []
        language = self.language
        if language is None:
            language = self._detect_language(path, duration)
            if language:
                self.log(f"Detected language: {language}")
        start = 0.0
        while start < duration - 0.5:
            self._check_cancel()
            window = min(CHUNK_SECONDS + OVERLAP_SECONDS, duration - start)
            is_last = start + window >= duration - 0.5
            audio = _decode_pcm(path, start, window, self.cancel_event)

            def chunk_progress(t: float, _start=start) -> None:
                if on_progress and duration:
                    on_progress(min(1.0, (_start + min(t, CHUNK_SECONDS)) / duration))

            chunk, detected = self._transcribe_array(audio, language, chunk_progress)
            if language is None and detected:
                language = detected  # lock the language after the first window
                self.log(f"Detected language: {language}")

            limit = CHUNK_SECONDS
            kept_end = None
            for seg in chunk:
                if not is_last and seg.start >= limit:
                    break
                seg.embedding = voice_features(audio, seg.start, seg.end)
                segments.append(_offset_segment(seg, start))
                kept_end = seg.end
            if is_last:
                break
            next_start = start + CHUNK_SECONDS
            if kept_end is not None:
                next_start = max(next_start, start + kept_end)
            start = next_start
            if on_progress:
                on_progress(min(1.0, start / duration))

        if on_progress:
            on_progress(1.0)
        return Transcript(_clean_segments(segments), language or "", self.backend, self.model_name)

    def _transcribe_array(self, audio: np.ndarray, language: Optional[str],
                          on_progress: Callable[[float], None]):
        if audio.size < SAMPLE_RATE // 2:
            return [], None
        prompt = build_prompt(language, self.vocabulary, self.profanity)
        if self.backend == "faster-whisper":
            # No VAD: Silero drops real speech over game audio/music. Hallucinations in
            # silent parts are filtered afterwards with the loudness envelope instead.
            # "hotwords" is applied to every 30 s window (initial_prompt only to the first one).
            options = dict(language=language, word_timestamps=True, vad_filter=False,
                           condition_on_previous_text=False, beam_size=self.beam_size)
            try:
                raw_segments, info = self._model.transcribe(audio, hotwords=prompt, **options)
            except TypeError:  # faster-whisper < 1.0.2
                raw_segments, info = self._model.transcribe(audio, initial_prompt=prompt, **options)
            result = []
            for seg in raw_segments:  # lazy generator: decoding happens here
                self._check_cancel()
                if getattr(seg, "no_speech_prob", 0) > 0.8 and getattr(seg, "avg_logprob", 0) < -1.0:
                    continue
                words = [Word(w.start, w.end, w.word, w.probability) for w in (seg.words or [])]
                result.append(Segment(seg.start, seg.end, seg.text.strip(), words))
                on_progress(seg.end)
            return result, info.language

        import torch

        # openai-whisper only keeps initial_prompt in context when it conditions on previous text.
        output = self._model.transcribe(
            audio, language=language, word_timestamps=True, verbose=None, initial_prompt=prompt,
            condition_on_previous_text=True, fp16=torch.cuda.is_available(), beam_size=self.beam_size,
        )
        result = []
        for seg in output.get("segments", []):
            if seg.get("no_speech_prob", 0) > 0.8 and seg.get("avg_logprob", 0) < -1.0:
                continue
            words = [Word(w["start"], w["end"], w["word"], w.get("probability", 1.0))
                     for w in seg.get("words", [])]
            result.append(Segment(seg["start"], seg["end"], seg["text"].strip(), words))
        return result, output.get("language")

    def preview(self, path: str, duration: float, window: float = 30.0,
                on_progress: Optional[Callable[[float], None]] = None) -> Tuple[Optional[str], str]:
        """Quickly transcribe three short windows spread over the recording.

        Returns the dominant language (a vote, more robust than one sample) and the text heard,
        used to recognise the game or topic before the full transcription.
        """
        self._load()
        if duration <= window * 1.5:
            positions = [0.0]
        else:
            positions = [max(0.0, min(duration - window, duration * f)) for f in (0.2, 0.5, 0.8)]
        votes: Dict[str, float] = {}
        texts: List[str] = []
        for index, start in enumerate(positions):
            self._check_cancel()
            audio = _decode_pcm(path, start, min(window, duration), self.cancel_event)
            if audio.size >= SAMPLE_RATE:
                try:
                    text, language, confidence = self._quick_transcribe(audio)
                except Exception as exc:  # noqa: BLE001 - the preview is optional
                    self.log(f"Speech preview skipped ({exc}).")
                    text, language, confidence = "", None, 0.0
                if text:
                    texts.append(text)
                if language:
                    votes[language] = votes.get(language, 0.0) + confidence * max(1, len(text))
            if on_progress:
                on_progress((index + 1) / len(positions))
        return (max(votes, key=votes.get) if votes else None), " ".join(texts)

    def _quick_transcribe(self, audio: np.ndarray) -> Tuple[str, Optional[str], float]:
        if self.backend == "faster-whisper":
            segments, info = self._model.transcribe(audio, language=self.language, beam_size=1, vad_filter=False,
                                                    condition_on_previous_text=False, without_timestamps=True)
            text = " ".join(seg.text.strip() for seg in segments)
            return text, info.language, float(getattr(info, "language_probability", 1.0) or 1.0)
        import torch

        output = self._model.transcribe(audio, language=self.language, fp16=torch.cuda.is_available(),
                                        condition_on_previous_text=False, verbose=None)
        return output.get("text", "").strip(), output.get("language"), 1.0

    def _detect_language(self, path: str, duration: float) -> Optional[str]:
        """Detect the language on 30 s taken from the first quarter of the recording (intros are
        often silent or music only)."""
        start = max(0.0, min(duration * 0.25, duration - 30.0))
        audio = _decode_pcm(path, start, min(30.0, duration), self.cancel_event)
        if audio.size < SAMPLE_RATE:
            return None
        try:
            if self.backend == "faster-whisper":
                detect = getattr(self._model, "detect_language", None)
                if detect is not None:
                    language, _prob, _all = detect(audio=audio)
                    return language
                _segments, info = self._model.transcribe(audio, beam_size=1)
                return info.language
            import whisper

            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=self._model.dims.n_mels)
            mel = mel.to(self._model.device)
            _tokens, probs = self._model.detect_language(mel)
            return max(probs, key=probs.get)
        except Exception as exc:  # noqa: BLE001 - fall back to per-chunk detection
            self.log(f"Language detection skipped ({exc}).")
            return None

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled("Cancelled by user")


def _offset_segment(seg: Segment, offset: float) -> Segment:
    return Segment(
        seg.start + offset, seg.end + offset, seg.text,
        [Word(w.start + offset, w.end + offset, w.text, w.probability) for w in seg.words],
        embedding=seg.embedding,
    )


def _clean_segments(segments: List[Segment]) -> List[Segment]:
    """Drop empty segments and the repeated-phrase loops Whisper sometimes hallucinates."""
    cleaned: List[Segment] = []
    repeats = 0
    for seg in sorted(segments, key=lambda s: s.start):
        if not seg.text.strip() or seg.end <= seg.start:
            continue
        if cleaned and seg.text.strip().lower() == cleaned[-1].text.strip().lower():
            repeats += 1
            if repeats >= 2:
                continue
        else:
            repeats = 0
        if cleaned and seg.start < cleaned[-1].end - 0.5:
            # Overlap from a chunk boundary: skip exact duplicates.
            if seg.text.strip() == cleaned[-1].text.strip():
                continue
        seg.words = _drop_stutter(w for w in seg.words if w.text.strip() and w.end >= w.start)
        if segments and not seg.words:
            continue
        seg.text = _join_words(seg.words) if seg.words else seg.text
        cleaned.append(seg)
    return cleaned


def _drop_stutter(words) -> List[Word]:
    """Remove hallucinated repetitions: a word repeated 3+ times whose first copy has ~zero
    confidence ("fuck, fuck, fuck, fuck" over a shout), and cap real ones at 6 in a row."""
    words = list(words)
    result: List[Word] = []
    i = 0
    while i < len(words):
        key = _normalize_token(words[i].text)
        j = i
        while j < len(words) and _normalize_token(words[j].text) == key:
            j += 1
        run = words[i:j]
        if len(run) >= 3 and run[0].probability < 0.2:
            run = []
        result.extend(run[:6])
        i = j
    return result


def _normalize_token(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _decode_pcm(path: str, start: float, duration: float,
                cancel_event: Optional[threading.Event]) -> np.ndarray:
    proc = popen_binary(pcm_reader_command(path, SAMPLE_RATE, start, duration))
    try:
        data, err = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.kill()
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled("Cancelled by user")
    if proc.returncode != 0:
        raise FFmpegError(f"Audio decoding failed: {err.decode('utf-8', 'replace')[-800:]}")
    return np.frombuffer(data[: len(data) - len(data) % 4], dtype=np.float32).copy()


# -- cache --------------------------------------------------------------------------------------

def load_cached(path: Path) -> Optional[Transcript]:
    try:
        return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_cached(path: Path, transcript: Transcript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
