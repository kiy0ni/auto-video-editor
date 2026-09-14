import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_video_editor.audio import AudioEnvelope  # noqa: E402
from auto_video_editor.transcribe import Segment, Transcript, Word  # noqa: E402


def make_envelope(duration=300.0, base=-30.0, loud=(), quiet=(), loud_db=-8.0, seed=0):
    hop = 0.05
    frames = int(duration / hop)
    rng = np.random.default_rng(seed)
    rms = base + rng.normal(0, 1.0, frames)
    for a, b in loud:
        rms[int(a / hop):int(b / hop)] = loud_db
    for a, b in quiet:
        rms[int(a / hop):int(b / hop)] = -70.0
    rms = rms.astype(np.float32)
    return AudioEnvelope(hop, rms, rms + 6)


def make_transcript(sentences):
    """sentences: list of (start, end, text); words are spread evenly."""
    segments = []
    for start, end, text in sentences:
        tokens = text.split()
        step = (end - start) / len(tokens)
        words = [Word(start + i * step, start + (i + 1) * step - 0.05, " " + t) for i, t in enumerate(tokens)]
        segments.append(Segment(start, end, text, words))
    return Transcript(segments, "en", "test", "test")


@pytest.fixture
def envelope_factory():
    return make_envelope


@pytest.fixture
def transcript_factory():
    return make_transcript
