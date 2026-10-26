"""Audio/video sync of rendered clips on recordings that are hard to cut: variable frame rate video and
audio starting later than the video. A white flash and a beep happen together every 2 seconds."""

import shutil
import subprocess

import numpy as np
import pytest

from auto_video_editor.ffmpeg import probe
from auto_video_editor.render import Renderer

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

FLASH = "color=c=black:s=160x90:r=60:d=40,drawbox=c=white:t=fill:enable='lt(mod(t\\,2)\\,0.1)'"
BEEP = "sine=f=1000:r=48000:d=40,volume='if(lt(mod(t\\,2)\\,0.1)\\,1\\,0)':eval=frame"
ENCODE = ["-c:v", "libx264", "-preset", "ultrafast", "-g", "120", "-c:a", "aac"]


def _make(path, video_filter="", audio_delay=0.0):
    video = FLASH + video_filter
    audio = BEEP + (f",atrim=start={audio_delay},asetpts=PTS-STARTPTS" if audio_delay else "")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", video]
    if audio_delay:
        cmd += ["-itsoffset", str(audio_delay)]
    cmd += ["-f", "lavfi", "-i", audio, "-fps_mode", "passthrough", *ENCODE, str(path)]
    subprocess.run(cmd, check=True)
    return path


def _onsets(values, times, threshold):
    above = values > threshold
    return [times[i] for i in range(len(values)) if above[i] and (i == 0 or not above[i - 1])]


def _sync_errors(path):
    info = probe(str(path))
    frames = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=8:8,format=gray",
                             "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    luma = np.frombuffer(frames, np.uint8).reshape(-1, 64).mean(axis=1)
    flashes = _onsets(luma, np.arange(len(luma)) / info.fps, 128)
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    samples = np.frombuffer(pcm, np.float32)
    window = 240  # 5 ms
    rms = np.sqrt((samples[: len(samples) // window * window].reshape(-1, window) ** 2).mean(axis=1))
    beeps = _onsets(rms, np.arange(len(rms)) * window / 48000, 0.05)
    assert len(flashes) >= 3 and abs(len(flashes) - len(beeps)) <= 1, (flashes, beeps)
    return [min(abs(b - f) for b in beeps) for f in flashes]


@pytest.mark.parametrize("kind", ["vfr", "late_audio"])
def test_clips_stay_in_sync(tmp_path, kind):
    if kind == "vfr":  # dropped frames, as in OBS recordings under load or phone videos
        source = _make(tmp_path / "vfr.mp4", video_filter=",select='not(between(mod(n\\,113)\\,3\\,40))"
                                                              "+lt(mod(t\\,2)\\,0.1)'")
    else:  # audio stream starting 0.4 s after the video, as in Twitch VODs (MPEG-TS)
        source = _make(tmp_path / "late.ts", audio_delay=0.4)
    renderer = Renderer(probe(str(source)), "libx264", "draft")
    # A clip starting just before a flash, and one with a jump cut and a teaser going back in time.
    for ranges in ([(7.3, 13.2)], [(30.3, 34.9), (12.2, 15.5), (17.1, 21.9)]):
        out = tmp_path / "clip.mp4"
        renderer.render_segment(ranges, out)
        errors = _sync_errors(out)
        assert max(errors) < 0.06, f"{kind} {ranges}: audio/video offsets {[round(e, 3) for e in errors]}"
