"""Choosing the audio track that is analyzed and the one heard in the output (multi-track recordings)."""

import shutil
import subprocess

import numpy as np
import pytest

from auto_video_editor.audio import analyze_audio
from auto_video_editor.ffmpeg import MediaInfo, audio_input, pcm_reader_command, probe
from auto_video_editor.render import Renderer
from auto_video_editor.settings import Settings

MEDIA = MediaInfo(path="obs.mkv", duration=600, has_video=True, has_audio=True, width=1920, height=1080, fps=60,
                  audio_tracks=["Track 1 · Mix (stereo)", "Track 2 · Mic (mono)", "Track 3 · Game (stereo)"])


def test_audio_input():
    assert audio_input(0, 1, 3) == "[0:a:0]"
    assert audio_input(2, 3, 3) == "[2:a:2]"
    assert audio_input(0, 0, 3) == "[0:a:0][0:a:1][0:a:2]amix=inputs=3:normalize=0,"
    assert audio_input(0, 0, 1) == "[0:a:0]", "nothing to mix with a single track"
    assert audio_input(0, 5, 2) == "[0:a:1]", "never points at a track the file does not have"


def test_pcm_reader_reads_the_chosen_track():
    cmd = pcm_reader_command("in.mkv", 16000, track=2, track_count=3)
    assert cmd[cmd.index("-filter_complex") + 1] == "[0:a:1]anull[pcm]"


def test_render_uses_the_output_track():
    for track, expected in ((2, "[0:a:1]aresample"), (0, "amix=inputs=3:normalize=0,aresample")):
        renderer = Renderer(MEDIA, "libx264", "draft", audio_track=track)
        graph = renderer.build_segment_command([(100, 110)], "o.mp4")
        assert expected in graph[graph.index("-filter_complex") + 1]


def test_track_settings_validation():
    assert Settings().audio_track == 1 and Settings().output_audio_track == 1
    assert Settings(audio_track=0, output_audio_track=4).validate() == []
    assert any("audio_track" in e for e in Settings(audio_track=-1).validate())
    assert Settings().with_value("output_audio_track", "0").output_audio_track == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_multitrack_file(tmp_path):
    """Track 1 is quiet, track 2 (the mic) is loud from 4 s to 6 s."""
    path = tmp_path / "obs.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=160x90:r=30:d=10",
        "-f", "lavfi", "-i", "sine=f=220:r=48000:d=10,volume=0.01",
        "-f", "lavfi", "-i", "sine=f=440:r=48000:d=10,volume='if(between(t,4,6),1,0.001)':eval=frame",
        "-map", "0", "-map", "1", "-map", "2", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
        "-metadata:s:a:1", "title=Mic", str(path),
    ], check=True)
    info = probe(str(path))
    assert info.track_count == 2
    assert info.audio_tracks[0].startswith("Track 1") and info.audio_tracks[1].startswith("Track 2 · Mic")

    loud = lambda env: float(np.mean(env.window(4.2, 5.8))) - float(np.mean(env.window(1, 3)))  # noqa: E731
    assert loud(analyze_audio(info, track=2)) > 20
    assert abs(loud(analyze_audio(info, track=1))) < 6
    assert loud(analyze_audio(info, track=0)) > 20, "the mix contains the mic"

    for track in (2, 0):
        out = tmp_path / f"out{track}.mp4"
        Renderer(info, "libx264", "draft", audio_track=track).render_segment([(3.0, 7.0)], out)
        assert probe(str(out)).has_audio
