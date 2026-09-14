"""End-to-end run on a generated video. Requires ffmpeg; skipped otherwise."""

import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from auto_video_editor.ffmpeg import probe
from auto_video_editor.pipeline import Pipeline, Project
from auto_video_editor.reporting import Reporter
from auto_video_editor.settings import Settings

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=40",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
        "-filter_complex", "[1:a]volume='if(between(t,20,24),1.0,0.01)':eval=frame[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    return path


def test_full_pipeline(sample_video, tmp_path, monkeypatch):
    monkeypatch.setattr("auto_video_editor.pipeline.cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr("auto_video_editor.settings.cache_dir", lambda: tmp_path / "cache")
    settings = Settings(
        transcriber="none", target_duration=10, min_clip=4, max_clip=20,
        shorts_count=1, shorts_auto=False, shorts_min=5, shorts_max=12, shorts_layout="crop",
        encoder="libx264", quality="draft", use_cache=False,
    )
    logs, progress = [], []
    reporter = Reporter(logs.append, lambda value, label: progress.append(value))
    pipeline = Pipeline(str(sample_video), settings, str(tmp_path / "out"), reporter)

    project = pipeline.analyze()
    assert project.highlight and project.shorts
    assert any(m.start <= 22 <= m.end for m in project.highlight), "the loud section should be selected"
    assert project.resolved["content_type"] in ("gaming", "talk", "vlog")
    assert project.resolved["shorts_layout"] == "crop", "an explicit layout is never replaced"

    # The project file round-trips and can be rendered after manual edits.
    project = Project.load(project.path)
    progress.clear()
    result = pipeline.render(project)

    assert result.highlight.exists()
    highlight = probe(str(result.highlight))
    assert 4 <= highlight.duration <= 15 and highlight.has_audio
    short = probe(str(result.shorts[0]))
    assert (short.width, short.height) == (1080, 1920)
    assert result.highlight.with_suffix(".edl").exists()
    ET.parse(result.highlight.with_suffix(".xml"))
    assert not (tmp_path / "out" / ".work").exists()
    assert list((tmp_path / "out").glob("*_thumbnail.jpg")) and list((tmp_path / "out").glob("*_publish.txt"))
    assert result.shorts[0].with_suffix(".jpg").exists() and result.shorts[0].with_suffix(".txt").exists()
    assert progress and progress[-1] == 1.0
    assert all(b >= a for a, b in zip(progress, progress[1:])), "progress must never go backwards"


def test_scan_video(sample_video):
    from auto_video_editor.insights import scan_video

    insights = scan_video(probe(str(sample_video)), motion_samples=6, face_samples=6)
    assert insights.motion > 0, "testsrc2 moves"
    assert insights.facecam is None
