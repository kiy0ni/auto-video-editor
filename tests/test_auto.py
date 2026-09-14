"""Automatic mode: detection helpers and the settings they decide."""

import json

import pytest

from auto_video_editor.captions import remap_words_sequence
from auto_video_editor.cli import build_parser, settings_from_args
from auto_video_editor.ffmpeg import MediaInfo
from auto_video_editor.glossary import detect_game, game_vocabulary, hashtags, hype_phrases, is_laughter
from auto_video_editor.insights import (VideoInsights, auto_shorts_count, auto_target, auto_trim, classify_content,
                                        describe_position, detect_facecam)
from auto_video_editor.moments import Moment, build_curve, compute_hook
from auto_video_editor.publish import suggest_title
from auto_video_editor.render import Renderer
from auto_video_editor.settings import Settings

MEDIA = MediaInfo(path="/v/stream.mp4", duration=3600, has_video=True, has_audio=True, width=1920, height=1080,
                  fps=60)


# -- glossary ----------------------------------------------------------------------------------------

def test_game_detection():
    assert detect_game(["Minecraft Hardcore Ep 12"], min_score=5)[0] == "Minecraft"
    assert detect_game(["on a vu un creeper dans le nether, puis un enderman"])[0] == "Minecraft"
    assert detect_game(["lol that was so funny"])[0] is None
    assert detect_game([""])[0] is None
    assert "creeper" in game_vocabulary("Minecraft") and game_vocabulary(None) == []


def test_phrases_laughter_and_tags():
    french = hype_phrases("fr")
    assert "c'est chaud" in french and "no way" in french
    assert is_laughter("Hahaha!") and is_laughter("MDR") and is_laughter("ptdrrr") and not is_laughter("hello")
    tags = hashtags("Minecraft", "gaming", "fr")
    assert tags[0] == "#shorts" and "#minecraft" in tags and "#twitchfr" in tags


def test_laughter_counts_as_a_hype_moment(envelope_factory, transcript_factory):
    env = envelope_factory()
    transcript = transcript_factory([(100, 103, "hahaha hahaha trop drole")])
    curve = build_curve(env, transcript, 300, keywords=[])
    assert "laughter" in curve.keyword_hits.values()


# -- picture ----------------------------------------------------------------------------------------

def test_facecam_detection():
    frames = []
    for i in range(20):
        boxes = [(0.84, 0.06, 0.06, 0.1)]
        if i % 3 == 0:
            boxes.append((0.3 + i * 0.01, 0.5, 0.05, 0.09))  # a character's face moving in the game
        frames.append(boxes)
    frames += [[] for _ in range(10)]
    rect = detect_facecam(frames, 1920, 1080)
    assert rect is not None
    x, y, w, h = rect
    assert x + w == pytest.approx(1.0, abs=0.01) and y == 0.0, "snapped to the top-right corner"
    assert x <= 0.84 and x + w >= 0.9
    assert describe_position(rect) == "top-right"
    assert detect_facecam([[(0.4, 0.3, 0.25, 0.4)]] * 20, 1920, 1080) is None, "big centered face = main shot"
    assert detect_facecam([[(0.1 * (i % 9), 0.5, 0.05, 0.08)] for i in range(20)], 1920, 1080) is None


def test_facecam_found_despite_full_screen_camera_moments():
    frames = []
    for i in range(48):
        if i % 2:
            frames.append([(0.3, 0.1, 0.35, 0.6)])      # camera full screen: big centered face
        else:
            frames.append([(0.85, 0.05, 0.05, 0.09)])   # webcam overlay in the top-right corner
    rect = detect_facecam(frames, 1920, 1080)
    assert rect is not None and describe_position(rect) == "top-right"


def test_overlay_borders_are_found():
    import numpy as np
    from auto_video_editor.insights import refine_overlay

    rng = np.random.default_rng(3)
    room = rng.integers(40, 200, size=(120, 160)).astype(np.uint8)   # static webcam picture
    frames = []
    for _ in range(12):
        frame = rng.integers(0, 256, size=(270, 480)).astype(np.uint8)  # the game changes all the time
        frame[0:120, 300:460] = room
        frame[40:75, 380:410] = rng.integers(90, 110, size=(35, 30))    # the face moves a little
        frames.append(frame)
    face = (380 / 480, 40 / 270, 30 / 480, 35 / 270)
    rect = refine_overlay(frames, face, [0.5, 0.0, 0.5, 0.6])
    assert rect == pytest.approx([300 / 480, 0.0, 160 / 480, 120 / 270], abs=0.01)
    assert refine_overlay(frames[:2], face, [0.5, 0.0, 0.5, 0.6]) == [0.5, 0.0, 0.5, 0.6], "not enough frames"


def test_content_classification():
    talk = VideoInsights(motion=0.01, faces_checked=True, face_ratio=0.9, face_size=0.2)
    stats = {"speech_ratio": 0.8, "background_ratio": 0.05, "active_ratio": 0.9}
    assert classify_content(talk, stats, 2, 3600, None)[0] == "talk"
    assert classify_content(VideoInsights(facecam=[0.7, 0, 0.3, 0.3]), stats, 1, 3600, None)[0] == "gaming"
    game_audio = {"speech_ratio": 0.2, "background_ratio": 0.6, "active_ratio": 0.9}
    assert classify_content(VideoInsights(), game_audio, 1, 3600, None)[0] == "gaming"
    vlog = VideoInsights(motion=0.08, faces_checked=True, face_ratio=0.7, face_size=0.15)
    assert classify_content(vlog, {"speech_ratio": 0.4, "background_ratio": 0.25, "active_ratio": 0.8}, 1, 900,
                            None)[0] == "vlog"
    assert classify_content(VideoInsights(), stats, 1, 100, "Minecraft") == ("gaming", "Minecraft detected")


# -- decisions ----------------------------------------------------------------------------------------

def test_auto_trim(transcript_factory):
    sentences = [(t, t + 10, "un deux trois quatre cinq six sept huit neuf dix") for t in range(300, 3300, 10)]
    start, end = auto_trim(transcript_factory(sentences), 3600)
    assert 200 < start < 300 and 200 < end < 300
    talking_from_start = [(t, t + 10, "un deux trois quatre cinq six sept huit neuf dix") for t in range(0, 3600, 10)]
    assert auto_trim(transcript_factory(talking_from_start), 3600) == (0.0, 0.0)


def test_auto_length_and_count():
    strong = [Moment(i * 300, i * 300 + 20, score=80) for i in range(10)]
    assert auto_target(strong, 7200, 20) == pytest.approx(180)
    assert auto_target([Moment(0, 20, score=30)], 7200, 20) == 60, "never below a minute"
    assert auto_target(strong * 20, 1800, 20) == pytest.approx(360), "capped to 20% of a 30 min video"
    assert auto_shorts_count(strong, 7200) == 10
    assert auto_shorts_count(strong, 600) == 3
    assert auto_shorts_count([Moment(0, 20, score=30)], 7200) == 1


def test_merge_auto_keeps_explicit_choices():
    resolved = Settings(content_type="talk", shorts_layout="split", facecam="0.7,0,0.3,0.3", target_duration=300,
                        shorts_count=4, jump_cuts=True, whisper_model="small").to_dict()
    merged = Settings.merge_auto(Settings(), resolved)
    assert (merged.shorts_layout, merged.target_duration, merged.shorts_count, merged.jump_cuts) == \
        ("split", 300, 4, True)
    explicit = Settings.merge_auto(Settings(shorts_layout="crop", shorts_auto=False, shorts_count=2), resolved)
    assert explicit.shorts_layout == "crop" and explicit.shorts_count == 2


def test_preset_only_fills_defaults():
    s = Settings(min_clip=12)
    s.fill_from_preset("talk")
    assert s.min_clip == 12 and s.jump_cuts and s.shorts_layout == "auto"


def test_old_settings_are_migrated(tmp_path):
    old = Settings(content_type="gaming", profile="medium", shorts_layout="blur", whisper_model="small",
                   caption_size=130).to_dict()
    old["keywords"] = ["no way", "wow", "my catchphrase"]
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(old))
    s = Settings.load(path)
    assert (s.content_type, s.profile, s.shorts_layout, s.whisper_model) == ("auto", "auto", "auto", "auto")
    assert s.caption_size == 130 and s.keywords == ["my catchphrase"]
    s.save(path)
    assert Settings.load(path).caption_size == 130


# -- shorts hook ----------------------------------------------------------------------------------------

def test_hook():
    words = [[114.2, 114.9, "no"], [117.3, 118.0, "way"]]
    assert compute_hook(100, 130, 115, words) == [(113.8, 116.8)]
    assert compute_hook(100, 110, 105, words) == [], "too short"
    assert compute_hook(100, 130, 102, words) == [], "already starts on its peak"
    sequence = remap_words_sequence([[10, 10.5, "hi"], [20, 20.5, "yo"]], [(19.8, 21), (9, 21)])
    assert [w.text for w in sequence] == ["yo", "hi", "yo"]
    assert sequence[1].start == pytest.approx(1.2 + 1.0)


def test_hook_renders_from_a_second_input():
    args = Renderer(MEDIA, "libx264", "draft").build_segment_command([(130, 133), (100, 125), (126, 140)], "o.mp4")
    assert args.count("-i") == 2
    graph = args[args.index("-filter_complex") + 1]
    assert "[0:v:0]trim=start=0.000:end=3.000" in graph
    assert "[1:v:0]split=2" in graph and "concat=n=3" in graph


# -- publishing -------------------------------------------------------------------------------------------

def test_titles():
    text = "A: we are just walking around here.  B: No way, that's insane! ok"
    assert suggest_title(text, ["no way"]) == "No way, that's insane!"
    long = "this is a very long sentence that keeps going and going without any punctuation at all for sure"
    title = suggest_title(long)
    assert len(title) <= 71 and title.endswith("…") and title[0] == "T"
    assert suggest_title("") == ""


def test_cli_auto_flags():
    s = settings_from_args(build_parser().parse_args(["a.mp4", "b.mp4", "--shorts", "3", "--hook"]))
    assert s.shorts_count == 3 and not s.shorts_auto and s.shorts_hook
    s = settings_from_args(build_parser().parse_args(["a.mp4", "--shorts", "auto", "--no-publish-kit"]))
    assert s.shorts_auto and not s.publish_kit
    assert Settings().content_type == "auto" and Settings().profile == "auto"
    with pytest.raises(ValueError):
        settings_from_args(build_parser().parse_args(["a.mp4", "--shorts", "many"]))
