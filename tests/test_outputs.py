import xml.etree.ElementTree as ET

from auto_video_editor.captions import TimedWord, build_ass, build_srt, group_words, remap_words
from auto_video_editor.exporters import frames_to_timecode, timebase, timecode_to_frames, write_edl, write_fcp_xml
from auto_video_editor.ffmpeg import MediaInfo
from auto_video_editor.render import Renderer, parse_loudnorm, vertical_filter
from auto_video_editor.settings import Settings
from auto_video_editor.timeline import TimeMap

MEDIA = MediaInfo(path="/videos/my stream.mp4", duration=3600, has_video=True, has_audio=True,
                  width=1920, height=1080, fps=30000 / 1001, audio_channels=2, sample_rate=48000)


# -- captions ---------------------------------------------------------------------------------

def test_group_words_breaks_on_pauses_and_punctuation():
    words = [TimedWord(0, 0.3, "Oh"), TimedWord(0.3, 0.6, "no!"), TimedWord(0.7, 1.0, "What"),
             TimedWord(2.5, 2.8, "happened")]
    groups = group_words(words)
    assert [[w.text for w in g] for g in groups] == [["Oh", "no!"], ["What"], ["happened"]]


def test_remap_words_drops_cut_words():
    words = [[1.0, 1.5, " hello"], [5.0, 5.5, " cut"], [10.2, 10.6, " world"]]
    mapped = remap_words(words, TimeMap([(0, 3), (10, 12)]))
    assert [w.text for w in mapped] == ["hello", "world"]
    assert abs(mapped[1].start - 3.2) < 1e-6


def test_srt_and_ass_are_well_formed():
    words = [TimedWord(i * 0.4, i * 0.4 + 0.35, w) for i, w in enumerate("this is a {weird} test sentence".split())]
    srt = build_srt(words)
    assert srt.startswith("1\n00:00:00,000 --> ")
    ass = build_ass(words)
    assert "[Events]" in ass and "PlayResY: 1920" in ass
    assert "{WEIRD}" not in ass  # braces are escaped so they are not parsed as override tags


# -- exporters ----------------------------------------------------------------------------------

def test_timecodes():
    assert timebase(30000 / 1001) == (30, True)
    assert timebase(25) == (25, False)
    assert frames_to_timecode(30 * 3661 + 5, 30) == "01:01:01:05"
    assert timecode_to_frames("01:00:00:00", 25) == 90000


def test_edl_and_xml(tmp_path):
    events = [(10.0, 20.0), (100.0, 104.5)]
    edl = tmp_path / "cut.edl"
    write_edl(edl, "My cut", MEDIA, events)
    lines = edl.read_text().splitlines()
    assert lines[0] == "TITLE: My cut"
    # 29.97 fps non-drop-frame: 20 s of wall clock = 599 frames = 00:00:19:29.
    assert lines[3] == "001  AX       AA/V  C        00:00:10:00 00:00:19:29 00:00:00:00 00:00:09:29"
    assert "* FROM CLIP NAME: my stream.mp4" in lines

    xml = tmp_path / "cut.xml"
    write_fcp_xml(xml, "My cut", MEDIA, events)
    root = ET.fromstring(xml.read_text().split("\n", 2)[2])
    clips = root.findall(".//video/track/clipitem")
    assert len(clips) == 2
    assert clips[1].find("start").text == clips[0].find("end").text
    assert root.find(".//pathurl").text == "file:///videos/my%20stream.mp4"
    assert len(root.findall(".//audio/track")) == 2


# -- render -------------------------------------------------------------------------------------

def test_segment_command_with_jump_cuts_and_vertical_layout():
    renderer = Renderer(MEDIA, "libx264", "draft")
    args = renderer.build_segment_command([(100, 105), (106, 110)], "out.mp4", layout="blur",
                                          captions_file="c.ass", audio_filter="loudnorm=I=-14")
    graph = args[args.index("-filter_complex") + 1]
    assert args[args.index("-ss") + 1] == "100.000"
    assert "concat=n=2:v=1:a=1" in graph
    assert "fps=30000/1001" in graph
    assert "ass=c.ass" in graph and "loudnorm=I=-14" in graph
    assert args[-1] == "out.mp4"
    assert args[args.index("-t", args.index("-filter_complex")) + 1] == "9.000"


def test_vertical_filters():
    crop = vertical_filter("[in]", "[out]", MEDIA, "smart", subject_x=0.9)
    assert crop.startswith("[in]crop=608:1080:1312:0")
    split = vertical_filter("[in]", "[out]", MEDIA, "split", facecam=(0.75, 0.0, 0.25, 0.3))
    assert "vstack" in split and "crop=480:324:1440:0" in split


def test_parse_loudnorm():
    stderr = 'noise\n[Parsed_loudnorm_0 @ 0x1]\n{\n "input_i" : "-23.5",\n "input_tp" : "-3.0",\n' \
             ' "input_lra" : "5.0",\n "input_thresh" : "-34.0",\n "target_offset" : "0.2"\n}\n'
    assert parse_loudnorm(stderr)["input_i"] == "-23.5"
    assert parse_loudnorm("nothing here") is None


# -- settings -----------------------------------------------------------------------------------

def test_settings_targets_and_validation():
    s = Settings()
    assert s.target_for(3 * 3600) == 18 * 60        # 10% of 3h
    assert s.target_for(10 * 60) == 2 * 60           # medium minimum
    assert s.target_for(60) == 30                    # never more than half the source
    s.profile = "long"
    assert s.target_for(10 * 3600) == 40 * 60        # long maximum
    s.shorts_layout = "split"
    assert any("facecam" in e for e in s.validate())
    s.facecam = "0.7,0.0,0.3,0.3"
    assert s.validate() == []
    restored = Settings.from_dict({**s.to_dict(), "unknown": 1, "min_clip": "oops"})
    assert restored.facecam == s.facecam and restored.min_clip == Settings().min_clip
