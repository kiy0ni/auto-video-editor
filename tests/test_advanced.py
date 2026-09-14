import pytest

from auto_video_editor.captions import TimedWord, ass_color, build_ass
from auto_video_editor.cli import build_parser, settings_from_args
from auto_video_editor.ffmpeg import MediaInfo
from auto_video_editor.moments import Moment, build_curve, select_highlight
from auto_video_editor.render import vertical_filter
from auto_video_editor.settings import Settings


def test_content_presets():
    s = Settings()
    s.apply_content_preset("talk")
    assert s.content_type == "talk" and s.jump_cuts and s.shorts_layout == "auto", "auto layout stays auto"
    assert s.validate() == []
    fixed = Settings(shorts_layout="blur")
    fixed.apply_content_preset("talk")
    assert fixed.shorts_layout == "smart"
    split = Settings(shorts_layout="split", facecam="0.7,0,0.3,0.3")
    split.apply_content_preset("gaming")
    assert split.shorts_layout == "split", "a configured facecam layout is kept"
    with pytest.raises(ValueError):
        s.apply_content_preset("cooking")


def test_with_value_parses_text():
    s = Settings().with_value("caption_uppercase", "false")
    assert s.caption_uppercase is False
    s = s.with_value("caption_size", "130").with_value("keywords", "wow, no way")
    assert s.caption_size == 130 and s.keywords == ["wow", "no way"]
    with pytest.raises(ValueError):
        s.with_value("nope", "1")
    with pytest.raises(ValueError):
        s.with_value("caption_size", "big")


def test_advanced_validation():
    s = Settings(diversity=2, caption_color="yellow", weight_loudness=0, weight_spikes=0, weight_speech=0,
                 weight_keywords=0, weight_exclamations=0, silence_threshold=-5)
    errors = " ".join(s.validate())
    for word in ("diversity", "caption_color", "scoring weight", "silence_threshold"):
        assert word in errors


def test_keyword_weight_changes_scores(envelope_factory, transcript_factory):
    env = envelope_factory()
    transcript = transcript_factory([(200, 204, "oh my god no way")])
    boosted = build_curve(env, transcript, 300, ["oh my god"], weights={"keywords": 2.0})
    muted = build_curve(env, transcript, 300, ["oh my god"], weights={"keywords": 0.0})
    assert boosted.raw[201] > muted.raw[201]


def test_diversity_spreads_selection():
    candidates = [Moment(100 + i * 12, 110 + i * 12, score=100 - i) for i in range(5)] + [Moment(3000, 3010, score=60)]
    greedy = select_highlight(candidates, 20, 3600, diversity=0.0)
    spread = select_highlight(candidates, 20, 3600, diversity=0.9)
    assert all(m.start < 1000 for m in greedy)
    assert any(m.start >= 3000 for m in spread)


def test_caption_options():
    words = [TimedWord(0, 0.4, "hello"), TimedWord(0.5, 0.9, "world")]
    ass = build_ass(words, uppercase=False, highlight="#00FF00", position="top", size_scale=1.5, max_words=1)
    assert "HELLO" not in ass and "hello" in ass
    assert "\\c&H00FF00&" in ass
    style = next(line for line in ass.splitlines() if line.startswith("Style:")).split(",")
    assert style[18] == "8"  # top alignment
    assert int(style[2]) == int(1920 * 0.047 * 1.5)
    assert ass.count("Dialogue:") == 2
    assert ass_color("#FFE600") == "&H00E6FF&"


def test_blur_strength_in_filter():
    media = MediaInfo(path="x.mp4", duration=10, has_video=True, has_audio=True, width=1920, height=1080)
    assert "boxblur=25:2" in vertical_filter("[a]", "[b]", media, "blur", blur=25)


def test_cli_preset_and_set():
    args = build_parser().parse_args(["in.mp4", "--preset", "talk", "--set", "caption_size=140",
                                      "--set", "diversity=0.5"])
    s = settings_from_args(args)
    assert s.content_type == "talk" and s.caption_size == 140 and s.diversity == 0.5
    with pytest.raises(ValueError):
        settings_from_args(build_parser().parse_args(["in.mp4", "--set", "caption_size"]))


# -- transcription context, censoring, speakers ---------------------------------------------------

def test_prompt_mentions_vocabulary_and_profanity():
    from auto_video_editor.transcribe import build_prompt
    prompt = build_prompt("fr", ["Minecraft", "creeper"], profanity=True)
    assert "Vocabulaire: Minecraft, creeper." in prompt and "Putain" in prompt
    assert "Putain" not in build_prompt("fr", [], profanity=False)
    assert "Vocabulary: mob." in build_prompt(None, ["mob"])


def test_censor():
    from auto_video_editor.censor import censor_text, censor_word, is_profane, profanity_set
    words = profanity_set(["chelou"])
    assert censor_word("putain") == "p*tain" and censor_word("merde") == "m*rde" and censor_word("shit") == "sh*t"
    assert censor_text("Putain, c'est trop chelou ce truc de merde !", words) == "P*tain, c'est trop ch*lou ce truc de m*rde !"
    assert censor_text("Nice play, well done.", words) == "Nice play, well done."
    assert is_profane("ENCULÉS", words) and not is_profane("bonjour", words)


def test_speaker_detection_separates_two_voices():
    import numpy as np
    from auto_video_editor.speakers import SR, assign_speakers, voice_features
    from auto_video_editor.transcribe import Segment, Transcript, Word

    rng = np.random.default_rng(1)

    def voice(f0, decay, seconds):
        t = np.arange(int(seconds * SR)) / SR
        vibrato = f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
        phase = 2 * np.pi * np.cumsum(vibrato) / SR
        signal = sum((decay ** k) * np.sin(k * phase) for k in range(1, 12))
        return (signal / np.abs(signal).max() * 0.5 + rng.normal(0, 0.01, t.size)).astype(np.float32)

    segments, audio_parts, cursor = [], [], 0.0
    for i in range(16):
        low_voice = i % 2 == 0
        clip = voice(110 if low_voice else 210, 0.85 if low_voice else 0.6, 1.5)
        audio_parts.append(clip)
        seg = Segment(cursor, cursor + 1.5, "bla", [Word(cursor, cursor + 1.5, " bla")])
        segments.append(seg)
        cursor += 1.5
    audio = np.concatenate(audio_parts)
    for seg in segments:
        seg.embedding = voice_features(audio, seg.start, seg.end)
    assert all(seg.embedding is not None for seg in segments)

    count = assign_speakers(segments, count=0)
    assert count == 2
    labels = [seg.speaker for seg in segments]
    assert labels[::2] == [labels[0]] * 8 and labels[1::2] == [labels[1]] * 8 and labels[0] != labels[1]

    transcript = Transcript(segments)
    for seg in segments:
        seg.set_speaker(seg.speaker)
    assert transcript.text_between(0, 3).startswith("A: bla  B: bla")
    assert assign_speakers(segments, count=1) == 1


def test_captions_color_by_speaker():
    from auto_video_editor.captions import SPEAKER_COLORS, TimedWord, ass_color, build_ass, group_words
    words = [TimedWord(0, 0.4, "hey", 0), TimedWord(0.5, 0.9, "you", 0), TimedWord(1.0, 1.4, "what", 1)]
    assert [[w.text for w in g] for g in group_words(words)] == [["hey", "you"], ["what"]]
    ass = build_ass(words, speaker_colors=True)
    blue = ass_color(SPEAKER_COLORS[1])
    what_line = next(line for line in ass.splitlines() if "WHAT" in line)
    assert what_line.endswith(",{\\c" + blue + "}{\\c&H00E6FF&\\fscx112\\fscy112\\t(0,90,\\fscx100\\fscy100)}WHAT{\\r\\c" + blue + "}")
    hey_line = next(line for line in ass.splitlines() if "HEY" in line)
    assert blue not in hey_line
    assert ass_color(SPEAKER_COLORS[1]) not in build_ass(words, speaker_colors=False)


def test_calm_talking_is_not_a_moment(envelope_factory):
    """A quiet intro where someone only talks must not beat real action elsewhere."""
    from auto_video_editor.moments import build_curve, find_candidates
    env = envelope_factory(duration=600, base=-25, loud=[(300, 304)], loud_db=-8)
    # Intro: silence, then 40 s of speech at a normal level (louder than the silence, quieter than the game).
    env.rms_db[: int(60 / env.hop)] = -70
    env.rms_db[int(60 / env.hop): int(100 / env.hop)] = -32
    curve = build_curve(env, None, 600)
    candidates = find_candidates(curve, env, None, 600, 8, 60, 10)
    best = candidates[0]
    assert best.start <= 300 <= best.end
    intro = [c for c in candidates if c.end <= 110]
    assert all(c.score < 20 for c in intro), "the calm intro must score far below the action"
    skipped = build_curve(env, None, 600, skip=(120, 0))
    assert skipped.raw[:120].max() == 0
