from auto_video_editor.moments import (Moment, build_curve, compute_keep_ranges, find_candidates, pad_moments,
                                       plan_shorts, select_highlight)


def test_finds_loud_event_without_transcript(envelope_factory):
    env = envelope_factory(loud=[(100, 106)])
    curve = build_curve(env, None, 300)
    candidates = find_candidates(curve, env, None, 300, 8, 60, 10)
    assert candidates, "the loud event should produce a candidate"
    best = candidates[0]
    assert best.start <= 100 and best.end >= 106
    assert best.score == 100
    assert "loud" in best.reasons
    ordered = sorted(candidates, key=lambda m: m.start)
    assert all(a.end <= b.start for a, b in zip(ordered, ordered[1:])), "candidates must not overlap"


def test_cuts_snap_to_sentence_boundaries(envelope_factory, transcript_factory):
    env = envelope_factory(loud=[(100, 106)])
    transcript = transcript_factory([
        (90, 97, "okay so we go in right now"),
        (98, 103, "no way no way"),
        (104, 113, "that was absolutely insane chat did you see it"),
    ])
    curve = build_curve(env, transcript, 300, keywords=["no way"])
    best = find_candidates(curve, env, transcript, 300, 8, 60, 10)[0]
    assert best.start == 90
    assert best.end == 113
    assert '"no way"' in best.reasons
    assert "no way" in best.text


def test_keywords_boost_quiet_moments(envelope_factory, transcript_factory):
    env = envelope_factory()
    transcript = transcript_factory([(60, 64, "just walking around here"), (200, 204, "oh my god no way")])
    curve = build_curve(env, transcript, 300, keywords=["oh my god"])
    best = find_candidates(curve, env, transcript, 300, 8, 60, 10)[0]
    assert best.start <= 200 <= best.end


def test_select_highlight_respects_target_and_order():
    candidates = [Moment(i * 60, i * 60 + 10, score=100 - i) for i in range(10)]
    chosen = select_highlight(candidates, target=30, duration=600, pad_before=0.5, pad_after=0.5)
    assert len(chosen) == 3
    assert [m.start for m in chosen] == sorted(m.start for m in chosen)
    assert sum(m.duration for m in chosen) <= 30 * 1.1 + 3


def test_pad_moments_never_overlaps():
    moments = [Moment(10, 20), Moment(20.2, 30)]
    padded = pad_moments(moments, 40, before=1.0, after=1.0)
    assert padded[0].start == 9
    assert padded[0].end <= padded[1].start
    assert padded[1].end == 31


def test_plan_shorts_extends_short_moments(envelope_factory, transcript_factory):
    env = envelope_factory(loud=[(100, 106), (200, 204)])
    transcript = transcript_factory([(95, 110, "what is happening here this is crazy")])
    curve = build_curve(env, transcript, 300)
    candidates = find_candidates(curve, env, transcript, 300, 6, 60, 10)
    shorts = plan_shorts(candidates, curve, transcript, env, 300, count=2, min_len=20, max_len=40)
    assert len(shorts) == 2
    for short in shorts:
        assert 15 <= short.duration <= 40
    assert shorts[0].end <= shorts[1].start or shorts[1].end <= shorts[0].start


def test_jump_cuts_remove_dead_air_but_keep_words(envelope_factory):
    env = envelope_factory(base=-20, quiet=[(10, 13), (16, 16.4)])
    keep = compute_keep_ranges(5, 20, env, words=[], min_silence=0.8)
    assert len(keep) == 2
    assert keep[0][0] == 5 and 10 <= keep[0][1] <= 10.3
    assert 12.7 <= keep[1][0] <= 13 and keep[1][1] == 20

    keep_with_word = compute_keep_ranges(5, 20, env, words=[[11.0, 12.0, "hey"]], min_silence=0.8)
    assert any(a <= 11.0 and b >= 12.0 for a, b in keep_with_word)
