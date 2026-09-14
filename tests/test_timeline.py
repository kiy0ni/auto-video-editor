import pytest

from auto_video_editor.timeline import (TimeMap, format_clock, format_srt_time, merge_ranges, parse_duration,
                                        slugify, subtract_ranges, total_duration)


def test_merge_ranges_overlapping_and_gap():
    assert merge_ranges([(5, 6), (0, 2), (1, 3)]) == [(0, 3), (5, 6)]
    assert merge_ranges([(0, 2), (2.5, 3)], gap=0.5) == [(0, 3)]
    assert merge_ranges([(3, 3), (4, 2)]) == []


def test_subtract_ranges():
    assert subtract_ranges((0, 10), [(2, 3), (5, 7)]) == [(0, 2), (3, 5), (7, 10)]
    assert subtract_ranges((0, 10), [(-1, 1), (9, 12)]) == [(1, 9)]
    assert subtract_ranges((0, 10), [(0, 10)]) == []
    assert total_duration([(0, 2), (3, 5)]) == 4


def test_time_map():
    tm = TimeMap([(10, 20), (30, 35)])
    assert tm.duration == 15
    assert tm.to_output(10) == 0
    assert tm.to_output(15) == 5
    assert tm.to_output(25) is None
    assert tm.to_output(32) == 12
    # A word straddling a cut is kept when its midpoint survives, clamped to the kept range.
    assert tm.map_interval(19, 20.5) == (9, 10)
    assert tm.map_interval(24, 26) is None


@pytest.mark.parametrize("text, seconds", [
    ("90", 90), ("90s", 90), ("10m", 600), ("1h30m", 5400), ("1:30", 90), ("1:02:03", 3723), ("2.5m", 150),
])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "abc", "10x", "1m30"])
def test_parse_duration_invalid(text):
    with pytest.raises(ValueError):
        parse_duration(text)


def test_formatting():
    assert format_clock(65) == "01:05"
    assert format_clock(3725) == "1:02:05"
    assert format_clock(5, always_hours=True) == "0:00:05"
    assert format_srt_time(3661.5) == "01:01:01,500"
    assert slugify("No way! Let's GO, c'est incroyable") == "no-way-lets-go-cest-incroyable"
