import random

from skryb.meetings.segmenter import SILENCE, Segment, Segmenter

FRAME = 0.01  # diarization frames are 10 ms


def frames(seconds: float, label: int) -> list[int]:
    return [label] * round(seconds / FRAME)


def run(labels: list[int], **kwargs) -> list[Segment]:
    segmenter = Segmenter(FRAME, **kwargs)
    return segmenter.push(labels) + segmenter.flush()


def test_a_half_second_pause_ends_an_utterance():
    segments = run(frames(2, 0) + frames(0.5, SILENCE) + frames(1, 0))
    assert [(s.start, s.end) for s in segments] == [(0, 2), (2.5, 3.5)]


def test_shorter_pauses_stay_inside_one_utterance():
    segments = run(frames(2, 0) + frames(0.4, SILENCE) + frames(1, 0))
    assert [(s.start, s.end) for s in segments] == [(0, 3.4)]


def test_a_speaker_change_that_lasts_splits_at_the_change():
    segments = run(frames(2, 0) + frames(1.5, 1))
    assert segments == [Segment(0, 2, 0), Segment(2, 3.5, 1)]


def test_a_brief_flicker_to_another_speaker_does_not_split():
    segments = run(frames(1, 0) + frames(0.2, 1) + frames(1, 0))
    assert segments == [Segment(0, 2.2, 0)]


def test_a_long_monologue_is_cut_at_the_length_cap():
    segments = run(frames(40, 2), max_length=15)
    assert [(s.start, s.end, s.speaker) for s in segments] == [(0, 15, 2), (15, 30, 2), (30, 40, 2)]


def test_blips_shorter_than_the_minimum_are_dropped():
    segments = run(frames(0.2, 0) + frames(1, SILENCE) + frames(0.3, 0) + frames(1, SILENCE))
    assert [(s.start, s.end) for s in segments] == [(1.2, 1.5)]


def test_flush_closes_an_utterance_still_open_at_stop():
    segmenter = Segmenter(FRAME)
    assert segmenter.push(frames(3, 0)) == []
    assert segmenter.flush() == [Segment(0, 3, 0)]


def test_labels_can_arrive_in_chunks_of_any_size():
    labels = frames(2, 0) + frames(0.6, SILENCE) + frames(1.2, 1) + frames(0.7, 0)
    whole = run(labels)
    segmenter = Segmenter(FRAME)
    chunked = []
    for i in range(0, len(labels), 7):
        chunked += segmenter.push(labels[i:i + 7])
    assert chunked + segmenter.flush() == whole


def test_an_hour_of_turn_taking_yields_one_segment_per_turn():
    rng = random.Random(7)
    labels, turns, speaker = [], [], 0
    while len(labels) < 360_000:  # one hour of 10 ms frames
        speaker = rng.choice([s for s in range(5) if s != speaker])
        length = round(rng.uniform(0.5, 14.5), 2)
        start = len(labels) * FRAME
        labels += frames(length, speaker)
        turns.append((round(start, 2), round(start + length, 2), speaker))
        if rng.random() < 0.6:
            labels += frames(rng.uniform(0.5, 3), SILENCE)
    segments = run(labels)
    assert [(round(s.start, 2), round(s.end, 2), s.speaker) for s in segments] == turns
