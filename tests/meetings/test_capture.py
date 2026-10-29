import numpy as np
import pytest

from skryb.meetings.capture import RATE, Resampler, Track, pick_output


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def stereo_block(rate: int, seconds: float, freq: float = 400.0) -> bytes:
    """400 Hz fits whole cycles in a 10 ms packet, so packets join without a phase jump."""
    t = np.arange(round(rate * seconds)) / rate
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.column_stack([tone, tone]).tobytes()


def play(track: Track, clock: FakeClock, rate: int, seconds: float) -> None:
    """Deliver audio in 10 ms packets the way WASAPI does."""
    for _ in range(round(seconds / 0.01)):
        clock.now += 0.01
        track.feed(stereo_block(rate, 0.01), rate, 2)


def peak_hz(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio))
    return float(np.argmax(spectrum) * RATE / len(audio))


def test_steady_48k_stereo_becomes_16k_mono_of_the_same_duration() -> None:
    clock = FakeClock()
    track = Track(clock)
    track.start()
    play(track, clock, 48000, 10)
    audio = track.read()
    assert abs(len(audio) - 10 * RATE) < 0.05 * RATE
    assert abs(peak_hz(audio[RATE:9 * RATE]) - 400) < 2


def test_a_silent_gap_in_loopback_is_filled_so_later_audio_stays_on_time() -> None:
    clock = FakeClock()
    track = Track(clock)
    track.start()
    play(track, clock, 48000, 3)
    first = track.read()
    clock.now += 7  # nothing plays for 7 s, so WASAPI sends nothing
    play(track, clock, 48000, 2)
    second = track.read()
    total = np.concatenate([first, second])
    assert abs(len(total) - 12 * RATE) < 0.1 * RATE
    gap = total[round(3.2 * RATE):round(9.8 * RATE)]
    assert np.all(gap == 0)
    assert np.abs(total[round(10.2 * RATE):round(11.8 * RATE)]).max() > 0.4


def test_reading_during_a_long_silence_keeps_the_clock_moving() -> None:
    clock = FakeClock()
    track = Track(clock)
    track.start()
    total = 0
    for _ in range(600):  # a minute of 100 ms reads with nothing playing
        clock.now += 0.1
        total += len(track.read())
    assert abs(total - 60 * RATE) <= 0.06 * RATE


def test_steady_audio_is_never_padded() -> None:
    clock = FakeClock()
    track = Track(clock)
    track.start()
    pieces = []
    for _ in range(300):  # 30 s read every 100 ms
        play(track, clock, 48000, 0.1)
        pieces.append(track.read())
    audio = np.concatenate(pieces)[RATE:-RATE]
    zero_runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], audio == 0, [0]]).astype(int))))[::2]
    assert zero_runs.max(initial=0) < 10


def test_a_sample_rate_change_mid_stream_keeps_the_timeline() -> None:
    clock = FakeClock()
    track = Track(clock)
    track.start()
    play(track, clock, 192000, 4)  # a 192 kHz output device...
    play(track, clock, 44100, 4)  # ...then a headset at 44.1 kHz
    audio = track.read()
    assert abs(len(audio) - 8 * RATE) < 0.1 * RATE


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000, 96000, 192000])
def test_every_common_device_rate_keeps_speech_tones_and_their_level(rate: int) -> None:
    t = np.arange(rate * 3) / rate
    audio = Resampler(rate)((0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))
    assert abs(len(audio) - 3 * RATE) < 64
    middle = audio[RATE // 2:-RATE // 2]
    assert abs(peak_hz(middle) - 1000) < 2
    assert abs(np.sqrt(np.mean(middle**2)) - 0.5 / np.sqrt(2)) < 0.01


def test_sounds_above_what_16k_can_hold_are_filtered_out_rather_than_folded_into_speech() -> None:
    rate = 48000
    t = np.arange(rate * 2) / rate
    audio = Resampler(rate)((0.5 * np.sin(2 * np.pi * 11000 * t)).astype(np.float32))
    assert np.abs(audio[RATE // 2:-RATE // 2]).max() < 0.005  # 11 kHz would alias to 5 kHz


@pytest.mark.parametrize("rate", [44100, 48000])
def test_resampling_block_by_block_gives_exactly_what_one_pass_gives(rate: int) -> None:
    audio = np.random.default_rng(5).standard_normal(rate * 4).astype(np.float32)
    whole = Resampler(rate)(audio)
    blocks = Resampler(rate)
    pieces = [blocks(audio[at:at + rate // 100]) for at in range(0, len(audio), rate // 100)]
    assert np.array_equal(np.concatenate(pieces), whole)


def test_ten_minutes_at_44_1k_in_odd_sized_blocks_neither_gains_nor_loses_a_sample() -> None:
    resampler = Resampler(44100)
    block = np.zeros(44100 * 60 + 37, np.float32)
    produced = sum(len(resampler(block)) for _ in range(10))
    expected = len(block) * 10 * RATE / 44100
    assert abs(produced - expected) <= 16  # only the filter's half-width is held back


def test_the_meeting_device_is_kept_while_it_is_still_playing() -> None:
    busy = ["Headset (Jabra)", "Speakers (Realtek)"]
    assert pick_output(busy, "Headset (Jabra)", "Speakers (Realtek)", "Speakers (Realtek)") == "Headset (Jabra)"


def test_an_idle_device_gives_way_to_the_busy_communications_default() -> None:
    busy = ["Speakers (Realtek)", "Headset (Jabra)"]
    assert pick_output(busy, "Monitor (HDMI)", "Headset (Jabra)", "Speakers (Realtek)") == "Headset (Jabra)"


def test_any_busy_device_beats_idle_defaults() -> None:
    assert pick_output(["Voicemeeter Input"], None, "Headset", "Speakers") == "Voicemeeter Input"


def test_nothing_playing_keeps_the_current_device() -> None:
    assert pick_output([], "Speakers", "Headset", "Speakers") == "Speakers"
    assert pick_output([], None, "Headset", "Speakers") == "Speakers"
    assert pick_output([], None, None, None) is None
