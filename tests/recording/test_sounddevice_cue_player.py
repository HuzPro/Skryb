

import sys
import types

import numpy as np
import pytest

from skryb.recording import sounddevice_cue_player
from skryb.recording.sounddevice_cue_player import SoundDeviceCuePlayer


class _Finished(Exception):
    pass


class _PrimingStream:
    """Stands in for sounddevice's OutputStream on Windows' default device: with priming asked for,
    start() runs the callback on the caller's thread until the buffers the latency asks for are full."""

    opened: list["_PrimingStream"] = []

    def __init__(self, samplerate, channels, dtype, callback, latency, prime_output_buffers_using_stream_callback) -> None:  # noqa: ANN001
        self.samplerate, self.callback, self.latency = samplerate, callback, latency
        self.primes = prime_output_buffers_using_stream_callback
        self.written: list[np.ndarray] = []
        self.finished = self.closed = False
        _PrimingStream.opened.append(self)

    def start(self) -> None:
        block = 448
        buffered = 0
        while self.primes and buffered < self.latency * self.samplerate and not self.finished:
            self.pull(block)
            buffered += block

    def pull(self, frames: int) -> None:
        outdata = np.full((frames, 1), np.nan, dtype=np.float32)
        try:
            self.callback(outdata, frames, None, None)
        except _Finished:
            self.finished = True
        self.written.append(outdata[:, 0].copy())

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> type[_PrimingStream]:
    _PrimingStream.opened = []
    fake = types.SimpleNamespace(OutputStream=_PrimingStream, CallbackStop=_Finished)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return _PrimingStream


class TestHandingTheCueToTheDevice:
    @pytest.mark.parametrize("cue", ["play_start", "play_stop", "play_ready", "play_warning"])
    def test_every_sample_of_the_cue_is_with_the_device_when_start_returns(self, device, cue: str) -> None:  # noqa: ANN001
        player = SoundDeviceCuePlayer()

        getattr(player, cue)()

        stream = device.opened[-1]
        played = np.concatenate(stream.written)
        expected = {"play_start": player._start, "play_stop": player._stop,
                    "play_ready": player._ready, "play_warning": player._warning}[cue]
        assert stream.primes and stream.finished
        np.testing.assert_array_equal(played[: len(expected)], expected)
        assert not played[len(expected):].any()

    def test_a_new_cue_cuts_off_the_one_still_playing(self, device) -> None:  # noqa: ANN001
        player = SoundDeviceCuePlayer()

        player.play_start()
        player.play_stop()

        first, second = device.opened
        assert first.closed and not second.closed


class TestFeedingTheDevice:
    def test_a_cue_goes_out_block_by_block_then_silence_and_the_stream_ends_on_its_last_sample(self) -> None:
        samples = np.arange(1, 1001, dtype=np.float32)
        fill = sounddevice_cue_player._feeder(samples, _Finished)
        blocks = []
        for _ in range(3):
            outdata = np.full((448, 1), np.nan, dtype=np.float32)
            try:
                fill(outdata, 448, None, None)
            except _Finished:
                blocks.append(outdata[:, 0])
                break
            blocks.append(outdata[:, 0])

        played = np.concatenate(blocks)
        assert len(blocks) == 3
        np.testing.assert_array_equal(played[:1000], samples)
        assert not played[1000:].any()

    def test_a_cue_exactly_one_block_long_ends_the_stream_on_that_block(self) -> None:
        fill = sounddevice_cue_player._feeder(np.ones(448, dtype=np.float32), _Finished)
        outdata = np.zeros((448, 1), dtype=np.float32)

        with pytest.raises(_Finished):
            fill(outdata, 448, None, None)
        assert outdata.all()


class TestPreviewingTheVolume:
    def test_a_preview_plays_the_asked_tone_at_the_asked_volume_even_with_sounds_off(self) -> None:
        import numpy as np

        from skryb.recording.sounddevice_cue_player import SoundDeviceCuePlayer

        played: list[np.ndarray] = []
        player = SoundDeviceCuePlayer(enabled=lambda: False, volume=lambda: 0.1)
        player._emit = played.append  # noqa: SLF001

        player.preview("start", 1.0)
        player.preview("stop", 0.5)
        player.preview("start", 0.0)
        player.play_start()

        assert len(played) == 2
        assert np.abs(played[0]).max() > np.abs(played[1]).max() > 0
        assert not np.array_equal(played[0][: len(played[1])] / np.abs(played[0]).max(),
                                  played[1] / np.abs(played[1]).max())
