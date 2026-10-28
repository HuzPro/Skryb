

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
