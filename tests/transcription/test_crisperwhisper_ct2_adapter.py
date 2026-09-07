from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.crisperwhisper_ct2_adapter import (
    CrisperWhisperCt2Adapter,
)


_MODEL_DIR = Path("/models/ct2")


def _audio(seconds: float = 1.0) -> AudioSegment:
    return AudioSegment(
        data=np.zeros(int(16000 * seconds), dtype=np.float32),
        sample_rate=16000,
    )


def _loaded(
    adapter: CrisperWhisperCt2Adapter, sequences: list[list[int]],
) -> MagicMock:
    model = MagicMock()
    model.generate.side_effect = [
        [MagicMock(sequences_ids=[ids])] for ids in sequences
    ]
    tokens = MagicMock()
    tokens.decoder_prompt.return_value = [51886, 50258, 50364]
    tokens.decode.side_effect = lambda ids: " ".join(str(i) for i in ids)
    with (
        patch(
            "voicetotext.transcription.crisperwhisper_ct2_adapter"
            "._import_backend",
            return_value=lambda *a, **k: model,
        ),
        patch(
            "voicetotext.transcription.crisperwhisper_ct2_adapter."
            "WhisperTokens",
            return_value=tokens,
        ),
    ):
        adapter.load_model()
    return model


class TestCrisperWhisperCt2Adapter:
    def test_should_report_not_loaded_before_load_model(self) -> None:
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en")

        assert adapter.is_model_loaded() is False

    def test_should_reject_transcribe_before_load_model(self) -> None:
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en")

        with pytest.raises(TranscriptionError, match="Model not loaded"):
            adapter.transcribe(_audio())

    @pytest.mark.parametrize("mode", ["intended", "verbatim"])
    def test_should_prompt_with_the_configured_language_and_mode(
        self, mode: str,
    ) -> None:
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "de", mode)
        model = _loaded(adapter, [[1, 2]])

        adapter.transcribe(_audio())

        assert model.generate.call_args.args[1] == [[51886, 50258, 50364]]

    def test_should_decode_greedily_within_the_models_budget(self) -> None:
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en")
        model = _loaded(adapter, [[7]])

        adapter.transcribe(_audio())

        assert model.generate.call_args.kwargs == {
            "beam_size": 1,
            "sampling_topk": 1,
            "suppress_blank": False,
            "max_length": 876,
        }

    def test_should_join_one_result_per_encoder_window(self) -> None:
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en")
        model = _loaded(adapter, [[1], [2], [3]])

        result = adapter.transcribe(_audio(seconds=70.0))

        assert len(model.generate.call_args_list) == 3
        assert result.text == "1 2 3"

    def test_should_ask_for_bfloat16_on_gpu_and_default_on_cpu(self) -> None:
        assert CrisperWhisperCt2Adapter(
            _MODEL_DIR, "cuda", "en",
        )._compute_type() == "bfloat16"
        assert CrisperWhisperCt2Adapter(
            _MODEL_DIR, "cpu", "en",
        )._compute_type() == "default"

    def test_should_report_not_loaded_after_dispose(self) -> None:
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en")
        _loaded(adapter, [[1]])

        adapter.dispose()

        assert adapter.is_model_loaded() is False
