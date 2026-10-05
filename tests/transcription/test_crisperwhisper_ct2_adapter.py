import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.crisperwhisper_ct2_adapter import (
    CrisperWhisperCt2Adapter,
)


_MODEL_DIR = Path("/models/ct2")


@pytest.fixture(autouse=True)
def fake_storage_view() -> Iterator[None]:
    with patch(
        "voicetotext.transcription.crisperwhisper_ct2_adapter._storage_view",
        side_effect=lambda features: features,
    ):
        yield


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

    def test_should_prompt_the_next_window_with_the_confirmed_words(
        self,
    ) -> None:
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en", "verbatim")
        _loaded(adapter, [[1, 2, 3, 4], [5]])

        adapter.transcribe(_audio(seconds=40.0))

        assert adapter._tokens.decoder_prompt.call_args_list[-2:] == [
            call("en", "verbatim", ""),
            call("en", "verbatim", "1 2 3 4"),
        ]

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


class _ThreadTrackingModel:
    """Records which thread created and freed it, and whether the creator still ran."""

    freed: list[tuple[str, bool]]

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.created_on = threading.current_thread()
        self.freed = _ThreadTrackingModel.freed

    def __del__(self) -> None:
        self.freed.append(
            (threading.current_thread().name, self.created_on.is_alive()),
        )


@pytest.fixture
def tracked_models() -> Iterator[list[tuple[str, bool]]]:
    _ThreadTrackingModel.freed = []
    with (
        patch(
            "voicetotext.transcription.crisperwhisper_ct2_adapter._import_backend",
            return_value=_ThreadTrackingModel,
        ),
        patch(
            "voicetotext.transcription.crisperwhisper_ct2_adapter.WhisperTokens",
        ),
    ):
        yield _ThreadTrackingModel.freed


def _on_a_thread_that_then_exits(step: Callable[[], object]) -> None:
    thread = threading.Thread(target=step)
    thread.start()
    thread.join()


class TestModelThreadLifetime:
    """CTranslate2 crashes the process when it frees a model whose creating
    thread has exited, so the model is created on a thread that outlives it."""

    def test_the_model_is_created_on_its_own_thread_not_the_callers(
        self, tracked_models,
    ) -> None:
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en")

        adapter.load_model()

        assert adapter._model.created_on.name == "ct2-model"
        assert adapter._model.created_on is not threading.current_thread()
        adapter.dispose()

    def test_a_model_loaded_by_an_exited_thread_is_freed_while_its_creator_still_runs(
        self, tracked_models,
    ) -> None:
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en")
        _on_a_thread_that_then_exits(adapter.load_model)

        adapter.dispose()

        assert tracked_models == [("ct2-model", True)]

    def test_a_model_disposed_from_another_exiting_thread_is_freed_on_its_creator(
        self, tracked_models,
    ) -> None:
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en")
        _on_a_thread_that_then_exits(adapter.load_model)

        _on_a_thread_that_then_exits(adapter.dispose)

        assert tracked_models == [("ct2-model", True)]

    def test_the_models_thread_ends_once_the_model_is_freed(
        self, tracked_models,
    ) -> None:
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en")
        adapter.load_model()
        model_thread = adapter._model.created_on

        adapter.dispose()
        model_thread.join(timeout=5)

        assert not model_thread.is_alive()

    def test_loading_again_frees_the_previous_model_first(
        self, tracked_models,
    ) -> None:
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en")
        adapter.load_model()

        adapter.load_model()

        assert tracked_models == [("ct2-model", True)]
        assert adapter.is_model_loaded()
        adapter.dispose()

    def test_a_load_that_fails_raises_and_leaves_no_thread_behind(self) -> None:
        before = {thread.ident for thread in threading.enumerate()}
        adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en")

        with (
            patch(
                "voicetotext.transcription.crisperwhisper_ct2_adapter._import_backend",
                return_value=MagicMock(side_effect=RuntimeError("hipErrorNoDevice")),
            ),
            patch(
                "voicetotext.transcription.crisperwhisper_ct2_adapter.WhisperTokens",
            ),
            pytest.raises(RuntimeError, match="hipErrorNoDevice"),
        ):
            adapter.load_model()

        _wait_for_threads_to_settle(before)
        assert not adapter.is_model_loaded()

    def test_disposing_an_engine_that_never_loaded_does_nothing(self) -> None:
        CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en").dispose()

    def test_a_thousand_swaps_from_short_lived_threads_free_every_model_safely(
        self, tracked_models,
    ) -> None:
        before = {thread.ident for thread in threading.enumerate()}
        for _ in range(1000):
            adapter = CrisperWhisperCt2Adapter(_MODEL_DIR, "cuda", "en")
            _on_a_thread_that_then_exits(adapter.load_model)
            _on_a_thread_that_then_exits(adapter.dispose)

        assert len(tracked_models) == 1000
        assert all(freed == ("ct2-model", True) for freed in tracked_models)
        _wait_for_threads_to_settle(before)


def _wait_for_threads_to_settle(before: set[int | None]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        extra = [t for t in threading.enumerate() if t.ident not in before]
        if not extra:
            return
        time.sleep(0.01)
    raise AssertionError(f"threads left running: {[t.name for t in extra]}")
