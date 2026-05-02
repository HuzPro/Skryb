"""Tests for :class:`NanoParakeetAdapter`.

``nano_parakeet`` is patched at the module boundary via
``sys.modules`` — the same seam strategy used by the transformers
adapter tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.result import TranscriptionResult


def _make_audio(samples: int = 16000) -> AudioSegment:
    return AudioSegment(
        data=np.zeros(samples, dtype=np.float32),
        sample_rate=16000,
    )


def _mocked_modules() -> dict:
    return {"nano_parakeet": MagicMock(), "torch": MagicMock()}


class TestNanoParakeetAdapterLoadModel:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_model_via_from_pretrained(self) -> None:
        import torch
        from nano_parakeet import from_pretrained

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        adapter.load_model()

        from_pretrained.assert_called_once_with(
            model_name="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            dtype=torch.float32,
        )
        assert adapter.is_model_loaded() is True

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_cuda_device_to_from_pretrained(self) -> None:
        import torch
        from nano_parakeet import from_pretrained

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v3",
            device="cuda",
            model_dir=Path("/tmp/models"),
        )

        adapter.load_model()

        from_pretrained.assert_called_once_with(
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            device="cuda",
            dtype=torch.float32,
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_force_float32_dtype_to_avoid_bf16_mismatch(self) -> None:
        # nano_parakeet auto-selects bfloat16 on bf16-capable GPUs (RDNA4
        # included). When the CUDA-graph fast path is disabled (which we
        # do to dodge cross-thread replay crashes on ROCm), the slow
        # `tdt_greedy_decode` path crashes with
        # "mat1 and mat2 must have the same dtype" because the encoder
        # output is cast back to float32 while decoder/joint stay bf16.
        # Forcing float32 avoids the entire bf16 code path.
        import torch
        from nano_parakeet import from_pretrained

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v3",
            device="cuda",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        kwargs = from_pretrained.call_args.kwargs
        assert kwargs["dtype"] is torch.float32

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_patch_hf_hub_download_to_use_app_models_dir(self) -> None:
        # nano_parakeet calls hf_hub_download() with no cache_dir, so it
        # falls back to the global default — typically ~/.cache/huggingface.
        # Our chunked downloader writes to model_dir, so without this patch
        # nano_parakeet would re-download the file from scratch.
        import nano_parakeet

        original_hf_download = MagicMock()
        nano_parakeet.hf_hub_download = original_hf_download

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/app/models"),
        )

        adapter.load_model()

        # The patched function should pass cache_dir on every call.
        nano_parakeet.hf_hub_download(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            filename="parakeet-tdt-0.6b-v2.nemo",
        )
        original_hf_download.assert_called_with(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            filename="parakeet-tdt-0.6b-v2.nemo",
            cache_dir=str(Path("/app/models")),
            local_files_only=True,
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_let_caller_override_cache_dir_explicitly(self) -> None:
        import nano_parakeet

        original_hf_download = MagicMock()
        nano_parakeet.hf_hub_download = original_hf_download

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/app/models"),
        )
        adapter.load_model()

        nano_parakeet.hf_hub_download(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            filename="parakeet-tdt-0.6b-v2.nemo",
            cache_dir="/explicit/override",
            local_files_only=False,
        )

        original_hf_download.assert_called_with(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            filename="parakeet-tdt-0.6b-v2.nemo",
            cache_dir="/explicit/override",
            local_files_only=False,
        )

    def test_should_report_not_loaded_before_load(self) -> None:
        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        assert adapter.is_model_loaded() is False


class TestNanoParakeetAdapterTranscribe:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_return_transcribed_text_in_result(self) -> None:
        from nano_parakeet import from_pretrained

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        model = from_pretrained.return_value
        model.transcribe.return_value = " Hello, world! "

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        result = adapter.transcribe(_make_audio())

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, world!"
        assert result.duration_seconds >= 0.0

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_numpy_audio_data_to_model(self) -> None:
        from nano_parakeet import from_pretrained

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        model = from_pretrained.return_value
        model.transcribe.return_value = "text"

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        audio = _make_audio()

        adapter.transcribe(audio)

        passed_audio = model.transcribe.call_args[0][0]
        assert np.array_equal(passed_audio, audio.data)


    @patch.dict("sys.modules", _mocked_modules())
    def test_should_raise_when_transcribing_before_load(self) -> None:
        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        with pytest.raises(TranscriptionError, match="not loaded"):
            adapter.transcribe(_make_audio())


class TestNanoParakeetAdapterDispose:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_clear_model_on_dispose(self) -> None:
        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        adapter.dispose()

        assert adapter.is_model_loaded() is False

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_release_gpu_cache_on_dispose(self) -> None:
        import sys
        torch_mock = MagicMock()
        sys.modules["torch"] = torch_mock

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cuda",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        adapter.dispose()

        torch_mock.cuda.empty_cache.assert_called_once()

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_skip_gpu_cache_release_on_cpu_dispose(self) -> None:
        import sys
        torch_mock = MagicMock()
        sys.modules["torch"] = torch_mock

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        adapter.dispose()

        torch_mock.cuda.empty_cache.assert_not_called()

    def test_should_not_raise_on_dispose_when_not_loaded(self) -> None:
        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        adapter.dispose()


class TestNanoParakeetAdapterIsATranscriptionEngine:
    def test_should_implement_transcription_engine_interface(self) -> None:
        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        assert isinstance(adapter, TranscriptionEngine)
