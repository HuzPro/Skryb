"""Tests for :class:`NanoParakeetAdapter`.

``nano_parakeet`` is patched at the module boundary via
``sys.modules`` — the same seam strategy used by the transformers
adapter tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.core.exceptions import (
    EngineDependencyMissingError,
    TranscriptionError,
)
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
    def test_should_route_hf_hub_download_through_app_models_dir_during_load(
        self,
    ) -> None:
        # nano_parakeet calls ``hf_hub_download`` from inside its
        # ``from_pretrained`` — we patch the module-level function so
        # those internal calls land in our app's models dir, with
        # ``local_files_only`` short-circuiting the upstream freshness
        # HEAD request. The patch is active only for the duration of
        # ``from_pretrained``; we simulate that here by making
        # ``from_pretrained`` exercise the patched function before
        # returning.
        import nano_parakeet
        from nano_parakeet import from_pretrained

        original_hf_download = MagicMock()
        nano_parakeet.hf_hub_download = original_hf_download

        def call_hf_download_during_load(**kwargs: object) -> object:
            nano_parakeet.hf_hub_download(
                repo_id=kwargs["model_name"],
                filename="parakeet.nemo",
            )
            return MagicMock()

        from_pretrained.side_effect = call_hf_download_during_load

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/app/models"),
        )
        adapter.load_model()

        original_hf_download.assert_called_once_with(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            filename="parakeet.nemo",
            cache_dir=str(Path("/app/models")),
            local_files_only=True,
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_restore_original_hf_hub_download_after_load(self) -> None:
        # Module-level mutation lasts only for the ``from_pretrained``
        # call. Without restoring, back-to-back adapter loads stomp
        # each other's closures and any code that imported the
        # original function after the first load runs against a stale
        # patch.
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

        assert nano_parakeet.hf_hub_download is original_hf_download

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_restore_original_hf_hub_download_when_load_raises(
        self,
    ) -> None:
        # Even on a failed model load, the module-global must come
        # back so the next attempt sees a clean slate. The restore
        # belongs in a finally block, not after the call.
        import nano_parakeet
        from nano_parakeet import from_pretrained

        original_hf_download = MagicMock()
        nano_parakeet.hf_hub_download = original_hf_download
        from_pretrained.side_effect = RuntimeError("kernel compile failed")

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/app/models"),
        )
        with pytest.raises(RuntimeError):
            adapter.load_model()

        assert nano_parakeet.hf_hub_download is original_hf_download

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
        assert result.transcription_seconds >= 0.0

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


class TestNanoParakeetAdapterWarmup:
    """Smoke test: the inherited default warmup runs end-to-end through
    parakeet's transcribe path so CUDA/ROCm kernels are compiled before
    the first user-triggered transcription."""

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_invoke_model_transcribe_during_warmup(self) -> None:
        from nano_parakeet import from_pretrained

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        model = from_pretrained.return_value
        model.transcribe.return_value = ""
        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v3",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        adapter.warmup()

        assert model.transcribe.call_count == 1


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


class TestNanoParakeetAdapterDependencyMissing:
    """When the ``nano_parakeet`` package is not installed,
    :meth:`NanoParakeetAdapter.load_model` must raise
    :class:`EngineDependencyMissingError` with an actionable install
    hint instead of leaking a bare ``ImportError``.
    """

    def test_should_raise_engine_dependency_missing_when_backend_not_installed(
        self,
    ) -> None:
        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        with patch.dict(
            "sys.modules", {"nano_parakeet": None, "torch": MagicMock()}
        ):
            with pytest.raises(EngineDependencyMissingError) as exc_info:
                adapter.load_model()

        message = str(exc_info.value).lower()
        assert "nano-parakeet" in message
        assert "pip install" in message
