"""Tests for :class:`ArkAsrAdapter`.

``transformers``, ``torch``, ``librosa``, ``soundfile``, and
``huggingface_hub`` are patched at the module boundary via
``sys.modules`` so tests never import the real backends and never
touch the network or a GPU.
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


_REPO_ID = "Audio8/ARK-ASR-3B"
_PAD_ID = 0
_EOS_ID = 5
_AUDIO_TOKEN_ID = 9
_PROMPT_LENGTH = 7


def _make_audio(samples: int = 16000) -> AudioSegment:
    return AudioSegment(
        data=np.zeros(samples, dtype=np.float32),
        sample_rate=16000,
    )


def _mocked_modules() -> dict:
    return {
        "transformers": MagicMock(),
        "torch": MagicMock(),
        "librosa": MagicMock(),
        "soundfile": MagicMock(),
        "huggingface_hub": MagicMock(),
    }


class _FakeBatch(dict):
    """Dict-shaped stand-in for the processor's ``BatchFeature``.

    ``model.generate(**inputs)`` requires a real mapping, which a bare
    ``MagicMock`` is not.
    """

    def __init__(self) -> None:
        input_ids = MagicMock()
        input_ids.shape = (1, _PROMPT_LENGTH)
        super().__init__(
            input_ids=input_ids,
            attention_mask=MagicMock(),
            audios=MagicMock(),
        )
        self.moved_to_device: str | None = None

    def to(self, device: str) -> "_FakeBatch":
        self.moved_to_device = device
        return self


def _wire_backend() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Configure the transformers mock far enough for a load to succeed."""
    import transformers

    model = (
        transformers.AutoModelForCausalLM.from_pretrained
        .return_value.to.return_value
    )
    processor = transformers.AutoProcessor.from_pretrained.return_value
    processor.apply_chat_template.return_value = _FakeBatch()
    tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
    tokenizer.pad_token_id = _PAD_ID
    tokenizer.eos_token_id = _EOS_ID
    tokenizer.all_special_ids = [_PAD_ID, _EOS_ID, _AUDIO_TOKEN_ID]
    tokenizer.get_added_vocab.return_value = {
        "<|audio|>": _AUDIO_TOKEN_ID,
        "hello": 11,
    }
    tokenizer.batch_decode.return_value = ["text"]
    return model, processor, tokenizer


def _make_adapter(device: str = "cuda"):
    from voicetotext.transcription.ark_asr_adapter import ArkAsrAdapter

    return ArkAsrAdapter(
        repo_id=_REPO_ID,
        device=device,
        model_dir=Path("/tmp/models"),
    )


class TestArkAsrAdapterLoadModel:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_resolve_snapshot_pinned_to_audited_revision(self) -> None:
        import huggingface_hub

        from voicetotext.transcription.ark_asr_adapter import ARK_ASR_REVISION

        _wire_backend()
        adapter = _make_adapter()

        adapter.load_model()

        huggingface_hub.snapshot_download.assert_called_once_with(
            repo_id=_REPO_ID,
            revision=ARK_ASR_REVISION,
            cache_dir=str(Path("/tmp/models")),
            local_files_only=True,
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_model_with_trust_remote_code_bf16_sdpa_on_gpu(
        self,
    ) -> None:
        import huggingface_hub
        import torch
        import transformers

        _wire_backend()
        snapshot_dir = str(huggingface_hub.snapshot_download.return_value)
        adapter = _make_adapter(device="cuda")

        adapter.load_model()

        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            snapshot_dir,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        assert adapter.is_model_loaded() is True

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_move_model_to_device_and_switch_to_eval(self) -> None:
        import transformers

        model, _, _ = _wire_backend()
        adapter = _make_adapter(device="cuda")

        adapter.load_model()

        transformers.AutoModelForCausalLM.from_pretrained.return_value.to.assert_called_once_with(
            "cuda",
        )
        model.eval.assert_called_once_with()

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_request_fp32_on_cpu(self) -> None:
        import torch
        import transformers

        _wire_backend()
        adapter = _make_adapter(device="cpu")

        adapter.load_model()

        kwargs = transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        assert kwargs["dtype"] is torch.float32

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_processor_and_tokenizer_from_pinned_snapshot(
        self,
    ) -> None:
        import huggingface_hub
        import transformers

        _wire_backend()
        snapshot_dir = str(huggingface_hub.snapshot_download.return_value)
        adapter = _make_adapter()

        adapter.load_model()

        transformers.AutoProcessor.from_pretrained.assert_called_once_with(
            snapshot_dir, trust_remote_code=True,
        )
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            snapshot_dir, trust_remote_code=True,
        )

    def test_should_report_not_loaded_before_load(self) -> None:
        adapter = _make_adapter()

        assert adapter.is_model_loaded() is False


class TestArkAsrAdapterTranscribe:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_return_transcribed_text_in_result(self) -> None:
        _, _, tokenizer = _wire_backend()
        tokenizer.batch_decode.return_value = [" Hello, world! "]
        adapter = _make_adapter()
        adapter.load_model()

        result = adapter.transcribe(_make_audio())

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, world!"
        assert result.transcription_seconds >= 0.0

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_send_audio_array_and_prompt_in_conversation(self) -> None:
        _, processor, _ = _wire_backend()
        adapter = _make_adapter()
        adapter.load_model()
        audio = _make_audio()

        adapter.transcribe(audio)

        call = processor.apply_chat_template.call_args
        conversation = call.args[0]
        audio_part, text_part = conversation[0]["content"]
        assert conversation[0]["role"] == "user"
        assert np.array_equal(audio_part["array"], audio.data)
        assert text_part["text"] == "Please transcribe this audio."
        assert call.kwargs["add_generation_prompt"] is True
        assert call.kwargs["return_tensors"] == "pt"
        assert call.kwargs["sampling_rate"] == 16000
        assert call.kwargs["audio_max_length"] == 30 * 16000

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_move_inputs_to_device(self) -> None:
        _, processor, _ = _wire_backend()
        adapter = _make_adapter(device="cuda")
        adapter.load_model()

        adapter.transcribe(_make_audio())

        batch = processor.apply_chat_template.return_value
        assert batch.moved_to_device == "cuda"

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_cast_audio_features_to_model_dtype(self) -> None:
        import torch

        _, processor, _ = _wire_backend()
        original_audios = processor.apply_chat_template.return_value["audios"]
        adapter = _make_adapter(device="cuda")
        adapter.load_model()

        adapter.transcribe(_make_audio())

        original_audios.to.assert_called_once_with(dtype=torch.bfloat16)

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_generate_greedily_with_control_tokens_banned(self) -> None:
        model, _, _ = _wire_backend()
        adapter = _make_adapter()
        adapter.load_model()

        adapter.transcribe(_make_audio())

        kwargs = model.generate.call_args.kwargs
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 256
        assert kwargs["pad_token_id"] == _PAD_ID
        assert kwargs["eos_token_id"] == _EOS_ID
        assert kwargs["bad_words_ids"] == [[_PAD_ID], [_AUDIO_TOKEN_ID]]

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_decode_only_generated_tokens(self) -> None:
        model, _, tokenizer = _wire_backend()
        adapter = _make_adapter()
        adapter.load_model()

        adapter.transcribe(_make_audio())

        model.generate.return_value.__getitem__.assert_called_once_with(
            (slice(None), slice(_PROMPT_LENGTH, None)),
        )
        assert tokenizer.batch_decode.call_args.kwargs == {
            "skip_special_tokens": True,
        }

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_raise_when_transcribing_before_load(self) -> None:
        adapter = _make_adapter()

        with pytest.raises(TranscriptionError, match="not loaded"):
            adapter.transcribe(_make_audio())


class TestArkAsrAdapterChunking:
    """Audio longer than the model's 30 s window is split into <=30 s
    chunks, each transcribed on its own, then joined with a single
    space. The reference inference truncates at ``audio_max_length``
    (30 s); chunking keeps long recordings intact instead."""

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_transcribe_once_when_audio_fits_in_single_chunk(
        self,
    ) -> None:
        _, processor, _ = _wire_backend()
        adapter = _make_adapter()
        adapter.load_model()

        adapter.transcribe(_make_audio(samples=16000))

        assert processor.apply_chat_template.call_count == 1

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_split_long_audio_into_30_second_chunks(self) -> None:
        _, processor, _ = _wire_backend()
        adapter = _make_adapter()
        adapter.load_model()

        # 70 s of audio at 16 kHz -> 30 s + 30 s + 10 s = 3 chunks
        adapter.transcribe(_make_audio(samples=70 * 16000))

        assert processor.apply_chat_template.call_count == 3
        chunk_sizes = [
            call.args[0][0]["content"][0]["array"].shape[0]
            for call in processor.apply_chat_template.call_args_list
        ]
        assert chunk_sizes == [30 * 16000, 30 * 16000, 10 * 16000]

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_concatenate_text_from_each_chunk_with_single_space(
        self,
    ) -> None:
        _, _, tokenizer = _wire_backend()
        tokenizer.batch_decode.side_effect = [
            ["  first chunk.  "],
            [" second chunk."],
            ["third chunk."],
        ]
        adapter = _make_adapter()
        adapter.load_model()

        result = adapter.transcribe(_make_audio(samples=70 * 16000))

        assert result.text == "first chunk. second chunk. third chunk."

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_skip_empty_chunk_text_when_joining(self) -> None:
        _, _, tokenizer = _wire_backend()
        tokenizer.batch_decode.side_effect = [["first."], ["   "], ["third."]]
        adapter = _make_adapter()
        adapter.load_model()

        result = adapter.transcribe(_make_audio(samples=70 * 16000))

        assert result.text == "first. third."


class TestArkAsrAdapterWarmup:
    """Smoke test: the inherited default warmup runs end-to-end through
    the generate path so kernels are compiled before the first real
    transcription."""

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_invoke_generate_during_warmup(self) -> None:
        model, _, tokenizer = _wire_backend()
        tokenizer.batch_decode.return_value = [""]
        adapter = _make_adapter()
        adapter.load_model()

        adapter.warmup()

        assert model.generate.call_count == 1


class TestArkAsrAdapterDispose:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_clear_model_on_dispose(self) -> None:
        _wire_backend()
        adapter = _make_adapter()
        adapter.load_model()

        adapter.dispose()

        assert adapter.is_model_loaded() is False

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_release_gpu_cache_on_dispose(self) -> None:
        import torch

        _wire_backend()
        adapter = _make_adapter(device="cuda")
        adapter.load_model()

        adapter.dispose()

        torch.cuda.empty_cache.assert_called_once()

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_skip_gpu_cache_release_on_cpu_dispose(self) -> None:
        import torch

        _wire_backend()
        adapter = _make_adapter(device="cpu")
        adapter.load_model()

        adapter.dispose()

        torch.cuda.empty_cache.assert_not_called()

    def test_should_not_raise_on_dispose_when_not_loaded(self) -> None:
        adapter = _make_adapter()

        adapter.dispose()


class TestArkAsrAdapterIsATranscriptionEngine:
    def test_should_implement_transcription_engine_interface(self) -> None:
        adapter = _make_adapter()

        assert isinstance(adapter, TranscriptionEngine)


class TestArkAsrAdapterDependencyMissing:
    """When a backend package is missing,
    :meth:`ArkAsrAdapter.load_model` must raise
    :class:`EngineDependencyMissingError` with an actionable install
    hint instead of leaking a bare ``ImportError``.
    """

    def test_should_raise_engine_dependency_missing_when_transformers_absent(
        self,
    ) -> None:
        adapter = _make_adapter()
        modules = _mocked_modules() | {"transformers": None}

        with patch.dict("sys.modules", modules):
            with pytest.raises(EngineDependencyMissingError) as exc_info:
                adapter.load_model()

        message = str(exc_info.value).lower()
        assert "transformers" in message
        assert "pip install" in message

    def test_should_raise_engine_dependency_missing_when_librosa_absent(
        self,
    ) -> None:
        # librosa is imported by the model's remote processing code, so
        # the adapter checks it up front for a clean error.
        adapter = _make_adapter()
        modules = _mocked_modules() | {"librosa": None}

        with patch.dict("sys.modules", modules):
            with pytest.raises(EngineDependencyMissingError) as exc_info:
                adapter.load_model()

        assert "pip install" in str(exc_info.value).lower()
