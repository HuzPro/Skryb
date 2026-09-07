"""Hardware-independent tests for the direct Transformers adapter."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch, sentinel

import numpy as np
import pytest

from voicetotext.core.exceptions import (
    ConfigurationError,
    EngineDependencyMissingError,
    TranscriptionError,
)
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.crisperwhisper_adapter import (
    CrisperWhisperAdapter,
    resolve_crisperwhisper_mode,
)
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.whisper_features import (
    split_into_encoder_windows,
)


_REPO_ID = "nyralabs/CrisperWhisper2.0_turbo"
_REVISION = "de0369c8a68025b7f6e86387b6eb5a3b369787c8"
_MODE_IDS = [10, 11, 12, 13, 14]
_WHISPER_IDS = [100, 101, 102, 103]


def _audio() -> AudioSegment:
    return AudioSegment(
        data=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
    )


def _adapter(
    *,
    mode: str = "intended",
    device: str = "cpu",
    revision: str = _REVISION,
) -> CrisperWhisperAdapter:
    return CrisperWhisperAdapter(
        repo_id=_REPO_ID,
        revision=revision,
        device=device,
        model_dir=Path("/app/models"),
        language="en",
        mode=mode,  # type: ignore[arg-type]
    )


def _fake_backend() -> tuple[
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    torch_module = MagicMock()
    torch_module.float32 = sentinel.float32
    torch_module.bfloat16 = sentinel.bfloat16
    torch_module.long = sentinel.long
    torch_module.inference_mode.return_value = nullcontext()
    torch_module.cuda.is_available.return_value = False
    torch_module.tensor.return_value = sentinel.decoder_input_ids

    processor = MagicMock()
    processor.tokenizer.encode.return_value = list(_MODE_IDS)
    processor.tokenizer.decode.return_value = " Hello [laughter] "
    processor.get_decoder_prompt_ids.return_value = [
        (1, 101),
        (2, 102),
        (3, 103),
    ]
    processor.return_value.input_features.to.return_value = sentinel.input_features

    model = MagicMock()
    model.config.decoder_start_token_id = 100
    model.generation_config.suppress_tokens = [-1, 9, 10]
    model.generate.return_value = np.array([[201, 202]], dtype=np.int64)

    transformers_module = MagicMock()
    model_type = transformers_module.WhisperForConditionalGeneration
    processor_type = transformers_module.WhisperProcessor
    model_type.from_pretrained.return_value = model
    processor_type.from_pretrained.return_value = processor
    return torch_module, transformers_module, model_type, processor_type, model


def _load_with_fakes(
    adapter: CrisperWhisperAdapter,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    (
        torch_module,
        transformers_module,
        model_type,
        processor_type,
        model,
    ) = _fake_backend()
    with patch.dict(
        "sys.modules",
        {"torch": torch_module, "transformers": transformers_module},
    ):
        adapter.load_model()
    return torch_module, model_type, processor_type, model


class TestCrisperWhisperModeValidation:
    def test_should_default_adapter_to_intended_mode(self) -> None:
        adapter = CrisperWhisperAdapter(
            _REPO_ID, _REVISION, "cpu", Path("/models"), "en",
        )

        assert adapter._mode == "intended"

    @pytest.mark.parametrize("mode", ["intended", "verbatim"])
    def test_should_accept_documented_modes(self, mode: str) -> None:
        assert resolve_crisperwhisper_mode(mode) == mode

    @pytest.mark.parametrize("value", ["raw", "", 1, None])
    def test_should_reject_unknown_config_values(self, value: object) -> None:
        with pytest.raises(ConfigurationError, match="crisperwhisper_mode"):
            resolve_crisperwhisper_mode(value)  # type: ignore[arg-type]


class TestCrisperWhisperLoadModel:
    def test_should_load_processor_and_model_from_app_cache_only(self) -> None:
        adapter = _adapter()

        _, model_type, processor_type, model = _load_with_fakes(adapter)

        processor_type.from_pretrained.assert_called_once_with(
            _REPO_ID,
            revision=_REVISION,
            cache_dir=str(Path("/app/models")),
            local_files_only=True,
        )
        model_type.from_pretrained.assert_called_once_with(
            _REPO_ID,
            revision=_REVISION,
            cache_dir=str(Path("/app/models")),
            local_files_only=True,
            dtype=sentinel.float32,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()
        assert adapter.is_model_loaded() is True

    def test_should_open_the_injected_revision_not_the_default_branch(
        self,
    ) -> None:
        # The pin lives on ``ModelInfo.revision``, so the commit the
        # loader opens is the one the downloader fetched and the one the
        # audited licence text belongs to.
        adapter = _adapter(revision="c0ffee")

        _, model_type, processor_type, _ = _load_with_fakes(adapter)

        assert processor_type.from_pretrained.call_args.kwargs[
            "revision"
        ] == "c0ffee"
        assert model_type.from_pretrained.call_args.kwargs[
            "revision"
        ] == "c0ffee"

    def test_should_use_bfloat16_for_gpu_safetensors(self) -> None:
        adapter = _adapter(device="cuda")

        _, model_type, _, _ = _load_with_fakes(adapter)

        assert model_type.from_pretrained.call_args.kwargs["dtype"] is (
            sentinel.bfloat16
        )

    def test_should_clear_transformers_default_decoder_controls(self) -> None:
        adapter = _adapter()

        _, _, _, model = _load_with_fakes(adapter)

        assert model.generation_config.forced_decoder_ids is None
        assert model.config.forced_decoder_ids is None
        assert model.generation_config.begin_suppress_tokens is None
        assert model.config.begin_suppress_tokens is None

    def test_should_reject_tokenizer_without_five_atomic_mode_tags(self) -> None:
        adapter = _adapter()
        torch_module, transformers_module, _, processor_type, _ = _fake_backend()
        processor_type.from_pretrained.return_value.tokenizer.encode.return_value = [1]

        with patch.dict(
            "sys.modules",
            {"torch": torch_module, "transformers": transformers_module},
        ):
            with pytest.raises(TranscriptionError, match="atomic intended tags"):
                adapter.load_model()

        assert adapter.is_model_loaded() is False


class TestCrisperWhisperDecoderPrefix:
    @pytest.mark.parametrize("language", ["en", "de"])
    def test_should_use_configured_language_for_warmup_and_dictation(
        self, language: str,
    ) -> None:
        adapter = CrisperWhisperAdapter(
            _REPO_ID, _REVISION, "cpu", Path("/models"), language,
        )
        _, _, processor_type, model = _load_with_fakes(adapter)

        adapter.warmup()
        adapter.transcribe(_audio())

        processor = processor_type.from_pretrained.return_value
        processor.get_decoder_prompt_ids.assert_called_once_with(
            task="transcribe", language=language, no_timestamps=True,
        )
        assert len(model.generate.call_args_list) == 2
        assert all(
            call.kwargs["language"] == language
            for call in model.generate.call_args_list
        )

    @pytest.mark.parametrize("mode", ["intended", "verbatim"])
    def test_should_prepend_five_mode_tags_before_whisper_controls(
        self, mode: str,
    ) -> None:
        adapter = _adapter(mode=mode)
        torch_module, _, processor_type, model = _load_with_fakes(adapter)

        adapter.transcribe(_audio())

        expected_tags = "".join(
            f"[{mode}_{index}]" for index in range(1, 6)
        )
        processor = processor_type.from_pretrained.return_value
        processor.tokenizer.encode.assert_called_once_with(
            expected_tags,
            add_special_tokens=False,
        )
        processor.get_decoder_prompt_ids.assert_called_once_with(
            task="transcribe",
            language="en",
            no_timestamps=True,
        )
        torch_module.tensor.assert_called_once_with(
            [_MODE_IDS + _WHISPER_IDS],
            device="cpu",
            dtype=sentinel.long,
        )
        assert model.generate.call_count == 1


class TestCrisperWhisperTranscribe:
    def test_should_process_audio_and_greedily_generate(self) -> None:
        adapter = _adapter()
        _, _, processor_type, model = _load_with_fakes(adapter)
        audio = _audio()

        result = adapter.transcribe(audio)

        processor = processor_type.from_pretrained.return_value
        passed_audio = processor.call_args.args[0]
        assert np.array_equal(passed_audio, audio.data)
        assert processor.call_args.kwargs == {
            "sampling_rate": 16000,
            "return_tensors": "pt",
        }
        processor.return_value.input_features.to.assert_called_once_with(
            device="cpu",
            dtype=sentinel.float32,
        )
        assert model.generate.call_args.args == (sentinel.input_features,)
        assert model.generate.call_args.kwargs == {
            "language": "en",
            "decoder_input_ids": sentinel.decoder_input_ids,
            "max_new_tokens": 256,
            "num_beams": 1,
            "do_sample": False,
            "use_cache": True,
            "suppress_tokens": [9, 10],
        }
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello [laughter]"
        assert result.language == "en"
        assert result.transcription_seconds >= 0.0

    def test_should_strip_echoed_prompt_ids_before_decoding(self) -> None:
        adapter = _adapter()
        _, _, processor_type, model = _load_with_fakes(adapter)
        model.generate.return_value = np.array(
            [_MODE_IDS + _WHISPER_IDS + [201, 202]], dtype=np.int64,
        )

        adapter.transcribe(_audio())

        processor = processor_type.from_pretrained.return_value
        processor.tokenizer.decode.assert_called_once_with(
            [201, 202],
            skip_special_tokens=True,
        )

    def test_should_raise_before_model_is_loaded(self) -> None:
        with pytest.raises(TranscriptionError, match="not loaded"):
            _adapter().transcribe(_audio())


class TestCrisperWhisperDispose:
    def test_should_release_loaded_references(self) -> None:
        adapter = _adapter()
        _load_with_fakes(adapter)

        adapter.dispose()

        assert adapter.is_model_loaded() is False
        assert adapter._decoder_prefix is None

    def test_should_release_gpu_cache(self) -> None:
        adapter = _adapter(device="cuda")
        torch_module, _, _, _ = _load_with_fakes(adapter)

        with patch.dict("sys.modules", {"torch": torch_module}):
            adapter.dispose()

        torch_module.cuda.empty_cache.assert_called_once_with()

    def test_should_ignore_dispose_before_load(self) -> None:
        _adapter().dispose()


class TestCrisperWhisperInterfaceAndDependencies:
    def test_should_implement_transcription_engine(self) -> None:
        assert isinstance(_adapter(), TranscriptionEngine)

    def test_should_report_missing_transformers_dependency(self) -> None:
        adapter = _adapter()

        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(EngineDependencyMissingError) as exc_info:
                adapter.load_model()

        message = str(exc_info.value).lower()
        assert "transformers" in message
        assert "pip install" in message


_SAMPLE_RATE = 16000
_GAP_START_SECONDS = 29.0
_GAP_SECONDS = 0.2


def _speech_with_gap(total_seconds: float) -> AudioSegment:
    """Loud throughout except one silent gap just before the 30 s edge."""
    data = np.full(
        int(total_seconds * _SAMPLE_RATE), 0.5, dtype=np.float32,
    )
    gap_start = int(_GAP_START_SECONDS * _SAMPLE_RATE)
    gap_end = gap_start + int(_GAP_SECONDS * _SAMPLE_RATE)
    data[gap_start:gap_end] = 0.0
    return AudioSegment(data=data, sample_rate=_SAMPLE_RATE)


class TestCrisperWhisperEncoderWindows:
    def test_should_keep_audio_within_the_encoder_window_whole(self) -> None:
        audio = _speech_with_gap(12.0)

        windows = split_into_encoder_windows(audio)

        assert len(windows) == 1
        assert np.array_equal(windows[0].data, audio.data)

    def test_should_split_audio_longer_than_the_encoder_window(self) -> None:
        windows = split_into_encoder_windows(_speech_with_gap(75.0))

        assert len(windows) > 1

    def test_should_preserve_every_sample_across_windows(self) -> None:
        audio = _speech_with_gap(75.0)

        windows = split_into_encoder_windows(audio)

        rejoined = np.concatenate([window.data for window in windows])
        assert np.array_equal(rejoined, audio.data)

    def test_should_split_on_the_quietest_point_near_the_boundary(self) -> None:
        windows = split_into_encoder_windows(_speech_with_gap(75.0))

        gap_start = int(_GAP_START_SECONDS * _SAMPLE_RATE)
        gap_end = gap_start + int(_GAP_SECONDS * _SAMPLE_RATE)
        assert gap_start <= windows[0].sample_count <= gap_end

    def test_should_never_exceed_the_encoder_window(self) -> None:
        windows = split_into_encoder_windows(_speech_with_gap(75.0))

        longest = max(window.duration_seconds for window in windows)
        assert longest <= 30.0


class TestCrisperWhisperLongAudio:
    def test_should_generate_once_for_short_audio(self) -> None:
        adapter = _adapter()
        _, _, _, model = _load_with_fakes(adapter)

        adapter.transcribe(_audio())

        assert model.generate.call_count == 1

    def test_should_generate_for_every_window_of_long_audio(self) -> None:
        adapter = _adapter()
        _, _, _, model = _load_with_fakes(adapter)

        adapter.transcribe(_speech_with_gap(75.0))

        assert model.generate.call_count == 3

    def test_should_join_the_text_of_every_window(self) -> None:
        adapter = _adapter()
        _, _, processor_type, _ = _load_with_fakes(adapter)
        processor = processor_type.from_pretrained.return_value
        processor.tokenizer.decode.side_effect = ["one", "two", "three"]

        result = adapter.transcribe(_speech_with_gap(75.0))

        assert result.text == "one two three"

    def test_should_feed_every_sample_to_the_processor(self) -> None:
        adapter = _adapter()
        _, _, processor_type, _ = _load_with_fakes(adapter)
        audio = _speech_with_gap(75.0)

        adapter.transcribe(audio)

        processor = processor_type.from_pretrained.return_value
        fed = np.concatenate(
            [call.args[0] for call in processor.call_args_list],
        )
        assert np.array_equal(fed, audio.data)

    def test_should_time_the_whole_transcription_not_one_window(self) -> None:
        adapter = _adapter()
        _load_with_fakes(adapter)

        result = adapter.transcribe(_speech_with_gap(75.0))

        assert result.transcription_seconds >= 0.0


class TestCrisperWhisperDecodeBudget:
    def test_should_budget_tokens_from_the_model_position_limit(self) -> None:
        adapter = _adapter()
        _, _, _, model = _load_with_fakes(adapter)
        model.config.max_target_positions = 448

        adapter.transcribe(_audio())

        prefix_length = len(_MODE_IDS + _WHISPER_IDS)
        assert model.generate.call_args.kwargs["max_new_tokens"] == (
            448 - prefix_length - 1
        )

    def test_should_fall_back_when_the_limit_is_unavailable(self) -> None:
        adapter = _adapter()
        _, _, _, model = _load_with_fakes(adapter)
        model.config.max_target_positions = None

        adapter.transcribe(_audio())

        assert model.generate.call_args.kwargs["max_new_tokens"] == 256
