"""Tests for :class:`GraniteSpeechAdapter`.

``transformers`` and ``torch`` are patched at the module boundary via
``sys.modules`` - the same seam strategy used by the nano-parakeet
adapter tests.

``torch`` is imported eagerly at the module level so its C-extension
docstrings register exactly once. Without this, an earlier test that
patches ``sys.modules["torch"]`` to a ``MagicMock`` and then has its
patch torn down can leave ``torch.overrides`` un-imported; a later
test doing ``import torch`` would then try to re-initialise the C
extension and crash with ``RuntimeError: function
'_has_torch_function' already has a docstring``.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch as _torch_eager_import  # noqa: F401 - see module docstring

from voicetotext.core.exceptions import (
    EngineDependencyMissingError,
    TranscriptionError,
)
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine


def _mocked_modules() -> dict[str, MagicMock]:
    return {"transformers": MagicMock(), "torch": MagicMock()}


def _make_audio(sample_rate: int = 16000, samples: int = 16000) -> AudioSegment:
    return AudioSegment(
        data=np.zeros(samples, dtype=np.float32),
        sample_rate=sample_rate,
    )


def _capture_warnings(caplog, logger, action) -> list:
    """Run ``action`` with caplog's handler attached to ``logger``.

    ``voicetotext.core.debug_logger`` disables propagation on the
    ``voicetotext`` subtree, so caplog's root-attached handler doesn't
    see records emitted from the adapter when the debug logger module
    has been imported (which happens any time ``bootstrap`` is loaded).
    Attaching caplog's handler directly to the target logger sidesteps
    the propagation block for the duration of the assertion.
    """
    import logging as stdlib_logging

    previous_level = logger.level
    logger.setLevel(stdlib_logging.WARNING)
    logger.addHandler(caplog.handler)
    try:
        action()
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous_level)
    return [r for r in caplog.records if r.levelno >= stdlib_logging.WARNING]


def _wire_successful_transcribe(
    *,
    n_input_tokens: int = 10,
    decoded_text: str = " hello world ",
) -> tuple[MagicMock, MagicMock, dict]:
    """Configure the mock chain so a transcribe() call completes.

    Returns ``(processor, model, fake_inputs)`` so the test can make
    assertions without re-deriving them from the imports.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained.return_value
    model = AutoModelForSpeechSeq2Seq.from_pretrained.return_value
    processor.tokenizer.apply_chat_template.return_value = "<chat-prompt>"
    fake_input_ids = MagicMock()
    fake_input_ids.shape = [1, n_input_tokens]
    fake_inputs = {"input_ids": fake_input_ids}
    processor.return_value.to.return_value = fake_inputs
    processor.tokenizer.batch_decode.return_value = [decoded_text]
    return processor, model, fake_inputs


class TestGraniteSpeechAdapterIsATranscriptionEngine:
    def test_should_implement_transcription_engine_interface(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        assert isinstance(adapter, TranscriptionEngine)


class TestGraniteSpeechAdapterLoadModel:
    def test_should_report_not_loaded_before_load(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        assert adapter.is_model_loaded() is False

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_processor_and_model_with_local_files_only(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/app/models"),
        )

        adapter.load_model()

        AutoProcessor.from_pretrained.assert_called_once_with(
            "ibm-granite/granite-speech-4.1-2b",
            cache_dir=str(Path("/app/models")),
            local_files_only=True,
        )
        load_kwargs = AutoModelForSpeechSeq2Seq.from_pretrained.call_args.kwargs
        load_args = AutoModelForSpeechSeq2Seq.from_pretrained.call_args.args
        assert load_args == ("ibm-granite/granite-speech-4.1-2b",)
        assert load_kwargs["cache_dir"] == str(Path("/app/models"))
        assert load_kwargs["local_files_only"] is True
        assert load_kwargs["device_map"] == "cpu"
        assert adapter.is_model_loaded() is True

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_default_to_bfloat16_dtype(self) -> None:
        # bfloat16 is Granite Speech 4.1's native training dtype. Unlike
        # nano-parakeet (which forces fp32 to dodge a kernel mismatch on
        # ROCm), Granite's bf16 path is well-trodden - keep the model
        # card's recommended dtype.
        import torch
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cuda",
            model_dir=Path("/app/models"),
        )

        adapter.load_model()

        kwargs = AutoModelForSpeechSeq2Seq.from_pretrained.call_args.kwargs
        assert kwargs["dtype"] is torch.bfloat16

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_honour_explicit_dtype_override(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cuda",
            model_dir=Path("/app/models"),
            dtype="float16",
        )

        adapter.load_model()

        kwargs = AutoModelForSpeechSeq2Seq.from_pretrained.call_args.kwargs
        assert kwargs["dtype"] is torch.float16


class TestGraniteSpeechAdapterTranscribeGuards:
    def test_should_raise_when_transcribing_before_load(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        with pytest.raises(TranscriptionError, match="not loaded"):
            adapter.transcribe(_make_audio())

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_raise_when_audio_sample_rate_is_not_16khz(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        audio_44k = _make_audio(sample_rate=44100, samples=44100)

        with pytest.raises(TranscriptionError, match="16000"):
            adapter.transcribe(audio_44k)


class TestGraniteSpeechAdapterTranscribe:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_build_chat_template_with_audio_token(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        processor, _, _ = _wire_successful_transcribe()

        adapter.transcribe(_make_audio())

        call = processor.tokenizer.apply_chat_template.call_args
        chat = call.args[0]
        assert call.kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        assert len(chat) == 1
        assert chat[0]["role"] == "user"
        assert "<|audio|>" in chat[0]["content"]

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_unsqueezed_torch_tensor_to_processor(self) -> None:
        import torch

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cuda",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        processor, _, _ = _wire_successful_transcribe()
        audio = _make_audio()

        adapter.transcribe(audio)

        torch.from_numpy.assert_called_once_with(audio.data)
        torch.from_numpy.return_value.unsqueeze.assert_called_once_with(0)
        expected_wav = torch.from_numpy.return_value.unsqueeze.return_value
        processor.assert_called_once_with(
            "<chat-prompt>",
            expected_wav,
            device="cuda",
            return_tensors="pt",
        )
        processor.return_value.to.assert_called_once_with("cuda")

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_decode_only_tokens_after_input_ids_length(self) -> None:
        # Granite is an autoregressive LLM: ``model.generate`` returns
        # ``[input_prompt_tokens || generated_tokens]``. Decoding the
        # full output would echo the chat template back. Slice off the
        # prompt before decoding.
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        processor, model, _ = _wire_successful_transcribe(n_input_tokens=42)

        adapter.transcribe(_make_audio())

        decoded_arg = processor.tokenizer.batch_decode.call_args.args[0]
        assert decoded_arg is model.generate.return_value.__getitem__.return_value
        getitem_key = model.generate.return_value.__getitem__.call_args.args[0]
        assert getitem_key == (slice(None, None, None), slice(42, None, None))
        decode_kwargs = processor.tokenizer.batch_decode.call_args.kwargs
        assert decode_kwargs == {"skip_special_tokens": True}

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_return_stripped_text_in_transcription_result(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )
        from voicetotext.transcription.result import TranscriptionResult

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        _wire_successful_transcribe(decoded_text="  Hello, world!  ")

        result = adapter.transcribe(_make_audio())

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, world!"
        assert result.language == "en"
        assert result.transcription_seconds >= 0.0


class TestGraniteSpeechAdapterMaxNewTokensBudget:
    """Output-token budget must scale with audio duration.

    A fixed budget silently truncates long recordings - a 121 s
    recording previously stopped mid-sentence at the 200-token cap
    even though the audio was captured fully.
    """

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_scale_budget_with_audio_duration_for_long_chunk(
        self,
    ) -> None:
        # 50 s fits in a single chunk so we can read the budget for that
        # duration directly off the (sole) ``generate`` call. Multi-chunk
        # behaviour is covered separately in the chunking tests.
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        _wire_successful_transcribe()
        fifty_seconds = _make_audio(samples=16000 * 50)

        adapter.transcribe(fifty_seconds)

        model = AutoModelForSpeechSeq2Seq.from_pretrained.return_value
        budget = model.generate.call_args.kwargs["max_new_tokens"]
        assert budget >= 500, (
            f"50 s of audio needs a generous output budget; got {budget}"
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_exceed_legacy_fixed_cap_for_thirty_second_chunk(
        self,
    ) -> None:
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        _wire_successful_transcribe()
        thirty_seconds = _make_audio(samples=16000 * 30)

        adapter.transcribe(thirty_seconds)

        model = AutoModelForSpeechSeq2Seq.from_pretrained.return_value
        budget = model.generate.call_args.kwargs["max_new_tokens"]
        assert budget > 200, (
            "Budget must exceed the legacy 200-token cap for a 30 s "
            f"chunk; got {budget}"
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_apply_floor_so_sub_second_audio_is_not_starved(
        self,
    ) -> None:
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        _wire_successful_transcribe()
        half_second = _make_audio(samples=8000)

        adapter.transcribe(half_second)

        model = AutoModelForSpeechSeq2Seq.from_pretrained.return_value
        budget = model.generate.call_args.kwargs["max_new_tokens"]
        assert budget >= 128, (
            f"Floor must protect short utterances; got {budget}"
        )


class TestGeneratedOnlyRepetitionPenalty:
    """Custom logits processor: scope the penalty to post-prompt tokens.

    Transformers' built-in ``RepetitionPenaltyLogitsProcessor`` also
    penalises prompt tokens, which the IBM team has documented as
    hurting transcription quality. Our variant skips the prompt slice
    entirely and only penalises tokens the model itself has emitted.
    """

    def test_should_return_scores_unchanged_when_only_prompt_present(
        self,
    ) -> None:
        import torch

        from voicetotext.transcription.granite_speech_adapter import (
            _GeneratedOnlyRepetitionPenalty,
        )

        processor = _GeneratedOnlyRepetitionPenalty(
            penalty=3.0, prompt_length=4,
        )
        input_ids = torch.tensor([[10, 20, 30, 40]])
        scores = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])

        result = processor(input_ids, scores.clone())

        assert torch.equal(result, scores)

    def test_should_divide_positive_scores_for_generated_tokens(
        self,
    ) -> None:
        # Prompt is the first 3 tokens; the model has just emitted tokens
        # 3 and 4. Only the score columns for those two tokens should be
        # divided by the penalty; the prompt-token column scores stay
        # untouched.
        import torch

        from voicetotext.transcription.granite_speech_adapter import (
            _GeneratedOnlyRepetitionPenalty,
        )

        processor = _GeneratedOnlyRepetitionPenalty(
            penalty=2.0, prompt_length=3,
        )
        input_ids = torch.tensor([[0, 1, 2, 3, 4]])
        scores = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

        result = processor(input_ids, scores)

        assert result[0, 0].item() == 1.0
        assert result[0, 1].item() == 1.0
        assert result[0, 2].item() == 1.0
        assert result[0, 3].item() == 0.5
        assert result[0, 4].item() == 0.5

    def test_should_multiply_negative_scores_for_generated_tokens(
        self,
    ) -> None:
        # For negative logits, multiplying by penalty pushes them further
        # negative - equivalent semantics to dividing positive logits.
        # Mirrors Transformers' built-in behaviour, just scoped.
        import torch

        from voicetotext.transcription.granite_speech_adapter import (
            _GeneratedOnlyRepetitionPenalty,
        )

        processor = _GeneratedOnlyRepetitionPenalty(
            penalty=2.0, prompt_length=1,
        )
        input_ids = torch.tensor([[0, 1]])
        scores = torch.tensor([[-1.0, -1.0]])

        result = processor(input_ids, scores)

        assert result[0, 0].item() == -1.0
        assert result[0, 1].item() == -2.0


class TestGraniteSpeechAdapterConfigurableBeamCount:
    """IBM's tech-report recipe sets ``num_beams=4`` for accuracy.
    Users who prefer the ~Nx speedup from greedy decoding can lower
    it via the existing ``transcription.beam_size`` config key -
    the adapter must honour whatever the engine factory passes in
    rather than hard-coding 4.
    """

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_default_to_four_beams_when_unspecified(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        _wire_successful_transcribe()

        adapter.transcribe(_make_audio())

        model = AutoModelForSpeechSeq2Seq.from_pretrained.return_value
        assert model.generate.call_args.kwargs["num_beams"] == 4

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_use_explicit_num_beams_when_passed(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
            num_beams=1,
        )
        adapter.load_model()
        _wire_successful_transcribe()

        adapter.transcribe(_make_audio())

        model = AutoModelForSpeechSeq2Seq.from_pretrained.return_value
        assert model.generate.call_args.kwargs["num_beams"] == 1


class TestGraniteSpeechAdapterIBMReferenceRecipe:
    """Inference must use IBM's reference recipe from the tech report:
    ``num_beams=4`` and ``repetition_penalty=3.0`` scoped to generated
    tokens. Greedy with no penalty is known to collapse into loops on
    audio outside the training distribution (~60 s).
    """

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_use_four_beams_per_ibm_reference_recipe(
        self,
    ) -> None:
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        _wire_successful_transcribe()

        adapter.transcribe(_make_audio())

        model = AutoModelForSpeechSeq2Seq.from_pretrained.return_value
        assert model.generate.call_args.kwargs["num_beams"] == 4

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_generated_only_repetition_penalty(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
            _GeneratedOnlyRepetitionPenalty,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        _wire_successful_transcribe(n_input_tokens=42)

        adapter.transcribe(_make_audio())

        model = AutoModelForSpeechSeq2Seq.from_pretrained.return_value
        processors = model.generate.call_args.kwargs["logits_processor"]
        scoped = [
            p for p in processors
            if isinstance(p, _GeneratedOnlyRepetitionPenalty)
        ]
        assert len(scoped) == 1, (
            "Expected exactly one generated-only repetition-penalty "
            f"processor; got {processors}"
        )
        assert scoped[0]._penalty == 3.0
        assert scoped[0]._prompt_length == 42


class TestGraniteSpeechAdapterChunking:
    """Granite is only trained on ≤60 s utterances. Long audio must be
    split into chunks below that threshold and transcribed sequentially,
    or the LLM hallucinates and degenerates into repetition.
    """

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_call_generate_once_for_audio_under_threshold(
        self,
    ) -> None:
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        _wire_successful_transcribe()
        fifty_seconds = _make_audio(samples=16000 * 50)

        adapter.transcribe(fifty_seconds)

        model = AutoModelForSpeechSeq2Seq.from_pretrained.return_value
        assert model.generate.call_count == 1

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_call_generate_multiple_times_for_long_audio(
        self,
    ) -> None:
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        _wire_successful_transcribe()
        two_minutes = _make_audio(samples=16000 * 120)

        adapter.transcribe(two_minutes)

        model = AutoModelForSpeechSeq2Seq.from_pretrained.return_value
        assert model.generate.call_count >= 2, (
            "120 s of audio must be split into ≥2 chunks below the "
            "Granite training-distribution limit"
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_join_chunk_transcripts_with_single_space(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        processor, _, _ = _wire_successful_transcribe()
        # One decoded text per chunk. 120 s of audio splits into at
        # least two chunks, and the helper rotates through this list as
        # successive ``batch_decode`` calls fire - so as long as the
        # adapter joins with a single space, the prefix of the final
        # text is deterministic regardless of the exact chunk count.
        processor.tokenizer.batch_decode.side_effect = [
            ["alpha"],
            [" beta "],
            ["gamma"],
            ["delta"],
            ["epsilon"],
        ]
        two_minutes = _make_audio(samples=16000 * 120)

        result = adapter.transcribe(two_minutes)

        assert result.text.startswith("alpha beta")
        assert "  " not in result.text

    def test_silence_aware_split_should_snap_to_quietest_window(self) -> None:
        # Construct 60 s of high-energy noise with a 0.5 s near-silence
        # gap at the 47 s mark. Asking the splitter for a target offset
        # of 50 s with a ±5 s search radius must snap the cut into that
        # quiet gap rather than the noisy area around 50 s.
        import numpy as np

        from voicetotext.transcription.granite_speech_adapter import (
            _find_quietest_offset,
        )

        rng = np.random.default_rng(0)
        sample_rate = 16000
        data = rng.standard_normal(sample_rate * 60).astype(np.float32)
        quiet_start = int(47 * sample_rate)
        quiet_end = int(47.5 * sample_rate)
        data[quiet_start:quiet_end] = 0.0
        target_offset = int(50 * sample_rate)

        split_at = _find_quietest_offset(data, target_offset, sample_rate)

        assert quiet_start <= split_at <= quiet_end, (
            f"split should fall inside the quiet zone "
            f"[{quiet_start}, {quiet_end}], got {split_at}"
        )


class TestGraniteSpeechAdapterRepetitionWarning:
    """Defence in depth: if a chunk still degenerates into repetition
    despite the recipe, surface it as a WARNING so the failure mode is
    debuggable instead of silently bad output.

    ``voicetotext.core.debug_logger`` sets ``propagate=False`` on the
    ``voicetotext`` logger at import time so production logs only land
    in the app's own debug file. That blocks pytest's ``caplog`` (which
    attaches to root) from seeing our records when any other test has
    imported the debug logger. We work around it by attaching the
    caplog handler directly to the adapter's logger for the duration of
    each test below.
    """

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_warn_when_chunk_output_loops(self, caplog) -> None:
        import logging as stdlib_logging

        from voicetotext.transcription import granite_speech_adapter

        adapter = granite_speech_adapter.GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        looping = (
            "the quick brown fox jumps over "
            "the quick brown fox jumps over "
            "the quick brown fox jumps over "
            "the quick brown fox jumps over"
        )
        _wire_successful_transcribe(decoded_text=looping)

        warnings = _capture_warnings(
            caplog, granite_speech_adapter._logger,
            lambda: adapter.transcribe(_make_audio()),
        )

        assert any(
            "repetit" in record.getMessage().lower() for record in warnings
        ), (
            "Expected a repetition warning; got records: "
            f"{[r.getMessage() for r in warnings]}"
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_not_warn_when_chunk_output_is_normal(
        self, caplog,
    ) -> None:
        from voicetotext.transcription import granite_speech_adapter

        adapter = granite_speech_adapter.GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        _wire_successful_transcribe(
            decoded_text="A normal English sentence with no looping.",
        )

        warnings = _capture_warnings(
            caplog, granite_speech_adapter._logger,
            lambda: adapter.transcribe(_make_audio()),
        )

        repetition_warnings = [
            r for r in warnings
            if "repetit" in r.getMessage().lower()
        ]
        assert not repetition_warnings, (
            "Expected no repetition warning; got "
            f"{[r.getMessage() for r in repetition_warnings]}"
        )


class TestGraniteSpeechAdapterWarmup:
    """Smoke test: the inherited default warmup must run end-to-end
    through Granite's chunked transcribe pipeline without crashing,
    exercising the beam search + repetition penalty kernels so the
    first user-triggered transcription is fast.
    """

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_invoke_model_generate_during_warmup(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        _wire_successful_transcribe()

        adapter.warmup()

        model = AutoModelForSpeechSeq2Seq.from_pretrained.return_value
        assert model.generate.call_count == 1


class TestGraniteSpeechAdapterDispose:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_clear_model_on_dispose(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        adapter.dispose()

        assert adapter.is_model_loaded() is False

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_release_gpu_cache_on_dispose_when_device_is_cuda(self) -> None:
        import sys

        torch_mock = MagicMock()
        sys.modules["torch"] = torch_mock

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
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

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        adapter.dispose()

        torch_mock.cuda.empty_cache.assert_not_called()

    def test_should_not_raise_on_dispose_when_not_loaded(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        adapter.dispose()


class TestGraniteSpeechAdapterDependencyMissing:
    """If ``transformers`` is missing the Granite adapter must raise
    :class:`EngineDependencyMissingError` with an actionable install
    hint rather than leaking a bare ``ImportError`` to the bootstrap
    layer.
    """

    def test_should_raise_engine_dependency_missing_when_transformers_not_installed(
        self,
    ) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        with patch.dict(
            "sys.modules", {"transformers": None, "torch": MagicMock()}
        ):
            with pytest.raises(EngineDependencyMissingError) as exc_info:
                adapter.load_model()

        message = str(exc_info.value).lower()
        assert "transformers" in message
        assert "pip install" in message
