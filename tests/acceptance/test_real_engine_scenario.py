"""End-to-end scenario with a real Whisper engine and a real MP3.

This test is the Dave-Farley ATDD layer: it reads as a business
scenario, uses the same ``VoiceToTextApp`` DSL as the rest of the
acceptance suite, and swaps in real components only where it matters
(the transcription engine and the recorded audio).  The rest of the
pipeline - state machine, event bus, hotkey wiring, clipboard
preservation, output decorator - stays the same as in production.

Skipped by default: gated behind ``VOICETOTEXT_RUN_REAL_MODEL_TESTS``
because importing torch and loading Whisper weights on Windows can
auto-expand the OS pagefile to fill the disk. Also skips when the
audio fixture or the ``base`` model is missing from the local
HuggingFace cache.
"""

import os

import pytest

from voicetotext.core.state_machine import AppState

from tests.acceptance.app_fixture import VoiceToTextApp
from tests.fixtures import require_short_fixture
from tests.helpers.gpu import require_cuda
from tests.helpers.model_cache import require_installed_model


pytestmark = [pytest.mark.acceptance, pytest.mark.slow]

_REAL_MODEL_OPT_IN = "VOICETOTEXT_RUN_REAL_MODEL_TESTS"


@pytest.mark.skipif(
    not os.environ.get(_REAL_MODEL_OPT_IN),
    reason=(
        "Real-engine acceptance test gated behind "
        f"{_REAL_MODEL_OPT_IN}=1 - opting in imports torch and loads "
        "Whisper weights, which can balloon the Windows pagefile to "
        "fill the disk."
    ),
)
class TestRealEngineTranscribesRealAudio:
    def test_should_transcribe_recorded_audio_and_type_text_when_user_toggles_hotkey(
        self, voice_to_text_app: VoiceToTextApp
    ) -> None:
        require_cuda()
        require_installed_model("base")
        voice_to_text_app.uses_real_engine(model="base")
        voice_to_text_app.microphone_captures_from_file(require_short_fixture())

        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.state == AppState.IDLE
        assert voice_to_text_app.typed_text is not None
        assert "testing" in voice_to_text_app.typed_text.lower()
