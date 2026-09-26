import logging

from voicetotext.output.key_sender import KeySender
from voicetotext.output.modifier_key_state_reader import ModifierKeyStateReader
from voicetotext.output.paste_method import PasteMethod
from voicetotext.output.sleeper import Sleeper


_logger = logging.getLogger(__name__)

# The hotkey that triggered the transcription (e.g. Ctrl+Shift+Alt+M)
# is usually still physically held when the synthetic Ctrl+V goes out
# ~150ms later. The target application then sees Ctrl+Shift+Alt+V and
# drops the paste. Poll until the user's fingers are off the modifiers,
# but never stall the pipeline longer than the timeout - a stuck or
# remapped key must not swallow the transcription entirely.
_POLL_INTERVAL_SECONDS = 0.01
_MAX_WAIT_SECONDS = 1.0
_MAX_POLLS = int(_MAX_WAIT_SECONDS / _POLL_INTERVAL_SECONDS)


class ModifierReleaseWaitingKeySender(KeySender):

    def __init__(
        self,
        inner: KeySender,
        modifier_state: ModifierKeyStateReader,
        sleeper: Sleeper,
    ) -> None:
        self._inner = inner
        self._modifier_state = modifier_state
        self._sleeper = sleeper

    def send_paste(self, method: PasteMethod, text: str) -> None:
        self._wait_for_modifier_release()
        self._inner.send_paste(method, text)

    def _wait_for_modifier_release(self) -> None:
        for _ in range(_MAX_POLLS):
            if not self._modifier_state.any_modifier_down():
                return
            self._sleeper.sleep(_POLL_INTERVAL_SECONDS)
        _logger.warning(
            "Modifier keys still held after %.1fs; pasting anyway.",
            _MAX_WAIT_SECONDS,
        )
