import logging
import queue
import threading
from dataclasses import dataclass

from voicetotext.edit_capture.recorder import EditRecorder
from voicetotext.edit_capture.replay import KeyStroke
from voicetotext.edit_capture.win32_foreground import (
    foreground_hwnd,
    foreground_window,
)
from voicetotext.edit_capture.win32_input import (
    clicked_since_last_check,
    key_character,
)
from voicetotext.output.win32_key_sender import VOICETOTEXT_INPUT_TAG

_logger = logging.getLogger(__name__)

_VK_V = 0x56
_VK_SHIFT = 0xA0, 0xA1, 0x10
_VK_CONTROL = 0xA2, 0xA3, 0x11
_VK_ALT = 0xA4, 0xA5, 0x12
_MODIFIER_KEYS = frozenset(_VK_SHIFT + _VK_CONTROL + _VK_ALT)

# How long the worker waits for a key before checking whether the open
# dictation has gone idle. Nothing depends on the exact value; it only
# bounds how late an idle block is closed.
_IDLE_POLL_SECONDS = 30.0
_STOP_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class _RawKey:
    is_down: bool
    vk: int
    extra: int


@dataclass(frozen=True)
class _Delivery:
    text: str
    pipeline_mode: str
    output_mode: str


class _Stop:
    pass


_QueueItem = _RawKey | _Delivery | _Stop


class Win32EditObserver:
    """Feeds the recorder from the keyboard hook, on its own thread.

    The hook procedure runs on every keystroke in the system and Windows
    silently removes a hook that takes too long, so ``observe_key`` only
    puts the event on a queue. Deliveries go through the same queue, which
    is what keeps the recorder single-threaded and in order: a delivery is
    always enqueued before the paste it produced.
    """

    def __init__(self, recorder: EditRecorder) -> None:
        self._recorder = recorder
        self._queue: queue.SimpleQueue[_QueueItem] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._held: set[int] = set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="edit-capture", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._thread = None
        self._queue.put(_Stop())
        thread.join(timeout=_STOP_TIMEOUT_SECONDS)

    def observe_key(
        self, is_key_down: bool, vk: int, _flags: int, extra: int,
    ) -> None:
        self._queue.put(_RawKey(is_down=is_key_down, vk=vk, extra=extra))

    def observe_delivery(
        self, text: str, pipeline_mode: str, output_mode: str,
    ) -> None:
        self._queue.put(_Delivery(text, pipeline_mode, output_mode))

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=_IDLE_POLL_SECONDS)
            except queue.Empty:
                self._handle_idle()
                continue
            if isinstance(item, _Stop):
                self._recorder.dispose()
                return
            self._handle(item)

    def _handle_idle(self) -> None:
        try:
            self._recorder.expire_idle_block()
        except Exception as exc:  # noqa: BLE001 - capture must never crash
            _logger.warning("Edit capture failed to close an idle block: %s", exc)

    def _handle(self, item: _QueueItem) -> None:
        try:
            self._dispatch(item)
        except Exception as exc:  # noqa: BLE001 - capture must never crash
            _logger.warning("Edit capture dropped an event: %s", exc)

    def _dispatch(self, item: _QueueItem) -> None:
        if isinstance(item, _Delivery):
            self._recorder.delivered(
                item.text, item.pipeline_mode, item.output_mode,
            )
            return
        if isinstance(item, _RawKey):
            self._dispatch_key(item)

    def _dispatch_key(self, raw: _RawKey) -> None:
        if raw.extra == VOICETOTEXT_INPUT_TAG:
            self._handle_own_input(raw)
            return
        self._track_modifier(raw)
        if not raw.is_down or raw.vk in _MODIFIER_KEYS:
            return
        watched = self._recorder.watching_hwnd
        if watched == 0 or foreground_hwnd() != watched:
            return
        self._recorder.keyed(self._stroke_for(raw.vk), watched)

    def _handle_own_input(self, raw: _RawKey) -> None:
        # Our own Ctrl+V, injected by the paste. It is the only reliable
        # signal that the transcription actually reached the window, and
        # its modifiers must not disturb the user's tracked state. The
        # discarded click check clears any click from before the paste, so
        # the first keystroke does not inherit one.
        if raw.is_down and raw.vk == _VK_V:
            clicked_since_last_check()
            self._recorder.pasted(foreground_window())

    def _track_modifier(self, raw: _RawKey) -> None:
        if raw.vk not in _MODIFIER_KEYS:
            return
        if raw.is_down:
            self._held.add(raw.vk)
        else:
            self._held.discard(raw.vk)

    def _stroke_for(self, vk: int) -> KeyStroke:
        shift = self._is_held(_VK_SHIFT)
        ctrl = self._is_held(_VK_CONTROL)
        alt = self._is_held(_VK_ALT)
        return KeyStroke(
            vk=vk,
            char=key_character(vk, shift, ctrl, alt),
            shift=shift,
            ctrl=ctrl,
            alt=alt,
            click_since_last=clicked_since_last_check(),
        )

    def _is_held(self, keys: tuple[int, ...]) -> bool:
        return any(key in self._held for key in keys)
