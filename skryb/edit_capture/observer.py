import logging
import queue
import threading
from dataclasses import dataclass

from skryb.edit_capture.recorder import EditRecorder
from skryb.edit_capture.replay import EditFlag, KeyStroke
from skryb.edit_capture.win32_foreground import (
    foreground_hwnd,
    foreground_window,
)
from skryb.edit_capture.win32_input import (
    caps_lock_enabled,
    clicked_since_last_check,
    key_character,
)
from skryb.output.win32_key_sender import SKRYB_INPUT_TAG

_logger = logging.getLogger(__name__)

_VK_V = 0x56
_VK_CAPITAL = 0x14
_VK_SHIFT = 0xA0, 0xA1, 0x10
_VK_CONTROL = 0xA2, 0xA3, 0x11
_VK_ALT = 0xA4, 0xA5, 0x12
_MODIFIER_KEYS = frozenset(_VK_SHIFT + _VK_CONTROL + _VK_ALT)

# ``LLKHF_INJECTED``. Set for our own paste, which the tag identifies, and
# for every other synthetic key: password-manager auto-type, AutoHotkey,
# the on-screen keyboard, a remote session. None of those are the user
# editing a dictation, and auto-type would write a password to the log.
_INJECTED = 0x10

# How long the worker waits for a key before checking whether the open
# dictation has gone idle. Nothing depends on the exact value; it only
# bounds how late an idle block is closed.
_IDLE_POLL_SECONDS = 30.0
_STOP_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class _RawKey:
    is_down: bool
    vk: int
    flags: int
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
        self._caps_lock = caps_lock_enabled()

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
        self, is_key_down: bool, vk: int, flags: int, extra: int,
    ) -> None:
        self._queue.put(
            _RawKey(is_down=is_key_down, vk=vk, flags=flags, extra=extra),
        )

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
            self._handle_idle()
            self._handle(item)

    def _handle_idle(self) -> None:
        # Checked on every event, not only when the queue goes quiet: the
        # queue sees every keystroke in the system, so an idle dictation
        # would otherwise stay open, and armed, until the user stopped
        # typing everywhere for a full poll interval.
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
        if raw.extra == SKRYB_INPUT_TAG:
            self._handle_own_input(raw)
            return
        if raw.vk == _VK_CAPITAL:
            self._track_caps_lock(raw)
        if raw.flags & _INJECTED:
            self._recorder.abandon(EditFlag.EXTERNAL_INPUT)
            return
        self._track_modifier(raw)
        if not raw.is_down or raw.vk in _MODIFIER_KEYS:
            return
        watched = self._recorder.watching_hwnd
        if watched == 0:
            return
        if foreground_hwnd() != watched:
            self._recorder.abandon(EditFlag.FOCUS_CHANGED)
            return
        self._recorder.keyed(self._stroke_for(raw.vk), watched)

    def _handle_own_input(self, raw: _RawKey) -> None:
        if raw.is_down and raw.vk == _VK_V:
            if clicked_since_last_check():
                self._recorder.abandon(EditFlag.MOUSE_CLICK)
            self._recorder.pasted(foreground_window())

    def _track_caps_lock(self, raw: _RawKey) -> None:
        if raw.is_down:
            if raw.vk not in self._held:
                self._caps_lock = not self._caps_lock
            self._held.add(raw.vk)
        else:
            self._held.discard(raw.vk)

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
            char=key_character(vk, shift, ctrl, alt, caps_lock=self._caps_lock),
            shift=shift,
            ctrl=ctrl,
            alt=alt,
            click_since_last=clicked_since_last_check(),
        )

    def _is_held(self, keys: tuple[int, ...]) -> bool:
        return any(key in self._held for key in keys)
