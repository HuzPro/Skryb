from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from voicetotext.core.timestamps import utc_iso
from voicetotext.edit_capture.replay import (
    VK_ESCAPE,
    VK_TAB,
    EditReplay,
    EditState,
    KeyStroke,
)
from voicetotext.edit_capture.window import ForegroundWindow

# A dictation left untouched for this long is treated as finished. The
# worker calls ``expire_idle_block`` when its queue goes quiet, so no
# timer is needed.
IDLE_TIMEOUT = timedelta(minutes=15)

# A delivery is reported directly above the Ctrl+V injection, so the wait
# is only the clipboard settle plus the wait for the hotkey's modifiers to
# come up. Anything older than this never became a paste: SendInput was
# refused by a higher-integrity window, or the clipboard write failed.
DELIVERY_TIMEOUT = timedelta(seconds=5)

# Typing in these is never dictation editing worth keeping.
# Tab moves focus to another control in the same window and Escape clears
# the box in many apps. Either way the keys that follow are no longer
# editing this dictation, so the block ends rather than logging them.
FOCUS_LEAVING_KEYS = frozenset({VK_TAB, VK_ESCAPE})

DENIED_PROCESSES = frozenset({
    "windowsterminal.exe",
    "wt.exe",
    "powershell.exe",
    "pwsh.exe",
    "cmd.exe",
    "conhost.exe",
    "keepass.exe",
    "keepassxc.exe",
    "1password.exe",
    "bitwarden.exe",
})


def _ending_for(stroke: KeyStroke, state: EditState) -> str:
    """Why this keystroke ends the dictation, or an empty string."""
    if state.committed:
        return "enter"
    if stroke.vk in FOCUS_LEAVING_KEYS:
        return "focus_left"
    return ""


@dataclass(frozen=True)
class _Delivery:
    seq: int
    text: str
    at: datetime


@dataclass
class _Block:
    identifier: int
    window: ForegroundWindow
    replay: EditReplay
    last_at: datetime
    deliveries: list[int] = field(default_factory=list)
    key_count: int = 0


class EditRecorder:
    """Writes what each pasted transcription became before it was sent.

    Deliveries and pastes are separate events joined by order: the text
    leaves the pipeline on the worker thread and the paste lands up to a
    second later on the GUI thread, so there is no single point that sees
    both.
    """

    def __init__(
        self,
        write: Callable[[dict[str, object]], None],
        now: Callable[[], datetime],
    ) -> None:
        self._write = write
        self._now = now
        # One slot, not a queue. The GUI thread runs report, clipboard
        # write, settle and SendInput as one sequence, so only one
        # delivery is ever in flight. A delivery whose paste never went
        # out is then overwritten by the next one instead of being
        # claimed by it, which would have labelled the audio with the
        # wrong dictation.
        self._pending: _Delivery | None = None
        self._block: _Block | None = None
        self._deliveries = 0
        self._blocks = 0

    def delivered(self, text: str, pipeline_mode: str, output_mode: str) -> None:
        self._deliveries += 1
        delivery = _Delivery(self._deliveries, text, self._now())
        self._pending = delivery
        self._write({
            "event": "delivered",
            "seq": delivery.seq,
            "at": utc_iso(delivery.at),
            "text": text,
            "pipeline_mode": pipeline_mode,
            "output_mode": output_mode,
        })

    def pasted(self, window: ForegroundWindow) -> None:
        delivery = self._claim_delivery()
        if delivery is None:
            return
        refusal = self._refusal_for(window)
        if refusal:
            self._write({
                "event": "skipped",
                "seq": delivery.seq,
                "at": utc_iso(self._now()),
                "process": window.process,
                "reason": refusal,
            })
            return
        block = self._block_for(window)
        block.replay.apply_paste(delivery.text)
        block.deliveries.append(delivery.seq)
        block.last_at = self._now()
        self._write({
            "event": "paste",
            "block": block.identifier,
            "seq": delivery.seq,
            "at": utc_iso(block.last_at),
            "hwnd": window.hwnd,
            "pid": window.pid,
            "process": window.process,
            "title": window.title,
        })

    def _refusal_for(self, window: ForegroundWindow) -> str:
        """Why this window must not be watched, or an empty string."""
        if window.hwnd == 0:
            # An elevated or otherwise unreadable window. Its process name
            # is empty, so the denylist cannot speak for it, and every such
            # window shares hwnd 0, which would merge unrelated dictations
            # into one block.
            return "unknown_window"
        if window.process in DENIED_PROCESSES:
            return "denied_process"
        return ""

    @property
    def watching_hwnd(self) -> int:
        """The window whose keys matter, or 0 when nothing is being edited."""
        block = self._block
        return 0 if block is None else block.window.hwnd

    def keyed(self, stroke: KeyStroke, hwnd: int) -> None:
        block = self._block
        if block is None or hwnd != block.window.hwnd:
            return
        block.last_at = self._now()
        block.key_count += 1
        state = block.replay.apply_key(stroke)
        ending = _ending_for(stroke, state)
        try:
            self._write_key(block, stroke)
        finally:
            if ending:
                self._finish(block, state, ending)

    def _write_key(self, block: _Block, stroke: KeyStroke) -> None:
        self._write({
            "event": "key",
            "block": block.identifier,
            "at": utc_iso(block.last_at),
            "vk": stroke.vk,
            "char": stroke.char,
            "shift": stroke.shift,
            "ctrl": stroke.ctrl,
            "alt": stroke.alt,
            "click": stroke.click_since_last,
        })

    def expire_idle_block(self) -> None:
        block = self._block
        if block is None or self._now() - block.last_at < IDLE_TIMEOUT:
            return
        self._finish(block, block.replay.snapshot(), "idle")

    def dispose(self) -> None:
        block = self._block
        if block is None:
            return
        self._finish(block, block.replay.snapshot(), "shutdown")

    def _claim_delivery(self) -> _Delivery | None:
        delivery = self._pending
        self._pending = None
        if delivery is None or delivery.at < self._now() - DELIVERY_TIMEOUT:
            return None
        return delivery

    def _block_for(self, window: ForegroundWindow) -> _Block:
        block = self._block
        if block is not None:
            if block.window.hwnd == window.hwnd:
                return block
            self._finish(block, block.replay.snapshot(), "paste_elsewhere")
        return self._open_block(window)

    def _open_block(self, window: ForegroundWindow) -> _Block:
        self._blocks += 1
        block = _Block(
            identifier=self._blocks,
            window=window,
            replay=EditReplay(),
            last_at=self._now(),
        )
        self._block = block
        return block

    def _finish(self, block: _Block, state: EditState, reason: str) -> None:
        self._block = None
        self._write({
            "event": "sent",
            "block": block.identifier,
            "at": utc_iso(self._now()),
            "reason": reason,
            "process": block.window.process,
            "title": block.window.title,
            "text": state.text,
            "reliable": state.reliable,
            "flags": sorted(flag.value for flag in state.flags),
            "key_count": block.key_count,
            "spans": [
                {
                    "seq": block.deliveries[span.ordinal],
                    "pasted_text": span.pasted_text,
                    "final_text": span.final_text,
                }
                for span in state.spans
                if span.ordinal < len(block.deliveries)
            ],
        })
