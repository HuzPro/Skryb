from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from voicetotext.core.timestamps import utc_iso
from voicetotext.edit_capture.replay import EditReplay, EditState, KeyStroke
from voicetotext.edit_capture.window import ForegroundWindow

# A dictation left untouched for this long is treated as finished. The
# worker calls ``expire_idle_block`` when its queue goes quiet, so no
# timer is needed.
IDLE_TIMEOUT = timedelta(minutes=15)

# The pipeline hands over the text on the worker thread, but the paste
# only reaches the target after the Tk dispatch, the clipboard settle and
# the wait for the hotkey's modifiers to come up. A delivery still
# unclaimed after this long never became a paste - the clipboard-only
# output mode, or a failed clipboard snapshot.
DELIVERY_TIMEOUT = timedelta(seconds=5)

# Typing in these is never dictation editing worth keeping.
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
        self._pending: deque[_Delivery] = deque()
        self._block: _Block | None = None
        self._deliveries = 0
        self._blocks = 0

    def delivered(self, text: str, pipeline_mode: str, output_mode: str) -> None:
        self._deliveries += 1
        delivery = _Delivery(self._deliveries, text, self._now())
        self._pending.append(delivery)
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
        if window.process in DENIED_PROCESSES:
            self._write({
                "event": "skipped",
                "seq": delivery.seq,
                "at": utc_iso(self._now()),
                "process": window.process,
                "reason": "denied_process",
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
        if state.committed:
            self._finish(block, state, "enter")

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
        deadline = self._now() - DELIVERY_TIMEOUT
        while self._pending:
            delivery = self._pending.popleft()
            if delivery.at >= deadline:
                return delivery
        return None

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
