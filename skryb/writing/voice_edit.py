"""Rewrite the selected text by voice: "make this shorter", "change Thursday to Friday".

The selection is read when the edit starts and read again just before the
replacement is pasted. If the window or the selection moved in between, the
replacement is dropped: a paste then would overwrite something else.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from skryb.edit_capture.recorder import DENIED_PROCESSES
from skryb.edit_capture.window import ForegroundWindow
from skryb.output.caret_context import NO_CONTEXT, CaretContext
from skryb.writing.cleanup import CompletionClient
from skryb.writing.client import WritingUnavailable
from skryb.writing.prompt import (
    EDIT_SYSTEM_PROMPT,
    edit_user_message,
    estimated_tokens,
)

MAX_SELECTION_CHARS = 6000
_LEFT_ALONE = "The selection was left unchanged."


class VoiceEditError(Exception):
    """Why a voice edit left the selection alone, in words for the user."""


@dataclass(frozen=True)
class Selection:
    text: str
    window: ForegroundWindow


class VoiceEditor:
    def __init__(
        self,
        client: CompletionClient,
        deadline: Callable[[], float],
        read_context: Callable[[], CaretContext],
        read_window: Callable[[], ForegroundWindow],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._deadline = deadline
        self._read_context = read_context
        self._read_window = read_window
        self._clock = clock

    def capture(self) -> Selection | None:
        """The selected text right now, or ``None`` when there is none to edit."""
        window = self._read_window()
        if window.process in DENIED_PROCESSES:
            return None
        selected = self._read_context().selected
        if not selected.strip():
            return None
        if len(selected) > MAX_SELECTION_CHARS:
            raise VoiceEditError("Selection too long to edit")
        return Selection(selected, window)

    def replacement(self, selection: Selection, instruction: str) -> str:
        """What to paste over ``selection``; raises ``VoiceEditError`` instead."""
        if not instruction.strip():
            raise VoiceEditError(f"No instruction was heard. {_LEFT_ALONE}")
        answer = self._rewrite(selection.text, instruction.strip())
        self._require_unchanged(selection)
        return answer

    def _rewrite(self, text: str, instruction: str) -> str:
        deadline = self._deadline()
        started = self._clock()
        try:
            answer = self._client.complete(
                EDIT_SYSTEM_PROMPT,
                edit_user_message(text, instruction),
                estimated_tokens(text + instruction) * 2 + 64,
                deadline,
            )
        except WritingUnavailable as exc:
            raise VoiceEditError(
                f"The writing model could not edit the selection ({exc}). {_LEFT_ALONE}",
            ) from exc
        if self._clock() - started > deadline:
            raise VoiceEditError(f"Editing took longer than {deadline:g} s. {_LEFT_ALONE}")
        cleaned = answer.strip()
        if not cleaned or "<think>" in cleaned:
            raise VoiceEditError(f"The writing model gave no usable text. {_LEFT_ALONE}")
        return cleaned

    def _require_unchanged(self, selection: Selection) -> None:
        window = self._read_window()
        if window.hwnd != selection.window.hwnd:
            raise VoiceEditError(f"The window changed while editing. {_LEFT_ALONE}")
        context = self._read_context()
        if context == NO_CONTEXT:
            if window.title != selection.window.title:
                raise VoiceEditError(f"The window changed while editing. {_LEFT_ALONE}")
            return
        if context.selected != selection.text:
            raise VoiceEditError(f"The selection changed while editing. {_LEFT_ALONE}")
