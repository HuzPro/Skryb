from voicetotext.output.handler import TextOutputHandler


class StreamingOutputWriter:
    """Pastes a growing committed transcript as append-only deltas.

    A streaming session re-emits its whole committed transcript each tick;
    pasting that full string every time would duplicate it at the cursor.
    Committed text is a monotonic prefix (LocalAgreement never retracts),
    so only the new suffix is pasted. A snapshot that doesn't extend the
    last one (e.g. a shorter string) is ignored.
    """

    def __init__(self, output: TextOutputHandler) -> None:
        self._output = output
        self._emitted = ""

    def emit(self, committed_text: str) -> None:
        if not committed_text.startswith(self._emitted):
            return
        delta = committed_text[len(self._emitted):]
        if not delta:
            return
        self._emitted = committed_text
        self._output.output(delta)

    def reset(self) -> None:
        self._emitted = ""
