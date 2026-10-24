from skryb.output.handler import TextOutputHandler


class StreamingOutputWriter:
    """Pastes a growing committed transcript as append-only word deltas.

    A streaming session re-emits its whole committed transcript each tick;
    pasting that full string every time would duplicate it at the cursor.
    The committed transcript only ever grows by whole words (LocalAgreement
    never retracts, and finalize only appends), so this writer tracks how
    many words it has already pasted and emits only the new trailing words.

    Tracking *words* - not a raw string prefix - is deliberate. The
    full-utterance decode at finalize can re-punctuate a word that was
    already committed and pasted (e.g. ``"Testing"`` becomes ``"Testing."``
    once the model hears the trailing pause). A naive string-prefix check
    would then see the new snapshot as a non-extension and silently drop the
    whole update, including the genuinely new final word. Counting words
    keeps the new tail flowing while never re-pasting (or retracting) the
    earlier words whose punctuation we can no longer change at the cursor.
    """

    def __init__(self, output: TextOutputHandler) -> None:
        self._output = output
        self._emitted_word_count = 0

    def emit(self, committed_text: str) -> None:
        words = committed_text.split()
        new_words = words[self._emitted_word_count:]
        if not new_words:
            return
        self._emitted_word_count = len(words)
        self._output.output(self._delta_for(new_words))

    def _delta_for(self, new_words: list[str]) -> str:
        suffix = " ".join(new_words)
        leads_the_line = self._emitted_word_count == len(new_words)
        return suffix if leads_the_line else " " + suffix

    def set_output(self, output: TextOutputHandler) -> None:
        """Send later deltas somewhere else without forgetting the earlier ones.

        Saving settings rebuilds the output chain, which can happen while a
        dictation is still running. Counting from zero again would re-paste
        the whole committed transcript at the cursor.
        """
        self._output = output

    def reset(self) -> None:
        self._emitted_word_count = 0
