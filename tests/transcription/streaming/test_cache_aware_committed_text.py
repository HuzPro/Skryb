"""Regression for the streaming truncation bug ('issue' typed as 'iss').

CPU-only: a fake SentencePiece + the real word-commit logic + the real output
writer. The model emits a word as subword tokens across chunks; live snapshots
must hold back the in-progress word so the word-count writer never types it
half-finished. Verifies the live-TYPED text equals the FINAL transcript.
"""
from voicetotext.transcription.streaming.cache_aware.streaming_session import (
    words_committed_text,
)
from voicetotext.transcription.streaming.streaming_output_writer import (
    StreamingOutputWriter,
)

from tests.fakes import FakeTextOutputHandler


class FakeSp:
    """Minimal SentencePiece stand-in; '▁' marks a word start."""

    def __init__(self, id_to_piece: dict[int, str]) -> None:
        self._pieces = id_to_piece

    def IdToPiece(self, token_id: int) -> str:  # noqa: N802 (SentencePiece name)
        return self._pieces[token_id]

    def DecodeIds(self, ids: list[int]) -> str:  # noqa: N802 (SentencePiece name)
        return "".join(self._pieces[i] for i in ids).replace("▁", " ").strip()


# "the issue next word"; 'issue' splits into subwords ▁iss + ue across chunks.
_PIECES = {1: "▁the", 2: "▁iss", 3: "ue", 4: "▁next", 5: "▁word"}


class TestWordsCommittedText:
    def test_should_hold_back_the_in_progress_word(self):
        sp = FakeSp(_PIECES)

        assert words_committed_text([1], sp) == ""            # 'the' in progress
        assert words_committed_text([1, 2], sp) == "the"      # 'iss…' in progress
        assert words_committed_text([1, 2, 3], sp) == "the"   # still 'issue'
        assert words_committed_text([1, 2, 3, 4], sp) == "the issue"

    def test_should_return_empty_for_no_tokens(self):
        assert words_committed_text([], FakeSp(_PIECES)) == ""


class TestTypedEqualsFinal:
    def test_live_typed_text_equals_final_through_the_writer(self):
        # Simulate per-chunk token growth driving the real output writer, then
        # the finalize flush - the typed text must equal the final transcript.
        sp = FakeSp(_PIECES)
        out = FakeTextOutputHandler()
        writer = StreamingOutputWriter(out)

        token_steps = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]
        for tokens in token_steps:
            writer.emit(words_committed_text(tokens, sp))
        writer.emit(sp.DecodeIds([1, 2, 3, 4, 5]).strip())  # finalize: all tokens

        typed = "".join(out.outputs)
        assert typed == "the issue next word"
        assert "iss " not in typed and not typed.endswith("iss")  # never truncated
