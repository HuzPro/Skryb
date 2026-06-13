from voicetotext.transcription.streaming.streaming_output_writer import (
    StreamingOutputWriter,
)

from tests.fakes import FakeTextOutputHandler


class TestStreamingOutputWriter:
    def test_should_emit_full_text_on_first_emit(self):
        out = FakeTextOutputHandler()

        StreamingOutputWriter(out).emit("Testing")

        assert out.outputs == ["Testing"]

    def test_should_emit_only_the_new_suffix_on_growth(self):
        out = FakeTextOutputHandler()
        writer = StreamingOutputWriter(out)

        writer.emit("Testing")
        writer.emit("Testing one two")

        assert out.outputs == ["Testing", " one two"]

    def test_should_keep_the_leading_space_in_the_delta(self):
        out = FakeTextOutputHandler()
        writer = StreamingOutputWriter(out)

        writer.emit("hello")
        writer.emit("hello world")

        assert out.outputs[-1] == " world"

    def test_should_emit_nothing_when_committed_is_unchanged(self):
        out = FakeTextOutputHandler()
        writer = StreamingOutputWriter(out)

        writer.emit("Testing")
        writer.emit("Testing")

        assert out.outputs == ["Testing"]

    def test_should_restart_from_empty_after_reset(self):
        out = FakeTextOutputHandler()
        writer = StreamingOutputWriter(out)

        writer.emit("first")
        writer.reset()
        writer.emit("second")

        assert out.outputs == ["first", "second"]

    def test_should_ignore_a_shorter_non_prefix_string(self):
        out = FakeTextOutputHandler()
        writer = StreamingOutputWriter(out)

        writer.emit("hello world")
        writer.emit("hello")  # never retract already-emitted text

        assert out.outputs == ["hello world"]

    def test_should_emit_nothing_for_empty_committed(self):
        out = FakeTextOutputHandler()

        StreamingOutputWriter(out).emit("")

        assert out.outputs == []

    def test_should_append_new_word_when_an_earlier_word_is_repunctuated(self):
        # Regression for the dropped final word: the full-utterance finalize
        # re-decode can re-punctuate an already-emitted word ("Testing" ->
        # "Testing." once the model hears the trailing pause). The genuinely
        # new final word must still be pasted, not discarded because the
        # whole string no longer prefix-matches what was already emitted.
        out = FakeTextOutputHandler()
        writer = StreamingOutputWriter(out)

        writer.emit("Testing")
        writer.emit("Testing one")
        writer.emit("Testing one two,")
        writer.emit("Testing. one two, three.")

        assert out.outputs[-1] == " three."

    def test_should_not_repaste_earlier_words_on_repunctuation(self):
        out = FakeTextOutputHandler()
        writer = StreamingOutputWriter(out)

        writer.emit("Testing one two,")
        writer.emit("Testing. one two, three.")

        assert "".join(out.outputs) == "Testing one two, three."

    def test_typed_output_equals_final_for_whole_word_growth(self):
        # The invariant a streaming session must uphold: feeding monotonic
        # WHOLE-WORD-growing snapshots types exactly the final text.
        out = FakeTextOutputHandler()
        writer = StreamingOutputWriter(out)

        for snapshot in ["", "this", "this is", "this is being", "this is being typed"]:
            writer.emit(snapshot)

        assert "".join(out.outputs) == "this is being typed"

    def test_cannot_extend_a_growing_last_word(self):
        # Documents WHY the cache-aware session holds back the in-progress
        # word: if a snapshot grows the LAST word in place (subword completion
        # '_iss' -> 'ue'), the word-count writer cannot extend it, so 'issue'
        # would be typed as 'iss'. The session must only emit complete words.
        out = FakeTextOutputHandler()
        writer = StreamingOutputWriter(out)

        writer.emit("the iss")
        writer.emit("the issue")

        assert "".join(out.outputs) == "the iss"
