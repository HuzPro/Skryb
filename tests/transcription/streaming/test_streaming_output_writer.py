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
