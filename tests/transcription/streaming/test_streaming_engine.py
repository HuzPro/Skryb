from skryb.transcription.streaming.streaming_engine import StreamingUpdate


class TestStreamingUpdateText:
    def test_should_join_committed_and_tentative_with_a_space(self):
        update = StreamingUpdate(
            committed_text="one two", tentative_text="three", is_final=False,
        )

        assert update.text == "one two three"

    def test_should_be_committed_only_when_no_tentative(self):
        update = StreamingUpdate(
            committed_text="one two", tentative_text="", is_final=False,
        )

        assert update.text == "one two"

    def test_should_be_tentative_only_when_no_committed(self):
        update = StreamingUpdate(
            committed_text="", tentative_text="three", is_final=False,
        )

        assert update.text == "three"

    def test_should_be_empty_when_both_empty(self):
        update = StreamingUpdate(
            committed_text="", tentative_text="", is_final=True,
        )

        assert update.text == ""

    def test_should_store_is_final_flag(self):
        assert StreamingUpdate("", "", is_final=True).is_final is True
