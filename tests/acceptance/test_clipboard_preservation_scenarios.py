"""Clipboard preservation - the user's clipboard contents survive
the transcription paste."""

from tests.acceptance.app_fixture import SkrybApp


class TestClipboardIsPreserved:
    def test_original_clipboard_contents_are_restored_after_paste(
        self, voice_to_text_app: SkrybApp
    ):
        voice_to_text_app.user_speaks("hello")

        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.clipboard_was_preserved

    def test_save_is_called_exactly_once_per_transcription(
        self, voice_to_text_app: SkrybApp
    ):
        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.clipboard_saved_count == 1
        assert voice_to_text_app.clipboard_restored_count == 1


class TestClipboardFailures:
    def test_restore_runs_even_when_paste_fails(
        self, voice_to_text_app: SkrybApp
    ):
        voice_to_text_app.output_fails_with(RuntimeError("paste crashed"))

        # The decorator re-raises the inner failure; tests shouldn't
        # crash the suite over it.
        try:
            voice_to_text_app.press_hotkey()
            voice_to_text_app.press_hotkey()
        except RuntimeError:
            pass

        assert voice_to_text_app.clipboard_restored_count == 1

    def test_save_failure_aborts_paste_and_does_not_type_anything(
        self, voice_to_text_app: SkrybApp
    ):
        voice_to_text_app.clipboard_save_fails_with(OSError("locked"))

        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.typed_text is None
        assert voice_to_text_app.clipboard_restored_count == 0
