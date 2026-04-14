from voicetotext.output.output_mode import OutputMode


class TestOutputModeMigrate:
    def test_should_map_direct_paste_string_to_enum_member(self):
        assert OutputMode.migrate("direct_paste") is OutputMode.DIRECT_PASTE

    def test_should_map_paste_and_clipboard_string_to_enum_member(self):
        assert OutputMode.migrate("paste_and_clipboard") is OutputMode.PASTE_AND_CLIPBOARD

    def test_should_map_clipboard_only_string_to_enum_member(self):
        assert OutputMode.migrate("clipboard_only") is OutputMode.CLIPBOARD_ONLY

    def test_should_rename_legacy_auto_paste_to_paste_and_clipboard(self):
        assert OutputMode.migrate("auto_paste") is OutputMode.PASTE_AND_CLIPBOARD

    def test_should_default_to_direct_paste_when_raw_is_none(self):
        assert OutputMode.migrate(None) is OutputMode.DIRECT_PASTE

    def test_should_default_to_direct_paste_for_unknown_string(self):
        assert OutputMode.migrate("not_a_real_mode") is OutputMode.DIRECT_PASTE


class TestOutputModePersistedValuesContract:
    """The enum values are what lands in config.json — changing them is a
    persisted-format break. These tests pin the contract."""

    def test_direct_paste_value_is_direct_paste(self):
        assert OutputMode.DIRECT_PASTE.value == "direct_paste"

    def test_paste_and_clipboard_value_is_paste_and_clipboard(self):
        assert OutputMode.PASTE_AND_CLIPBOARD.value == "paste_and_clipboard"

    def test_clipboard_only_value_is_clipboard_only(self):
        assert OutputMode.CLIPBOARD_ONLY.value == "clipboard_only"


class TestOutputModeDisplayLabels:
    def test_every_member_has_a_non_empty_display_label(self):
        for mode in OutputMode:
            assert mode.display_label
            assert isinstance(mode.display_label, str)
