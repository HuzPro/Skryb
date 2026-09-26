"""Save & Apply from the shell takes effect at once, through the app's own ``apply_settings``."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

from tests.ui.rig import Rig
from voicetotext.bootstrap import apply_settings
from voicetotext.settings.settings_draft import ChangeSet


def _applied_live(make_rig: Callable[..., Rig]) -> tuple[Rig, MagicMock, MagicMock]:
    use_case, app = MagicMock(), MagicMock()
    rigs: list[Rig] = []

    def apply(changes: ChangeSet) -> None:
        apply_settings(changes, rigs[0].config, use_case, app, gui_dispatcher=MagicMock())

    rigs.append(make_rig(on_save=apply))
    return rigs[0], use_case, app


@patch("voicetotext.bootstrap._build_hotkey_listener")
def test_new_shortcuts_rebuild_the_hotkey_listener(build_listener: MagicMock, make_rig) -> None:
    rig, use_case, app = _applied_live(make_rig)
    status, settings = rig.call("GET", "/api/settings")
    settings["hotkey_bindings"] = [
        {"keys": "ctrl+alt+space", "mode": "hold", "action": "dictate"},
        {"keys": "mouse4", "mode": "toggle", "action": "dictate"},
    ]

    assert rig.call("PUT", "/api/settings", settings)[0] == 200

    app.set_hotkey_listener.assert_called_once_with(build_listener.return_value)
    assert build_listener.call_args.args[0] is rig.config
    use_case.set_output.assert_not_called()


@patch("voicetotext.bootstrap._build_output")
def test_a_new_output_mode_rebuilds_the_text_pipeline(build_output: MagicMock, make_rig) -> None:
    rig, use_case, app = _applied_live(make_rig)
    status, settings = rig.call("GET", "/api/settings")
    settings["output_mode"] = "clipboard_only"

    assert rig.call("PUT", "/api/settings", settings)[0] == 200

    use_case.set_output.assert_called_once_with(build_output.return_value)
    assert build_output.call_args.args[0].value == "clipboard_only"
    app.set_hotkey_listener.assert_not_called()


@patch("voicetotext.bootstrap._build_output")
def test_a_changed_dictionary_rebuilds_the_text_pipeline(build_output: MagicMock, make_rig) -> None:
    rig, use_case, _app = _applied_live(make_rig)
    dictionary = {
        "options": {
            "remove_fillers": True, "spoken_commands": True, "numbers": False,
            "punctuation_model": False,
        },
        "rules": [{"kind": "literal", "spoken": ["jose"], "replacement": "José", "case_sensitive": False,
                   "whole_words": True, "phonetic": False, "enabled": True, "learned": False}],
    }

    assert rig.call("PUT", "/api/dictionary", dictionary)[0] == 204

    use_case.set_output.assert_called_once_with(build_output.return_value)
