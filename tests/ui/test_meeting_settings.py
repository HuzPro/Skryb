"""The three meeting settings, saved through the same Settings endpoint as everything else."""

from typing import Any

import pytest

from skryb.meetings.storage import default_meetings_folder
from tests.ui.rig import Rig


def _settings(rig: Rig) -> dict[str, Any]:
    status, settings = rig.call("GET", "/api/settings")
    assert status == 200
    return settings


def test_a_fresh_install_shows_the_default_folder_the_microphone_on_and_english(rig: Rig) -> None:
    assert _settings(rig)["meetings"] == {
        "folder": str(default_meetings_folder()), "record_microphone": True, "language": "en",
    }


def test_meeting_settings_save_without_reloading_the_dictation_model(rig: Rig) -> None:
    settings = _settings(rig)
    settings["meetings"] = {"folder": "  E:\\Calls\\میٹنگز  ", "record_microphone": False, "language": "ur"}

    status, reply = rig.call("PUT", "/api/settings", settings)

    assert status == 200
    assert reply["engine_reload_started"] is False
    assert rig.config.get("meetings.folder") == "E:\\Calls\\میٹنگز"
    assert rig.config.get("meetings.record_microphone") is False
    assert rig.config.get("meetings.language") == "ur"
    assert _settings(rig)["meetings"]["folder"] == "E:\\Calls\\میٹنگز"


@pytest.mark.parametrize("meetings, problem", [
    ({"folder": "   ", "record_microphone": True, "language": "en"}, "settings.meetings.folder cannot be blank."),
    ({"folder": "D:\\Meetings", "record_microphone": True, "language": "fr"}, "settings.meetings.language must be one of en, ur, auto."),
    ({"folder": "D:\\Meetings", "record_microphone": "yes", "language": "en"}, "settings.meetings.record_microphone must be true or false."),
    ({"folder": "D:\\Meetings", "language": "en"}, "settings.meetings.record_microphone is missing."),
])
def test_meeting_settings_that_make_no_sense_are_refused_whole(rig: Rig, meetings: dict[str, Any], problem: str) -> None:
    settings = _settings(rig)
    settings["meetings"] = meetings

    status, reply = rig.call("PUT", "/api/settings", settings)

    assert status == 400
    assert reply["detail"] == problem
    assert rig.config.get("meetings.language") is None
