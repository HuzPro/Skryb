"""The UI contract, end to end over loopback HTTP, against real services on fakes."""

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dataclasses import dataclass

import pytest

from tests.fakes.hardware import FakeHardwareDetector
from tests.ui.rig import CRISPERWHISPER_LICENCE, NEMOTRON, PARAKEET, TURBO, Rig
from skryb.core.events import DownloadProgressEvent
from skryb.core.release_update import UpdateStatus
from skryb.core.state_machine import AppState
from skryb.output.output_mode import OutputMode
from skryb.recording.pill import DEFAULT_PILL_POSITION
from skryb.settings.settings_draft import SettingsDraft
from skryb.transcription.engine_types import HardwareProfile
from skryb.ui.shapes import draft_from_json
from skryb.writing.install import INSTALL_BYTES
from skryb.writing.transforms import BUILTIN_TRANSFORMS

_SETTINGS_KEYS = {
    "model_size", "language", "device", "engine", "output_mode", "transcription_mode",
    "crisperwhisper_mode", "capture_content", "input_device", "boost_quiet_speech",
    "writing", "hotkey_bindings", "double_tap_locks_hold", "recording",
    "history_retention_days", "paste_method_overrides", "check_for_updates",
}
_WRITING_KEYS = {
    "cleanup", "lists", "deadline_seconds", "context", "learn_words", "learn_style",
    "style_notes", "after_dictation_transform", "runtime_path", "model_path",
}
_RECORDING_KEYS = {
    "warn_when_silent", "preferred_devices", "cues_enabled", "cue_volume",
    "duck_other_audio", "duck_level", "max_duration_seconds", "warn_before_cap_seconds",
}
_CHANGE_KEYS = {
    "engine_reload_required", "hotkey_changed", "output_changed", "mode_changed",
    "diagnostics_changed", "writing_server_changed", "learning_turned_on",
    "history_retention_changed",
}


def _settings(rig: Rig) -> dict[str, Any]:
    status, settings = rig.call("GET", "/api/settings")
    assert status == 200
    return settings


# -- transport ------------------------------------------------------------------


def test_a_request_without_the_apps_token_is_refused(rig: Rig) -> None:
    assert rig.call("GET", "/api/app", token="")[0] == 401
    assert rig.call("GET", "/api/settings", token="not-" + rig.server.token)[0] == 401
    assert rig.call("PUT", "/api/settings", {"model_size": "x"}, token="")[0] == 401
    with pytest.raises(urllib.error.HTTPError) as refused:
        rig.events(token="wrong")
    with refused.value:
        assert refused.value.code == 401
    assert rig.call("GET", "/api/app")[0] == 200


def test_the_token_never_reaches_the_session_log(rig: Rig, app_caplog) -> None:
    app_caplog.set_level(logging.DEBUG, logger="skryb")

    stream = rig.events()
    stream.until("app")
    rig.call("GET", "/api/app")
    stream.close()

    assert "GET /api/events?token=" in app_caplog.text
    assert rig.server.token not in app_caplog.text


def test_the_shells_preflight_is_answered_without_a_token_and_allows_its_origin(rig: Rig) -> None:
    request = urllib.request.Request(rig.url("/api/settings"), method="OPTIONS")
    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status == 204
        assert response.headers["Access-Control-Allow-Origin"] == "*"
        assert "Authorization" in response.headers["Access-Control-Allow-Headers"]


def test_an_unknown_address_and_a_body_that_is_not_json_get_plain_errors(rig: Rig) -> None:
    status, body = rig.call("GET", "/api/nothing-here")
    assert status == 404 and body["error"]

    request = urllib.request.Request(
        rig.url("/api/settings"), data=b"{not json", method="PUT",
        headers={"Authorization": f"Bearer {rig.server.token}"},
    )
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=10)
    with refused.value:
        assert refused.value.code == 400
        assert json.loads(refused.value.read())["error"] == "The request body is not valid JSON."


# -- app, settings ------------------------------------------------------------------


def test_the_app_and_its_settings_answer_in_the_contracts_shapes(rig: Rig) -> None:
    status, app = rig.call("GET", "/api/app")

    assert status == 200
    assert app["state"] == "idle"
    assert app["first_run"] is False
    assert set(app["engine"]) == {"ready", "loading", "error", "model"}
    assert app["engine"]["model"] == PARAKEET
    assert app["history_directory"] == str(rig.history_dir)
    assert app["pill_default_position"] == list(DEFAULT_PILL_POSITION)
    assert isinstance(app["version"], str) and app["version"]
    assert app["update"] is None

    settings = _settings(rig)
    assert set(settings) == _SETTINGS_KEYS
    assert set(settings["writing"]) == _WRITING_KEYS
    assert set(settings["recording"]) == _RECORDING_KEYS
    assert settings["model_size"] == PARAKEET
    assert settings["output_mode"] == "direct_paste"
    assert all(set(binding) == {"keys", "mode", "action"} for binding in settings["hotkey_bindings"])


def test_saving_settings_applies_them_and_tells_every_open_window(rig: Rig) -> None:
    stream = rig.events()
    settings = _settings(rig)
    settings["output_mode"] = "clipboard_only"
    settings["hotkey_bindings"] = [
        {"keys": "Ctrl + Alt + Space", "mode": "hold", "action": "dictate"},
        {"keys": "f13", "mode": "toggle", "action": "dictate"},
        {"keys": "", "mode": "toggle", "action": "paste_last"},
        {"keys": "ctrl+alt+c", "mode": "hold", "action": "copy_last"},
    ]
    settings["writing"]["style_notes"] = "  British spelling, café, 東京  "
    settings["recording"]["preferred_devices"] = ["Maono PD300X", " ", "Maono PD300X", "Jabra"]
    settings["paste_method_overrides"] = [["Mintty", "shift_insert"], ["code.exe", "ctrl_v"]]

    status, reply = rig.call("PUT", "/api/settings", settings)

    assert status == 200
    assert set(reply["changes"]) == _CHANGE_KEYS
    assert reply["changes"]["output_changed"] is True
    assert reply["changes"]["hotkey_changed"] is True
    assert reply["changes"]["engine_reload_required"] is False
    assert reply["engine_reload_started"] is False
    assert len(rig.saved) == 1 and rig.saved[0].hotkey_changed and rig.saved[0].output_changed
    assert rig.config.get("output.mode") == "clipboard_only"
    assert rig.config.get("audio.preferred_devices") == ["Maono PD300X", "Jabra"]
    pushed = stream.until("settings")
    stream.close()
    assert pushed["output_mode"] == "clipboard_only"
    assert pushed["writing"]["style_notes"] == "British spelling, café, 東京"
    assert pushed["hotkey_bindings"] == [
        {"keys": "ctrl+alt+space", "mode": "hold", "action": "dictate"},
        {"keys": "f13", "mode": "toggle", "action": "dictate"},
        {"keys": "ctrl+alt+c", "mode": "press", "action": "copy_last"},
    ]
    assert rig.config.get_all()["output"]["paste_method_overrides"] == {
        "code.exe": "ctrl_v", "mintty.exe": "shift_insert",
    }


def test_a_gated_model_is_refused_until_its_licence_is_accepted_then_it_loads(make_rig) -> None:
    rig = make_rig(installed=(PARAKEET, TURBO))
    stream = rig.events()
    settings = _settings(rig)
    settings["model_size"] = TURBO

    status, refused = rig.call("PUT", "/api/settings", settings)

    assert status == 409
    assert refused["license"]["id"] == CRISPERWHISPER_LICENCE
    assert "CrisperWhisper" in refused["license"]["text"]
    assert refused["license"]["title"]
    assert rig.config.get("transcription.model_size") == PARAKEET
    assert rig.saved == []

    assert rig.call("POST", "/api/license/accept", {"id": CRISPERWHISPER_LICENCE})[0] == 204
    status, reply = rig.call("PUT", "/api/settings", settings)

    assert status == 200
    assert reply["engine_reload_started"] is True
    assert stream.until("engine") == {
        "state": "loading", "loading": True, "ready": False, "error": None, "model": TURBO,
    }
    assert stream.until("engine")["state"] == "ready"
    stream.close()
    assert [changes.engine_reload_required for changes in rig.saved] == [True]
    assert rig.call("GET", "/api/app")[1]["engine"] == {
        "loading": False, "ready": True, "error": None, "model": TURBO,
    }


def test_an_unknown_licence_cannot_be_accepted(rig: Rig) -> None:
    assert rig.call("POST", "/api/license/accept", {"id": "no-such-licence"})[0] == 404
    assert rig.call("POST", "/api/license/accept", {})[0] == 400


@pytest.mark.parametrize(
    ("bindings", "message"),
    [
        (
            [{"keys": "ctrl+alt+m", "mode": "toggle", "action": "dictate"},
             {"keys": "ctrl+alt+m", "mode": "hold", "action": "copy_last"}],
            "Shortcut ctrl+alt+m is used twice. Give each shortcut its own keys.",
        ),
        (
            [{"keys": "ctrl+tab", "mode": "toggle", "action": "dictate"}],
            "Shortcut 'ctrl+tab' is not valid: Unknown key: tab.",
        ),
        (
            [{"keys": "ctrl+é", "mode": "toggle", "action": "dictate"}],
            "Shortcut 'ctrl+é' is not valid: Unknown key: é.",
        ),
        (
            [{"keys": "shift", "mode": "hold", "action": "dictate"}],
            "Shortcut shift needs another key: a modifier on its own fires with "
            "every shortcut that uses it.",
        ),
        (
            [{"keys": "", "mode": "toggle", "action": "dictate"},
             {"keys": "f9", "mode": "toggle", "action": "paste_last"}],
            "Add at least one dictation shortcut.",
        ),
    ],
    ids=["twice", "unknown-key", "unicode-key", "modifier-alone", "no-dictation"],
)
def test_shortcuts_that_cannot_work_are_refused_with_the_reason(
    rig: Rig, bindings: list[dict[str, str]], message: str,
) -> None:
    settings = _settings(rig)
    settings["hotkey_bindings"] = bindings

    assert rig.call("PUT", "/api/settings", settings) == (400, {"error": message})
    assert rig.call("POST", "/api/shortcuts/validate", {"bindings": bindings}) == (400, {"error": message})
    assert rig.saved == []


def test_shortcuts_that_work_together_validate(rig: Rig) -> None:
    bindings = [
        {"keys": "ctrl+alt+m", "mode": "toggle", "action": "dictate"},
        {"keys": "mouse4", "mode": "hold", "action": "dictate"},
        {"keys": "ctrl+alt+e", "mode": "hold", "action": "edit_selection"},
    ]
    assert rig.call("POST", "/api/shortcuts/validate", {"bindings": bindings}) == (200, {"ok": True})


@pytest.mark.parametrize(
    ("change", "named"),
    [
        (lambda settings: settings.pop("language"), "settings.language is missing."),
        (
            lambda settings: settings.update(capture_content="yes"),
            "settings.capture_content must be true or false.",
        ),
        (lambda settings: settings.update(output_mode="fax"), "settings.output_mode must be one of"),
        (lambda settings: settings.update(history_retention_days=-7), "cannot be negative"),
        (
            lambda settings: settings["recording"].update(cue_volume=True),
            "settings.recording.cue_volume must be a number.",
        ),
        (
            lambda settings: settings.update(paste_method_overrides=[["code.exe", "shout"]]),
            "paste_method_overrides[0]",
        ),
    ],
    ids=["missing", "wrong-type", "unknown-choice", "negative", "bool-as-number", "bad-override"],
)
def test_settings_that_cannot_be_read_are_refused_naming_the_field(rig: Rig, change, named: str) -> None:
    settings = _settings(rig)
    change(settings)

    status, body = rig.call("PUT", "/api/settings", settings)

    assert status == 400
    assert body["error"] == "Those settings could not be read."
    assert named in body["detail"]
    assert rig.saved == []


@dataclass(frozen=True)
class _DraftWithALaterSetting(SettingsDraft):
    pill_size: str = "large"


def test_a_setting_the_shell_does_not_send_yet_keeps_its_saved_value(rig: Rig) -> None:
    saved = _DraftWithALaterSetting(
        model_size=PARAKEET, language="en", device="auto", engine="auto",
        output_mode=OutputMode.DIRECT_PASTE, pill_size="small",
    )
    sent = _settings(rig)
    sent["language"] = "de"

    draft = draft_from_json(sent, saved)

    assert isinstance(draft, _DraftWithALaterSetting)
    assert (draft.pill_size, draft.language) == ("small", "de")


def test_settings_wait_while_a_dictation_is_in_progress(rig: Rig) -> None:
    rig.state_machine.transition_to(AppState.RECORDING)
    settings = _settings(rig)
    settings["output_mode"] = "clipboard_only"

    status, body = rig.call("PUT", "/api/settings", settings)

    assert status == 409
    assert "recording" in body["error"]
    assert rig.config.get("output.mode") == "direct_paste"


def test_a_model_that_is_not_installed_cannot_be_chosen(rig: Rig) -> None:
    settings = _settings(rig)
    settings["model_size"] = "crisperwhisper-2.0-large"

    assert rig.call("PUT", "/api/settings", settings) == (
        400, {"error": "'crisperwhisper-2.0-large' is not installed. Download it first."},
    )


def test_settings_export_and_import_round_trip_through_a_file_and_apply_at_once(
    rig: Rig, tmp_path: Path,
) -> None:
    exported = tmp_path / "Réglages exportés.json"
    assert rig.call("POST", "/api/settings/export", {"path": str(exported)})[0] == 204
    document = json.loads(exported.read_text(encoding="utf-8"))
    assert document["transcription"]["model_size"] == PARAKEET

    document["output"]["mode"] = "paste_and_clipboard"
    document["hotkey"] = {"bindings": [{"keys": "ctrl+shift+f9", "mode": "hold", "action": "dictate"}]}
    exported.write_text(json.dumps(document), encoding="utf-8")
    stream = rig.events()
    assert rig.call("POST", "/api/settings/import", {"path": str(exported)})[0] == 204
    assert stream.until("settings")["output_mode"] == "paste_and_clipboard"
    stream.close()
    assert [(c.output_changed, c.hotkey_changed, c.engine_reload_required) for c in rig.saved] == [
        (True, True, False),
    ]

    rig.state_machine.transition_to(AppState.RECORDING)
    assert rig.call("POST", "/api/settings/import", {"path": str(exported)})[0] == 409
    rig.state_machine.transition_to(AppState.TRANSCRIBING)
    rig.state_machine.transition_to(AppState.IDLE)

    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    status, body = rig.call("POST", "/api/settings/import", {"path": str(broken)})
    assert status == 400 and body["error"].startswith("Could not import settings")
    assert rig.call("POST", "/api/settings/export", {"path": "  "})[0] == 400


def test_the_logs_folder_the_pill_and_setup_are_one_call_each(rig: Rig) -> None:
    rig.config.set("overlay.position.x", 0.1)

    assert rig.call("POST", "/api/logs/open")[0] == 204
    assert rig.call("POST", "/api/pill/reset-position")[0] == 204
    assert rig.call("POST", "/api/setup/done")[0] == 204

    assert rig.opened == [rig.history_dir]
    position = (rig.config.get("overlay.position.x"), rig.config.get("overlay.position.y"))
    assert position == DEFAULT_PILL_POSITION


def test_a_new_logs_folder_is_kept_for_the_next_start_and_one_that_is_not_there_is_refused(
    rig: Rig, tmp_path: Path,
) -> None:
    chosen = tmp_path / "Journaux dictée"
    chosen.mkdir()
    (chosen / "notes.txt").write_text("x", encoding="utf-8")
    running_in = rig.call("GET", "/api/app")[1]["diagnostics_directory"]

    for refused in (tmp_path / "missing", "relative\\logs", "   ", chosen / "notes.txt"):
        assert rig.call("PUT", "/api/logs/directory", {"path": str(refused)})[0] == 400
    assert rig.config.get("diagnostics.directory", "") == ""

    assert rig.call("PUT", "/api/logs/directory", {"path": str(chosen)})[0] == 204

    assert rig.config.get("diagnostics.directory") == str(chosen)
    assert rig.call("GET", "/api/app")[1]["diagnostics_directory"] == running_in


def test_first_run_setup_needs_a_model_and_has_no_history_or_writing_yet(make_rig) -> None:
    rig = make_rig(first_run=True, installed=())
    settings = _settings(rig)

    assert rig.call("GET", "/api/app")[1]["first_run"] is True
    assert rig.call("PUT", "/api/settings", settings) == (
        400, {"error": "Please download at least one model before saving."},
    )
    assert rig.call("GET", "/api/history")[0] == 404
    assert rig.call("GET", "/api/writing")[0] == 404
    assert rig.call("GET", "/api/transforms")[0] == 404
    assert rig.call("POST", "/api/setup/done")[0] == 204
    assert rig.setup_done.is_set()


# -- models, engines, hardware ---------------------------------------------------------


def test_models_list_the_whole_catalogue_with_what_each_can_do(rig: Rig) -> None:
    rig.bus.publish(DownloadProgressEvent(job_id=NEMOTRON, bytes_done=512, bytes_total=2048))

    status, models = rig.call("GET", "/api/models")

    assert status == 200
    by_name = {model["name"]: model for model in models}
    assert set(by_name) == {
        PARAKEET, "parakeet-ultra", "phonon-2", NEMOTRON, TURBO, "crisperwhisper-2.0-large", "qwen3-asr-1.7b",
    }
    assert by_name[PARAKEET]["installed"] is True
    assert by_name[PARAKEET]["license_id"] is None
    assert by_name[TURBO]["license_id"] == CRISPERWHISPER_LICENCE
    assert by_name[TURBO]["has_speech_styles"] is True
    assert (by_name[NEMOTRON]["supports_streaming"], by_name[NEMOTRON]["supports_batch"]) == (True, False)
    assert by_name[NEMOTRON]["download"] == {
        "state": "downloading", "bytes_done": 512, "bytes_total": 2048, "error": "",
    }
    assert by_name[PARAKEET]["download"] is None
    assert set(by_name[PARAKEET]) == {
        "name", "display_size", "description", "installed", "supports_batch",
        "supports_streaming", "has_speech_styles", "license_id", "download",
    }


def test_first_run_setup_lists_this_computers_default_model_first(make_rig) -> None:
    rig = make_rig(first_run=True, installed=())
    rig.config.set("transcription.model_size", "phonon-2")

    names = [model["name"] for model in rig.call("GET", "/api/models")[1]]

    assert names[0] == "phonon-2"
    assert names[1:3] == [PARAKEET, "parakeet-ultra"]
    assert len(names) == len(set(names))


def test_the_settings_window_keeps_the_catalogue_order(rig: Rig) -> None:
    rig.config.set("transcription.model_size", "phonon-2")

    names = [model["name"] for model in rig.call("GET", "/api/models")[1]]

    assert names[:3] == [PARAKEET, "parakeet-ultra", "phonon-2"]


def test_model_downloads_and_deletes_answer_at_once_and_report_over_the_stream(make_rig) -> None:
    rig = make_rig(installed=(PARAKEET, NEMOTRON))
    stream = rig.events()

    assert rig.call("POST", f"/api/models/{NEMOTRON}/download")[0] == 202
    assert rig.call("POST", f"/api/models/{NEMOTRON}/pause")[0] == 202
    assert rig.call("POST", f"/api/models/{NEMOTRON}/resume")[0] == 202
    assert rig.call("POST", f"/api/models/{NEMOTRON}/cancel")[0] == 202
    assert rig.call("POST", "/api/models/crisperwhisper-2.0-large/cancel")[0] == 202
    assert rig.manager.download_calls == [NEMOTRON, NEMOTRON]

    status, refused = rig.call("POST", f"/api/models/{TURBO}/download")
    assert status == 409 and refused["license"]["id"] == CRISPERWHISPER_LICENCE
    assert rig.call("POST", f"/api/models/{PARAKEET}/delete")[0] == 409
    assert rig.call("POST", "/api/models/no-such-model/download")[0] == 404

    assert rig.call("POST", f"/api/models/{NEMOTRON}/delete")[0] == 202
    assert stream.until("download") == {
        "name": NEMOTRON, "state": "deleted", "bytes_done": 0, "bytes_total": 0, "error": "",
    }
    stream.close()
    assert rig.manager.delete_calls == [NEMOTRON]


def test_engines_answer_before_the_gpu_probe_and_the_probe_reports_later(make_rig) -> None:
    profile = HardwareProfile(kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200")
    detector = FakeHardwareDetector(profile, probed=False)
    rig = make_rig(hardware=detector)
    stream = rig.events()

    status, engines = rig.call("GET", f"/api/engines?model={TURBO}")

    assert status == 200
    assert engines["hardware"] == {
        "probe_pending": True, "streaming_supported": False, "kind": "", "device_name": "",
    }
    ranked = [candidate["id"] for candidate in engines["candidates"]]
    assert ranked == ["crisperwhisper_ct2", "crisperwhisper"]
    assert set(engines["candidates"][0]) == {"id", "label", "recommended", "reason"}
    assert detector.detect_calls == 0

    assert rig.call("POST", "/api/hardware/probe")[0] == 202
    assert stream.until("hardware") == {
        "probe_pending": False, "streaming_supported": True,
        "kind": "amd_rocm", "device_name": "AMD Radeon RX 9060 XT",
    }
    stream.close()
    assert rig.call("GET", "/api/engines")[0] == 400


# -- microphones and shortcuts -------------------------------------------------------------


def test_microphones_list_the_connected_ones_and_a_chosen_one_now_unplugged_but_not_the_windows_default(
    rig: Rig,
) -> None:
    status, with_default_chosen = rig.call("GET", "/api/microphones")
    rig.config.set("audio.input_device", "USB Mic (unplugged) ✓")

    microphones = rig.call("GET", "/api/microphones")[1]
    rig.call("GET", "/api/microphones?rescan=1")

    assert status == 200
    assert with_default_chosen == microphones[:2]
    assert microphones == [
        _microphone("Microphone (Maono PD300X)", "Microphone (Maono PD300X)", connected=True),
        _microphone(
            "Headset (Jabra Evolve2 65 Mon", "Headset (Jabra Evolve2 65 Mono)", connected=True,
        ),
        _microphone("USB Mic (unplugged) ✓", "USB Mic (unplugged) ✓", connected=False),
    ]
    assert rig.microphones.rescans == 1


def _microphone(name: str, label: str, *, connected: bool) -> dict[str, Any]:
    return {"name": name, "label": label, "connected": connected, "is_default": False}


def test_the_level_meter_streams_the_loudest_level_and_stops_when_asked(rig: Rig) -> None:
    stream = rig.events()
    maono = {"device": "Microphone (Maono PD300X)", "on": True}

    assert rig.call("POST", "/api/microphones/meter", maono)[0] == 204
    assert rig.microphones.metered == "Microphone (Maono PD300X)"
    assert rig.microphones.level is not None
    rig.microphones.level(0.42)
    assert stream.until("meter") == {"level": 0.42}

    assert rig.call("POST", "/api/microphones/meter", {"device": "", "on": False})[0] == 204
    stream.close()
    assert rig.microphones.level is None

    rig.microphones.opens = False
    assert rig.call("POST", "/api/microphones/meter", {"device": "Broken", "on": True}) == (
        400, {"error": "This microphone could not be opened."},
    )


def test_recording_a_shortcut_pauses_the_apps_shortcuts_and_streams_each_combo(rig: Rig) -> None:
    stream = rig.events()

    assert rig.call("POST", "/api/shortcuts/capture", {"on": True})[0] == 204
    assert rig.call("POST", "/api/shortcuts/capture", {"on": True})[0] == 204
    assert (rig.listener.pause_calls, rig.bridge.running) == (1, True)

    rig.bridge.press("ctrl", "ctrl", "shift", "m")
    captured = [stream.until("capture") for _ in range(3)]

    assert rig.call("POST", "/api/shortcuts/capture", {"on": False})[0] == 204
    stream.close()
    assert captured == [
        {"keys": "ctrl", "display": "Ctrl"},
        {"keys": "ctrl+shift", "display": "Ctrl + Shift"},
        {"keys": "ctrl+shift+m", "display": "Ctrl + Shift + M"},
    ]
    assert (rig.listener.resume_calls, rig.bridge.running) == (1, False)


def test_escape_cancels_recording_a_shortcut_and_gives_the_keys_back(rig: Rig) -> None:
    stream = rig.events()
    rig.call("POST", "/api/shortcuts/capture", {"on": True})

    rig.bridge.press("alt", "esc", "m")

    assert stream.until("capture") == {"keys": "alt", "display": "Alt"}
    assert stream.until("capture") == {"error": "Recording the shortcut was cancelled.", "cancelled": True}
    stream.close()
    assert (rig.bridge.running, rig.listener.resume_calls) == (False, 1)


def test_a_shell_that_goes_away_mid_recording_leaves_no_hook_or_meter_behind(make_rig) -> None:
    rig = make_rig(heartbeat_seconds=0.05)
    stream = rig.events()
    stream.until("app")
    rig.call("POST", "/api/shortcuts/capture", {"on": True})
    rig.call("POST", "/api/microphones/meter", {"device": "", "on": True})

    stream.close()

    _wait_for(lambda: not rig.bridge.running and rig.microphones.level is None)
    assert rig.listener.resume_calls == 1


# -- dictionary, transforms, writing ------------------------------------------------------------

_DICTIONARY: dict[str, Any] = {
    "options": {
        "remove_fillers": True, "spoken_commands": False, "numbers": True, "punctuation_model": False,
    },
    "rules": [
        {"kind": "literal", "spoken": ["aisha", "ayesha"], "replacement": "Aïsha", "case_sensitive": False,
         "whole_words": True, "phonetic": True, "enabled": True, "learned": False},
        {"kind": "literal", "spoken": ["farhan"], "replacement": "Farhān", "case_sensitive": False,
         "whole_words": True, "phonetic": False, "enabled": True, "learned": True},
        {"kind": "literal", "spoken": ["tokyo"], "replacement": "東京", "case_sensitive": True,
         "whole_words": False, "phonetic": False, "enabled": False, "learned": True},
        {"kind": "regex", "pattern": r"\bteh\b", "replacement": "the", "enabled": True},
    ],
}


def test_the_dictionary_round_trips_and_rebuilds_the_pipeline(rig: Rig) -> None:
    status, empty = rig.call("GET", "/api/dictionary")
    assert status == 200
    assert empty == {
        "options": {
            "remove_fillers": False, "spoken_commands": False, "numbers": False,
            "punctuation_model": False,
        },
        "rules": [],
    }

    assert rig.call("PUT", "/api/dictionary", _DICTIONARY)[0] == 204

    assert rig.call("GET", "/api/dictionary") == (200, _DICTIONARY)
    assert [changes.output_changed for changes in rig.saved] == [True]


def test_removing_learned_words_stops_them_being_learned_again(rig: Rig, monkeypatch) -> None:
    forgotten: list[tuple[tuple[str, str], ...]] = []
    monkeypatch.setattr(rig.writing, "forget_learned_words", forgotten.append)
    rig.call("PUT", "/api/dictionary", _DICTIONARY)
    kept = [rule for rule in _DICTIONARY["rules"] if rule.get("spoken") != ["farhan"]]
    without_farhan = {**_DICTIONARY, "rules": kept}

    rig.call("PUT", "/api/dictionary", without_farhan)
    assert rig.call("POST", "/api/dictionary/forget-learned")[0] == 204

    assert forgotten == [(), (("farhan", "Farhān"),), (("tokyo", "東京"),)]
    kinds = [(rule["kind"], rule.get("learned")) for rule in rig.call("GET", "/api/dictionary")[1]["rules"]]
    assert kinds == [("literal", False), ("regex", None)]


def test_forgetting_one_learned_word_removes_only_it_and_stops_it_being_learned_again(
    rig: Rig, monkeypatch,
) -> None:
    forgotten: list[tuple[tuple[str, str], ...]] = []
    monkeypatch.setattr(rig.writing, "forget_learned_words", forgotten.append)
    slashed = {
        "kind": "literal", "spoken": ["ümit / o'neil"], "replacement": "Ümit O'Neil",
        "case_sensitive": False, "whole_words": True, "phonetic": False, "enabled": True,
        "learned": True,
    }
    rig.call("PUT", "/api/dictionary", {**_DICTIONARY, "rules": [*_DICTIONARY["rules"], slashed]})
    rig.saved.clear()

    assert rig.call("DELETE", "/api/dictionary/learned/%C3%BCmit%20%2F%20o'neil")[0] == 204
    assert rig.call("DELETE", "/api/dictionary/learned/tokyo")[0] == 204

    assert forgotten[1:] == [(("ümit / o'neil", "Ümit O'Neil"),), (("tokyo", "東京"),)]
    rules = rig.call("GET", "/api/dictionary")[1]["rules"]
    assert [rule.get("spoken", rule.get("pattern")) for rule in rules] == [
        ["aisha", "ayesha"], ["farhan"], r"\bteh\b",
    ]
    assert [changes.output_changed for changes in rig.saved] == [True, True]


def test_forgetting_a_word_that_was_not_learned_changes_nothing(rig: Rig) -> None:
    rig.call("PUT", "/api/dictionary", _DICTIONARY)
    rig.saved.clear()

    assert rig.call("DELETE", "/api/dictionary/learned/aisha")[0] == 404
    assert rig.call("DELETE", "/api/dictionary/learned/nobody")[0] == 404

    assert rig.call("GET", "/api/dictionary") == (200, _DICTIONARY)
    assert rig.saved == []


def test_a_dictionary_with_a_broken_rule_is_refused_and_the_saved_one_is_kept(rig: Rig) -> None:
    rig.call("PUT", "/api/dictionary", _DICTIONARY)
    unclosed = {"kind": "regex", "pattern": "(", "replacement": "x", "enabled": True}
    broken = {**_DICTIONARY, "rules": [*_DICTIONARY["rules"], unclosed]}

    status, body = rig.call("PUT", "/api/dictionary", broken)

    assert status == 400
    assert body["detail"] == "Dictionary rule 5 is not valid."
    assert rig.call("GET", "/api/dictionary") == (200, _DICTIONARY)


def test_transformations_can_be_edited_and_the_built_ins_restored(rig: Rig) -> None:
    status, transforms = rig.call("GET", "/api/transforms")
    assert status == 200
    assert transforms == [
        {"name": name, "instruction": instruction, "builtin": True}
        for name, instruction in BUILTIN_TRANSFORMS.items()
    ]

    edited = [
        {"name": "Concise", "instruction": "Shorter, keep every name like Aïsha."},
        {"name": "Emoji ✨", "instruction": "Add one fitting emoji."},
        {"name": "   ", "instruction": "Dropped: it has no name."},
    ]
    assert rig.call("PUT", "/api/transforms", edited)[0] == 204
    assert rig.call("GET", "/api/transforms")[1] == [
        {"name": "Concise", "instruction": "Shorter, keep every name like Aïsha.", "builtin": True},
        {"name": "Emoji ✨", "instruction": "Add one fitting emoji.", "builtin": False},
    ]

    assert rig.call("POST", "/api/transforms/restore-builtins")[0] == 204
    names = [transform["name"] for transform in rig.call("GET", "/api/transforms")[1]]
    assert names == ["Concise", "Emoji ✨", "Bullet points", "Professional", "Casual"]
    assert rig.call("GET", "/api/transforms")[1][0]["instruction"] == BUILTIN_TRANSFORMS["Concise"]
    assert [changes.output_changed for changes in rig.saved] == [True, True]
    assert rig.call("PUT", "/api/transforms", {"Concise": "not a list"})[0] == 400


def test_the_writing_model_reports_its_status_and_acts_on_request(rig: Rig) -> None:
    status, writing = rig.call("GET", "/api/writing")

    assert status == 200
    assert writing["state"] == "not_installed"
    assert writing["install_bytes"] == INSTALL_BYTES
    assert writing["runtime_path"].endswith("llama-server.exe")
    assert writing["learned_style_summary"] == "nothing yet"
    assert set(writing) == {
        "state", "detail", "bytes_done", "bytes_total", "install_bytes", "runtime_path",
        "model_path", "last_cleanup_seconds", "learned_style_summary",
    }

    stream = rig.events()
    assert rig.call("POST", "/api/writing/forget-style")[0] == 204
    assert stream.until("writing")["state"] == "not_installed"
    assert rig.call("POST", "/api/writing/install")[0] == 202
    failed = _next_writing_state(stream, "failed")
    stream.close()
    assert "internet connection" in failed["detail"]
    assert rig.call("POST", "/api/writing/reset-learning")[0] == 204
    assert rig.call("POST", "/api/writing/levitate")[0] == 404


def _next_writing_state(stream, state: str) -> dict[str, Any]:
    while True:
        status = stream.until("writing")
        if status["state"] == state:
            return status


def _wait_for(condition, seconds: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.01)


# -- updates -------------------------------------------------------------------------


class _Updates:
    """Stands in for the release updater: a 0.3.0 download that is ready, or nothing yet."""

    def __init__(self, ready: str = "") -> None:
        self.ready = ready
        self.checks = 0
        self.installs = 0

    def status(self) -> UpdateStatus:
        if self.ready:
            return UpdateStatus(state="ready", version=self.ready)
        return UpdateStatus(state="up_to_date")

    def offered_version(self) -> str:
        return self.ready

    def check_now(self) -> None:
        self.checks += 1

    def install(self) -> None:
        self.installs += 1


def test_a_git_checkout_says_it_updates_through_git(rig: Rig) -> None:
    status, body = rig.call("POST", "/api/updates/check")

    assert status == 404
    assert "updates through git" in body["error"]
    assert rig.call("POST", "/api/updates/install")[0] == 404


def test_an_installed_copy_reports_its_update_in_the_app_info(make_rig) -> None:
    rig = make_rig(updates=_Updates(ready="0.3.0"))

    update = rig.call("GET", "/api/app")[1]["update"]

    assert update == {"state": "ready", "version": "0.3.0", "bytes_done": 0, "bytes_total": 0, "error": ""}


def test_checking_by_hand_asks_the_updater_every_time(make_rig) -> None:
    updates = _Updates()
    rig = make_rig(updates=updates)

    statuses = [rig.call("POST", "/api/updates/check")[0] for _ in range(20)]

    assert statuses == [202] * 20
    assert updates.checks == 20


def test_installing_with_nothing_downloaded_is_refused(make_rig) -> None:
    updates = _Updates()
    rig = make_rig(updates=updates)

    status, body = rig.call("POST", "/api/updates/install")

    assert status == 409
    assert body["error"] == "There is no downloaded update to install."
    assert updates.installs == 0


def test_installing_a_downloaded_update_hands_it_to_the_updater(make_rig) -> None:
    updates = _Updates(ready="0.3.0")
    rig = make_rig(updates=updates)

    assert rig.call("POST", "/api/updates/install")[0] == 202
    assert updates.installs == 1


def test_turning_automatic_checks_off_is_saved(rig: Rig) -> None:
    settings = _settings(rig)
    assert settings["check_for_updates"] is True

    status, _ = rig.call("PUT", "/api/settings", {**settings, "check_for_updates": False})

    assert status == 200
    assert rig.config.get("updates.check_automatically") is False
    assert _settings(rig)["check_for_updates"] is False


@pytest.mark.parametrize("value", ["false", 0, None, "no"])
def test_automatic_checks_take_only_true_or_false(rig: Rig, value: object) -> None:
    settings = _settings(rig)

    status, body = rig.call("PUT", "/api/settings", {**settings, "check_for_updates": value})

    assert status == 400
    assert body["detail"] == "settings.check_for_updates must be true or false."
    assert rig.config.get("updates.check_automatically") is None

