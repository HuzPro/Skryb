"""Telemetry over the shell's API: off until switched on, counted, exported as one file, and deletable."""

import json
from pathlib import Path
from typing import Any

from tests.ui.rig import Rig

CLICKS = [{"kind": "click", "name": "history.copy"}] * 12 + [{"kind": "click", "name": "history.play"}] * 3
VISIT = {"kind": "page", "name": "history", "ms": 42_000}


def switch_on(rig: Rig) -> None:
    status, settings = rig.call("GET", "/api/settings")
    assert status == 200
    settings["telemetry"] = True
    assert rig.call("PUT", "/api/settings", settings)[0] == 200


def status_of(rig: Rig) -> dict[str, Any]:
    status, body = rig.call("GET", "/api/telemetry")
    assert status == 200
    return body


def test_telemetry_starts_off_and_what_the_shell_sends_meanwhile_is_not_kept(rig: Rig) -> None:
    assert rig.call("GET", "/api/settings")[1]["telemetry"] is False

    assert rig.call("POST", "/api/telemetry", {"events": CLICKS})[0] == 204

    assert status_of(rig) == {"events": 0, "counts": {}}
    assert not rig.telemetry_path.exists()


def test_switched_on_it_is_saved_and_events_are_kept_and_counted_most_used_first(rig: Rig) -> None:
    switch_on(rig)
    assert rig.config.get("telemetry.enabled") is True

    assert rig.call("POST", "/api/telemetry", {"events": [*CLICKS, VISIT]})[0] == 204

    status = status_of(rig)
    assert status["events"] == 16
    assert list(status["counts"].items()) == [("click:history.copy", 12), ("click:history.play", 3), ("page:history", 1)]


def test_switched_off_again_it_stops_keeping_events_and_keeps_what_it_has(rig: Rig) -> None:
    switch_on(rig)
    rig.call("POST", "/api/telemetry", {"events": CLICKS})
    settings = rig.call("GET", "/api/settings")[1]
    settings["telemetry"] = False
    assert rig.call("PUT", "/api/settings", settings)[0] == 200

    rig.call("POST", "/api/telemetry", {"events": CLICKS})

    assert status_of(rig)["events"] == 15


def test_speech_paths_and_names_sent_as_events_never_reach_the_file(rig: Rig) -> None:
    switch_on(rig)
    sent = [
        {"kind": "click", "name": "Renew the car insurance before the end of the month"},
        {"kind": "click", "name": "history.copy", "value": "C:\\Users\\someone\\notes.txt", "text": "Sounds good, see you then."},
        {"kind": "setting", "name": "input_device", "value": "Microphone (Maono PD300X)"},
        {"kind": "page", "name": "/meetings/2026-11-20-weekly-review"},
    ]

    assert rig.call("POST", "/api/telemetry", {"events": sent})[0] == 204

    stored = rig.telemetry_path.read_text(encoding="utf-8")
    assert status_of(rig)["counts"] == {"click:history.copy": 1, "setting:input_device": 1}
    for secret in ("insurance", "someone", "Sounds good", "Maono", "weekly-review"):
        assert secret not in stored


def test_a_body_without_a_list_of_events_is_refused_in_a_sentence(rig: Rig) -> None:
    switch_on(rig)

    for body in ({}, {"events": "history.copy"}, {"events": None}, ["history.copy"]):
        status, answer = rig.call("POST", "/api/telemetry", body)
        assert status == 400 and answer["error"] == "Send the events as a list."


def test_an_export_is_one_file_with_the_install_described_and_nothing_personal_in_it(rig: Rig, tmp_path: Path) -> None:
    rig.config.set("audio.input_device", "Microphone (Maono PD300X)")
    rig.config.set("meetings.folder", str(tmp_path / "Private Meetings"))
    switch_on(rig)
    rig.call("POST", "/api/telemetry", {"events": [*CLICKS, VISIT]})
    target = tmp_path / "skryb-telemetry.json"

    status, body = rig.call("POST", "/api/telemetry/export", {"path": str(target)})

    assert (status, body) == (200, {"events": 16})
    text = target.read_text(encoding="utf-8")
    document = json.loads(text)
    assert document["format"] == "skryb-telemetry/1" and len(document["events"]) == 16
    assert document["summary"]["by_name"]["click:history.copy"] == 12
    assert document["summary"]["ms_on_pages"] == {"history": 42_000}
    assert document["app"]["version"]
    assert set(document["hardware"]) == {"kind", "device_name"}
    assert document["settings"]["model_size"] == "parakeet-ultra"
    assert document["settings"]["recording"]["preferred_devices"] == 0
    assert document["dictation"]["all"] == {"words": 0, "seconds": 0.0, "dictations": 0, "lost_seconds": 0.0}
    for secret in ("Maono", "Private Meetings", str(tmp_path), json.dumps(str(tmp_path))[1:-1], rig.server.token):
        assert secret not in text


def test_an_export_to_a_folder_that_is_not_there_is_refused_and_says_why(rig: Rig, tmp_path: Path) -> None:
    switch_on(rig)

    status, body = rig.call("POST", "/api/telemetry/export", {"path": str(tmp_path / "missing" / "out.json")})

    assert status == 400 and body["error"].startswith("Could not export the usage data:")
    for blank in ({}, {"path": ""}, {"path": "   "}, {"path": 7}):
        assert rig.call("POST", "/api/telemetry/export", blank)[0] == 400


def test_deleting_removes_everything_kept_and_leaves_telemetry_on(rig: Rig) -> None:
    switch_on(rig)
    rig.call("POST", "/api/telemetry", {"events": CLICKS})

    assert rig.call("DELETE", "/api/telemetry")[0] == 204

    assert status_of(rig) == {"events": 0, "counts": {}}
    assert not rig.telemetry_path.exists()
    rig.call("POST", "/api/telemetry", {"events": [VISIT]})
    assert status_of(rig)["events"] == 1


def test_telemetry_is_refused_without_the_apps_token(rig: Rig) -> None:
    for method, path in (("GET", "/api/telemetry"), ("POST", "/api/telemetry"), ("DELETE", "/api/telemetry"),
                         ("POST", "/api/telemetry/export")):
        assert rig.call(method, path, {"events": []}, token="")[0] == 401
