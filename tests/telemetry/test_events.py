"""Telemetry keeps what was used and never what was said, on this computer, until it is exported."""

import json
import random
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from skryb.telemetry import events as telemetry
from skryb.telemetry.events import TelemetryStore, scrub

NOW = datetime(2026, 11, 23, 14, 32, 5, tzinfo=UTC)

# What must never be stored, as a name or as a value: speech, paths, device and people's names, and text
# in other scripts, next to the blank and boundary cases.
NOT_A_TOKEN = [
    "Thanks for sending the draft over",
    "see you then",
    "C:\\Users\\someone\\Documents\\notes.txt",
    "D:/Meetings/weekly-review",
    "/home/someone/.config",
    "Microphone (Blue Yeti)",
    "Priya",
    "History.Copy",
    "مرحبا",
    "naïve",
    "日本語",
    "copy\n",
    "",
    " ",
    "-leading-dash",
    ".hidden",
    "a" * 65,
]
PAGES = ["home", "history", "dictionary", "shortcuts", "microphone", "output", "transcription", "updates"]
CONTROLS = ["copy", "play", "rotate-ccw", "download", "trash-2", "file-down", "search", "row"]


def on(path: Path, **options: object) -> TelemetryStore:
    return TelemetryStore(path, enabled=lambda: True, clock=lambda: NOW, **options)  # type: ignore[arg-type]


def many_events(count: int, seed: int = 7) -> list[dict[str, object]]:
    """A mixed day of use: page visits with times, clicks that repeat, and saved settings."""
    rng = random.Random(seed)
    events: list[dict[str, object]] = []
    for _ in range(count):
        roll = rng.random()
        if roll < 0.25:
            events.append({"kind": "page", "name": rng.choice(PAGES), "ms": rng.randint(0, 600_000)})
        elif roll < 0.9:
            events.append({"kind": "click", "name": f"{rng.choice(PAGES)}.{rng.choice(CONTROLS)}"})
        else:
            events.append({"kind": "setting", "name": "theme", "value": rng.choice(["default", "glass", "paper"])})
    return events


def test_an_event_keeps_only_its_kind_name_value_and_time() -> None:
    event = scrub(
        {"kind": "click", "name": "history.copy", "value": True, "text": "what was said", "path": "C:\\x", "user": "Priya"},
        NOW,
    )

    assert event == {"at": "2026-11-23T14:32:05+00:00", "kind": "click", "name": "history.copy", "value": True}


@pytest.mark.parametrize("text", NOT_A_TOKEN)
def test_an_event_named_with_speech_a_path_or_someones_name_is_dropped(text: str) -> None:
    assert scrub({"kind": "click", "name": text}, NOW) is None


@pytest.mark.parametrize("text", NOT_A_TOKEN)
def test_a_value_holding_speech_a_path_or_someones_name_is_left_out_of_its_event(text: str) -> None:
    assert scrub({"kind": "setting", "name": "input_device", "value": text}, NOW) == {
        "at": "2026-11-23T14:32:05+00:00", "kind": "setting", "name": "input_device",
    }


@pytest.mark.parametrize("event", [None, "click", 7, [], ["click", "history.copy"], {}, {"kind": "click"},
                                   {"name": "history.copy"}, {"kind": "keystroke", "name": "a"},
                                   {"kind": "click", "name": 5}, {"kind": "click", "name": None}])
def test_anything_that_is_not_a_page_click_or_setting_event_is_dropped(event: object) -> None:
    assert scrub(event, NOW) is None


def test_a_name_of_sixty_four_characters_is_kept_and_sixty_five_is_not() -> None:
    assert scrub({"kind": "page", "name": "a" * 64}, NOW) is not None
    assert scrub({"kind": "page", "name": "a" * 65}, NOW) is None


@pytest.mark.parametrize("value", [True, False, 0, -1, 90, 0.6, "parakeet-ultra", "qwen3-asr-1.7b", "direct_paste"])
def test_a_switch_a_number_or_a_fixed_choice_is_kept_as_the_value(value: object) -> None:
    event = scrub({"kind": "setting", "name": "x", "value": value}, NOW)

    assert event is not None and event["value"] == value and type(event["value"]) is type(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), ["a"], {"a": 1}, None])
def test_a_value_that_is_not_a_plain_number_switch_or_choice_is_left_out(value: object) -> None:
    event = scrub({"kind": "setting", "name": "x", "value": value}, NOW)

    assert event is not None and "value" not in event


@pytest.mark.parametrize(("ms", "kept"), [(0, 0), (1499.6, 1500), (600_000, 600_000), (-1, None),
                                          (True, None), ("900", None), (float("inf"), None)])
def test_time_on_a_page_is_kept_in_whole_milliseconds_and_only_when_it_makes_sense(ms: object, kept: int | None) -> None:
    event = scrub({"kind": "page", "name": "history", "ms": ms}, NOW)

    assert event is not None and event.get("ms") == kept


@pytest.mark.parametrize(("ago", "at"), [(0, "14:32:05"), (5_000, "14:32:00"), (3_600_000, "13:32:05"),
                                         (3_600_001, "14:32:05"), (-5_000, "14:32:05"), ("5000", "14:32:05"),
                                         (float("nan"), "14:32:05")])
def test_an_event_is_stamped_with_when_it_happened_unless_the_shell_claims_an_impossible_wait(ago: object, at: str) -> None:
    event = scrub({"kind": "click", "name": "history.copy", "ago": ago}, NOW)

    assert event is not None and event["at"] == f"2026-11-23T{at}+00:00"


def test_nothing_is_written_while_telemetry_is_off(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    switch = {"on": False}
    store = TelemetryStore(path, enabled=lambda: switch["on"], clock=lambda: NOW)

    assert store.record(many_events(50)) == 0
    assert not path.exists()

    switch["on"] = True
    assert store.record(many_events(50)) == 50
    switch["on"] = False
    assert store.record(many_events(50)) == 0
    assert len(path.read_text(encoding="utf-8").splitlines()) == 50


def test_thousands_of_events_with_repeats_are_all_kept_and_counted_most_used_first(tmp_path: Path) -> None:
    store = on(tmp_path / "telemetry.jsonl")
    sent = many_events(5_000)

    for start in range(0, len(sent), 250):
        assert store.record(sent[start:start + 250]) == 250

    counts = store.counts()
    expected: dict[str, int] = {}
    for event in sent:
        key = f"{event['kind']}:{event['name']}"
        expected[key] = expected.get(key, 0) + 1
    assert counts == expected
    assert list(counts.values()) == sorted(counts.values(), reverse=True)
    assert sum(counts.values()) == 5_000 and len(store.events()) == 5_000


def test_events_that_do_not_pass_are_dropped_from_a_batch_and_the_rest_kept(tmp_path: Path) -> None:
    store = on(tmp_path / "telemetry.jsonl")
    batch = [{"kind": "click", "name": "history.copy"}, {"kind": "click", "name": "Renew the car insurance"},
             None, {"kind": "click", "name": "history.copy"}, {"kind": "page", "name": ""}]

    assert store.record(batch) == 2
    assert store.counts() == {"click:history.copy": 2}


def test_one_batch_holds_five_hundred_events_at_most(tmp_path: Path) -> None:
    store = on(tmp_path / "telemetry.jsonl")

    assert store.record(many_events(501)) == 500


def test_counts_come_back_after_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    for seed in range(3):
        on(path).record(many_events(400, seed))

    assert sum(on(path).counts().values()) == 1_200


def test_a_file_past_its_limit_loses_its_oldest_events_and_keeps_the_newest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry, "MAX_EVENTS", 1_000)
    path = tmp_path / "telemetry.jsonl"
    on(path).record([{"kind": "click", "name": "old.one"}] * 500)
    for _ in range(4):
        on(path).record([{"kind": "click", "name": "new.one"}] * 500)

    counts = on(path).counts()

    assert counts == {"click:new.one": 1_000}
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1_000


def test_a_line_cut_short_or_edited_by_hand_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    on(path).record([{"kind": "click", "name": "history.copy"}])
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"at":"2026-11-23T14:00:00","kind":"click","na\n')
        stream.write("\n")
        stream.write('["not", "an", "event"]\n')
        stream.write('{"at":"x","kind":"click","name":"Somebody typed a sentence here"}\n')
        stream.write('{"at":"2026-11-23T14:01:00","kind":"page","name":"history","ms":4000}\n')

    store = on(path)

    assert store.counts() == {"click:history.copy": 1, "page:history": 1}
    assert [event["name"] for event in store.events()] == ["history.copy", "history"]


def test_events_recorded_from_several_threads_at_once_all_land_whole(tmp_path: Path) -> None:
    store = on(tmp_path / "telemetry.jsonl")
    threads = [threading.Thread(target=lambda: [store.record(many_events(25)) for _ in range(10)]) for _ in range(8)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(store.events()) == 2_000 and sum(store.counts().values()) == 2_000


def test_an_export_is_one_json_file_that_explains_itself_and_holds_every_event(tmp_path: Path) -> None:
    store = on(tmp_path / "telemetry.jsonl")
    for seed in range(6):
        store.record(many_events(500, seed))
    target = tmp_path / "skryb-telemetry.json"

    assert store.export(target, {"app": {"version": "0.16.0"}, "settings": {"theme": "paper"}}) == 3_000

    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["format"] == "skryb-telemetry/1"
    assert "no dictated text" in document["about"]
    assert document["exported_at"] == "2026-11-23T14:32:05+00:00"
    assert document["app"] == {"version": "0.16.0"} and document["settings"] == {"theme": "paper"}
    assert document["events"] == store.events() and len(document["events"]) == 3_000
    summary = document["summary"]
    assert summary["events"] == 3_000 and summary["days_active"] == 1
    assert summary["first"] == summary["last"] == "2026-11-23T14:32:05+00:00"
    assert summary["by_name"] == store.counts()
    assert summary["ms_on_pages"] == {
        page: sum(event["ms"] for event in document["events"] if event["kind"] == "page" and event["name"] == page)
        for page in summary["ms_on_pages"]
    }
    assert set(summary["ms_on_pages"]) <= set(PAGES)


def test_an_export_with_nothing_recorded_is_still_a_whole_file(tmp_path: Path) -> None:
    target = tmp_path / "skryb-telemetry.json"

    assert on(tmp_path / "telemetry.jsonl").export(target, {"app": {"version": "0.16.0"}}) == 0

    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["events"] == []
    assert document["summary"] == {"events": 0, "first": None, "last": None, "days_active": 0, "by_name": {}, "ms_on_pages": {}}


def test_an_export_counts_the_days_skryb_was_used(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    days = [datetime(2026, 11, 20, 9, tzinfo=UTC), datetime(2026, 11, 20, 23, 59, 59, tzinfo=UTC),
            datetime(2026, 11, 21, 0, tzinfo=UTC), datetime(2026, 11, 23, 8, tzinfo=UTC)]
    for day in days:
        TelemetryStore(path, enabled=lambda: True, clock=lambda day=day: day).record([{"kind": "click", "name": "history.copy"}])
    target = tmp_path / "out.json"

    on(path).export(target, {})

    summary = json.loads(target.read_text(encoding="utf-8"))["summary"]
    assert (summary["days_active"], summary["first"], summary["last"]) == (3, "2026-11-20T09:00:00+00:00", "2026-11-23T08:00:00+00:00")


def test_deleting_removes_the_file_and_the_counts_and_recording_carries_on_after(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    store = on(path)
    store.record(many_events(400))

    store.clear()

    assert not path.exists() and store.counts() == {} and store.events() == []
    store.clear()
    assert store.record([{"kind": "click", "name": "history.copy"}]) == 1
    assert store.counts() == {"click:history.copy": 1}
