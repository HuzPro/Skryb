"""The shell's one event stream: the app's events, in order, to every window."""

import threading

from tests.ui.rig import NEMOTRON, TURBO, Rig
from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
    MicrophoneConnectedEvent,
    RecordingCancelledEvent,
    TranscriptionCompletedEvent,
)
from voicetotext.core.state_machine import AppState


def _names_until(stream, last: str) -> list[str]:
    names: list[str] = []
    while True:
        name, _data = stream.next()
        if name == "heartbeat":
            continue
        names.append(name)
        if name == last:
            return names


def test_two_windows_see_the_same_events_in_the_order_they_happened(rig: Rig) -> None:
    first, second = rig.events(), rig.events()
    assert first.next()[0] == second.next()[0] == "app"

    rig.bus.publish(DownloadProgressEvent(job_id=NEMOTRON, bytes_done=1_000, bytes_total=2_470_000_000))
    rig.bus.publish(EngineLoadStartedEvent(model_name=TURBO))
    rig.state_machine.transition_to(AppState.RECORDING)
    rig.state_machine.transition_to(AppState.TRANSCRIBING)
    rig.bus.publish(TranscriptionCompletedEvent(text="Café at 9 with 東京 team"))
    rig.call("POST", "/api/writing/forget-style")

    expected = ["download", "engine", "app", "app", "app", "history", "writing"]
    seen = [_names_until(first, "writing"), _names_until(second, "writing")]
    first.close()
    second.close()
    assert seen == [expected, expected]


def test_the_stream_says_hello_with_the_app_and_then_keeps_the_line_alive(make_rig) -> None:
    rig = make_rig(heartbeat_seconds=0.05)
    stream = rig.events()

    name, app = stream.next()
    heartbeat = stream.next()

    stream.close()
    assert name == "app" and app["state"] == "idle"
    assert heartbeat == ("heartbeat", {})


def test_a_download_is_followed_from_progress_to_its_end(rig: Rig) -> None:
    stream = rig.events()

    rig.bus.publish(DownloadStateChangedEvent(job_id=TURBO, state="queued"))
    rig.bus.publish(DownloadProgressEvent(job_id=TURBO, bytes_done=10, bytes_total=100))
    rig.bus.publish(DownloadStateChangedEvent(job_id=TURBO, state="paused"))
    rig.bus.publish(DownloadStateChangedEvent(job_id=TURBO, state="failed"))
    rig.bus.publish(DownloadFailedEvent(job_id=TURBO, error="HTTP 503 from huggingface.co"))
    rig.bus.publish(DownloadStateChangedEvent(job_id=NEMOTRON, state="completed"))
    rig.bus.publish(DownloadCompletedEvent(job_id=NEMOTRON))

    states = [stream.until("download") for _ in range(4)]
    stream.close()
    assert [(state["name"], state["state"]) for state in states] == [
        (TURBO, "downloading"), (TURBO, "paused"), (TURBO, "failed"), (NEMOTRON, "completed"),
    ]
    assert states[2] == {
        "name": TURBO, "state": "failed", "bytes_done": 10, "bytes_total": 100,
        "error": "HTTP 503 from huggingface.co",
    }


def test_thousands_of_progress_reports_reach_the_shell_as_a_few_updates(rig: Rig) -> None:
    stream = rig.events()

    for done in range(5_000):
        rig.bus.publish(DownloadProgressEvent(job_id=TURBO, bytes_done=done, bytes_total=5_000))
    rig.bus.publish(DownloadCompletedEvent(job_id=TURBO))

    updates = []
    while not updates or updates[-1]["state"] != "completed":
        updates.append(stream.until("download"))
    stream.close()
    assert len(updates) < 100
    assert updates[-1]["name"] == TURBO


def test_a_failed_model_load_and_a_new_microphone_are_announced(rig: Rig) -> None:
    stream = rig.events()

    rig.bus.publish(EngineLoadFailedEvent(model_name=TURBO, error="CUDA out of memory"))
    rig.bus.publish(MicrophoneConnectedEvent(name="Jabra Evolve2 65", will_be_used=True))
    rig.bus.publish(RecordingCancelledEvent(recoverable=True))

    engine = stream.until("engine")
    microphone = stream.until("microphones")
    history = stream.until("history")
    stream.close()
    assert engine == {
        "state": "failed", "loading": False, "ready": False, "error": "CUDA out of memory", "model": TURBO,
    }
    assert microphone == {"name": "Jabra Evolve2 65", "will_be_used": True}
    assert history == {"event": "changed"}


def test_many_windows_connecting_at_once_all_get_every_event(rig: Rig) -> None:
    streams = [rig.events() for _ in range(8)]
    for stream in streams:
        stream.until("app")
    heard: list[dict] = []
    lock = threading.Lock()

    def listen(stream) -> None:
        data = stream.until("history")
        with lock:
            heard.append(data)

    listeners = [threading.Thread(target=listen, args=(stream,)) for stream in streams]
    for listener in listeners:
        listener.start()
    rig.bus.publish(TranscriptionCompletedEvent(text="hello"))
    for listener in listeners:
        listener.join(10)
    for stream in streams:
        stream.close()
    assert heard == [{"event": "changed"}] * 8
