import threading
import time

from skryb.transcription.engine_turns import EngineTurns

_WAIT_SECONDS = 5.0


def _waiting_for_turn(turns: EngineTurns, kind: str, log: list[str], name: str) -> threading.Thread:
    def run() -> None:
        with getattr(turns, kind)():
            log.append(name)

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def _until(condition, seconds: float = _WAIT_SECONDS) -> None:  # noqa: ANN001
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_a_dictation_waits_only_for_the_meeting_clip_already_decoding_and_goes_before_queued_clips() -> None:
    turns = EngineTurns()
    log: list[str] = []
    release_clip = threading.Event()

    def first_clip() -> None:
        with turns.meeting():
            log.append("clip 1 started")
            release_clip.wait(_WAIT_SECONDS)
            log.append("clip 1 done")

    decoding = threading.Thread(target=first_clip)
    decoding.start()
    _until(lambda: log == ["clip 1 started"])
    queued = [_waiting_for_turn(turns, "meeting", log, f"clip {n}") for n in (2, 3, 4)]
    time.sleep(0.05)
    dictation = _waiting_for_turn(turns, "dictation", log, "dictation")
    _until(lambda: turns._dictations_waiting == 1)
    release_clip.set()
    for thread in (decoding, dictation, *queued):
        thread.join(_WAIT_SECONDS)

    assert log[:3] == ["clip 1 started", "clip 1 done", "dictation"]
    assert sorted(log[3:]) == ["clip 2", "clip 3", "clip 4"]


def test_meeting_clips_wait_while_any_dictation_is_decoding() -> None:
    turns = EngineTurns()
    log: list[str] = []
    with turns.dictation():
        clip = _waiting_for_turn(turns, "meeting", log, "clip")
        time.sleep(0.05)
        assert log == []
    clip.join(_WAIT_SECONDS)
    assert log == ["clip"]


def test_only_one_decode_runs_at_a_time_among_hundreds_of_clips_and_dictations() -> None:
    turns = EngineTurns()
    inside = 0
    most = 0
    lock = threading.Lock()

    def decode(kind: str) -> None:
        nonlocal inside, most
        with getattr(turns, kind)():
            with lock:
                inside += 1
                most = max(most, inside)
            time.sleep(0.0005)
            with lock:
                inside -= 1

    threads = [threading.Thread(target=decode, args=("dictation" if n % 7 == 0 else "meeting",)) for n in range(300)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(_WAIT_SECONDS)

    assert most == 1
    assert turns._busy is False and turns._dictations_waiting == 0


def test_a_decode_that_fails_still_hands_the_turn_on() -> None:
    turns = EngineTurns()
    try:
        with turns.meeting():
            raise RuntimeError("out of GPU memory")
    except RuntimeError:
        pass
    with turns.dictation():
        pass
