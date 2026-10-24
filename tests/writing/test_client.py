import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from skryb.writing.client import Endpoint, WritingClient, WritingUnavailable

_KEY = "k3y-for-tests"


class _Model:
    def __init__(self) -> None:
        self.delay = 0.0
        self.body = _completion("Let's meet on Monday.")
        self.requests: list[dict[str, object]] = []
        self.authorizations: list[str] = []


def _completion(content: str, finish_reason: str = "stop") -> bytes:
    return json.dumps({
        "choices": [{"finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}],
    }).encode()


@pytest.fixture
def model() -> Iterator[tuple[_Model, Endpoint]]:
    state = _Model()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server name)
            length = int(self.headers["Content-Length"])
            state.requests.append(json.loads(self.rfile.read(length)))
            state.authorizations.append(self.headers.get("Authorization", ""))
            time.sleep(state.delay)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(state.body)))
                self.end_headers()
                self.wfile.write(state.body)
            except OSError:
                pass

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield state, Endpoint(f"http://{host}:{port}", _KEY)
    server.shutdown()
    server.server_close()


def test_a_completion_sends_the_key_the_greedy_settings_and_the_cached_prompt_flag(model):
    state, endpoint = model
    client = WritingClient(lambda: endpoint)

    answer = client.complete("rules", '{"dictation": "meet friday actually monday"}', 64, 5.0)

    assert answer == "Let's meet on Monday."
    assert state.authorizations == [f"Bearer {_KEY}"]
    request = state.requests[0]
    assert request["temperature"] == 0
    assert request["cache_prompt"] is True
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["max_tokens"] == 64


def test_the_deadline_gives_up_on_a_slow_answer_instead_of_waiting_for_it(model):
    state, endpoint = model
    state.delay = 1.5
    client = WritingClient(lambda: endpoint)
    started = time.monotonic()

    with pytest.raises(WritingUnavailable):
        client.complete("rules", "{}", 64, 0.3)

    assert time.monotonic() - started < 1.2


def test_a_cut_off_answer_is_not_used(model):
    state, endpoint = model
    state.body = _completion("Let's meet on", finish_reason="length")

    with pytest.raises(WritingUnavailable, match="cut short"):
        WritingClient(lambda: endpoint).complete("rules", "{}", 4, 5.0)


def test_an_oversized_answer_is_not_read_into_memory(model):
    state, endpoint = model
    state.body = _completion("x" * 1_200_000)

    with pytest.raises(WritingUnavailable):
        WritingClient(lambda: endpoint).complete("rules", "{}", 64, 5.0)


def test_a_model_that_is_not_running_is_unavailable_without_a_request():
    client = WritingClient(lambda: None)

    assert not client.is_ready()
    with pytest.raises(WritingUnavailable, match="not running"):
        client.complete("rules", "{}", 64, 5.0)


def test_only_one_request_is_in_flight_and_a_second_gives_up_at_once(model):
    state, endpoint = model
    state.delay = 0.25
    client = WritingClient(lambda: endpoint)
    first = threading.Thread(target=client.complete, args=("rules", "{}", 64, 5.0))
    first.start()
    time.sleep(0.05)

    with pytest.raises(WritingUnavailable, match="busy"):
        client.complete("rules", "{}", 64, 5.0)
    first.join()
