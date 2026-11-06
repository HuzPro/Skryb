"""A stand-in for llama.cpp's ``llama-server`` on loopback: no model and no GPU, it answers by rule.

A translation is ``EN: `` plus the text asked about. A request with a grammar is answered with the
words that grammar spells out, read back out of it, each glossed ``gloss of <word>``, so a test that
gets every word back in order has also shown the grammar holds every word in order.
"""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType

from skryb.translation.llama_server import Endpoint

_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')


def words_in_grammar(grammar: str) -> list[str]:
    """The fixed word of each ``wN`` rule, in order."""
    words = []
    for line in grammar.splitlines():
        if re.match(r"w\d+ ::=", line):
            literal = _LITERAL.findall(line)[1]
            words.append(json.loads(re.sub(r"\\(.)", r"\1", literal)))
    return words


class FakeLlamaServer:
    def __init__(self, api_key: str = "test-key") -> None:
        self.api_key = api_key
        self.requests: list[dict[str, object]] = []
        self.finish_reason = "stop"
        # Requests about any of these texts come back cut short, as a runaway answer would.
        self.cut_short: set[str] = set()
        # Every answer comes back as reasoning with no content, as Jais-3's thinking template files a grammar-locked one.
        self.answers_as_reasoning = False
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, args=(0.02,), daemon=True)

    @property
    def endpoint(self) -> Endpoint:
        return Endpoint(f"http://127.0.0.1:{self._server.server_address[1]}", self.api_key)

    def __enter__(self) -> "FakeLlamaServer":
        self._thread.start()
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None) -> None:
        self._server.shutdown()
        self._server.server_close()

    def answer(self, payload: dict[str, object]) -> str:
        grammar = payload.get("grammar")
        if isinstance(grammar, str):
            words = words_in_grammar(grammar)
            return json.dumps([{"w": word, "g": f"gloss of {word}"} for word in words], ensure_ascii=False, separators=(",", ":"))
        messages = payload["messages"]
        assert isinstance(messages, list)
        return "EN: " + str(messages[-1]["content"]).rsplit("\n\n", 1)[-1]

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return None

            def do_GET(self) -> None:
                self._reply(200 if self._allowed() else 401, {"status": "ok"})

            def do_POST(self) -> None:
                if not self._allowed():
                    self._reply(401, {"error": "no key"})
                    return
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(payload)
                content = fake.answer(payload)
                cut = any(text in json.dumps(payload["messages"], ensure_ascii=False) for text in fake.cut_short)
                message = {"content": "", "reasoning_content": content} if fake.answers_as_reasoning else {"content": content}
                self._reply(200, {"choices": [{"finish_reason": "length" if cut else fake.finish_reason, "message": message}]})

            def _allowed(self) -> bool:
                return self.headers.get("Authorization") == f"Bearer {fake.api_key}"

            def _reply(self, status: int, body: object) -> None:
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler
