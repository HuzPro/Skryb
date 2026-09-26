"""The loopback HTTP server the UI shell talks to.

Standard library only. Every request carries the bearer token the shell was
started with; the event stream, which ``EventSource`` cannot add a header
to, takes it as ``?token=``. The shell's page is served from its own origin,
so responses allow any origin: the token, not the origin, is what grants
access. Handlers run on the server's threads, never on the GUI thread.
"""

import hmac
import json
import logging
import queue
import re
import secrets
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from voicetotext.ui.event_hub import EventHub

_logger = logging.getLogger(__name__)

_HOST = "127.0.0.1"
_EVENTS_PATH = "/api/events"
_MAX_BODY_BYTES = 8 * 1024 * 1024
_PARAMETER = re.compile(r"\{(\w+)\}")
_CORS_HEADERS = (
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET, PUT, POST, DELETE, OPTIONS"),
    ("Access-Control-Allow-Headers", "Authorization, Content-Type"),
    ("Access-Control-Allow-Private-Network", "true"),
    ("Access-Control-Max-Age", "600"),
)


@dataclass(frozen=True)
class ApiRequest:
    params: dict[str, str]
    query: dict[str, str]
    body: object = None


@dataclass(frozen=True)
class ApiResponse:
    status: int = 204
    body: object = None


class ApiError(Exception):
    """A request the API refuses; ``error`` is a sentence the shell shows as is."""

    def __init__(
        self,
        status: int,
        error: str,
        detail: str = "",
        extra: dict[str, object] | None = None,
    ) -> None:
        super().__init__(error)
        self.status = status
        self.error = error
        self.detail = detail
        self.extra = extra or {}

    def body(self) -> dict[str, object]:
        body: dict[str, object] = {"error": self.error, **self.extra}
        if self.detail:
            body["detail"] = self.detail
        return body


Handler = Callable[[ApiRequest], ApiResponse]
# Method, path with ``{name}`` parameters, handler.
Route = tuple[str, str, Handler]
_CompiledRoute = tuple[str, re.Pattern[str], Handler]


def sse_frame(name: str, data: object) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


class ApiServer:
    def __init__(
        self, hub: EventHub, token: str | None = None, *, poll_seconds: float = 0.5,
    ) -> None:
        self.hub = hub
        self.token = token or secrets.token_urlsafe(32)
        # How long ``stop`` can wait for the serving loop to notice.
        self._poll_seconds = poll_seconds
        self._routes: tuple[_CompiledRoute, ...] = ()
        self._httpd: _Server | None = None

    @property
    def port(self) -> int:
        return 0 if self._httpd is None else int(self._httpd.server_address[1])

    def start(self) -> None:
        """Bind a free loopback port and serve on a daemon thread."""
        httpd = _Server((_HOST, 0), _RequestHandler, self)
        threading.Thread(
            target=httpd.serve_forever, args=(self._poll_seconds,), name="ui-api", daemon=True,
        ).start()
        self._httpd = httpd
        _logger.info("UI API listening on %s:%d", _HOST, self.port)

    def bind(self, routes: Iterable[Route]) -> None:
        """Serve *routes* from now on, in place of whatever was bound before."""
        self._routes = tuple(
            (method, re.compile(_PARAMETER.sub(r"(?P<\1>[^/]+)", path) + "$"), handler)
            for method, path, handler in routes
        )

    def stop(self) -> None:
        self.hub.close()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def token_matches(self, given: str) -> bool:
        return hmac.compare_digest(given.encode(), self.token.encode())

    def handle(self, method: str, path: str, query: dict[str, str], body: object) -> ApiResponse:
        for route_method, pattern, handler in self._routes:
            match = pattern.match(path)
            if match and route_method == method:
                params = {key: unquote(value) for key, value in match.groupdict().items()}
                return handler(ApiRequest(params, query, body))
        raise ApiError(404, "There is nothing at that address.")


class _Server(ThreadingHTTPServer):
    # An event stream only ends when its client goes; stopping must not wait on it.
    block_on_close = False
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        api: ApiServer,
    ) -> None:
        super().__init__(address, handler)
        self.api = api


class _RequestHandler(BaseHTTPRequestHandler):
    server: _Server

    def do_GET(self) -> None:
        self._serve("GET")

    def do_PUT(self) -> None:
        self._serve("PUT")

    def do_POST(self) -> None:
        self._serve("POST")

    def do_DELETE(self) -> None:
        self._serve("DELETE")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        # The windowless app has no stderr for the default to write to.
        _logger.debug("UI API: " + format, *args)

    def _serve(self, method: str) -> None:
        url = urlsplit(self.path)
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        api = self.server.api
        if method == "GET" and url.path == _EVENTS_PATH:
            if not api.token_matches(query.get("token", "")):
                self._send_json(401, {"error": "The events stream needs the app's token."})
                return
            self._stream_events()
            return
        if not api.token_matches(self._bearer_token()):
            self._send_json(401, {"error": "This request needs the app's token."})
            return
        try:
            response = api.handle(method, url.path, query, self._read_body())
        except ApiError as error:
            self._send_json(error.status, error.body())
            return
        except Exception as exc:  # the shell gets a sentence, the log the traceback
            _logger.exception("UI API %s %s failed", method, url.path)
            self._send_json(500, {"error": "Something went wrong.", "detail": str(exc)})
            return
        self._send_json(response.status, response.body)

    def _bearer_token(self) -> str:
        scheme, _space, token = self.headers.get("Authorization", "").partition(" ")
        return token.strip() if scheme.lower() == "bearer" else ""

    def _read_body(self) -> object:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ApiError(400, "The request has an unreadable length.") from None
        if length <= 0:
            return None
        if length > _MAX_BODY_BYTES:
            raise ApiError(400, "The request is too large.")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ApiError(400, "The request body is not valid JSON.", str(exc)) from None

    def _send_json(self, status: int, body: object) -> None:
        payload = b"" if body is None else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self._send_cors()
        if payload:
            self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_cors(self) -> None:
        for name, value in _CORS_HEADERS:
            self.send_header(name, value)

    def _stream_events(self) -> None:
        # Joined before the headers go out, so nothing published after the
        # shell sees the stream open can be missed.
        hub = self.server.api.hub
        client = hub.connect()
        try:
            self._write_events(client, hub.heartbeat_seconds)
        except OSError:
            _logger.debug("UI event stream closed by the client.")
        finally:
            hub.disconnect(client)

    def _write_events(
        self, client: "queue.Queue[tuple[str, object] | None]", heartbeat: float,
    ) -> None:
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        while True:
            try:
                event = client.get(timeout=heartbeat)
            except queue.Empty:
                event = ("heartbeat", {})
            if event is None:
                return
            self.wfile.write(sse_frame(*event))
            self.wfile.flush()
