"""Telemetry's endpoints: the shell reports what was used, reads the counts back, and exports or deletes it all."""

from collections.abc import Callable
from pathlib import Path

from skryb.telemetry.events import TelemetryStore
from skryb.ui.api_server import ApiError, ApiRequest, ApiResponse, Route
from skryb.ui.shapes import Fields

Json = dict[str, object]


class TelemetryApi:
    def __init__(self, store: TelemetryStore, *, context: Callable[[], Json]) -> None:
        self._store = store
        self._context = context

    def routes(self) -> list[Route]:
        return [
            ("GET", "/api/telemetry", self._status),
            ("POST", "/api/telemetry", self._record),
            ("POST", "/api/telemetry/export", self._export),
            ("DELETE", "/api/telemetry", self._clear),
        ]

    def _status(self, _request: ApiRequest) -> ApiResponse:
        counts = self._store.counts()
        return ApiResponse(200, {"events": sum(counts.values()), "counts": counts})

    def _record(self, request: ApiRequest) -> ApiResponse:
        """``{"events": [...]}``; whatever isn't a telemetry event is dropped, and all of it while telemetry is off."""
        self._store.record(_events_of(request.body))
        return ApiResponse(204)

    def _export(self, request: ApiRequest) -> ApiResponse:
        path = _path_of(request.body)
        try:
            events = self._store.export(path, self._context())
        except OSError as exc:
            raise ApiError(400, f"Could not export the usage data: {exc}") from exc
        return ApiResponse(200, {"events": events})

    def _clear(self, _request: ApiRequest) -> ApiResponse:
        try:
            self._store.clear()
        except OSError as exc:
            raise ApiError(400, f"Could not delete the usage data: {exc}") from exc
        return ApiResponse(204)


def _events_of(body: object) -> list[object]:
    try:
        return Fields(body, "body").items("events")
    except ValueError as exc:
        raise ApiError(400, "Send the events as a list.", str(exc)) from exc


def _path_of(body: object) -> Path:
    try:
        text = Fields(body, "body").text("path")
    except ValueError as exc:
        raise ApiError(400, "Say which file to use.", str(exc)) from exc
    if not text.strip():
        raise ApiError(400, "Say which file to use.")
    return Path(text)
