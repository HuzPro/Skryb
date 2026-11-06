"""Show English's endpoints: the model this computer uses and its download, and English for History's lines."""

from collections.abc import Callable

from skryb.translation.llama_server import ServerError
from skryb.translation.service import TranslationService
from skryb.translation.translator import has_arabic
from skryb.ui.api_server import ApiError, ApiRequest, ApiResponse, Route
from skryb.ui.event_hub import EventHub
from skryb.ui.shapes import Fields

# A dictation longer than this is not sent; it would hold the model for minutes.
_MAX_CHARS = 60_000


class TranslationApi:
    def __init__(self, service: TranslationService, hub: EventHub, *, enabled: Callable[[], bool]) -> None:
        self._service = service
        self._hub = hub
        self._enabled = enabled

    def routes(self) -> list[Route]:
        return [
            ("GET", "/api/translation", self._status),
            ("POST", "/api/translation", self._translate),
            ("POST", "/api/translation/download", self._download),
        ]

    def _status(self, _request: ApiRequest) -> ApiResponse:
        view = self._service.view()
        return ApiResponse(200, {**view, "download": self._hub.download_of(str(view["name"]))})

    def _download(self, _request: ApiRequest) -> ApiResponse:
        """Fetch the model again after a failed or cancelled download; only while Show English is on."""
        if not self._enabled():
            raise ApiError(409, "Turn on Show English first.")
        self._service.install()
        return ApiResponse(202)

    def _translate(self, request: ApiRequest) -> ApiResponse:
        """``{"lines": [...]}``, one per line of the text: its English and gloss, or null for a line with no Arabic."""
        text = _text_of(request.body)
        if not self._enabled():
            raise ApiError(409, "Show English is off.")
        if not self._service.installed():
            raise ApiError(409, "The translation model is still downloading.")
        try:
            lines = self._annotated_lines(text)
        except (ServerError, ValueError) as exc:
            raise ApiError(503, f"Couldn't translate it: {exc}") from exc
        return ApiResponse(200, {"lines": lines})

    def _annotated_lines(self, text: str) -> list[dict[str, object] | None]:
        return [self._service.annotate(line).json() if has_arabic(line) else None for line in text.split("\n")]


def _text_of(body: object) -> str:
    try:
        text = Fields(body, "body").text("text")
    except ValueError as exc:
        raise ApiError(400, "Say which text to translate.", str(exc)) from exc
    if len(text) > _MAX_CHARS:
        raise ApiError(400, "That dictation is too long to translate here.")
    return text
