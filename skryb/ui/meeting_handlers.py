"""The Meetings window's endpoints: start and stop, the live meeting, and the saved ones.

The live meeting's lines, names and status reach the window as ``meeting``
server events; a saved meeting is read from its folder on request.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

from skryb.core.action_dispatcher import ThreadedActionDispatcher
from skryb.meetings.model_files import MeetingModelFiles
from skryb.meetings.recorder import MeetingError, MeetingRecorder
from skryb.meetings.storage import list_meetings, meeting_folder, vocabulary_file
from skryb.meetings.transcript import Transcript, is_speaker_key
from skryb.ui.api_server import ApiError, ApiRequest, ApiResponse, Route
from skryb.ui.event_hub import EventHub
from skryb.ui.shapes import Fields

_logger = logging.getLogger(__name__)

_NAME_CHARS = 60
_NO_CONTENT = ApiResponse(204)
_ACCEPTED = ApiResponse(202)
_IN_BACKGROUND = ThreadedActionDispatcher()


class MeetingApi:
    def __init__(
        self,
        recorder: MeetingRecorder,
        hub: EventHub,
        *,
        folder: Callable[[], Path],
        model_files: MeetingModelFiles,
        open_path: Callable[[Path], None],
        run_in_background: Callable[[Callable[[], None]], None] = _IN_BACKGROUND.dispatch,
    ) -> None:
        self._recorder = recorder
        self._hub = hub
        self._folder = folder
        self._model_files = model_files
        self._open_path = open_path
        self._run_in_background = run_in_background

    def routes(self) -> list[Route]:
        return [
            ("GET", "/api/meetings", self._overview),
            ("GET", "/api/meetings/live", self._live),
            ("POST", "/api/meetings/start", self._start),
            ("POST", "/api/meetings/stop", self._stop),
            ("POST", "/api/meetings/models/download", self._download_models),
            ("POST", "/api/meetings/vocabulary/open", self._open_vocabulary),
            ("POST", "/api/meetings/folder/open", self._open_meetings_folder),
            ("GET", "/api/meetings/{id}", self._read),
            ("POST", "/api/meetings/{id}/rename", self._rename),
            ("POST", "/api/meetings/{id}/open", self._open_meeting),
        ]

    def publish(self, event: dict[str, object]) -> None:
        self._hub.publish("meeting", event)

    def recording(self) -> bool:
        return self._recorder.is_recording()

    def _overview(self, _request: ApiRequest) -> ApiResponse:
        missing = self._model_files.missing()
        return ApiResponse(200, {
            "recording": self._recorder.is_recording(),
            "folder": str(self._folder()),
            "models_ready": not missing,
            "models": [
                {**view, "download": self._hub.download_of(str(view["name"]))}
                for view in self._model_files.views()
            ],
            "meetings": [summary.json() for summary in list_meetings(self._folder())],
        })

    def _live(self, _request: ApiRequest) -> ApiResponse:
        return ApiResponse(200, {"meeting": self._recorder.live()})

    def _start(self, _request: ApiRequest) -> ApiResponse:
        try:
            self._recorder.start()
        except MeetingError as exc:
            raise ApiError(400, str(exc)) from exc
        return _ACCEPTED

    def _stop(self, _request: ApiRequest) -> ApiResponse:
        self._run_in_background(self._recorder.stop)
        return _ACCEPTED

    def _download_models(self, _request: ApiRequest) -> ApiResponse:
        try:
            self._model_files.download_missing()
        except Exception as exc:
            _logger.exception("The meeting models' download did not start")
            raise ApiError(400, f"Could not start the download: {exc}") from exc
        return _ACCEPTED

    def _open_vocabulary(self, _request: ApiRequest) -> ApiResponse:
        try:
            path = vocabulary_file(self._folder())
        except OSError as exc:
            raise ApiError(400, f"Could not create the vocabulary file: {exc}") from exc
        self._open_path(path)
        return _NO_CONTENT

    def _open_meetings_folder(self, _request: ApiRequest) -> ApiResponse:
        folder = self._folder()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ApiError(400, f"Could not create {folder}: {exc}") from exc
        self._open_path(folder)
        return _NO_CONTENT

    def _read(self, request: ApiRequest) -> ApiResponse:
        live = self._recorder.live_session(request.params["id"])
        if live is not None:
            return ApiResponse(200, live.transcript.view())
        return ApiResponse(200, self._saved(request).view())

    def _rename(self, request: ApiRequest) -> ApiResponse:
        key, name = _rename_request(request.body)
        if not is_speaker_key(key):
            raise ApiError(400, "That is not a speaker in this meeting.")
        live = self._recorder.live_session(request.params["id"])
        transcript = live.transcript if live is not None else self._saved(request, publish=self.publish)
        try:
            transcript.rename(key, name)
        except OSError as exc:
            raise ApiError(400, f"Could not save the new name: {exc}") from exc
        return _NO_CONTENT

    def _open_meeting(self, request: ApiRequest) -> ApiResponse:
        self._open_path(self._meeting_folder(request))
        return _NO_CONTENT

    def _meeting_folder(self, request: ApiRequest) -> Path:
        folder = meeting_folder(self._folder(), request.params["id"])
        if folder is None:
            raise ApiError(404, "That meeting is no longer in the meetings folder.")
        return folder

    def _saved(self, request: ApiRequest, publish: Callable[[dict[str, object]], None] | None = None) -> Transcript:
        folder = self._meeting_folder(request)
        try:
            if publish is None:
                return Transcript.load(folder)
            meeting_id = folder.name
            return Transcript.load(folder, lambda event: publish({**event, "id": meeting_id}))
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, f"That meeting's transcript could not be read: {exc}") from exc


def _rename_request(body: object) -> tuple[str, str]:
    try:
        fields = Fields(body, "body")
        return fields.text("key"), fields.text("name")[:_NAME_CHARS]
    except ValueError as exc:
        raise ApiError(400, "Say which speaker to rename and the new name.", str(exc)) from exc
