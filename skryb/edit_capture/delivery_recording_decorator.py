import logging
from collections.abc import Callable

from skryb.output.handler import TextOutputHandler

_logger = logging.getLogger(__name__)


class DeliveryRecordingDecorator(TextOutputHandler):
    """Reports the text leaving the pipeline, on the way to the paste.

    This is the only point that sees the transcription after the
    dictionary, filler removal, spoken commands and number formatting have
    run. It cannot see the paste land: every handler below it hands the
    work to the GUI thread and returns.
    """

    def __init__(
        self, inner: TextOutputHandler, record: Callable[[str], None],
    ) -> None:
        self._inner = inner
        self._record = record

    def output(self, text: str) -> None:
        self._record_quietly(text)
        self._inner.output(text)

    def _record_quietly(self, text: str) -> None:
        try:
            self._record(text)
        except Exception as exc:  # noqa: BLE001 - never block a transcription
            _logger.warning("Edit capture could not record a delivery: %s", exc)
