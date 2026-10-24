"""Hand-rolled test fakes for Skryb ports.

Each module in this package provides concrete in-memory implementations
of a production port ABC.  Unlike ``unittest.mock.MagicMock``, these
fakes obey their port's contract exactly and expose observable state
via plain attributes (lists, counters, flags) so tests can make
behaviour assertions without patching or spec-inspection.
"""

from tests.fakes.config import FakeConfig
from tests.fakes.dispatcher import DeferredActionDispatcher, InlineActionDispatcher
from tests.fakes.downloads import FakeHttpClient
from tests.fakes.events import RecordingEventBus
from tests.fakes.hardware import FakeHardwareDetector
from tests.fakes.hotkeys import FakeHotkeyListener
from tests.fakes.licensing import AllowAllLicensePolicy
from tests.fakes.model_manager import FakeModelManager
from tests.fakes.output import FakeClipboardPreserver, FakeTextOutputHandler, InstantSleeper
from tests.fakes.recording import FakeAudioRecorder
from tests.fakes.transcription import FakeTranscriptionEngine

__all__ = [
    "AllowAllLicensePolicy",
    "DeferredActionDispatcher",
    "FakeAudioRecorder",
    "FakeClipboardPreserver",
    "FakeConfig",
    "FakeHardwareDetector",
    "FakeHotkeyListener",
    "FakeHttpClient",
    "FakeModelManager",
    "FakeTextOutputHandler",
    "FakeTranscriptionEngine",
    "InlineActionDispatcher",
    "InstantSleeper",
    "RecordingEventBus",
]
