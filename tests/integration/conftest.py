"""Integration-test fixtures that need a real Whisper model on the GPU.

``real_whisper_engine`` is parametrized dynamically over every model
that is already present in the user's HuggingFace cache at collection
time.  Each model is loaded once per session on the CUDA/ROCm device
(engine construction is expensive) and disposed at session teardown.

Tests are skipped cleanly in three situations:

* the HuggingFace cache is empty — no parametrize values, nothing runs;
* ``torch.cuda.is_available()`` is ``False`` — :func:`require_cuda`
  skips each test instance with a clear reason;
* an audio fixture is missing — the fixture-side helpers skip.
"""

from collections.abc import Iterator

import pytest

from voicetotext.transcription.engine import TranscriptionEngine

from tests.helpers.gpu import require_cuda
from tests.helpers.model_cache import installed_model_names, models_dir


INSTALLED_WHISPER_MODELS = installed_model_names()


@pytest.fixture(scope="session", params=INSTALLED_WHISPER_MODELS)
def real_whisper_engine(
    request: pytest.FixtureRequest,
) -> Iterator[TranscriptionEngine]:
    require_cuda()
    from voicetotext.transcription.transformers_whisper_adapter import (
        TransformersWhisperAdapter,
    )

    model_name: str = request.param
    engine = TransformersWhisperAdapter(
        model_size=model_name,
        device="cuda",
        language="en",
        download_root=str(models_dir()),
    )
    engine.load_model()
    try:
        yield engine
    finally:
        engine.dispose()
