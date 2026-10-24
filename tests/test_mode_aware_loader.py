from unittest.mock import MagicMock

import skryb.bootstrap as bootstrap
from tests.fakes import AllowAllLicensePolicy, FakeConfig
from skryb.diagnostics.session import SessionDiagnostics


def _config(mode: str) -> FakeConfig:
    return FakeConfig(
        {
            "transcription.mode": mode,
            "transcription.model_size": "parakeet-tdt-0.6b-v3",
        }
    )


class TestModeAwareLoader:
    def test_streaming_mode_installs_the_streaming_engine(self, monkeypatch):
        fake_streaming = MagicMock()
        monkeypatch.setattr(
            bootstrap,
            "_load_streaming_engine",
            lambda config, paths, hardware, policy, diagnostics: fake_streaming,
        )
        monkeypatch.setattr(
            bootstrap, "_load_engine_with_fallback",
            MagicMock(side_effect=AssertionError("batch load in streaming mode")),
        )

        installer = bootstrap._load_active_engine_installer(
            _config("streaming"), MagicMock(), MagicMock(), MagicMock(),
            AllowAllLicensePolicy(), MagicMock(spec=SessionDiagnostics),
        )
        use_case = MagicMock()
        installer(use_case)

        use_case.set_streaming_engine.assert_called_once_with(fake_streaming)
        use_case.set_engine.assert_not_called()

    def test_batch_mode_installs_the_batch_engine(self, monkeypatch):
        fake_batch = MagicMock()
        monkeypatch.setattr(
            bootstrap, "_load_engine_with_fallback",
            lambda config, paths, registry, hardware, policy, diagnostics, preload=None: (
                fake_batch
            ),
        )
        monkeypatch.setattr(
            bootstrap, "_load_streaming_engine",
            MagicMock(side_effect=AssertionError("streaming load in batch mode")),
        )

        installer = bootstrap._load_active_engine_installer(
            _config("batch"), MagicMock(), MagicMock(), MagicMock(),
            AllowAllLicensePolicy(), MagicMock(spec=SessionDiagnostics),
        )
        use_case = MagicMock()
        installer(use_case)

        use_case.set_engine.assert_called_once_with(fake_batch)
        use_case.set_streaming_engine.assert_not_called()
