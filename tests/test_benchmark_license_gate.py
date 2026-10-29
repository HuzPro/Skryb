"""The benchmark harness may not touch gated weights either.

A sweep downloads and runs the same models the app gates, and the licence
covers use and generated outputs rather than the download alone. Both
benchmark paths - the pre-download step and the per-combo runner -
therefore consult the same acceptance record the app writes.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from tests.fakes.config import FakeConfig
from tests.performance._combos import BenchmarkCombo
from tests.performance._pre_download import ensure_models_downloaded
from tests.performance._runner import BenchmarkResult, run_benchmark
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.license_policy import LicensePolicy
from skryb.transcription.model_info import ALL_MODELS


_GATED_COMBO = BenchmarkCombo("crisperwhisper-2.0-turbo", "crisperwhisper", "cpu")
_UNGATED_COMBO = BenchmarkCombo("parakeet-ultra", "nano_parakeet", "cpu")


def _refusing_policy() -> LicensePolicy:
    return LicensePolicy(FakeConfig())


def _accepting_policy() -> LicensePolicy:
    policy = LicensePolicy(FakeConfig())
    policy.accept(ALL_MODELS["crisperwhisper-2.0-turbo"])
    return policy


def _audio() -> AudioSegment:
    return AudioSegment(data=np.zeros(16000, dtype=np.float32), sample_rate=16000)


def _benchmark(
    combo: BenchmarkCombo, policy: LicensePolicy, registry: MagicMock,
) -> BenchmarkResult:
    return run_benchmark(
        combo=combo,
        audio=_audio(),
        n_timed_runs=1,
        registry=registry,
        models_dir=Path("/app/models"),
        license_policy=policy,
    )


class TestBenchmarkPreDownload:
    def test_should_not_fetch_gated_weights_without_acceptance(
        self, tmp_path: Path,
    ) -> None:
        logged: list[str] = []

        with patch("huggingface_hub.snapshot_download") as download:
            ensure_models_downloaded(
                [_GATED_COMBO], tmp_path, _refusing_policy(), log=logged.append,
            )

        download.assert_not_called()
        assert any("LicenseNotAcceptedError" in line for line in logged)

    def test_should_fetch_gated_weights_once_accepted(
        self, tmp_path: Path,
    ) -> None:
        with patch("huggingface_hub.snapshot_download") as download:
            ensure_models_downloaded(
                [_GATED_COMBO], tmp_path, _accepting_policy(), log=lambda _: None,
            )

        assert download.call_args.kwargs["revision"] == (
            ALL_MODELS["crisperwhisper-2.0-turbo"].revision
        )

    def test_should_leave_an_ungated_model_alone(self, tmp_path: Path) -> None:
        with patch("huggingface_hub.snapshot_download") as download:
            ensure_models_downloaded(
                [_UNGATED_COMBO], tmp_path, _refusing_policy(), log=lambda _: None,
            )

        download.assert_called_once()


class TestBenchmarkRunner:
    def test_should_refuse_to_build_an_engine_for_an_unaccepted_licence(
        self,
    ) -> None:
        registry = MagicMock()

        result = _benchmark(_GATED_COMBO, _refusing_policy(), registry)

        assert result.status == "error"
        assert result.error_type == "LicenseNotAcceptedError"
        registry.build.assert_not_called()

    def test_should_build_the_engine_once_the_licence_is_accepted(self) -> None:
        registry = MagicMock()

        result = _benchmark(_GATED_COMBO, _accepting_policy(), registry)

        assert result.status == "ok"
        registry.build.assert_called_once()

    def test_should_leave_an_ungated_model_benchmarkable(self) -> None:
        registry = MagicMock()

        result = _benchmark(_UNGATED_COMBO, _refusing_policy(), registry)

        assert result.status == "ok"
