import json
import logging
from pathlib import Path

import pytest

from voicetotext.transcription.ct2_conversion import (
    ct2_model_dir,
    ensure_ct2_model,
    prebuild_ct2_model,
    snapshot_has_ct2_model,
)


_MODULE = "voicetotext.transcription.ct2_conversion"

_BUILT_FILENAMES = ("model.bin", "config.json", "vocabulary.json")
_PINNED_SHA = "de0369c8a68025b7f6e86387b6eb5a3b369787c8"
_OTHER_SHA = "f4334f6e8193f2691212d49b20fa12d370e13896"


def _write_snapshot(
    root: Path, sha: str = _PINNED_SHA, model_type: str = "whisper",
) -> Path:
    snapshot = root / "snapshots" / sha
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"WEIGHTS")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    return snapshot


@pytest.fixture()
def log_capture(caplog):
    """Attach caplog handler to the module logger (propagate=False here)."""
    logger = logging.getLogger(_MODULE)
    logger.addHandler(caplog.handler)
    caplog.handler.setLevel(logging.DEBUG)
    yield caplog
    logger.removeHandler(caplog.handler)


class _RecordingConverter:
    """Stand-in for CTranslate2's converter - writes the files it would."""

    def __init__(self) -> None:
        self.outputs: list[Path] = []

    def __call__(self, snapshot: Path, output: Path) -> None:
        self.outputs.append(output)
        output.mkdir(parents=True)
        for name in _BUILT_FILENAMES:
            (output / name).write_text(f"built from {snapshot.name}")


def _explode(snapshot: Path, output: Path) -> None:
    output.mkdir(parents=True)
    (output / "model.bin").write_bytes(b"half a model")
    raise RuntimeError("converter blew up")


class TestBuildingTheConvertedModel:
    def test_should_build_when_the_snapshot_has_none(self, tmp_path: Path) -> None:
        snapshot = _write_snapshot(tmp_path)
        convert = _RecordingConverter()

        built = ensure_ct2_model(snapshot, convert)

        assert built == ct2_model_dir(snapshot)
        assert snapshot_has_ct2_model(snapshot)
        assert (built / "model.bin").exists()

    def test_should_reuse_an_existing_build_without_converting_again(
        self, tmp_path: Path,
    ) -> None:
        snapshot = _write_snapshot(tmp_path)
        first = _RecordingConverter()
        ensure_ct2_model(snapshot, first)
        second = _RecordingConverter()

        ensure_ct2_model(snapshot, second)

        assert second.outputs == []

    def test_should_leave_the_source_snapshot_untouched(
        self, tmp_path: Path,
    ) -> None:
        snapshot = _write_snapshot(tmp_path)

        ensure_ct2_model(snapshot, _RecordingConverter())

        assert (snapshot / "model.safetensors").read_bytes() == b"WEIGHTS"
        assert (snapshot / "config.json").exists()


class TestBuildCompletionBarrier:
    def test_a_failed_conversion_should_leave_no_build_behind(
        self, tmp_path: Path,
    ) -> None:
        snapshot = _write_snapshot(tmp_path)

        with pytest.raises(RuntimeError):
            ensure_ct2_model(snapshot, _explode)

        assert not snapshot_has_ct2_model(snapshot)
        assert not ct2_model_dir(snapshot).exists()
        assert list(snapshot.glob(".ct2*")) == []

    def test_should_rebuild_after_a_failure(self, tmp_path: Path) -> None:
        snapshot = _write_snapshot(tmp_path)
        with pytest.raises(RuntimeError):
            ensure_ct2_model(snapshot, _explode)

        ensure_ct2_model(snapshot, _RecordingConverter())

        assert snapshot_has_ct2_model(snapshot)

    def test_should_discard_a_partial_build_from_an_earlier_crash(
        self, tmp_path: Path,
    ) -> None:
        snapshot = _write_snapshot(tmp_path)
        partial = ct2_model_dir(snapshot)
        partial.mkdir()
        (partial / "model.bin").write_bytes(b"truncated")

        ensure_ct2_model(snapshot, _RecordingConverter())

        assert snapshot_has_ct2_model(snapshot)
        assert (partial / "model.bin").read_text().startswith("built from")


class TestBuildIsKeyedToTheSourceRevision:
    def test_a_different_revision_should_not_be_reported_as_built(
        self, tmp_path: Path,
    ) -> None:
        pinned = _write_snapshot(tmp_path, sha=_PINNED_SHA)
        ensure_ct2_model(pinned, _RecordingConverter())

        repinned = _write_snapshot(tmp_path, sha=_OTHER_SHA)

        assert snapshot_has_ct2_model(pinned)
        assert not snapshot_has_ct2_model(repinned)

    def test_should_build_separately_for_each_revision(
        self, tmp_path: Path,
    ) -> None:
        pinned = _write_snapshot(tmp_path, sha=_PINNED_SHA)
        repinned = _write_snapshot(tmp_path, sha=_OTHER_SHA)
        convert = _RecordingConverter()

        ensure_ct2_model(pinned, convert)
        ensure_ct2_model(repinned, convert)

        assert (ct2_model_dir(pinned) / "model.bin").read_text().endswith(
            _PINNED_SHA,
        )
        assert (ct2_model_dir(repinned) / "model.bin").read_text().endswith(
            _OTHER_SHA,
        )


class TestPostInstallPrebuild:
    def test_should_ignore_a_snapshot_that_is_not_a_whisper_model(
        self, tmp_path: Path,
    ) -> None:
        snapshot = _write_snapshot(tmp_path, model_type="nemo")
        convert = _RecordingConverter()

        prebuild_ct2_model(snapshot, convert)

        assert convert.outputs == []
        assert not ct2_model_dir(snapshot).exists()

    def test_should_ignore_a_snapshot_without_a_config(
        self, tmp_path: Path,
    ) -> None:
        snapshot = tmp_path / "snapshots" / _PINNED_SHA
        snapshot.mkdir(parents=True)
        (snapshot / "parakeet.nemo").touch()
        convert = _RecordingConverter()

        prebuild_ct2_model(snapshot, convert)

        assert convert.outputs == []

    def test_should_warn_instead_of_raising_when_the_build_fails(
        self, tmp_path: Path, log_capture: pytest.LogCaptureFixture,
    ) -> None:
        snapshot = _write_snapshot(tmp_path)

        prebuild_ct2_model(snapshot, _explode)

        assert not ct2_model_dir(snapshot).exists()
        assert any(
            "CTranslate2 build" in record.getMessage()
            for record in log_capture.records
        )

    def test_should_build_a_whisper_snapshot(self, tmp_path: Path) -> None:
        snapshot = _write_snapshot(tmp_path)

        prebuild_ct2_model(snapshot, _RecordingConverter())

        assert snapshot_has_ct2_model(snapshot)
