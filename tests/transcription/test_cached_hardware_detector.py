from pathlib import Path

from tests.fakes.config import FakeConfig
from tests.fakes.hardware import FakeHardwareDetector
from skryb.core.json_config_adapter import JsonConfigAdapter
from skryb.transcription.cached_hardware_detector import (
    CachedHardwareDetector,
)
from skryb.transcription.engine_types import HardwareProfile

_RX_9060 = HardwareProfile(
    kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200",
)
_RTX_4090 = HardwareProfile(
    kind="nvidia_cuda", device_name="NVIDIA GeForce RTX 4090", gfx_arch=None,
)
_AMD_DESKTOP = "AMD Radeon RX 9060 XT|AMD Radeon(TM) Graphics"
_NVIDIA_DESKTOP = "NVIDIA GeForce RTX 4090|Écran Radeon™ intégré"


def _launch(
    config_path: Path, probe: FakeHardwareDetector, fingerprint: str,
) -> CachedHardwareDetector:
    config = JsonConfigAdapter(config_path)
    config.load()
    return CachedHardwareDetector(probe, config, lambda: fingerprint)


def test_a_probe_is_reused_after_a_restart_while_the_display_adapters_are_unchanged(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    assert _launch(
        config_path, FakeHardwareDetector(_RX_9060, probed=False), _AMD_DESKTOP,
    ).detect() == _RX_9060

    next_probe = FakeHardwareDetector(_RX_9060, probed=False)
    detector = _launch(config_path, next_probe, _AMD_DESKTOP)

    assert detector.peek() == _RX_9060
    assert detector.detect() == _RX_9060
    assert next_probe.detect_calls == 0


def test_a_changed_display_adapter_forces_a_fresh_probe_that_replaces_the_memory(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    _launch(
        config_path, FakeHardwareDetector(_RX_9060, probed=False), _AMD_DESKTOP,
    ).detect()

    swapped_probe = FakeHardwareDetector(_RTX_4090, probed=False)
    detector = _launch(config_path, swapped_probe, _NVIDIA_DESKTOP)

    assert detector.peek() is None
    assert detector.detect() == _RTX_4090
    assert swapped_probe.detect_calls == 1
    later_probe = FakeHardwareDetector(_RX_9060, probed=False)
    assert _launch(config_path, later_probe, _NVIDIA_DESKTOP).detect() == _RTX_4090
    assert later_probe.detect_calls == 0


def test_an_empty_fingerprint_never_remembers_or_reuses_a_probe() -> None:
    config = FakeConfig({
        "hardware.probe.kind": "amd_rocm",
        "hardware.probe.device_name": "AMD Radeon RX 9060 XT",
        "hardware.probe.gfx_arch": "gfx1200",
        "hardware.probe.fingerprint": "",
    })
    probe = FakeHardwareDetector(_RTX_4090, probed=False)
    detector = CachedHardwareDetector(probe, config, lambda: "")

    assert detector.peek() is None
    assert detector.detect() == _RTX_4090
    assert probe.detect_calls == 1
    assert config.save_count == 0
    assert config.get("hardware.probe.kind") == "amd_rocm"


def test_a_hand_edited_hardware_kind_is_probed_again() -> None:
    config = FakeConfig({
        "hardware.probe.kind": "quantum",
        "hardware.probe.device_name": "",
        "hardware.probe.fingerprint": _AMD_DESKTOP,
    })
    probe = FakeHardwareDetector(_RX_9060, probed=False)

    assert CachedHardwareDetector(probe, config, lambda: _AMD_DESKTOP).detect() == (
        _RX_9060
    )
    assert probe.detect_calls == 1
    assert config.get("hardware.probe.kind") == "amd_rocm"


class _ReadOnlyConfig(FakeConfig):
    def save(self) -> None:
        raise PermissionError("config.json is read-only")


def test_the_probe_still_answers_when_the_config_cannot_be_saved() -> None:
    probe = FakeHardwareDetector(_RX_9060, probed=False)
    detector = CachedHardwareDetector(probe, _ReadOnlyConfig(), lambda: _AMD_DESKTOP)

    assert detector.detect() == _RX_9060
    assert detector.peek() == _RX_9060
