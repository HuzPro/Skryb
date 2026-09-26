import logging
from collections.abc import Callable
from typing import cast

from voicetotext.core.config import ConfigurationStore
from voicetotext.transcription.engine_types import (
    ALL_HARDWARE_KINDS,
    HardwareKind,
    HardwareProfile,
)
from voicetotext.transcription.hardware_detector import HardwareDetector

_logger = logging.getLogger(__name__)

_KIND_KEY = "hardware.probe.kind"
_DEVICE_NAME_KEY = "hardware.probe.device_name"
_GFX_ARCH_KEY = "hardware.probe.gfx_arch"
_FINGERPRINT_KEY = "hardware.probe.fingerprint"


class CachedHardwareDetector(HardwareDetector):
    """Remembers the GPU probe in config until the display adapters change.

    A cold probe imports torch, which takes up to ~19 s on a ROCm install,
    yet the answer only changes when the GPU does. The display-adapter
    fingerprint stands in for "same GPU as last time"; an empty one means
    that cannot be told, so nothing is remembered or reused.
    """

    def __init__(
        self,
        inner: HardwareDetector,
        config: ConfigurationStore,
        fingerprint: Callable[[], str],
    ) -> None:
        self._inner = inner
        self._config = config
        self._fingerprint = fingerprint

    def detect(self) -> HardwareProfile:
        fingerprint = self._fingerprint()
        remembered = self._remembered(fingerprint)
        if remembered is not None:
            return remembered
        profile = self._inner.detect()
        if fingerprint:
            self._remember(profile, fingerprint)
        return profile

    def peek(self) -> HardwareProfile | None:
        return self._inner.peek() or self._remembered(self._fingerprint())

    def _remembered(self, fingerprint: str) -> HardwareProfile | None:
        if not fingerprint or self._config.get(_FINGERPRINT_KEY) != fingerprint:
            return None
        kind = self._config.get(_KIND_KEY)
        if kind not in ALL_HARDWARE_KINDS:
            return None
        gfx_arch = self._config.get(_GFX_ARCH_KEY)
        return HardwareProfile(
            kind=cast(HardwareKind, kind),
            device_name=str(self._config.get(_DEVICE_NAME_KEY, "")),
            gfx_arch=None if gfx_arch is None else str(gfx_arch),
        )

    def _remember(self, profile: HardwareProfile, fingerprint: str) -> None:
        self._config.set(_KIND_KEY, profile.kind)
        self._config.set(_DEVICE_NAME_KEY, profile.device_name)
        self._config.set(_GFX_ARCH_KEY, profile.gfx_arch)
        self._config.set(_FINGERPRINT_KEY, fingerprint)
        try:
            self._config.save()
        except OSError:
            _logger.warning("Could not save the hardware probe", exc_info=True)
