from abc import ABC, abstractmethod

from voicetotext.transcription.engine_types import HardwareProfile


class HardwareDetector(ABC):
    """Port for probing the local accelerator landscape.

    The detector returns a :class:`HardwareProfile` describing the most
    capable available device. Adapters wrap third-party probes (e.g.
    PyTorch's ``torch.cuda.*`` namespace, which serves both CUDA and
    ROCm builds) so the rest of the app stays free of those imports.
    """

    @abstractmethod
    def detect(self) -> HardwareProfile:
        pass
