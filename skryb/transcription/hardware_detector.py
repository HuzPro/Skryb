from abc import ABC, abstractmethod

from skryb.transcription.engine_types import HardwareProfile


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

    @abstractmethod
    def peek(self) -> HardwareProfile | None:
        """The profile if it is already known without probing, else None.

        Safe on the GUI thread: it never imports torch or waits on a probe.
        """
