"""The translation models and llama.cpp builds Skryb can download, and which pair each computer runs.

Every computer runs llama.cpp's ``llama-server`` on one of four paths: the ROCm build on an AMD
GPU, the CUDA build on an NVIDIA GPU, the OpenVINO build on an Intel NPU, and the CPU build
everywhere else. GPUs get the larger model; the NPU and the CPU get the small one. Weights come
from the maintainers' own Hugging Face repos, and builds from llama.cpp's own GitHub releases,
both pinned.
"""

from dataclasses import dataclass

from skryb.transcription.engine_types import HardwareProfile

_LLAMA_BUILD = "b12144"
_RELEASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{_LLAMA_BUILD}/"


@dataclass(frozen=True)
class Archive:
    url: str
    size: int
    sha256: str

    @property
    def file_name(self) -> str:
        return self.url.rsplit("/", 1)[1]


@dataclass(frozen=True)
class TranslationModel:
    name: str
    label: str
    description: str
    repo_id: str
    revision: str
    file_name: str
    size: int
    # Sent to the model's chat template with every request.
    chat_options: tuple[tuple[str, str], ...] = ()

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{self.file_name}"


@dataclass(frozen=True)
class TranslationPath:
    """One way to run translation: a device, the llama.cpp build for it and the model it runs."""

    name: str
    device: str
    runtime: tuple[Archive, ...]
    model: TranslationModel
    # Picked by the shared llama.cpp flags in ``llama_server``; these add the device's own.
    flags: tuple[str, ...] = ()

    @property
    def runtime_name(self) -> str:
        return f"llama-{_LLAMA_BUILD}-{self.name}"

    @property
    def download_bytes(self) -> int:
        return self.model.size + sum(archive.size for archive in self.runtime)

    @property
    def display_size(self) -> str:
        return f"{self.download_bytes / 1e9:.1f} GB"


# Jais-3's template thinks at length unless told not to, and these were picked with thinking off.
_NO_THINKING = (("reasoning_effort", "low"),)
GPU_MODEL = TranslationModel(
    name="jais-3-3.7b",
    label="Jais-3 3.7B",
    description="Inception's Arabic model, run on the graphics card",
    repo_id="IFM/Jais-3-3.7B-GGUF",
    revision="c8c71420945d7a1ad5a93c8dfcbebeb1ea559578",
    file_name="Jais-3-3.7B-Q4_K_M.gguf",
    size=3_156_606_176,
    chat_options=_NO_THINKING,
)
SMALL_MODEL = TranslationModel(
    name="jais-3-0.9b",
    label="Jais-3 0.9B",
    description="Inception's small Arabic model, for the processor or an Intel NPU",
    repo_id="IFM/Jais-3-0.9B-GGUF",
    revision="7f015af7ba57fc481ea477ff0e106d4cad4eeabb",
    file_name="Jais-3-0.9B-Q4_K_M.gguf",
    size=666_192_384,
    chat_options=_NO_THINKING,
)

_CPU = Archive(
    _RELEASE + f"llama-{_LLAMA_BUILD}-bin-win-cpu-x64.zip", 24_731_650,
    "569990fcbacbe305aef41490f71df70088474a59d514ae3b1ee65c130ee74555",
)
# The HIP backend; hipBLAS and rocBLAS come from the ROCm libraries Skryb's AMD install already has.
_ROCM = Archive(
    _RELEASE + f"llama-{_LLAMA_BUILD}-bin-win-rocm-10.0-x64.zip", 267_077_158,
    "349eaeab6e61279b4498fa02ee485298eb1a7dd204b92bc4e45ce9ef179a8486",
)
# CUDA 13, like Skryb's own NVIDIA install: RTX 20 to RTX 50 series, driver 580 or newer.
_CUDA = Archive(
    _RELEASE + f"llama-{_LLAMA_BUILD}-bin-win-cuda-13.4-x64.zip", 170_311_229,
    "19809fbc335671be6e964c874a19ce879b64ef0e1f9431a23ce9654bf71f44ce",
)
_CUDA_RUNTIME = Archive(
    _RELEASE + "cudart-llama-bin-win-cuda-13.4-x64.zip", 423_535_356,
    "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
)
_OPENVINO = Archive(
    _RELEASE + f"llama-{_LLAMA_BUILD}-bin-win-openvino-2026.4.1-x64.zip", 104_725_103,
    "08e56548acfec91802942428f8a06b4b33a4608eab31b5934177419bea70c853",
)

# Every layer on the chosen device, or a load that fails and says why: ``--fit off`` stops llama.cpp
# from quietly moving layers that don't fit onto the processor.
_ALL_LAYERS = ("-ngl", "999", "--fit", "off")
AMD = TranslationPath("rocm", "GPU", (_ROCM,), GPU_MODEL, (*_ALL_LAYERS, "-fa", "on"))
NVIDIA = TranslationPath("cuda", "GPU", (_CUDA, _CUDA_RUNTIME), GPU_MODEL, (*_ALL_LAYERS, "-fa", "on"))
NPU = TranslationPath("openvino", "NPU", (_OPENVINO,), SMALL_MODEL, _ALL_LAYERS)
CPU = TranslationPath("cpu", "CPU", (_CPU,), SMALL_MODEL, ("-ngl", "0"))
PATHS = (AMD, NVIDIA, NPU, CPU)


def path_for(hardware: HardwareProfile) -> TranslationPath:
    """The GPU's build when there is one, else the tuned Intel NPU's, else the CPU's."""
    if hardware.kind == "amd_rocm":
        return AMD
    if hardware.kind == "nvidia_cuda":
        return NVIDIA
    return NPU if hardware.npu else CPU
