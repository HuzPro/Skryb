def detect_available_backends() -> list[str]:
    """Return available engine names.

    Always returns ``["transformers"]`` — the PyTorch-based engine supports
    NVIDIA (CUDA), AMD (ROCm), and CPU with a single model format.
    """
    return ["transformers"]


def get_device_info() -> str:
    """Return human-readable GPU name for logging and UI."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU only"
