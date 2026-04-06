def detect_available_backends() -> list[str]:
    """Return available engine names in priority order.

    Priority:
        1. faster_whisper – CTranslate2 with CUDA (fastest on NVIDIA)
        2. transformers   – PyTorch GPU (works on AMD ROCm and NVIDIA)
        3. cpu            – faster_whisper on CPU (always available)
    """
    backends: list[str] = []

    # CTranslate2 CUDA (NVIDIA) — fastest option
    try:
        import ctranslate2

        ctranslate2.get_supported_compute_types("cuda")
        backends.append("faster_whisper")
    except Exception:
        pass

    # PyTorch GPU (ROCm or CUDA)
    try:
        import torch

        if torch.cuda.is_available():
            backends.append("transformers")
    except Exception:
        pass

    # CPU always available
    backends.append("cpu")
    return backends


def get_device_info() -> str:
    """Return human-readable GPU name for logging and UI."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU only"
