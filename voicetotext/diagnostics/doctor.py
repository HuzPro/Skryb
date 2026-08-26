"""Diagnostic probe for ``voicetotext --doctor``.

Renders a human-readable report describing the Python environment,
PyTorch state, optional engine backends, audio device defaults, and
HuggingFace cache path. Every probe is wrapped so a single broken
module degrades to a ``MISSING`` / ``ERROR`` line rather than crashing
the whole report - the friend can run this on a misbehaving machine
and share the output without needing extra diagnostic skill.
"""

import importlib
import os
import sys


_BACKEND_INSTALL_HINTS: dict[str, str] = {
    "nano_parakeet": "pip install nano-parakeet",
    "huggingface_hub": "pip install huggingface-hub",
    "sounddevice": "pip install sounddevice",
}


def run_doctor() -> str:
    sections: list[list[str]] = [
        ["=== VoiceToText diagnostic report ==="],
        _probe_python(),
        _probe_torch(),
        _probe_engine_backends(),
        _probe_audio(),
        _probe_hf_cache(),
    ]
    return "\n".join(_join_sections(sections))


def _join_sections(sections: list[list[str]]) -> list[str]:
    output: list[str] = []
    for section in sections:
        output.extend(section)
        output.append("")
    return output


def _probe_python() -> list[str]:
    return [
        "[Python]",
        f"  version: {sys.version.split()[0]}",
        f"  executable: {sys.executable}",
    ]


def _probe_torch() -> list[str]:
    try:
        return ["[torch]", *_read_torch_info()]
    except ImportError as exc:
        return ["[torch]", f"  MISSING: {exc}"]
    except Exception as exc:
        return ["[torch]", f"  ERROR: {exc}"]


def _read_torch_info() -> list[str]:
    import torch

    lines = [
        f"  version: {getattr(torch, '__version__', '(unknown)')}",
        f"  cuda available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"  device count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            lines.append(
                f"  device[{index}]: {torch.cuda.get_device_name(index)}"
            )
    return lines


def _probe_engine_backends() -> list[str]:
    lines = ["[Engine backends]"]
    for module_name in (
        "nano_parakeet",
        "huggingface_hub",
    ):
        lines.append(_probe_one_backend(module_name))
    return lines


def _probe_one_backend(module_name: str) -> str:
    try:
        version = _read_module_version(module_name)
    except ImportError:
        hint = _BACKEND_INSTALL_HINTS.get(
            module_name, f"pip install {module_name}",
        )
        return f"  {module_name}: MISSING ({hint})"
    except Exception as exc:
        return f"  {module_name}: ERROR ({exc})"
    return f"  {module_name}: {version}"


def _read_module_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "(no __version__)"))


def _probe_audio() -> list[str]:
    try:
        return ["[Audio]", *_read_audio_info()]
    except ImportError as exc:
        return ["[Audio]", f"  MISSING: {exc}"]
    except Exception as exc:
        return ["[Audio]", f"  ERROR: {exc}"]


def _read_audio_info() -> list[str]:
    import sounddevice as sd

    lines = [f"  sounddevice version: {getattr(sd, '__version__', '(unknown)')}"]
    default = getattr(sd, "default", None)
    if default is not None:
        device = getattr(default, "device", None)
        if device is not None:
            lines.append(f"  default device: {device}")
    return lines


def _probe_hf_cache() -> list[str]:
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    return [
        "[HuggingFace cache]",
        f"  HF_HUB_CACHE: {cache or '(not set)'}",
    ]
