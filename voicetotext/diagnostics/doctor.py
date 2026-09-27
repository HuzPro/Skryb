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
import tempfile
import time
from pathlib import Path

from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.core.paths import AppPaths
from voicetotext.writing.client import post_completion
from voicetotext.writing.install import WRITING_MODEL, WritingPaths, managed_root, server_files
from voicetotext.writing.llama_server import (
    LlamaServer,
    ServerFiles,
    ServerState,
    last_line,
)
from voicetotext.writing.prompt import Hints, cleanup_system_prompt, cleanup_user_message
from voicetotext.writing.settings import WritingSettings, read_writing_settings

_CTRANSLATE2_INSTALL_HINT = "see INSTALL.md - the wheel differs per GPU vendor"

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
        _probe_ctranslate2(),
        _probe_engine_backends(),
        _probe_audio(),
        _probe_hf_cache(),
        _probe_diagnostics_path(),
        _probe_writing_model(),
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


def _probe_ctranslate2() -> list[str]:
    """Report which CTranslate2 build is installed, not just its version.

    The PyPI wheel and the ROCm wheel share a version number, so the
    device count is the only thing that tells them apart - and a wheel
    built for the wrong runtime reports zero devices instead of failing
    to import.
    """
    try:
        return ["[ctranslate2]", *_read_ctranslate2_info()]
    except ImportError:
        return ["[ctranslate2]", f"  MISSING ({_CTRANSLATE2_INSTALL_HINT})"]
    except Exception as exc:
        return ["[ctranslate2]", f"  ERROR: {exc}"]


def _read_ctranslate2_info() -> list[str]:
    import ctranslate2

    device_count = ctranslate2.get_cuda_device_count()
    lines = [
        f"  version: {getattr(ctranslate2, '__version__', '(unknown)')}",
        f"  gpu devices: {device_count}",
    ]
    if device_count:
        compute_types = sorted(ctranslate2.get_supported_compute_types("cuda"))
        lines.append(f"  gpu compute types: {', '.join(compute_types)}")
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


def _probe_diagnostics_path() -> list[str]:
    if os.environ.get("APPDATA") is None:
        return ["[Session diagnostics]", "  directory: (APPDATA not set)"]
    paths = AppPaths()
    paths.load_configured_diagnostics_root()
    return [
        "[Session diagnostics]",
        f"  directory: {paths.diagnostics_dir}",
    ]


_WRITING_READY_SECONDS = 90.0
_WRITING_SAMPLE = "um so let's meet on friday actually no monday at four"


def _probe_writing_model() -> list[str]:
    try:
        return ["[Writing model]", *_read_writing_model_info()]
    except Exception as exc:
        return ["[Writing model]", f"  ERROR: {exc}"]


def _read_writing_model_info() -> list[str]:
    config = JsonConfigAdapter(AppPaths().config_path)
    config.load()
    return writing_model_report(
        read_writing_settings(config),
        WritingPaths(managed_root()),
        Path(tempfile.gettempdir()) / "voicetotext-doctor-writing-server.log",
    )


def writing_model_report(
    settings: WritingSettings, paths: WritingPaths, log_path: Path,
) -> list[str]:
    """Install state, paths, and one timed cleanup if the model is installed."""
    lines = [
        f"  cleanup: {settings.cleanup}",
        f"  runtime: {settings.runtime_path or paths.runtime_exe}",
        f"  pinned model: {WRITING_MODEL.id} ({WRITING_MODEL.gguf.file_name})",
        f"  model: {settings.model_path or paths.model_file}",
    ]
    files = server_files(paths, settings)
    if files is None:
        return [*lines, "  installed: no (Settings > Writing > Install)"]
    return [*lines, "  installed: yes", *_writing_round_trip(files, settings, log_path)]


def _writing_round_trip(
    files: ServerFiles, settings: WritingSettings, log_path: Path,
) -> list[str]:
    server = LlamaServer(lambda: files, log_path, warm_up=lambda _endpoint: None)
    server.start()
    try:
        return _time_one_cleanup(server, settings, log_path)
    finally:
        server.stop()


def _time_one_cleanup(
    server: LlamaServer, settings: WritingSettings, log_path: Path,
) -> list[str]:
    started = time.monotonic()
    while time.monotonic() - started < _WRITING_READY_SECONDS:
        if server.endpoint() is not None or server.status.state in (
            ServerState.FAILED, ServerState.NOT_INSTALLED,
        ):
            break
        time.sleep(0.2)
    endpoint = server.endpoint()
    if endpoint is None:
        reason = last_line(server.status.detail) or server.status.state.value
        return [f"  server: did not start ({reason}); log: {log_path}"]
    loaded = time.monotonic() - started
    asked = time.monotonic()
    answer = post_completion(
        endpoint,
        cleanup_system_prompt(settings),
        cleanup_user_message(_WRITING_SAMPLE, Hints()),
        64,
        30.0,
    )
    return [
        f"  server start: {loaded:.1f} s",
        f"  round trip: {time.monotonic() - asked:.2f} s -> {answer!r}",
    ]
