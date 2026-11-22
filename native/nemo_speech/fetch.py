"""Fetches NVIDIA's NeMo-Speech.cpp Windows CPU release: the DLLs the meeting diarizer loads on computers without a
supported GPU, and their licences.

The official release zip, checked against its pinned sha256. Only the C ABI DLL, what it links against and the
licence files are kept; the CLI, TTS and translation parts are left out.

    python native/nemo_speech/fetch.py [dest]    # dest defaults to native/nemo_speech/build
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

VERSION = "0.3.0"
URL = (f"https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v{VERSION}/"
       f"nemo-speech-{VERSION}-windows-x86_64-cpu.zip")
SHA256 = "4ed9740e361e4653f56c3dce7d1e81883a1c8b831c1bd7d01567fff1130b5b05"
# nemo_speech_asr_c.dll and everything it imports beyond Windows itself, read from the DLLs' import tables.
DLLS = (
    "nemo_speech_asr_c.dll", "nemo_speech_asr.dll", "nemo_speech_speechlm.dll", "llama.dll", "llama-common.dll",
    "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll",
    "vcomp140.dll",
)
LICENCES = "share/licenses/nemo-speech/"


def kept(name: str) -> PurePosixPath | None:
    """Where a zip member goes under dest, or None when it isn't needed."""
    inner = PurePosixPath(*PurePosixPath(name).parts[1:])
    if inner.parent == PurePosixPath("bin") and inner.name.lower() in DLLS:
        return PurePosixPath(inner.name)
    if str(inner).startswith(LICENCES) and not name.endswith("/"):
        return PurePosixPath("licenses") / inner.relative_to(LICENCES)
    return None


def main() -> None:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "build"
    if all((dest / dll).is_file() for dll in DLLS):
        print(f"NeMo-Speech.cpp {VERSION} already at {dest}")
        return
    with urllib.request.urlopen(URL) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit(f"{URL}: sha256 mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            target = kept(member.filename)
            if target is None:
                continue
            path = dest / target
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(member))
    missing = [dll for dll in DLLS if not (dest / dll).is_file()]
    if missing:
        raise SystemExit(f"NeMo-Speech.cpp {VERSION} has no {', '.join(missing)}")
    print(f"NeMo-Speech.cpp {VERSION} at {dest}")


if __name__ == "__main__":
    main()
