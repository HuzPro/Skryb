"""Fetches the parts of the CUDA toolkit the native engines build with: nvcc, the runtime, CCCL.

NVIDIA's per-component redistributable archives, checked against the sha256 in their manifest and merged
into one toolkit root (bin, include, lib, nvvm). No installer, no admin rights, no GPU needed.

    python native/cuda/fetch_toolkit.py [dest]    # dest defaults to native/cuda/build/toolkit
"""

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

VERSION = "12.9.1"
ROOT = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = ("cuda_nvcc", "cuda_cudart", "cuda_cccl")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def main() -> None:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "build" / "toolkit"
    if (dest / "bin" / "nvcc.exe").is_file():
        print(f"CUDA {VERSION} already at {dest}")
        return
    manifest = json.loads(fetch(f"{ROOT}redistrib_{VERSION}.json"))
    for name in COMPONENTS:
        entry = manifest[name]["windows-x86_64"]
        data = fetch(ROOT + entry["relative_path"])
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise SystemExit(f"{name}: sha256 mismatch")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                inner = Path(*Path(member.filename).parts[1:])
                if member.is_dir() or not inner.parts:
                    continue
                target = dest / inner
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
        print(f"{name} {manifest[name]['version']}")
    print(f"CUDA {VERSION} at {dest}")


if __name__ == "__main__":
    main()
