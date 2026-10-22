#!/usr/bin/env bash
# Builds qasr.dll and qasr-bench.exe for gfx1200 with the ROCm SDK's hipcc.
# HIPCC defaults to the rocm-sdk-core pip package in the global Python 3.12.
set -euo pipefail
cd "$(dirname "$0")"
HIPCC="${HIPCC:-$(python -c 'import _rocm_sdk_core, os; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "bin", "hipcc.exe"))')}"
mkdir -p build
FLAGS=(--offload-arch=gfx1200 -O3 -std=c++20 -Wno-unused-result)
"$HIPCC" "${FLAGS[@]}" -shared kernels.hip engine.hip -o build/qasr.dll
"$HIPCC" -O2 -std=c++20 -x c++ bench.cpp -x none build/qasr.lib -o build/qasr-bench.exe
echo "built build/qasr.dll build/qasr-bench.exe"
