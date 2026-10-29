#!/usr/bin/env bash
# Builds npk.dll for gfx1200 with the ROCm SDK's hipcc; shares the GEMM, norm and attention kernels of ../engine.
set -euo pipefail
cd "$(dirname "$0")"
HIPCC="${HIPCC:-$(python -c 'import _rocm_sdk_core, os; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "bin", "hipcc.exe"))')}"
mkdir -p build
"$HIPCC" --offload-arch=gfx1200 -O3 -std=c++20 -Wno-unused-result -shared ../engine/kernels.hip pk_kernels.hip engine.hip -o build/npk.dll
echo "built build/npk.dll"
