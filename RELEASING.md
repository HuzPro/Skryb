# Releasing VoiceToText

The release artifact is a Windows installer (`VoiceToText-Setup-<version>.exe`).
It does **not** bundle PyTorch: it ships a standalone CPython 3.12 plus the
app, detects the user's GPU at install time, and pip-installs the matching
runtime (NVIDIA CUDA / AMD ROCm / CPU) on their machine. That keeps the
published download small instead of a multi-GB per-target bundle.

## Cut a release

Land a commit on `main` whose message starts with `release:` and a version:

```
release: v0.2.0
```

The [`Release` workflow](.github/workflows/release.yml) then runs the tests,
builds the installer, tags `v0.2.0`, and publishes a GitHub Release with the
`setup.exe`. Any other commit only runs the cheap gate job and stops, so
ordinary merges never release.

The leading `v` is optional, a trailing squash-merge `(#NN)` is ignored, and
the version is read from the first line only. `workflow_dispatch` (Actions
tab) is an escape hatch for re-runs; leave its version blank to use
`pyproject.toml`.

## Build the installer locally

Requires [Inno Setup 6](https://jrsoftware.org/isdl.php) and a relocatable
CPython 3.12 (a [python-build-standalone](https://github.com/astral-sh/python-build-standalone)
`install_only` build):

```powershell
python build_installer.py --python-dir C:\path\to\standalone\python
python build_installer.py --python-dir ... --version 0.2.0 --iscc "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
```

Output: `dist/VoiceToText-Setup-<version>.exe`. The `--iscc` path is
auto-detected from the default install location when omitted.

## What the installer does on the user's machine

1. Detects the GPU (`Win32_VideoController`) and pre-selects a runtime on a
   radio page: NVIDIA (CUDA) / AMD (ROCm) / CPU.
2. `pip install`s the chosen `requirements-*.txt` plus the app wheel into the
   bundled Python (this is the multi-GB download).
3. Adds `voicetotext` to the system PATH (runnable from any terminal) and
   creates Start-Menu / optional desktop shortcuts (windowless).
4. Writes `THIRD-PARTY-NOTICES.txt` for the installed packages; uninstall
   removes everything, including the pip-installed files.

## Silent / unattended install

```
VoiceToText-Setup-<version>.exe /VERYSILENT /TARGET=cpu
```

`/TARGET=cpu|cuda|rocm` overrides hardware detection (handy for IT
deployment). The installer's pip runs `--isolated`, so a machine's global
`pip.ini` or `PIP_*` environment variables cannot break the dependency
install.

## Licensing

VoiceToText is open source under the Apache License 2.0 (`LICENSE.txt`,
copyright statement in `NOTICE`). Python packages and downloaded model weights
have separate licenses. The application's license does not grant rights to
use a model or its outputs.

CrisperWhisper models and outputs have non-commercial restrictions. The app
requires acceptance of the bundled terms before downloading or loading these
models. Commercial or production use requires a separate license from the
model publisher. See `voicetotext/licenses/crisperwhisper-nc.md`.

Because PyTorch and the other Python packages are downloaded from their
official indices at install time, the installer does not redistribute them,
and model weights are downloaded separately. The installer ships:

- `LICENSE.txt` - the Apache License 2.0 covering VoiceToText itself.
- `NOTICE.txt` - covers the bundled CPython runtime (PSF license) and
  states the app's own license and copyright.
- `THIRD-PARTY-NOTICES.txt` - generated post-install from the actual
  installed environment by `release/notices.py`.
