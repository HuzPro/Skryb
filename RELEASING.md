# Releasing Skryb

The release artifact is a Windows installer (`Skryb-Setup-<version>.exe`).
It does **not** bundle PyTorch: it ships a standalone CPython 3.12 plus the
app, detects the user's GPU at install time, and pip-installs the matching
runtime (NVIDIA CUDA / AMD ROCm / CPU) on their machine. That keeps the
published download small instead of a multi-GB per-target bundle.

## Pick the version

Versions are `MAJOR.MINOR.PATCH`, decided by the commit types since the last
release:

| Since the last release | Bump | Example |
|---|---|---|
| Only `fix:`, `perf:`, `docs:`, `test:`, `chore:` and the like | PATCH | 0.2.0 -> 0.2.1 |
| At least one `feat:`: something people can see or turn on | MINOR | 0.2.1 -> 0.3.0 |
| A breaking change, `feat!:` or a `BREAKING CHANGE:` line: settings that don't carry over, a dropped GPU or Windows version | MAJOR | 1.4.2 -> 2.0.0 |

Before 1.0.0 a breaking change bumps MINOR instead. 1.0.0 is cut by hand
once Skryb is stable.

The helper applies those rules (run `git fetch --tags` first):

```powershell
python -m release.next_version           # prints e.g. "0.3.0 (minor: 12 commits since v0.2.1)"
python -m release.next_version --apply   # also writes it into pyproject.toml
```

## Cut a release

Land a commit on `main` that sets the new version in `pyproject.toml` and
whose message starts with `release:` and that same version:

```
release: v0.2.0
```

The release gate fails if the two differ: the installed app reports
`pyproject.toml`'s version, so a mismatched tag would have its updater offer
the same update forever.

The [`Release` workflow](.github/workflows/release.yml) then runs the tests,
builds the installer, tags `v0.2.0`, and publishes a GitHub Release with the
`setup.exe`. Any other commit only runs the cheap gate job and stops, so
ordinary merges never release.

The leading `v` is optional, a trailing squash-merge `(#NN)` is ignored, and
the version is read from the first line only. `workflow_dispatch` (Actions
tab) is an escape hatch for re-runs; leave its version blank to use
`pyproject.toml`.

## Build the installer locally

Requires [Inno Setup 6](https://jrsoftware.org/isdl.php), a relocatable
CPython 3.12 (a [python-build-standalone](https://github.com/astral-sh/python-build-standalone)
`install_only` build), and the settings window built first. The build stops
if `ui\src-tauri\target\release\skryb-ui.exe` is missing, because it
ships inside the app wheel:

```powershell
.\scripts\build_ui.ps1
python build_installer.py --python-dir C:\path\to\standalone\python
python build_installer.py --python-dir ... --version 0.2.0 --iscc "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
```

Output: `dist/Skryb-Setup-<version>.exe`. The `--iscc` path is
auto-detected from the default install location when omitted.

## What the installer does on the user's machine

1. Detects the GPU (`Win32_VideoController`) and pre-selects a runtime on a
   radio page: NVIDIA (CUDA) / AMD (ROCm) / CPU.
2. `pip install`s the chosen `requirements-*.txt` plus the app wheel into the
   bundled Python (this is the multi-GB download).
3. Adds `skryb` to the user's PATH (runnable from any terminal) and
   creates Start-Menu / optional desktop shortcuts (windowless). It
   installs per user, in `%LOCALAPPDATA%\Programs\Skryb`, with no admin
   prompt.
4. Writes `THIRD-PARTY-NOTICES.txt` for the installed packages; uninstall
   removes everything, including the pip-installed files.

## Silent / unattended install

```
Skryb-Setup-<version>.exe /VERYSILENT /TARGET=cpu
```

`/TARGET=cpu|cuda|rocm` overrides hardware detection (handy for IT
deployment). Without it, a reinstall keeps the runtime chosen last time. The installer's pip runs `--isolated`, so a machine's global
`pip.ini` or `PIP_*` environment variables cannot break the dependency
install.

## In-app updates

An installed copy (one with `unins000.exe` next to its bundled Python) asks
`GET /repos/HuzPro/Skryb/releases/latest` 15 s after start and then hourly,
unless Settings > Updates turns that off. A newer `Skryb-Setup-<version>.exe`
is downloaded to `%LOCALAPPDATA%\Skryb\updates` and checked against the
SHA-256 GitHub records for the asset; only then is it offered, in a notice,
the tray menu and Settings > Updates. **Restart and update** waits for any
dictation to finish, then Skryb starts:

```
Skryb-Setup-<version>.exe /SILENT /NORESTART /FORCECLOSEAPPLICATIONS /WAITPID=<skryb pid> /LOG=<updates>\install-<version>.log
```

and quits. Setup waits up to a minute for that process to exit, force-closes
anything still holding its files after that (a stuck copy), installs over it
with the same runtime, and its `[Run]` entry starts Skryb again. If
the new copy doesn't start, the next launch of the old one says the update
didn't finish and names the log. A copy running from a git checkout keeps
updating through git instead.

The first release with the updater has to be installed by hand once.

## Licensing

Skryb is open source under the Apache License 2.0 (`LICENSE.txt`,
copyright statement in `NOTICE`). Python packages and downloaded model weights
have separate licenses. The application's license does not grant rights to
use a model or its outputs.

CrisperWhisper models and outputs have non-commercial restrictions. The app
requires acceptance of the bundled terms before downloading or loading these
models. Commercial or production use requires a separate license from the
model publisher. See `skryb/licenses/crisperwhisper-nc.md`.

Because PyTorch and the other Python packages are downloaded from their
official indices at install time, the installer does not redistribute them,
and model weights are downloaded separately. The installer ships:

- `LICENSE.txt` - the Apache License 2.0 covering Skryb itself.
- `NOTICE.txt` - covers the bundled CPython runtime (PSF license) and
  states the app's own license and copyright.
- `THIRD-PARTY-NOTICES.txt` - generated post-install from the actual
  installed environment by `release/notices.py`.
