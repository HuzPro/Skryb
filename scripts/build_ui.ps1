# Builds the UI shell (Tauri + WebView2) in ui\ and prints where the exe landed.
# The app finds it there on its own; set SKRYB_UI_EXE to use another build.
# Beside it goes skryb_ui.dll, the same window as an installed Skryb.exe loads it.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ui = Resolve-Path (Join-Path $PSScriptRoot "..\ui")
Push-Location $ui
try {
    pnpm install --frozen-lockfile
    if ($LASTEXITCODE -ne 0) { throw "pnpm install failed with exit code $LASTEXITCODE." }
    pnpm tauri build
    if ($LASTEXITCODE -ne 0) { throw "pnpm tauri build failed with exit code $LASTEXITCODE." }
    # tauri build asks cargo for the exe only, which leaves the DLL among the intermediate files.
    cargo build --release --lib --features custom-protocol --manifest-path src-tauri\Cargo.toml
    if ($LASTEXITCODE -ne 0) { throw "cargo build of skryb_ui.dll failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}

$exe = Join-Path $ui "src-tauri\target\release\skryb-ui.exe"
$dll = Join-Path $ui "src-tauri\target\release\skryb_ui.dll"
foreach ($built in $exe, $dll) {
    if (-not (Test-Path $built)) { throw "The build finished but $built is missing." }
}
Write-Output $exe
