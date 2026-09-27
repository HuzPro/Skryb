# Builds the UI shell (Tauri + WebView2) in ui\ and prints where the exe landed.
# The app finds it there on its own; set VOICETOTEXT_UI_EXE to use another build.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ui = Resolve-Path (Join-Path $PSScriptRoot "..\ui")
Push-Location $ui
try {
    pnpm install --frozen-lockfile
    if ($LASTEXITCODE -ne 0) { throw "pnpm install failed with exit code $LASTEXITCODE." }
    pnpm tauri build
    if ($LASTEXITCODE -ne 0) { throw "pnpm tauri build failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}

$exe = Join-Path $ui "src-tauri\target\release\voicetotext-ui.exe"
if (-not (Test-Path $exe)) { throw "The build finished but $exe is missing." }
Write-Output $exe
