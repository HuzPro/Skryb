<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/skryb-wordmark-dark.svg">
    <img alt="Skryb" src="brand/skryb-wordmark-light.svg" height="72">
  </picture>
</h1>

*Said like "scribe".* Local, offline speech-to-text dictation for Windows. Press a hotkey, speak,
and the transcription is typed straight into whatever window has focus. All
transcription runs on your own machine. There's no cloud and no account, and
your audio never leaves the computer.

## Download

Get `Skryb-Setup-<version>.exe` from the
[latest release](https://github.com/HuzPro/Skryb/releases/latest) and
run it. The installer detects your GPU, lets you pick NVIDIA (CUDA), AMD
(ROCm) or CPU, and downloads the matching PyTorch build during setup. On a GPU
machine that download is several GB. It installs for your Windows account
only, so it never asks for admin rights.

Skryb checks GitHub for a new version once an hour and downloads it in the
background. When one is ready, it asks you; click **Restart and update** and
it installs and opens again. Turn the checks off in Settings > Updates.

The installer isn't code-signed yet, so Windows SmartScreen may say it
protected your PC. Click **More info**, then **Run anyway**.

You need 64-bit Windows 11, or Windows 10 with the Microsoft Edge WebView2
Runtime, which most Windows 10 PCs already have. A GPU is recommended but not
required. On AMD, the ROCm runtime needs Adrenalin driver 26.2.2 or newer.

To run from source instead, follow
[INSTALL.md](https://github.com/HuzPro/Skryb/blob/main/INSTALL.md).

## Getting started

Launch Skryb from the Start Menu, or run `skryb` in a terminal.
The first run opens a setup window where you pick a model, download it and
set your hotkey. Then dictate anywhere. Press the hotkey, talk, press it
again, and the text is typed into the active window.

If something misbehaves, run `skryb --doctor` and read the report. It
covers Python, PyTorch, the GPU and the model cache in one go, and it's the
first thing to include in a bug report.

## Features

- **Dictate into any app.** Set up as many toggle or hold shortcuts as you
  like, on keys (punctuation and the Copilot key included) or mouse buttons 3
  to 5. The default is `Ctrl+Alt+M`. Double-tap a hold shortcut to go
  hands-free. The shortcut's last key never reaches the focused app, and your
  clipboard is preserved.
- **Local models.** Parakeet (`parakeet-tdt-0.6b-v3`) is the default on a GPU.
  `parakeet-ultra` is an optional download with fewer errors, and
  CrisperWhisper is there too, under a non-commercial license.
- **Live streaming mode (experimental).** Text appears while you speak, using
  Parakeet or the cache-aware `nemotron-speech-streaming-en-0.6b`. GPU only.
- **GPU first, CPU works too.** CUDA and ROCm, Windows ROCm included, are
  first-class. On a computer with no GPU the default is `phonon-2.5-english`, Parakeet
  v3 shrunk to 164 MB (English only). It runs on Fermion Research's CPU engine
  at roughly 15 to 30 times faster than real time. Underdog Bark 3.0
  (163 MB, English only) is more accurate at the same size, and on a laptop
  with an Intel NPU it runs on the NPU.
- **Learns your spellings.** Fix a word it got wrong after it pastes. Once two
  separate fixes agree, it uses your spelling from then on.
- **Small footprint.** A tray icon, plus a pill on screen while you dictate.
  Settings and History open in a separate 3.5 MB window only when you ask for
  them, and it closes completely when you close it.

## Privacy

Audio and transcripts aren't saved by default. The **Save history** switch on
the History page lets you opt in, and saved sessions go to the logs folder.
Ordinary logs hold timing and status information, never transcript text.

The default folder is `%APPDATA%\Skryb\diagnostics`. Use a local folder
that isn't synced or shared. Turning history off keeps the files already
saved, so delete them yourself if you no longer want them. Automatic cleanup
keeps the last 50 marked sessions. Older, unmarked folders stay until you
remove them.

Edit capture is also off by default. It needs history turned on and the
separate `diagnostics.capture_edits=true` setting in the config file before
launch. It saves your edits to dictated text along with window titles, so only
turn it on deliberately. Turning history off stops edit capture writing too.

## License

The application is licensed under
[Apache-2.0](https://github.com/HuzPro/Skryb/blob/main/LICENSE.txt).
Downloaded models have their own licenses. CrisperWhisper weights and outputs
are restricted to non-commercial use, and the app asks you to accept those
terms before downloading or using them. Commercial or production use needs
separate permission under its
[model license](https://github.com/HuzPro/Skryb/blob/main/skryb/licenses/crisperwhisper-nc.md).

## Sponsorship

The open-source release of Skryb was partly sponsored by
[BitSmith365](https://github.com/BitSmith365).
