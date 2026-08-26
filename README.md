# VoiceToText

Local, offline speech-to-text dictation for Windows. Press a hotkey, speak,
and the transcription is typed straight into whatever window has focus. All
transcription runs on your own machine: no cloud, no accounts, no audio ever
leaving the box.

## Features

- **System-wide dictation.** Toggle or push-to-talk hotkey (default
  `Ctrl+Alt+M`), text lands in the focused app; your clipboard is preserved.
- **Local models.** Parakeet provides accurate batch transcription and
  Nemotron provides low-latency streaming.
- **Live streaming mode (experimental).** Text appears while you speak,
  using `parakeet-tdt-0.6b-v3` or the cache-aware
  `nemotron-speech-streaming-en-0.6b`. GPU only.
- **GPU-focused.** CUDA and ROCm (including Windows ROCm) are first-class.
  A GPU is expected; Parakeet technically runs on CPU but is about 15x slower
  than real time.
- **Native and lightweight UI.** Win32 tray icon plus a Tkinter settings
  window with OS-aware dark mode. No browser, no Electron, no Qt.
- **Self-diagnosis.** `python -m voicetotext --doctor` reports Python,
  PyTorch, GPU, and model-cache state in one shot.

## Install

Grab `VoiceToText-Setup-<version>.exe` from
[Releases](https://github.com/HuzPro/VoiceToText/releases): it detects your
GPU and installs the matching PyTorch runtime for you.

To run from source (Windows, and what to pick for NVIDIA / AMD / CPU), see
[INSTALL.md](INSTALL.md).

## Quickstart

```powershell
python -m voicetotext
```

For a source checkout installed with `pip install -e .`, create a console-free
Start Menu entry with `voicetotext --create-shortcut`. Optional development
autostart is managed with `voicetotext --enable-autostart` and
`voicetotext --disable-autostart`. The dev Run entry is intentionally not
removed by the packaged installer or its uninstaller.

First run opens a setup window: pick a model, download it, set your hotkey.
Then dictate anywhere: hit the hotkey, talk, hit it again, and the text is
typed into the active window. If anything misbehaves, run
`python -m voicetotext --doctor` and read the report.

## License

[Apache-2.0](LICENSE.txt).

## Sponsorship

The open-source release of VoiceToText was sponsored by Andrew
([BitSmith365](https://github.com/BitSmith365)).
