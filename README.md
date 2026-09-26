# VoiceToText

Local, offline speech-to-text dictation for Windows. Press a hotkey, speak,
and the transcription is typed straight into whatever window has focus. All
transcription runs on your own machine: no cloud, no accounts, no audio ever
leaving the box.

## Features

- **System-wide dictation.** Any number of toggle or hold shortcuts, on keys
  or mouse buttons 3-5 (default `Ctrl+Alt+M`); double-tap a hold shortcut to
  go hands-free. Text lands in the focused app; your clipboard is preserved.
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

Install from source on Windows using
[INSTALL.md](https://github.com/HuzPro/VoiceToText/blob/main/INSTALL.md).
It covers Python setup and the NVIDIA, AMD, and CPU dependency options.
No packaged installer has been published yet.

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

## Privacy

Audio and transcripts are not saved by default. Settings > Output > Privacy
lets you opt into saving them in the logs folder. Changes apply when you save.
Ordinary logs contain timing and status information, without transcript text.

The default folder is `%APPDATA%\VoiceToText\diagnostics`. Use a local folder
that is not synced or shared. Turning recording off keeps existing files;
delete those files separately if you no longer want them. Automatic cleanup
keeps 50 marked sessions. Older, unmarked folders remain for manual cleanup.

Edit capture is also off by default. It requires recording consent and the
separate `diagnostics.capture_edits=true` configuration setting before launch.
It saves edits and window titles for dictation, so only enable it deliberately.
Disabling recording consent stops further edit writes too.

## License

The application is licensed under
[Apache-2.0](https://github.com/HuzPro/VoiceToText/blob/main/LICENSE.txt).
Downloaded models have their own licenses. CrisperWhisper weights and outputs
have non-commercial restrictions and require acceptance before download or use.
Commercial or production use requires separate permission under its
[model license](https://github.com/HuzPro/VoiceToText/blob/main/voicetotext/licenses/crisperwhisper-nc.md).

## Sponsorship

The open-source release of VoiceToText was sponsored by Andrew
([BitSmith365](https://github.com/BitSmith365)).
