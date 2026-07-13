"""Release-engineering tooling for VoiceToText.

This package builds the distributable Windows binary, generates the
bundled third-party license notice, and decides, from a commit message,
whether a push should be cut as a GitHub release. It is deliberately
decoupled from the ``voicetotext`` application package: nothing here
imports the app or PyTorch, so the build logic stays fast to test and
free of GPU/runtime concerns.
"""

# The product name. Used for the PyInstaller bundle, the download
# archive, and the version resource. Single source of truth.
APP_NAME = "VoiceToText"
