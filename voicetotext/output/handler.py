from abc import ABC, abstractmethod


# TODO: formatting layer.
#
# Add a TextOutputHandler decorator that applies configurable
# formatting rules to the transcription text before delegating to
# the inner handler — e.g. "always capitalise 'Anthropic'", "replace
# 'new line' with '\n'", brand-name casing, punctuation fixups, etc.
# Rules should be user-configurable (probably via settings) and
# applied to every single transcription regardless of which output
# mode is active.
#
# Slot it into bootstrap._build_output as the outermost decorator
# so it wraps both PasteAndClipboardAdapter and the clipboard-
# preserving path.  This is a natural GoF Decorator and sits cleanly
# at the output-layer seam.


class TextOutputHandler(ABC):

    @abstractmethod
    def output(self, text: str) -> None:
        pass
