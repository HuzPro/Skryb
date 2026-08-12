from enum import Enum


# Legacy names that must be rewritten to a current canonical value when
# a user config file from an earlier release is loaded. Kept minimal -
# only rename, do not delete or silently repurpose.
_LEGACY_RENAMES: dict[str, str] = {
    "auto_paste": "paste_and_clipboard",
}


_DISPLAY_LABELS: dict[str, str] = {
    "direct_paste": "Paste only (does not modify your clipboard)",
    "paste_and_clipboard": "Paste and copy to clipboard",
    "clipboard_only": "Clipboard only (no paste)",
}


class OutputMode(Enum):
    """How transcribed text is delivered to the focused window.

    Enum values are the strings that land in ``config.json``. Changing
    a value is a persisted-format break - add a rename to
    :data:`_LEGACY_RENAMES` and keep the value intact instead.
    """

    DIRECT_PASTE = "direct_paste"
    PASTE_AND_CLIPBOARD = "paste_and_clipboard"
    CLIPBOARD_ONLY = "clipboard_only"

    @classmethod
    def migrate(cls, raw: str | None) -> "OutputMode":
        """Map a persisted string to its canonical :class:`OutputMode`.

        Unknown values and ``None`` fall back to :attr:`DIRECT_PASTE`,
        the default for a fresh config. Legacy names listed in
        :data:`_LEGACY_RENAMES` are rewritten to their current spelling
        before lookup.
        """
        if raw is None:
            return cls.DIRECT_PASTE
        canonical = _LEGACY_RENAMES.get(raw, raw)
        try:
            return cls(canonical)
        except ValueError:
            return cls.DIRECT_PASTE

    @property
    def display_label(self) -> str:
        return _DISPLAY_LABELS[self.value]
