from enum import Enum


class PasteMethod(Enum):
    """How the text is put into the target app once it is on the clipboard."""

    CTRL_V = "ctrl_v"
    CTRL_SHIFT_V = "ctrl_shift_v"
    SHIFT_INSERT = "shift_insert"
    # Keystrokes instead of a paste, for apps that block pasting.
    TYPE = "type"

    @property
    def display_label(self) -> str:
        return _LABELS[self]


_LABELS = {
    PasteMethod.CTRL_V: "Ctrl+V",
    PasteMethod.CTRL_SHIFT_V: "Ctrl+Shift+V",
    PasteMethod.SHIFT_INSERT: "Shift+Insert",
    PasteMethod.TYPE: "Type the text",
}

_BY_VALUE = {method.value: method for method in PasteMethod}

# mintty (Git Bash, MSYS2, Cygwin) sends Ctrl+V to the shell as a raw ^V.
_AUTOMATIC = {"mintty.exe": PasteMethod.SHIFT_INSERT}

# SendInput takes a batch at a time; smaller batches let other input
# interleave instead of freezing the queue behind a long dictation.
TYPE_CHUNK_UNITS = 64


def normalized_process_name(name: str) -> str:
    """``"Mintty"`` and ``" mintty.EXE "`` both become ``"mintty.exe"``."""
    process = name.strip().lower()
    if process and not process.endswith(".exe"):
        process += ".exe"
    return process


def paste_method_overrides(configured: object) -> dict[str, PasteMethod]:
    """The valid entries of ``output.paste_method_overrides``; the rest is ignored."""
    if not isinstance(configured, dict):
        return {}
    overrides: dict[str, PasteMethod] = {}
    for process, method in configured.items():
        name = normalized_process_name(process) if isinstance(process, str) else ""
        if name and isinstance(method, str) and method in _BY_VALUE:
            overrides[name] = _BY_VALUE[method]
    return overrides


def paste_method_for(process: str, configured_overrides: object) -> PasteMethod:
    """The user's choice for *process*, else the automatic rule, else Ctrl+V."""
    name = normalized_process_name(process)
    overrides = paste_method_overrides(configured_overrides)
    return overrides.get(name) or _AUTOMATIC.get(name, PasteMethod.CTRL_V)


def utf16_chunks(text: str, max_units: int = TYPE_CHUNK_UNITS) -> list[list[int]]:
    """*text* as UTF-16 code units, in batches that never split a character.

    ``KEYEVENTF_UNICODE`` takes one code unit per key event, so a character
    outside the Basic Multilingual Plane (most emoji) is two events that
    must reach the target together.
    """
    chunks: list[list[int]] = []
    current: list[int] = []
    for character in text:
        encoded = character.encode("utf-16-le")
        units = [
            int.from_bytes(encoded[offset:offset + 2], "little")
            for offset in range(0, len(encoded), 2)
        ]
        if current and len(current) + len(units) > max_units:
            chunks.append(current)
            current = []
        current.extend(units)
    if current:
        chunks.append(current)
    return chunks
