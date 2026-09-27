"""The UI contract's JSON shapes, to and from the app's own types.

Anything read from the shell is checked whole: a missing field, a wrong
type or an unknown choice raises ``ValueError`` with a sentence naming it,
which the server answers with a 400.
"""

import math
from collections.abc import Iterable
from dataclasses import asdict, replace
from pathlib import Path
from typing import TypeVar, cast

from voicetotext.history.entries import HistoryEntry
from voicetotext.hotkeys.bindings import Binding, BindingAction, BindingMode, binding_for
from voicetotext.output.dictionary_config import (
    DictionaryConfig,
    DictionaryRule,
    LiteralRule,
    dictionary_from_json,
)
from voicetotext.output.output_mode import OutputMode
from voicetotext.output.paste_method import PasteMethod, normalized_process_name
from voicetotext.recording.preferences import DUCK_MODES
from voicetotext.settings.service import CRISPERWHISPER_MODES
from voicetotext.settings.settings_draft import SettingsDraft
from voicetotext.settings.transcription_mode import TranscriptionMode
from voicetotext.writing.install import INSTALL_BYTES
from voicetotext.writing.llama_server import last_line
from voicetotext.writing.service import WritingStatus
from voicetotext.writing.settings import CLEANUP_LEVELS, clamp_deadline

Json = dict[str, object]
_T = TypeVar("_T")

_WRITING_STATES = {"downloading": "installing", "install_failed": "failed"}


def shown_combination(combo: str) -> str:
    """``"ctrl+shift+m"`` as the key caps read: ``"Ctrl + Shift + M"``."""
    return " + ".join(
        part.upper() if len(part) == 1 else part[:1].upper() + part[1:]
        for part in combo.split("+")
        if part
    )


def entry_id(entry: HistoryEntry) -> str:
    """Session folder and log line, which stay put when other entries are deleted."""
    return f"{entry.key[0]}.{entry.key[1]}"


def has_audio(entry: HistoryEntry) -> bool:
    return entry.audio_path is not None and entry.audio_path.is_file()


def history_entry_json(entry: HistoryEntry) -> Json:
    return {
        "id": entry_id(entry),
        "at": entry.at.astimezone().isoformat(timespec="seconds"),
        "text": entry.text,
        "audio_seconds": entry.audio_seconds,
        "status": entry.status,
        "error": entry.error,
        "has_audio": has_audio(entry),
        "kind": entry.kind,
    }


def binding_json(binding: Binding) -> Json:
    return {"keys": binding.combo, "mode": binding.mode.value, "action": binding.action.value}


def settings_json(draft: SettingsDraft) -> Json:
    return {
        "model_size": draft.model_size,
        "language": draft.language,
        "device": draft.device,
        "engine": draft.engine,
        "output_mode": draft.output_mode.value,
        "transcription_mode": draft.transcription_mode.value,
        "crisperwhisper_mode": draft.crisperwhisper_mode,
        "capture_content": draft.capture_content,
        "input_device": draft.input_device,
        "boost_quiet_speech": draft.boost_quiet_speech,
        "writing": asdict(draft.writing),
        "hotkey_bindings": [binding_json(binding) for binding in draft.hotkey_bindings],
        "double_tap_locks_hold": draft.double_tap_locks_hold,
        "recording": {
            **asdict(draft.recording),
            "preferred_devices": list(draft.recording.preferred_devices),
        },
        "history_retention_days": draft.history_retention_days,
        "paste_method_overrides": [list(pair) for pair in draft.paste_method_overrides],
    }


def draft_from_json(document: object, saved: SettingsDraft) -> SettingsDraft:
    """The settings the shell sent, over *saved*.

    Starting from what is saved keeps a setting this contract does not
    carry yet as it is, rather than resetting it to its default.
    """
    settings = Fields(document, "settings")
    writing = settings.section("writing")
    recording = settings.section("recording")
    retention = settings.whole("history_retention_days")
    if retention < 0:
        raise ValueError("settings.history_retention_days cannot be negative.")
    return replace(
        saved,
        model_size=settings.text("model_size"),
        language=settings.text("language"),
        device=settings.text("device"),
        engine=settings.text("engine"),
        output_mode=OutputMode(settings.one_of("output_mode", (m.value for m in OutputMode))),
        transcription_mode=TranscriptionMode(
            settings.one_of("transcription_mode", (m.value for m in TranscriptionMode)),
        ),
        crisperwhisper_mode=settings.one_of("crisperwhisper_mode", CRISPERWHISPER_MODES),
        capture_content=settings.flag("capture_content"),
        input_device=settings.text("input_device"),
        boost_quiet_speech=settings.flag("boost_quiet_speech"),
        writing=replace(
            saved.writing,
            cleanup=writing.one_of("cleanup", CLEANUP_LEVELS),
            lists=writing.flag("lists"),
            deadline_seconds=clamp_deadline(writing.number("deadline_seconds")),
            context=writing.flag("context"),
            learn_words=writing.flag("learn_words"),
            learn_style=writing.flag("learn_style"),
            style_notes=writing.text("style_notes").strip(),
            after_dictation_transform=writing.text("after_dictation_transform"),
            runtime_path=writing.text("runtime_path").strip(),
            model_path=writing.text("model_path").strip(),
        ),
        hotkey_bindings=bindings_from_json(settings.items("hotkey_bindings")),
        double_tap_locks_hold=settings.flag("double_tap_locks_hold"),
        recording=replace(
            saved.recording,
            warn_when_silent=recording.flag("warn_when_silent"),
            preferred_devices=tuple(dict.fromkeys(
                name.strip() for name in recording.texts("preferred_devices") if name.strip()
            )),
            cues_enabled=recording.flag("cues_enabled"),
            cue_volume=_clamped(recording.number("cue_volume"), 0.0, 1.0),
            duck_other_audio=recording.one_of("duck_other_audio", DUCK_MODES),
            duck_level=_clamped(recording.number("duck_level"), 0.0, 1.0),
            max_duration_seconds=max(recording.number("max_duration_seconds"), 1.0),
            warn_before_cap_seconds=max(recording.number("warn_before_cap_seconds"), 0.0),
        ),
        history_retention_days=retention,
        paste_method_overrides=_overrides_from_json(settings.items("paste_method_overrides")),
    )


def bindings_from_json(rows: list[object]) -> tuple[Binding, ...]:
    """The shortcuts sent, leaving out any with no keys: that turns it off."""
    bindings: list[Binding] = []
    for index, row in enumerate(rows):
        fields = Fields(row, f"hotkey_bindings[{index}]")
        keys = _typed_combination(fields.text("keys"))
        action = BindingAction(fields.one_of("action", (a.value for a in BindingAction)))
        mode = fields.one_of("mode", (m.value for m in BindingMode))
        if keys:
            bindings.append(binding_for(keys, mode, action))
    return tuple(bindings)


def dictionary_json(dictionary: DictionaryConfig) -> Json:
    return {
        "options": asdict(dictionary.options),
        "rules": [_rule_json(rule) for rule in dictionary.rules],
    }


def dictionary_from_contract(document: object) -> DictionaryConfig:
    """The contract names a rule's type ``kind``; the saved file calls it ``type``."""
    fields = Fields(document, "dictionary")
    rules: list[object] = []
    for index, rule in enumerate(fields.items("rules")):
        raw = Fields(rule, f"rules[{index}]").raw()
        rules.append({**raw, "type": raw.get("kind")})
    return dictionary_from_json({"options": fields.section("options").raw(), "rules": rules})


def writing_status_json(
    status: WritingStatus, *, runtime: Path, model: Path, learned_style: str,
) -> Json:
    return {
        "state": _WRITING_STATES.get(status.state, status.state),
        "detail": last_line(status.detail) if status.state == "failed" else status.detail,
        "bytes_done": status.bytes_done,
        "bytes_total": status.bytes_total,
        "install_bytes": INSTALL_BYTES,
        "runtime_path": str(runtime),
        "model_path": str(model),
        "last_cleanup_seconds": status.last_cleanup_seconds,
        "learned_style_summary": learned_style,
    }


def learned_pairs(dictionary: DictionaryConfig) -> set[tuple[str, str]]:
    """Each learned word as the learner blocks it: first spoken form, written form."""
    return {
        (rule.spoken[0].strip(), rule.replacement)
        for rule in dictionary.rules
        if isinstance(rule, LiteralRule) and rule.learned
    }


class Fields:
    """One JSON object from the shell, read field by field with its type checked."""

    def __init__(self, document: object, name: str) -> None:
        if not isinstance(document, dict):
            raise ValueError(f"{name} must be an object.")
        self._document = cast(dict[str, object], document)
        self._name = name

    def raw(self) -> dict[str, object]:
        return dict(self._document)

    def section(self, key: str) -> "Fields":
        return Fields(self._get(key), f"{self._name}.{key}")

    def text(self, key: str) -> str:
        return self._typed(key, str, "text")

    def flag(self, key: str) -> bool:
        return self._typed(key, bool, "true or false")

    def items(self, key: str) -> list[object]:
        return cast(list[object], self._typed(key, list, "a list"))

    def texts(self, key: str) -> list[str]:
        values = self.items(key)
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"{self._name}.{key} must be a list of text.")
        return cast(list[str], values)

    def number(self, key: str) -> float:
        value = self._get(key)
        number = None if isinstance(value, bool) else value
        if not isinstance(number, int | float) or not math.isfinite(number):
            raise ValueError(f"{self._name}.{key} must be a number.")
        return float(number)

    def whole(self, key: str) -> int:
        value = self._get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{self._name}.{key} must be a whole number.")
        return value

    def one_of(self, key: str, choices: Iterable[str]) -> str:
        allowed = tuple(choices)
        value = self.text(key)
        if value not in allowed:
            raise ValueError(f"{self._name}.{key} must be one of {', '.join(allowed)}.")
        return value

    def _typed(self, key: str, kind: type[_T], described: str) -> _T:
        value = self._get(key)
        if not isinstance(value, kind) or (kind is not bool and isinstance(value, bool)):
            raise ValueError(f"{self._name}.{key} must be {described}.")
        return value

    def _get(self, key: str) -> object:
        if key not in self._document:
            raise ValueError(f"{self._name}.{key} is missing.")
        return self._document[key]


def _rule_json(rule: DictionaryRule) -> Json:
    if isinstance(rule, LiteralRule):
        return {
            "kind": "literal",
            "spoken": list(rule.spoken),
            "replacement": rule.replacement,
            "case_sensitive": rule.case_sensitive,
            "whole_words": rule.whole_words,
            "phonetic": rule.phonetic,
            "enabled": rule.enabled,
            "learned": rule.learned,
        }
    return {
        "kind": "regex",
        "pattern": rule.pattern,
        "replacement": rule.replacement,
        "enabled": rule.enabled,
    }


def _typed_combination(text: str) -> str:
    """What the user typed, in the form the config stores: ``"ctrl+m"``."""
    if not text.strip():
        return ""
    return "+".join(part.strip().lower() for part in text.strip().split("+"))


def _overrides_from_json(rows: list[object]) -> tuple[tuple[str, str], ...]:
    methods = {method.value for method in PasteMethod}
    overrides: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not (
            isinstance(row, list) and len(row) == 2
            and all(isinstance(part, str) for part in row)
            and row[1] in methods
        ):
            raise ValueError(
                f"paste_method_overrides[{index}] must be [process, method] with a known method.",
            )
        process = normalized_process_name(row[0])
        if process:
            overrides[process] = row[1]
    return tuple(sorted(overrides.items()))


def _clamped(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
