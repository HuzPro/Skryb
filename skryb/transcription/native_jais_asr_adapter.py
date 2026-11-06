"""Jais-ASR on the native engine (qasr.dll), the same DLL as Qwen3-ASR, packed with its decoder in 4 bits.

Jais-ASR (IFM / MBZUAI) is the most accurate open model on Gulf Arabic. It
opens every transcript with a bracketed decision tag, "[SA]" or "[msa]", or
"[IFM-NoTranscript]" when it hears no speech; only the words after it are
kept.
"""

import logging
import re
import time
from pathlib import Path

from skryb.transcription.native_qwen3_asr_adapter import NativeQwen3AsrAdapter

_logger = logging.getLogger(__name__)

_PACK_FILENAME = "jais-asr.bin"
_LEADING_TAGS = re.compile(r"^(?:\s*\[([^\]\n]{0,40})\])+")


def without_decision_tag(text: str) -> str:
    """The transcript after Jais-ASR's leading tags; nothing when a tag says it heard no speech."""
    match = _LEADING_TAGS.match(text)
    if match is None:
        return text.strip()
    tags = re.findall(r"\[([^\]\n]{0,40})\]", match.group(0))
    if any(tag.strip().upper().startswith("IF") for tag in tags):
        return ""
    return text[match.end():].strip()


class NativeJaisAsrAdapter(NativeQwen3AsrAdapter):
    """Batch Jais-ASR through ``qasr.dll``.

    Names to spell are not passed on: Jais-ASR's system turn is its mode switch,
    and any other system text turns code-switching off.
    """

    def __init__(self, snapshot: Path, dll_path: Path | None = None, cuda: bool = False, code_switch: bool = True) -> None:
        super().__init__(snapshot, dll_path=dll_path, question_threshold=None, cuda=cuda)
        self._code_switch = code_switch

    def _packed_path(self) -> Path:
        return self._snapshot / ".qasr" / _PACK_FILENAME

    def _pack_snapshot(self, packed: Path) -> None:
        from skryb.transcription.jais_asr_pack import pack

        _logger.info("Packing Jais-ASR for the native engine (once per download): %s", packed)
        started_at = time.perf_counter()
        pack(self._snapshot, packed)
        _logger.info("Packed in %.1fs", time.perf_counter() - started_at)

    def _prompt(self, context: str | None, language: str | None) -> tuple[str, str]:
        from skryb.transcription.jais_asr_pack import prompt_text

        return prompt_text(self._code_switch)

    def _finish(self, text: str, language: str | None) -> str:
        return without_decision_tag(text)
