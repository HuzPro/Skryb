"""Jais-ASR on an Intel NPU through OpenVINO, or on OpenVINO's CPU device, with no torch.

The same two graphs and the same stateful decoder as Qwen3-ASR's OpenVINO engine
(``openvino_qwen3_asr_graph`` builds both models); only the prompt and the
decision tag Jais-ASR opens its transcripts with differ.
"""

from pathlib import Path

from skryb.transcription.jais_asr_pack import prompt_text
from skryb.transcription.native_jais_asr_adapter import transcript_text
from skryb.transcription.openvino_qwen3_asr_adapter import OpenVinoQwen3AsrAdapter


class OpenVinoJaisAsrAdapter(OpenVinoQwen3AsrAdapter):
    """Batch and meeting Jais-ASR. Names to spell are not passed on: its system turn is its mode switch."""

    name = "Jais-ASR"

    def __init__(self, model_dir: Path, repo_id: str, revision: str = "", code_switch: bool = True) -> None:
        super().__init__(model_dir, repo_id, revision, language="ar")
        self._code_switch = code_switch

    def _prompt(self, context: str | None, language: str | None) -> tuple[str, str]:
        return prompt_text(self._code_switch)

    def _finish(self, text: str) -> str:
        return transcript_text(text)
