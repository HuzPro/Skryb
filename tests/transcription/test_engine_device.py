from pathlib import Path

from skryb.transcription.nano_parakeet_adapter import NanoParakeetAdapter
from skryb.transcription.native_parakeet_adapter import NativeParakeetAdapter
from skryb.transcription.native_qwen3_asr_adapter import NativeQwen3AsrAdapter
from skryb.transcription.phonon2_cpu_adapter import Phonon2CpuAdapter
from skryb.transcription.qwen3_asr_adapter import Qwen3AsrAdapter


def test_every_batch_engine_says_whether_dictation_runs_on_the_gpu_or_the_cpu(tmp_path: Path) -> None:
    assert NanoParakeetAdapter("moondream/parakeet-ultra", "cuda", tmp_path).encoder_device == "GPU"
    assert NanoParakeetAdapter("moondream/parakeet-ultra", "cpu", tmp_path).encoder_device == "CPU"
    assert Qwen3AsrAdapter(tmp_path, "cuda", "en").encoder_device == "GPU"
    assert Qwen3AsrAdapter(tmp_path, "cpu", "ur").encoder_device == "CPU"
    assert NativeParakeetAdapter(tmp_path, "moondream/parakeet-ultra").encoder_device == "GPU"
    assert NativeQwen3AsrAdapter(tmp_path, cuda=True).encoder_device == "GPU"
    assert Phonon2CpuAdapter(tmp_path).encoder_device == "CPU"
