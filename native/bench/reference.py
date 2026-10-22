"""Run Qwen3-ASR-1.7B through transformers on the corpus: the accuracy and timing baseline.

Needs a venv with ROCm torch and transformers >= 5.13 (MeetingTranscriber's works):

    <venv>/python native/bench/reference.py --model <dir> [--limit N]

Writes reference.jsonl next to corpus.jsonl: text, generated token ids and
seconds per clip, timed with perf_counter around the whole audio-to-text call.
"""
import argparse
import json
import os
import time
import wave
from pathlib import Path

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")  # exhaustive find hangs on RDNA 4 + Windows

import numpy as np
import torch


def _read(path: str) -> np.ndarray:
    with wave.open(path) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _sync() -> None:
    torch.cuda.synchronize()


class Reference:
    def __init__(self, model_dir: Path, attn: str) -> None:
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, attn_implementation=attn,
        ).to("cuda").eval()
        _sync()
        self.load_seconds = time.perf_counter() - started

    def transcribe(self, audio: np.ndarray) -> tuple[str, list[int], float]:
        started = time.perf_counter()
        inputs = self.processor.apply_transcription_request(
            audio=audio, language="en",
        ).to("cuda", torch.bfloat16)
        prompt_length = inputs["input_ids"].shape[1]
        max_new = int(len(audio) / 16000 * 8) + 32
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        tokens = out[0, prompt_length:].tolist()
        text = self.processor.decode(
            out[0, prompt_length:], return_format="transcription_only",
        ).strip()
        _sync()
        return text, tokens, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bench", type=Path, default=Path("D:/VoiceToText_bench"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--attn", default="sdpa", help="eager gives a second valid bf16 run: the noise floor")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    clips = [json.loads(l) for l in (args.bench / "corpus.jsonl").open(encoding="utf-8")]
    clips = clips[: args.limit] if args.limit else clips

    ref = Reference(args.model, args.attn)
    print(f"load {ref.load_seconds:.2f}s", flush=True)
    ref.transcribe(_read(clips[0]["audio"]))  # warm-up, not timed
    torch.cuda.reset_peak_memory_stats()

    out_path = args.bench / (args.out or ("reference.jsonl" if not args.limit else "reference-sample.jsonl"))
    with out_path.open("w", encoding="utf-8") as f:
        for i, clip in enumerate(clips):
            text, tokens, seconds = ref.transcribe(_read(clip["audio"]))
            f.write(json.dumps({
                "id": clip["id"], "seconds": seconds, "audio_seconds": clip["seconds"],
                "tokens": tokens, "text": text,
            }, ensure_ascii=False) + "\n")
            print(f"{i + 1}/{len(clips)} {clip['seconds']:6.1f}s audio {seconds:6.3f}s", flush=True)
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
