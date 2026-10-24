# Native Qwen3-ASR engine (RX 9060 XT, gfx1200)

Hand-written HIP kernels behind `qasr.dll`, loaded by
`skryb/transcription/native_qwen3_asr_adapter.py` for the `qwen3-asr-1.7b` model. No torch.

Build (needs the ROCm 7.2.1 SDK wheels in the global Python 3.12, `rocm-sdk-devel` included):

```bash
bash native/engine/build.sh
```

Run HIP binaries from a short path: the HIP runtime crashes (0xC0000409) when its module path is long.

Benchmark against the PyTorch reference (clips stay outside the repo, in `D:/VoiceToText_bench`):

```bash
python native/bench/corpus.py
<venv with ROCm torch + transformers>/python native/bench/reference.py --model <Qwen3-ASR-1.7B-hf dir>
native/engine/build/qasr-bench.exe <snapshot>/.qasr/qwen3-asr.bin D:/VoiceToText_bench/corpus.jsonl D:/VoiceToText_bench/native.jsonl
python native/bench/compare.py
venv/Scripts/python.exe native/bench/app_engine_check.py
```

`QASR_DRAFT=0` turns off speculative decoding; `QASR_PROFILE=1` adds per-op prefill times to the stats.
