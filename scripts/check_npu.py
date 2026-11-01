"""Prove the NPU engine runs on the NPU, and race it against the Intel GPU and the CPU.

    python scripts\\check_npu.py [--model phonon-2.5-english] [--models DIR] [clip.wav | folder ...]

Uses the app's models folder (or --models) and the engine Skryb runs, so the
model must already be downloaded. Clips are 16 kHz mono 16-bit wavs, like the
ones the diagnostics folder keeps. Without any, it speaks a few passages with
Windows' own voice, and their words become the reference transcript.

It reports:
- how long the engine takes to start taking dictation, and to bring every length up;
- where OpenVINO says each compiled length runs;
- this process's NPU busy time from Windows' "GPU Engine" counters, the ones
  Task Manager draws, which only move when the work really reaches the NPU;
- per device on the same clips: latency, this process's CPU time, NPU and GPU
  busy time, battery drain when unplugged, and word error rate.

Windows' counters run in this process's view, so run it outside other work
on the NPU (camera effects, other AI apps) for clean numbers.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from skryb.core.paths import AppPaths
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription import openvino_parakeet_adapter as npu_engine
from skryb.transcription.hf_model_manager import loadable_snapshot_dir
from skryb.transcription.model_info import get_model_info
from skryb.transcription.parakeet_numpy import bundled_tokenizer

_RATE = 16_000
# Dictation-like passages, about 3 to 45 seconds spoken.
_PASSAGES = (
    "Remind me to call the bank tomorrow morning.",
    "Hi Sarah, thanks for sending the draft over. I had a quick look and the numbers in the second table "
    "don't match the summary, so could you check them before Friday?",
    "The meeting moved to half past three on Thursday. Please bring the printed agenda, the budget sheet "
    "and the notes from last week, and let me know if the room on the fourth floor is still free.",
    "I tried the new build on the laptop this afternoon. It starts much faster than before, but the settings "
    "window still forgets its size when I close it, and the tray icon stays grey for a few seconds after "
    "the model is ready. Everything else looked fine.",
    "For the report, start with a short paragraph on what changed this quarter, then list the three biggest "
    "risks with one sentence each. Keep the charts at the end, and make sure every figure has a source and a "
    "date under it. If something is an estimate, say so plainly instead of rounding it to look exact. Send "
    "me the first version by Wednesday evening so there is time to fix it before the review.",
    "Here is the plan for the trip. We leave on Saturday at seven, stop for breakfast near the lake, and "
    "should reach the cabin before lunch. Each of us brings our own sleeping bag and two days of food. The "
    "cabin has a stove but no fridge, so nothing that needs to stay cold. On Sunday we walk the ridge path if "
    "the weather holds, and if it rains we take the shorter trail through the forest instead. We drive back "
    "on Monday after breakfast and should be home by mid afternoon. Tell me by Thursday if you can't come, "
    "so I can cancel the second room and get the deposit back.",
)


def main() -> None:
    args = _arguments()
    models_dir = Path(args.models) if args.models else AppPaths().models_dir
    repo_id = get_model_info(args.model).repo_id
    engine = _start(models_dir, repo_id)
    openvino = npu_engine.import_openvino()
    _describe(openvino)
    clips = _read_clips(args.clips) if args.clips else _spoken_passages()
    print(f"\n{len(clips)} clips, {sum(len(c[1]) for c in clips) / _RATE:.0f} s of audio")
    _prove(engine, clips)
    engine.dispose()
    results = [_race(openvino, models_dir, repo_id, device, clips) for device in args.devices]
    _summarise([result for result in results if result is not None])


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clips", nargs="*", help="16 kHz mono wavs, or folders of them")
    parser.add_argument("--model", default="phonon-2.5-english")
    parser.add_argument("--models", help="models folder (default: the app's)")
    parser.add_argument("--devices", nargs="+", default=["NPU", "GPU", "CPU"])
    return parser.parse_args()


def _start(models_dir: Path, repo_id: str) -> npu_engine.OpenVinoParakeetAdapter:
    """Load the engine the way Skryb does, first thing, so OpenVINO's import counts too."""
    engine = npu_engine.OpenVinoParakeetAdapter(models_dir, repo_id)
    started = time.monotonic()
    engine.load_model()
    print(f"Taking dictation after {time.monotonic() - started:.2f}s on {engine.encoder_device}")
    if engine.device_note:
        print(f"  note: {engine.device_note}")
    deadline = started + 15 * 60
    while not _settled(engine) and time.monotonic() < deadline:
        time.sleep(0.05)
    print(f"{engine.lengths_up} lengths up after {time.monotonic() - started:.2f}s on {engine.encoder_device}")
    return engine


def _settled(engine: npu_engine.OpenVinoParakeetAdapter) -> bool:
    """Every length up on the NPU, or the engine settled on the CPU for good."""
    if engine.encoder_device == "NPU":
        return engine.lengths_up == len(npu_engine.BUCKET_FRAMES)
    note = engine.device_note or ""
    return engine.lengths_up > 0 or not note.startswith("Preparing")


def _describe(openvino: object) -> None:
    core = openvino.Core()
    print(f"OpenVINO {openvino.__version__}, devices: {core.available_devices}")
    for device in ("NPU", "GPU"):
        if device in core.available_devices:
            print(f"{device}: {core.get_property(device, 'FULL_DEVICE_NAME')}")
    if "NPU" in core.available_devices:
        print(f"NPU driver {core.get_property('NPU', 'NPU_DRIVER_VERSION')}")


def _prove(engine: npu_engine.OpenVinoParakeetAdapter, clips: list) -> None:
    """Transcribe every clip through the engine and read the NPU's busy time off Windows' own counter."""
    before, cpu_before, started = _busy_seconds(), time.process_time(), time.monotonic()
    for _, samples, _ in clips:
        engine.transcribe(AudioSegment(data=samples, sample_rate=_RATE))
    wall, cpu = time.monotonic() - started, time.process_time() - cpu_before
    busy = _difference(_busy_seconds(), before)
    print(f"\nThrough the engine: {wall:.2f}s wall, {cpu:.2f}s of CPU, "
          f"NPU busy {busy['npu']:.2f}s, GPU busy {busy['gpu']:.2f}s")
    if engine.encoder_device == "NPU" and busy["npu"] > 0:
        print("PROOF: Windows saw this process keep the NPU busy; the encoder ran there.")
    else:
        print(f"NOT ON THE NPU: the engine reports {engine.encoder_device} and Windows saw "
              f"{busy['npu']:.2f}s of NPU work.")


def _race(openvino: object, models_dir: Path, repo_id: str, device: str, clips: list) -> dict | None:
    core = openvino.Core()
    if device not in core.available_devices:
        print(f"\n{device}: not on this computer")
        return None
    ir_path = loadable_snapshot_dir(models_dir, repo_id) / ".openvino" / "encoder-int8.xml"
    model = core.read_model(str(ir_path))
    started = time.monotonic()
    lengths = npu_engine.BUCKET_FRAMES if device != "CPU" else (0,)
    # The NPU reuses Skryb's own compiled lengths; the others compile into a folder that goes away after.
    with tempfile.TemporaryDirectory(prefix=f"skryb-{device.lower()}-", ignore_cleanup_errors=True) as scratch:
        cache = ir_path.parent / npu_engine.CACHE_DIRECTORY_NAME if device == "NPU" else Path(scratch)
        compiled = {frames: npu_engine.compile_length(core, model, cache, device, frames) for frames in lengths}
    where = sorted({npu_engine.CompiledEncoder({0: c}).device for c in compiled.values()})
    print(f"\n{device}: {len(compiled)} lengths up in {time.monotonic() - started:.1f}s, "
          f"OpenVINO says they run on {where}")
    decoder = npu_engine._load_decoder(openvino, ir_path.parent / "decoder.npz")
    pipeline = npu_engine._NpuParakeet(npu_engine.CompiledEncoder(compiled), decoder, bundled_tokenizer())
    for _, samples, _ in clips[:2]:
        npu_engine._text_of(pipeline, AudioSegment(data=samples, sample_rate=_RATE))
    drain = [_battery_drain_mw()]
    before, cpu_before = _busy_seconds(), time.process_time()
    latencies, errors, words = [], 0, 0
    for _, samples, reference in clips:
        started = time.monotonic()
        text = npu_engine._text_of(pipeline, AudioSegment(data=samples, sample_rate=_RATE))
        latencies.append(time.monotonic() - started)
        if reference:
            errors += _word_errors(reference, text)
            words += len(_words(reference))
    drain.append(_battery_drain_mw())
    result = {
        "device": device,
        "median_ms": statistics.median(latencies) * 1000,
        "total_s": sum(latencies),
        "cpu_s": time.process_time() - cpu_before,
        **_difference(_busy_seconds(), before),
        "wer": errors / words if words else None,
        "drain_mw": None if None in drain else sum(drain) / len(drain),
    }
    print(f"  {result['total_s']:.2f}s for all clips, median {result['median_ms']:.0f} ms, "
          f"{result['cpu_s']:.2f}s of CPU, NPU busy {result['npu']:.2f}s, GPU busy {result['gpu']:.2f}s")
    return result


def _summarise(results: list[dict]) -> None:
    print("\n device | all clips | median | CPU time | NPU busy | GPU busy |  WER  | battery")
    for r in results:
        wer = "  -  " if r["wer"] is None else f"{r['wer'] * 100:4.1f}%"
        drain = "plugged in" if r["drain_mw"] is None else f"{r['drain_mw'] / 1000:.1f} W"
        print(f" {r['device']:>6} | {r['total_s']:8.2f}s | {r['median_ms']:4.0f}ms | {r['cpu_s']:7.2f}s "
              f"| {r['npu']:7.2f}s | {r['gpu']:7.2f}s | {wer} | {drain}")


def _busy_seconds() -> dict[str, float]:
    """This process's busy time on the NPU and the GPU, from Windows' "GPU Engine" counters (100 ns ticks)."""
    kinds = _adapter_kinds()
    busy = {"npu": 0.0, "gpu": 0.0}
    for luid, _, ticks in _engine_counters(f"pid_{os.getpid()}_*"):
        if luid in kinds:
            busy[kinds[luid]] += ticks / 1e7
    return busy


def _adapter_kinds() -> dict[str, str]:
    """Each adapter's LUID as "npu" or "gpu".

    The NPU is the adapter with only compute engines. Intel's GPU has a
    "neural" engine of its own besides 3D and video, so the engine name
    alone doesn't tell them apart. Display-only adapters (3D alone) are left out.
    """
    engines: dict[str, set[str]] = {}
    for luid, engine, _ in _engine_counters("*"):
        engines.setdefault(luid, set()).add(engine)
    kinds = {}
    for luid, names in engines.items():
        if names <= {"compute", "copy"}:
            kinds[luid] = "npu"
        elif "videodecode" in names:
            kinds[luid] = "gpu"
    return kinds


def _engine_counters(instances: str) -> list[tuple[str, str, int]]:
    """(adapter LUID, engine type, running time) for each matching "GPU Engine" counter instance."""
    command = (
        f"(Get-Counter -Counter '\\GPU Engine({instances})\\Running Time' -ErrorAction SilentlyContinue)"
        ".CounterSamples | ForEach-Object { $_.InstanceName + '=' + $_.RawValue }"
    )
    counters = []
    for line in _powershell(command).splitlines():
        name, _, raw = line.rpartition("=")
        found = re.search(r"luid_(0x[0-9a-f]+_0x[0-9a-f]+).*engtype_(\w*)", name)
        if found and raw.strip().isdigit():
            counters.append((found.group(1), found.group(2), int(raw)))
    return counters


def _difference(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before.get(key, 0.0) for key in after}


def _battery_drain_mw() -> float | None:
    """The battery's discharge rate, or None when the laptop is plugged in."""
    reading = _powershell(
        "Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | "
        "Select-Object -First 1 | ForEach-Object { \"$($_.PowerOnline) $($_.DischargeRate)\" }"
    ).split()
    if len(reading) != 2 or reading[0] == "True":
        return None
    return float(reading[1])


def _powershell(command: str) -> str:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=120,
    )
    return completed.stdout


def _spoken_passages() -> list[tuple[str, np.ndarray, str]]:
    """The passages above in Windows' own voice, at 16 kHz mono."""
    folder = Path(tempfile.mkdtemp(prefix="skryb-npu-check-"))
    jobs = [{"path": str(folder / f"passage-{index}.wav"), "text": text} for index, text in enumerate(_PASSAGES)]
    (folder / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    _powershell(
        "Add-Type -AssemblyName System.Speech; "
        "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 16000, 'Sixteen', 'Mono'; "
        f"foreach ($job in (Get-Content -Raw '{folder / 'jobs.json'}' | ConvertFrom-Json)) {{ "
        "$voice.SetOutputToWaveFile($job.path, $format); $voice.Speak($job.text) }; $voice.Dispose()"
    )
    return [(Path(job["path"]).name, _read_wav(job["path"]), job["text"]) for job in jobs]


def _read_clips(paths: list[str]) -> list[tuple[str, np.ndarray, str | None]]:
    files = [file for path in map(Path, paths) for file in (sorted(path.glob("*.wav")) if path.is_dir() else [path])]
    return [(file.name, _read_wav(str(file)), None) for file in files]


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path) as recording:
        if (recording.getframerate(), recording.getnchannels(), recording.getsampwidth()) != (_RATE, 1, 2):
            raise SystemExit(f"{path} must be 16 kHz mono 16-bit")
        frames = recording.readframes(recording.getnframes())
    return np.frombuffer(frames, np.int16).astype(np.float32) / 32768


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _word_errors(reference: str, hypothesis: str) -> int:
    """Word-level edit distance: substitutions, insertions and deletions."""
    ref, hyp = _words(reference), _words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (word != other))
    return row[-1]


if __name__ == "__main__":
    sys.exit(main())
