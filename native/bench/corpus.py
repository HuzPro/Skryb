"""Pick a fixed, length-stratified sample of saved dictations as the benchmark set.

Reads session.jsonl from every diagnostics session folder and writes
corpus.jsonl to the bench folder (outside the repo: the clips are private).

    python native/bench/corpus.py [--logs D:/VoiceToText_logs] [--out D:/VoiceToText_bench]
"""
import argparse
import json
import random
import wave
from pathlib import Path

# (upper bound in seconds, clips to take). Dictation is mostly short, but long
# clips stress the encoder and the KV cache, so every bucket gets a fair share.
_BUCKETS = ((3, 60), (8, 80), (20, 80), (60, 60), (10_000, 40))
_SEED = 20261022


def _clips(logs: Path) -> list[dict]:
    clips = []
    for jsonl in sorted(logs.glob("*/session.jsonl")):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("event") != "transcription" or not row.get("audio_path"):
                continue
            audio = jsonl.parent / row["audio_path"]
            if not audio.exists() or row.get("is_empty") or row.get("error"):
                continue
            clips.append({
                "id": f"{jsonl.parent.name}/{audio.stem}",
                "audio": str(audio),
                "seconds": _seconds(audio),
                "source_model": row.get("model"),
                "source_text": row.get("text") or "",
            })
    return clips


def _seconds(audio: Path) -> float:
    with wave.open(str(audio)) as w:
        return w.getnframes() / w.getframerate()


def _stratify(clips: list[dict]) -> list[dict]:
    rng = random.Random(_SEED)
    picked, lower = [], 0.0
    for upper, count in _BUCKETS:
        bucket = [c for c in clips if lower <= c["seconds"] < upper]
        picked += rng.sample(bucket, min(count, len(bucket)))
        lower = upper
    return sorted(picked, key=lambda c: c["id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("D:/VoiceToText_logs"))
    parser.add_argument("--out", type=Path, default=Path("D:/VoiceToText_bench"))
    args = parser.parse_args()
    clips = _clips(args.logs)
    picked = _stratify(clips)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for clip in picked:
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")
    total = sum(c["seconds"] for c in picked)
    print(f"{len(clips)} usable clips, picked {len(picked)} ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
