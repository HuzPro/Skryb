"""Compare a candidate run against the PyTorch reference: same words, how much faster.

    python native/bench/compare.py [candidate.jsonl] [--bench D:/VoiceToText_bench]

Word error rate here is the candidate measured against the reference output,
after lowercasing and dropping punctuation, so 0% means no regression at all.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

_BUCKETS = ((0, 3), (3, 8), (8, 20), (20, 60), (60, 10_000))


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def _edits(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.open(encoding="utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", nargs="?", default="native.jsonl")
    parser.add_argument("--bench", type=Path, default=Path("D:/VoiceToText_bench"))
    args = parser.parse_args()
    ref = _load(args.bench / "reference.jsonl")
    cand = _load(args.bench / args.candidate)
    ids = [i for i in ref if i in cand]

    exact = sum(ref[i]["text"] == cand[i]["text"] for i in ids)
    edits = sum(_edits(_words(ref[i]["text"]), _words(cand[i]["text"])) for i in ids)
    words = sum(len(_words(ref[i]["text"])) for i in ids)
    print(f"{len(ids)} clips: {exact} identical ({exact / len(ids):.1%}), word error rate vs reference {edits / max(words, 1):.2%}")

    print(f"{'audio':>9} {'n':>4} {'reference':>10} {'candidate':>10} {'speedup':>8}")
    for lo, hi in _BUCKETS:
        b = [i for i in ids if lo <= ref[i]["audio_seconds"] < hi]
        if not b:
            continue
        r = statistics.median(ref[i]["seconds"] for i in b)
        c = statistics.median(cand[i]["seconds"] for i in b)
        print(f"{lo:>3}-{hi if hi < 10_000 else '':<4}s {len(b):>4} {r:>9.3f}s {c:>9.3f}s {r / c:>7.2f}x")
    r, c = sum(ref[i]["seconds"] for i in ids), sum(cand[i]["seconds"] for i in ids)
    print(f"{'total':>9} {len(ids):>4} {r:>9.1f}s {c:>9.1f}s {r / c:>7.2f}x")

    worst = sorted(ids, key=lambda i: -_edits(_words(ref[i]["text"]), _words(cand[i]["text"])))
    for i in worst[:8]:
        if ref[i]["text"] == cand[i]["text"]:
            break
        print(f"\n{i} ({ref[i]['audio_seconds']:.1f}s)\n  ref:  {ref[i]['text'][:300]}\n  cand: {cand[i]['text'][:300]}")


if __name__ == "__main__":
    main()
