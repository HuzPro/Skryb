"""Measure where the speech model puts its full stops and commas, from saved dictations.

    python -m scripts.measure_punctuation checks --logs D:/VoiceToText_logs --out D:/Skryb-punctuation-eval
    "%LOCALAPPDATA%/Programs/Skryb/python/python.exe" -B scripts/measure_punctuation.py replay \
        --logs D:/VoiceToText_logs --out D:/Skryb-punctuation-eval --min-seconds 15 --control

``checks`` decodes nothing. It counts, over every logged dictation of one
model, the signs of a misplaced mark that need no labels: sentences ending on
a word no sentence ends on, sentences opening with And/But/So/Or, fragments,
dictations with no closing mark, questions before the end. Both the engine's
own text (``text``) and what was typed after the cleanup rules (``typed_text``)
are counted, because the rules already move a few marks and the gap between
the two says how much they catch.

``replay`` decodes each saved clip again, once as recorded and once per cap
with every long pause inside it shortened to that cap, and compares the marks.
Shortening pauses cut mid-sentence splits for Parakeet; whether Qwen3-ASR
splits at pauses at all is what this measures. The words must not change: a
trim that changes them has damaged recognition, whatever it did to the marks.
The engine also moves on its own: taking 40 ms off one clip changed a word.
So ``--control`` decodes each clip twice more with its pauses untouched, once
exactly as it is and once without its first 20 ms, and a cap only means
something where it moves the output more than those do. Every spot where the two disagree on a mark is written out for a later blind
listening pass, which is the only fair judge: the user's own edits were made
from the raw output and so favour it.

The replay needs the engine DLLs, which only the installed app ships, so it
runs in the app's own Python and imports ``skryb`` from there. Nothing here
puts the checkout on ``sys.path``: run as a script, Python adds only
``scripts/``, so the app's package is the one that loads.

Everything written goes under ``--out``, which must sit outside any git
checkout and outside the logs: the files hold dictated text, and a folder
inside the repo is one ``git add`` away from publishing it. The console only
ever shows counts.
"""

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import TextIO

import numpy as np

from skryb.output.dictation_trace import TYPED_FIELD
from skryb.output.text_pipeline import _NEVER_FINAL_WORDS
from skryb.recording.audio_segment import AudioSegment, load_wav

DEFAULT_MODEL = "qwen3-asr-1.7b"
DEFAULT_CAPS_MS = (300, 500, 800)

Decode = Callable[[np.ndarray, int], str]
DecoderFactory = Callable[[str, str], tuple[Decode, Callable[[], None]]]
_Event = dict[str, object]

# Words a full stop after is almost always a thinking pause, not a sentence end.
# English does end sentences on some of them ("turn it on"), so these are
# counted apart from the app's own list rather than mixed into it.
NEVER_FINAL_APP = frozenset(_NEVER_FINAL_WORDS)
NEVER_FINAL_WIDE = NEVER_FINAL_APP | {
    "to", "of", "with", "for", "from", "in", "on", "at", "is", "are", "was",
    "were", "that", "which", "because", "if", "when", "so",
}
_CONJUNCTIONS = frozenset({"and", "but", "so", "or"})

# Urdu writes "۔" for a full stop, Hindi "।", Arabic and Urdu "؟" for a
# question; Chinese and Japanese use full-width marks with no space after.
_FULL_STOPS = ".۔।。"
_QUESTION_MARKS = "?؟？"
_CLOSERS = "\"'”’)\\]"
_SENTENCE_END = re.compile(rf"[.!?؟۔।]+(?=[{_CLOSERS}]*(?:\s|$))|[。？！]+")
_WORD = re.compile(r"\w+(?:['’-]\w+)*")
# "1,000" is a number, not a comma in a sentence.
_COMMA = re.compile(r"[,،、，](?!\d)")
_COMMA_BEFORE_CONJUNCTION = re.compile(r"[,،，、]\s*(?:and|but|so|or)\b", re.IGNORECASE)
_STOP_BEFORE_CONJUNCTION = re.compile(rf"[.۔।。][{_CLOSERS}]*\s+(?:and|but|so|or)\b", re.IGNORECASE)

_BUCKETS = ((15.0, "under 15 s"), (30.0, "15-30 s"), (60.0, "30-60 s"), (math.inf, "60 s and over"))
_UNKNOWN_LENGTH = "length unknown"
_ALL = "all"
_FIELDS = ("text", TYPED_FIELD)

_LABEL_WIDTH = 52
_CELL_WIDTH = 18


@dataclass(frozen=True)
class _Dictation:
    id: str
    model: str
    engine: str
    audio: Path | None
    audio_seconds: float | None
    text: str
    typed_text: str


@dataclass(frozen=True)
class _Sentence:
    text: str
    words: tuple[str, ...]
    mark: str


@dataclass(frozen=True)
class _Options:
    command: str
    logs: Path
    out: Path
    model: str
    seed: int
    examples: int
    caps: tuple[int, ...]
    limit: int | None
    min_seconds: float | None
    max_seconds: float | None
    control: bool


def main(argv: Sequence[str] | None = None, build_decoder: DecoderFactory | None = None) -> None:
    options = _parse_options(argv)
    if not options.logs.is_dir():
        raise SystemExit(f"No such logs folder: {options.logs}")
    out = _checked_output_dir(options.out, options.logs)
    if options.command == "checks":
        _checks_command(options, out)
    else:
        _replay_command(options, out, build_decoder or _app_decoder)


def _parse_options(argv: Sequence[str] | None) -> _Options:
    parser = argparse.ArgumentParser(description="Measure punctuation placement in saved Skryb dictations.")
    commands = parser.add_subparsers(dest="command", required=True)
    checks = commands.add_parser("checks", help="label-free counts over the logged text, no decoding")
    replay = commands.add_parser("replay", help="decode saved clips raw and with long pauses shortened")
    for command in (checks, replay):
        command.add_argument("--logs", type=Path, required=True, help="folder with one subfolder per session")
        command.add_argument("--out", type=Path, required=True, help="folder for the results, outside the repo")
        command.add_argument("--model", default=DEFAULT_MODEL, help="the model field to keep")
        command.add_argument("--seed", type=int, default=0, help="seed for picking examples or clips")
    checks.add_argument(
        "--examples", type=int, default=0, help="write up to N example sentences per check to a local file",
    )
    replay.add_argument("--cap", type=int, action="append", help="longest pause kept, in ms (repeatable)")
    replay.add_argument("--limit", type=int, help="replay at most N clips, picked at random with --seed")
    replay.add_argument("--min-seconds", type=float, help="keep only clips logged at least this long")
    replay.add_argument("--max-seconds", type=float, help="keep only clips logged shorter than this")
    replay.add_argument(
        "--control", action="store_true",
        help="also decode each clip a second time as it is (repeat) and without its first 20 ms (shift)",
    )
    arguments = parser.parse_args(argv)
    caps = tuple(getattr(arguments, "cap", None) or DEFAULT_CAPS_MS)
    if any(cap <= 0 for cap in caps):
        parser.error("--cap must be a positive number of milliseconds")
    shortest, longest = getattr(arguments, "min_seconds", None), getattr(arguments, "max_seconds", None)
    if any(bound is not None and bound < 0 for bound in (shortest, longest)):
        parser.error("--min-seconds and --max-seconds can't be negative")
    if shortest is not None and longest is not None and longest <= shortest:
        parser.error("--max-seconds must be more than --min-seconds")
    return _Options(
        command=arguments.command,
        logs=arguments.logs,
        out=arguments.out,
        model=arguments.model,
        seed=arguments.seed,
        examples=getattr(arguments, "examples", 0),
        caps=tuple(dict.fromkeys(caps)),
        limit=getattr(arguments, "limit", None),
        min_seconds=shortest,
        max_seconds=longest,
        control=getattr(arguments, "control", False),
    )


def _checked_output_dir(out: Path, logs: Path) -> Path:
    out = out.expanduser().resolve()
    logs = logs.expanduser().resolve()
    if out.is_relative_to(logs):
        raise SystemExit(f"--out {out} is inside the logs folder {logs}. Pick a folder outside it.")
    checkout = next((folder for folder in (out, *out.parents) if (folder / ".git").exists()), None)
    if checkout is not None:
        raise SystemExit(
            f"--out {out} is inside the git checkout {checkout}. The results hold dictated text, "
            "so pick a folder outside any repository."
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


# --- Reading the logs ---------------------------------------------------------


def read_dictations(logs_dir: Path) -> list[_Dictation]:
    """Every logged transcription, oldest session first, each one once.

    A session log can end in a half-written line while the app is running,
    and an old one can hold a line that was cut short by a crash; both are
    skipped rather than ending the run. A line logged twice counts once.
    """
    dictations: dict[str, _Dictation] = {}
    for session in sorted(path for path in logs_dir.iterdir() if path.is_dir()):
        for event in _events(session / "session.jsonl"):
            if event.get("event") != "transcription":
                continue
            dictation = _dictation(session, event)
            dictations.setdefault(dictation.id, dictation)
    return list(dictations.values())


def _dictation(session: Path, event: _Event) -> _Dictation:
    audio_path = _text(event, "audio_path")
    return _Dictation(
        id=f"{session.name}/{_count(event, 'index'):04d}",
        model=_text(event, "model"),
        engine=_text(event, "engine"),
        audio=session / audio_path if audio_path else None,
        audio_seconds=_seconds(event, "audio_seconds"),
        text=_text(event, "text"),
        typed_text=_text(event, TYPED_FIELD),
    )


def _events(path: Path) -> list[_Event]:
    if not path.is_file():
        return []
    # A line cut off mid-character must not stop the rest of the file being read.
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [event for event in map(_parsed, lines) if event is not None]


def _parsed(line: str) -> _Event | None:
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _text(event: _Event, key: str) -> str:
    value = event.get(key, "")
    return value if isinstance(value, str) else ""


def _count(event: _Event, key: str) -> int:
    value = event.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _seconds(event: _Event, key: str) -> float | None:
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _bucket(seconds: float | None) -> str:
    if seconds is None:
        return _UNKNOWN_LENGTH
    return next(name for limit, name in _BUCKETS if seconds < limit)


# --- Reading the text ---------------------------------------------------------


def sentences(text: str) -> list[_Sentence]:
    """*text* split after each sentence mark, keeping the mark; pieces without a word are dropped.

    A mark ends a sentence only when a space, a closing quote or the end
    follows it, so "3.5" and "0.17.1" stay whole.
    """
    pieces: list[tuple[str, str]] = []
    start = 0
    for end in _SENTENCE_END.finditer(text):
        pieces.append((text[start:end.end()], end.group()))
        start = end.end()
    pieces.append((text[start:], ""))
    return [
        _Sentence(text=body.strip(), words=tuple(_WORD.findall(body)), mark=mark)
        for body, mark in pieces
        if _WORD.search(body)
    ]


def _mark_kind(mark: str) -> str:
    if any(character in _QUESTION_MARKS for character in mark):
        return "question"
    if any(character in _FULL_STOPS for character in mark):
        return "full_stop"
    return "exclamation" if mark else ""


def _opens_with_conjunction(sentence: _Sentence) -> bool:
    return sentence.words[0].casefold() in _CONJUNCTIONS


def _lexical(text: str) -> list[str]:
    return [word.casefold() for word in _WORD.findall(text)]


# --- checks -------------------------------------------------------------------

# Each row: the key, its label, and the key it is a share of (None for a plain count).
_CHECK_ROWS: tuple[tuple[str, str, str | None], ...] = (
    ("dictations", "dictations", None),
    ("sentences", "sentences", None),
    ("words", "words", None),
    ("never_final_app", "sentences ending on a word on the app's list", "sentences"),
    ("never_final_wide", "sentences ending on a word on the wider list", "sentences"),
    ("starts", "sentences starting And/But/So/Or", "sentences"),
    ("inner_starts", "  the same, after the dictation's first sentence", "inner_sentences"),
    ("fragments", "fragments under 3 words (dictations of 2+ sentences)", "sentences_in_multi"),
    ("no_final_mark", "dictations of 5+ words with no closing mark", "five_word_dictations"),
    ("ends_question", "dictations ending in a question mark", "dictations"),
    ("inner_questions", "question marks before the last sentence", "question_marks"),
    ("comma_before", "commas before and/but/so/or", None),
    ("stop_before", "full stops before And/But/So/Or", None),
)
_DENOMINATORS = ("inner_sentences", "sentences_in_multi", "five_word_dictations", "question_marks")
_METRICS = tuple(dict.fromkeys([key for key, _, _ in _CHECK_ROWS] + list(_DENOMINATORS)))


def checks(dictations: Iterable[_Dictation]) -> tuple[dict[str, dict[str, dict[str, int]]], list[_Event]]:
    """Counts per text field and take length, and every sentence each check caught."""
    tallies = {field: {_ALL: Counter[str]()} for field in _FIELDS}
    caught: list[_Event] = []
    for dictation in dictations:
        for field, text in (("text", dictation.text), (TYPED_FIELD, dictation.typed_text)):
            counts, hits = _dictation_counts(text)
            if not counts:
                continue
            tallies[field][_ALL].update(counts)
            tallies[field].setdefault(_bucket(dictation.audio_seconds), Counter()).update(counts)
            caught += [{"field": field, "check": check, "id": dictation.id, "sentence": s} for check, s in hits]
    report = {
        field: {bucket: {metric: tally[metric] for metric in _METRICS} for bucket, tally in _in_bucket_order(by_bucket)}
        for field, by_bucket in tallies.items()
    }
    return report, caught


def _in_bucket_order(by_bucket: dict[str, Counter[str]]) -> list[tuple[str, Counter[str]]]:
    order = [_ALL, *(name for _, name in _BUCKETS), _UNKNOWN_LENGTH]
    return [(name, by_bucket[name]) for name in order if name in by_bucket]


def _dictation_counts(text: str) -> tuple[Counter[str], list[tuple[str, str]]]:
    found = sentences(text)
    counts: Counter[str] = Counter()
    hits: list[tuple[str, str]] = []
    if not found:
        return counts, hits
    counts.update(
        dictations=1,
        sentences=len(found),
        inner_sentences=len(found) - 1,
        words=sum(len(sentence.words) for sentence in found),
        comma_before=len(_COMMA_BEFORE_CONJUNCTION.findall(text)),
        stop_before=len(_STOP_BEFORE_CONJUNCTION.findall(text)),
    )
    for position, sentence in enumerate(found):
        for check in _sentence_checks(sentence, position, found):
            counts[check] += 1
            # Every sentence start and question would swamp the examples; the splits are the interest.
            if check not in ("starts", "question_marks"):
                hits.append((check, sentence.text))
    if len(found) > 1:
        counts["sentences_in_multi"] = len(found)
    if counts["words"] >= 5:
        counts["five_word_dictations"] = 1
        if not found[-1].mark:
            counts["no_final_mark"] = 1
            hits.append(("no_final_mark", found[-1].text))
    if _mark_kind(found[-1].mark) == "question":
        counts["ends_question"] = 1
    return counts, hits


def _sentence_checks(sentence: _Sentence, position: int, found: list[_Sentence]) -> list[str]:
    caught: list[str] = []
    last_word = sentence.words[-1].casefold()
    # As in the app's rule, a word alone in its sentence is left out: "And." may be the whole thought.
    if sentence.mark and len(sentence.words) > 1:
        caught += ["never_final_app"] if last_word in NEVER_FINAL_APP else []
        caught += ["never_final_wide"] if last_word in NEVER_FINAL_WIDE else []
    if _opens_with_conjunction(sentence):
        # The first sentence of a dictation may carry on from the one before it,
        # so only a later one is a split the model made inside this clip.
        caught += ["starts", "inner_starts"] if position else ["starts"]
    if len(found) > 1 and len(sentence.words) < 3:
        caught.append("fragments")
    if _mark_kind(sentence.mark) == "question":
        caught += ["question_marks", "inner_questions"] if position < len(found) - 1 else ["question_marks"]
    return caught


def _checks_command(options: _Options, out: Path) -> None:
    every = read_dictations(options.logs)
    chosen = [dictation for dictation in every if dictation.model == options.model]
    report, caught = checks(chosen)
    slug = _slug(options.model)
    models = Counter(dictation.model for dictation in every).most_common()
    document = {
        "model": options.model,
        "logs": str(options.logs),
        "dictations_logged": len(chosen),
        "models_in_logs": dict(models),
        "fields": report,
    }
    _write_json(out / f"checks-{slug}.json", document)
    examples_path = out / f"checks-{slug}-examples.jsonl"
    if options.examples > 0:
        _write_jsonl(examples_path, _examples(caught, options.examples, options.seed))
    print(f"Punctuation checks for {options.model}: {len(chosen)} dictations logged")
    print("Models in these logs: " + ", ".join(f"{name or '(none)'} {count}" for name, count in models))
    _print_checks(report)
    print(f"\nWritten to {out / f'checks-{slug}.json'}")
    if options.examples > 0:
        print(f"Example sentences (local only): {examples_path}")


def _examples(caught: list[_Event], per_check: int, seed: int) -> list[_Event]:
    by_check: dict[tuple[object, object], list[_Event]] = {}
    for hit in caught:
        by_check.setdefault((hit["field"], hit["check"]), []).append(hit)
    rng = random.Random(seed)
    return [
        hit
        for hits in by_check.values()
        for hit in rng.sample(hits, min(per_check, len(hits)))
    ]


_BY_LENGTH_ROWS = ("dictations", "sentences", "starts", "inner_starts", "fragments", "no_final_mark")


def _print_checks(report: dict[str, dict[str, dict[str, int]]]) -> None:
    print()
    _print_table(["engine text", "typed text"], [
        (label, [_share_cell(report[field][_ALL], key, of) for field in _FIELDS])
        for key, label, of in _CHECK_ROWS
    ])
    for field, title in zip(_FIELDS, ("engine text", "typed text")):
        print(f"\nBy take length, {title}")
        buckets = [name for name in report[field] if name != _ALL]
        _print_table(buckets, [
            (label, [_share_cell(report[field][name], key, of) for name in buckets])
            for key, label, of in _CHECK_ROWS
            if key in _BY_LENGTH_ROWS
        ])


def _share_cell(tally: dict[str, int], key: str, of: str | None) -> str:
    count = tally[key]
    if of is None:
        return f"{count:,}"
    total = tally[of]
    return f"{count:,} ({100 * count / total:.1f}%)" if total else f"{count:,}"


# --- Pause shortening ---------------------------------------------------------

# Levels are read in the app's own 20 ms frames, and the room is the quietest
# tenth of them: the reading the session log keeps as audio_noise_floor.
_FRAME_SECONDS = 0.02
_ROOM_QUANTILE = 0.1
# A frame is pause at up to twice the room's level (+6 dB). Room noise wavers
# well inside that, while a whispered syllable rises above it.
_PAUSE_OVER_ROOM = 2.0
# And never within 20 dB of the loudest speech: in a noisy room a whisper sits
# near the noise, so there nothing is taken for a pause and nothing is cut.
_SPEECH_QUANTILE = 0.99
_PAUSE_UNDER_SPEECH = 0.1
# At least 100 ms of each pause stays either side of the cut, whatever the
# cap, so a soft consonant or breath at the edge of a word is never removed.
_MIN_EDGE_SECONDS = 0.1
# Equal-power, because the room noise either side of a join is unrelated: a
# linear fade would dip it by 3 dB in the middle of the join.
_CROSSFADE_SECONDS = 0.005


def shorten_pauses(samples: np.ndarray, sample_rate: int, cap_ms: float) -> np.ndarray:
    """*samples* with each pause inside the clip longer than *cap_ms* cut down to *cap_ms*.

    A pause is a stretch quieter than a level set from the clip's own room
    noise. The audio comes out of the middle of the pause, so real room noise
    is left either side of the join, and the join is crossfaded so it does not
    click. Nothing is inserted: no zeros, no made-up silence. Silence before
    the first word and after the last is left as it is, and a clip with no
    pause over the cap comes back as it went in.
    """
    cuts = _pause_cuts(samples, sample_rate, cap_ms)
    if not cuts:
        return samples
    return _joined(samples, cuts, max(round(sample_rate * _CROSSFADE_SECONDS), 1))


def _pause_cuts(samples: np.ndarray, sample_rate: int, cap_ms: float) -> list[tuple[int, int]]:
    """Where to cut, as (first sample removed, first sample kept) pairs, in order."""
    frame = max(round(sample_rate * _FRAME_SECONDS), 1)
    whole = samples.size // frame
    if whole == 0:
        return []
    framed = samples[: whole * frame].reshape(whole, frame).astype(np.float64)
    levels = np.sqrt(np.mean(np.square(framed), axis=1))
    threshold = min(
        float(np.quantile(levels, _ROOM_QUANTILE)) * _PAUSE_OVER_ROOM,
        float(np.quantile(levels, _SPEECH_QUANTILE)) * _PAUSE_UNDER_SPEECH,
    )
    kept = max(round(sample_rate * cap_ms / 1000), 2 * round(sample_rate * _MIN_EDGE_SECONDS))
    fade = max(round(sample_rate * _CROSSFADE_SECONDS), 1)
    cuts: list[tuple[int, int]] = []
    for first, last in _quiet_runs(levels <= threshold):
        start, end = first * frame, last * frame
        removed = end - start - kept
        # Touching either end of the clip is lead-in or tail, not a pause inside speech.
        if first == 0 or last == whole or removed <= fade:
            continue
        cut_from = start + kept // 2
        cuts.append((cut_from, cut_from + removed))
    return cuts


def _quiet_runs(quiet: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], quiet.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _joined(samples: np.ndarray, cuts: list[tuple[int, int]], fade: int) -> np.ndarray:
    position = (np.arange(fade) + 0.5) / fade
    fade_out, fade_in = np.cos(position * np.pi / 2), np.sin(position * np.pi / 2)
    pieces: list[np.ndarray] = []
    start = 0
    for cut_from, cut_to in cuts:
        pieces.append(samples[start:cut_from])
        pieces.append(samples[cut_from:cut_from + fade] * fade_out + samples[cut_to:cut_to + fade] * fade_in)
        start = cut_to + fade
    pieces.append(samples[start:])
    return np.concatenate(pieces).astype(samples.dtype, copy=False)


# --- replay -------------------------------------------------------------------

_COUNT_LABELS = {
    "full_stops": "full stops",
    "question_marks": "question marks",
    "commas": "commas",
    "starts": "sentences starting And/But/So/Or",
    "inner_starts": "  the same, after the dictation's first sentence",
    "stop_before": "full stops before And/But/So/Or",
    "comma_before": "commas before and/but/so/or",
}
_COUNT_KEYS = tuple(_COUNT_LABELS)
_CONTEXT_WORDS = 5
# A pause result means nothing without knowing how far the output moves when
# the pauses are left alone. "repeat" decodes the very same samples again: the
# engine's own run-to-run wobble. "shift" drops the first 20 ms, before the
# first word: about as small a change as the smallest trim, at no pause.
_REPEAT = "repeat"
_SHIFT = "shift"
_SHIFT_SECONDS = 0.02
_PROGRESS_EVERY = 25


def _replay_command(options: _Options, out: Path, build_decoder: DecoderFactory) -> None:
    started = time.monotonic()
    of_model = [
        dictation
        for dictation in read_dictations(options.logs)
        if dictation.model == options.model and _in_length_range(dictation.audio_seconds, options)
    ]
    saved = [dictation for dictation in of_model if dictation.audio is not None]
    on_disk = [dictation for dictation in saved if dictation.audio is not None and dictation.audio.is_file()]
    if not on_disk:
        raise SystemExit(f"No saved audio for {options.model} in that length range under {options.logs}.")
    # One adapter for the whole run: the one most of this model's dictations went through.
    engine = Counter(dictation.engine for dictation in on_disk).most_common(1)[0][0]
    chosen = sample_clips([d for d in on_disk if d.engine == engine], options.limit, options.seed)
    keys = _variant_keys(options)
    slug = _slug(options.model)
    results_path = out / f"replay-{slug}.jsonl"
    earlier = replayed_records(results_path)
    pending = [dictation for dictation in chosen if _missing_variants(earlier.get(dictation.id), keys)]
    # The engine is only loaded when there is something left to decode.
    failures = _replay_with_engine(pending, build_decoder, options, engine, results_path, earlier) if pending else []
    replayed = replayed_records(results_path)
    records = [
        replayed[dictation.id]
        for dictation in chosen
        if dictation.id in replayed and not _missing_variants(replayed[dictation.id], keys)
    ]
    entries = [_per_dictation(record, keys) for record in records]
    summary = _replay_summary(entries, options, keys, engine, len(failures), len(saved) - len(on_disk))
    _write_jsonl(out / f"replay-{slug}-per-dictation.jsonl", entries)
    _write_jsonl(out / f"replay-{slug}-differences.jsonl", _differences(records, keys))
    _write_json(out / f"replay-{slug}-summary.json", summary)
    _print_replay(summary, keys, failures)
    print(f"\nWritten to {out}")
    print(_wall_time(time.monotonic() - started, len(pending)))


def _in_length_range(seconds: float | None, options: _Options) -> bool:
    if options.min_seconds is None and options.max_seconds is None:
        return True
    if seconds is None:
        return False
    return (options.min_seconds is None or seconds >= options.min_seconds) and (
        options.max_seconds is None or seconds < options.max_seconds
    )


def _variant_keys(options: _Options) -> tuple[str, ...]:
    """The variants of each clip besides raw, as their keys in the results file."""
    controls = (_REPEAT, _SHIFT) if options.control else ()
    return (*(str(cap) for cap in options.caps), *controls)


def _variant_name(key: str) -> str:
    return f"{key} ms" if key.isdigit() else key


def _wall_time(seconds: float, clips: int) -> str:
    each = f", {seconds / clips:.1f} s a clip with the engine load" if clips else ""
    return f"Took {seconds:.0f} s for {clips} clips decoded this run{each}"


def sample_clips(dictations: Sequence[_Dictation], limit: int | None, seed: int) -> list[_Dictation]:
    """A seeded shuffle, cut to *limit*.

    Shuffled even without a limit, so a run stopped halfway has replayed a
    random sample rather than the oldest sessions.
    """
    shuffled = list(dictations)
    random.Random(seed).shuffle(shuffled)
    return shuffled if limit is None else shuffled[:limit]


def _replay_with_engine(
    pending: list[_Dictation],
    build_decoder: DecoderFactory,
    options: _Options,
    engine: str,
    results_path: Path,
    earlier: dict[str, _Event],
) -> list[str]:
    decode, close = build_decoder(options.model, engine)
    try:
        return replay(pending, decode, _variant_keys(options), results_path, earlier)
    finally:
        close()


def replay(
    dictations: Sequence[_Dictation],
    decode: Decode,
    keys: Sequence[str],
    results_path: Path,
    earlier: dict[str, _Event],
) -> list[str]:
    """Decode each clip's missing variants, appending one record per clip; returns what failed.

    Each record is flushed as soon as it is made, so a run that is stopped
    loses at most the clip it was on.
    """
    failures: list[str] = []
    started = time.monotonic()
    with _appending(results_path) as stream:
        for done, dictation in enumerate(dictations, start=1):
            try:
                record = _replay_clip(dictation, decode, keys, earlier.get(dictation.id))
            except Exception as error:  # noqa: BLE001 - one bad clip must not end a long run
                failures.append(_failure(dictation, error))
            else:
                _append_record(stream, record)
            if done % _PROGRESS_EVERY == 0:
                print(f"replayed {done} of {len(dictations)} clips, {time.monotonic() - started:.0f} s", flush=True)
    return failures


def _replay_clip(dictation: _Dictation, decode: Decode, keys: Sequence[str], earlier: _Event | None) -> _Event:
    assert dictation.audio is not None
    audio = load_wav(dictation.audio)
    # A clip replayed before keeps its raw text and every variant it has, so
    # asking for more decodes only what is new.
    raw = str(earlier["raw"]) if earlier else decode(audio.data, audio.sample_rate)
    variants = dict(earlier["trimmed"]) if earlier else {}
    for key in _missing_variants(earlier, keys):
        variants[key] = _variant(key, audio, raw, decode)
    return {
        "id": dictation.id,
        "audio_path": str(dictation.audio),
        "audio_seconds": dictation.audio_seconds,
        "logged_text": dictation.text,
        "raw": raw,
        # Named when caps were the only variants; the controls sit beside them.
        "trimmed": variants,
    }


def _variant(key: str, audio: AudioSegment, raw: str, decode: Decode) -> _Event:
    rate = audio.sample_rate
    if key == _REPEAT:
        return {"text": decode(audio.data, rate), "removed_seconds": 0.0}
    if key == _SHIFT:
        changed = audio.data[round(rate * _SHIFT_SECONDS):]
    else:
        changed = shorten_pauses(audio.data, rate, int(key))
    removed = audio.data.size - changed.size
    # Nothing cut is the same audio, and decoding it again would only add the
    # engine's run-to-run wobble, which "repeat" measures on its own.
    text = decode(changed, rate) if removed else raw
    return {"text": text, "removed_seconds": round(removed / rate, 3)}


def _failure(dictation: _Dictation, error: Exception) -> str:
    return f"{dictation.id}: {type(error).__name__}: {error}"


def _append_record(stream: TextIO, record: _Event) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def _appending(path: Path) -> TextIO:
    # A run killed mid-write leaves a line with no newline; the next record
    # must start on a line of its own or both are lost.
    if path.is_file() and path.stat().st_size and not _ends_with_newline(path):
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    return path.open("a", encoding="utf-8")


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as stream:
        stream.seek(-1, 2)
        return stream.read(1) == b"\n"


def replayed_records(results_path: Path) -> dict[str, _Event]:
    """Every clip replayed so far, by id, whatever variants it has; a later record wins."""
    replayed: dict[str, _Event] = {}
    for record in _events(results_path):
        if (
            isinstance(record.get("id"), str)
            and isinstance(record.get("raw"), str)
            and isinstance(record.get("trimmed"), dict)
        ):
            replayed[str(record["id"])] = record
    return replayed


def _missing_variants(record: _Event | None, keys: Sequence[str]) -> list[str]:
    if record is None:
        return list(keys)
    variants = record["trimmed"]
    assert isinstance(variants, dict)
    return [key for key in keys if not _is_variant(variants.get(key))]


def _is_variant(variant: object) -> bool:
    return (
        isinstance(variant, dict)
        and isinstance(variant.get("text"), str)
        and isinstance(variant.get("removed_seconds"), (int, float))
    )


def _per_dictation(record: _Event, keys: Sequence[str]) -> _Event:
    raw, logged = str(record["raw"]), _text(record, "logged_text")
    trimmed = record["trimmed"]
    assert isinstance(trimmed, dict)
    entry: _Event = {
        "id": record["id"],
        "length": _bucket(_seconds(record, "audio_seconds")),
        "logged": bool(logged.strip()),
        "reproduced": raw.strip() == logged.strip(),
        "same_words_as_logged": _lexical(raw) == _lexical(logged),
        "raw": text_counts(raw),
    }
    for key in keys:
        variant = trimmed[key]
        text = str(variant["text"])
        entry[_variant_name(key)] = {
            **text_counts(text),
            "removed_seconds": variant["removed_seconds"],
            # A cap that cut nothing reused the raw text; only the rest were decoded again.
            "decoded": key == _REPEAT or variant["removed_seconds"] > 0,
            "words_changed": _lexical(raw) != _lexical(text),
            "spots": len(punctuation_spots(raw, text)),
        }
    return entry


def text_counts(text: str) -> dict[str, int]:
    found = sentences(text)
    kinds = Counter(_mark_kind(sentence.mark) for sentence in found)
    return {
        "full_stops": kinds["full_stop"],
        "question_marks": kinds["question"],
        "commas": len(_COMMA.findall(text)),
        "starts": sum(1 for sentence in found if _opens_with_conjunction(sentence)),
        "inner_starts": sum(1 for sentence in found[1:] if _opens_with_conjunction(sentence)),
        "stop_before": len(_STOP_BEFORE_CONJUNCTION.findall(text)),
        "comma_before": len(_COMMA_BEFORE_CONJUNCTION.findall(text)),
    }


def punctuation_spots(raw: str, trimmed: str) -> list[_Event]:
    """Each word the two texts share whose following mark differs, with a few words around it.

    Words are matched ignoring case and marks, so a spot is found even where
    the trim also changed a word nearby.
    """
    raw_tokens, trimmed_tokens = _tokens(raw), _tokens(trimmed)
    matcher = SequenceMatcher(
        None,
        [word.casefold() for word, _ in raw_tokens],
        [word.casefold() for word, _ in trimmed_tokens],
        autojunk=False,
    )
    spots: list[_Event] = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            at_raw, at_trimmed = block.a + offset, block.b + offset
            if raw_tokens[at_raw][1] != trimmed_tokens[at_trimmed][1]:
                spots.append({
                    "word": at_raw,
                    "raw_mark": raw_tokens[at_raw][1],
                    "trimmed_mark": trimmed_tokens[at_trimmed][1],
                    "raw": _around(raw_tokens, at_raw),
                    "trimmed": _around(trimmed_tokens, at_trimmed),
                })
    return spots


def _tokens(text: str) -> list[tuple[str, str]]:
    """Each word with the marks between it and the next word."""
    matches = list(_WORD.finditer(text))
    ends = [match.start() for match in matches[1:]] + [len(text)]
    return [(match.group(), "".join(text[match.end():end].split())) for match, end in zip(matches, ends)]


def _around(tokens: list[tuple[str, str]], at: int) -> str:
    window = tokens[max(at - _CONTEXT_WORDS, 0): at + _CONTEXT_WORDS + 1]
    return " ".join(word + mark for word, mark in window)


def _differences(records: list[_Event], keys: Sequence[str]) -> list[_Event]:
    differences: list[_Event] = []
    for record in records:
        trimmed = record["trimmed"]
        assert isinstance(trimmed, dict)
        for key in keys:
            text = str(trimmed[key]["text"])
            differences += [
                {
                    "id": record["id"],
                    "audio_path": record.get("audio_path"),
                    "variant": _variant_name(key),
                    "cap_ms": int(key) if key.isdigit() else None,
                    "raw_text": record["raw"],
                    "trimmed_text": text,
                    "spot": spot,
                }
                for spot in punctuation_spots(str(record["raw"]), text)
            ]
    return differences


def _replay_summary(
    entries: list[_Event], options: _Options, keys: Sequence[str], engine: str, failed: int, audio_gone: int,
) -> _Event:
    logged = [entry for entry in entries if entry["logged"]]
    raw = [entry["raw"] for entry in entries]
    variants: dict[str, _Event] = {"raw": _totals(entries, "raw")}
    for key in keys:
        name = _variant_name(key)
        changed = [entry[name] for entry in entries]
        variants[name] = {
            **_totals(entries, name),
            "clips_decoded": sum(1 for variant in changed if variant["decoded"]),
            "seconds_removed": round(sum(variant["removed_seconds"] for variant in changed), 1),
            "clips_fewer_full_stops": sum(1 for v, r in zip(changed, raw) if v["full_stops"] < r["full_stops"]),
            "clips_more_full_stops": sum(1 for v, r in zip(changed, raw) if v["full_stops"] > r["full_stops"]),
            "clips_words_changed": sum(1 for variant in changed if variant["words_changed"]),
            "spots": sum(variant["spots"] for variant in changed),
        }
    return {
        "model": options.model,
        "engine": engine,
        "min_seconds": options.min_seconds,
        "max_seconds": options.max_seconds,
        "clips": len(entries),
        "failed": failed,
        "audio_gone": audio_gone,
        "logged_text": len(logged),
        "reproduced_exactly": sum(1 for entry in logged if entry["reproduced"]),
        "reproduced_word_for_word": sum(1 for entry in logged if entry["same_words_as_logged"]),
        "variants": variants,
    }


def _totals(entries: list[_Event], variant: str) -> _Event:
    return {key: sum(entry[variant][key] for entry in entries) for key in _COUNT_KEYS}


def _print_replay(summary: _Event, keys: Sequence[str], failures: list[str]) -> None:
    print(
        f"Replay of {summary['model']} on {summary['engine']}: {summary['clips']} clips, "
        f"{summary['failed']} failed this run, {summary['audio_gone']} with their audio gone"
    )
    print(
        f"Raw replay matches the logged text: {summary['reproduced_exactly']} of {summary['logged_text']} exactly, "
        f"{summary['reproduced_word_for_word']} of {summary['logged_text']} word for word"
    )
    for failure in failures[:5]:
        print(f"  failed: {failure}")
    variants = summary["variants"]
    assert isinstance(variants, dict)
    names = list(variants)
    rows = [(label, [f"{variants[name][key]:,}" for name in names]) for key, label in _COUNT_LABELS.items()]
    for field, label in (
        ("clips_decoded", "clips decoded again"),
        ("seconds_removed", "seconds of audio removed"),
        ("clips_fewer_full_stops", "clips with fewer full stops than raw"),
        ("clips_more_full_stops", "clips with more full stops than raw"),
        ("clips_words_changed", "clips whose words changed"),
        ("spots", "places a mark differs from raw"),
    ):
        rows.append((label, ["-", *(f"{variants[_variant_name(key)][field]:,}" for key in keys)]))
    print()
    _print_table(names, rows)


# --- The installed app's engine -----------------------------------------------


def _app_decoder(model: str, engine_id: str) -> tuple[Decode, Callable[[], None]]:
    """*model* on *engine_id*, built and called the way the app's dictation does it.

    Settings, the people list and the models come from the app's own folders
    and are only read. The quiet-speech boost, the language and the names
    hint are applied as ``RecordAndTranscribeUseCase._transcribe`` applies
    them, with ``bootstrap._prepare_audio`` and ``bootstrap._dictation_language``
    mirrored here because importing the bootstrap would start half the app.
    """
    from skryb.core.json_config_adapter import JsonConfigAdapter
    from skryb.core.paths import AppPaths
    from skryb.people.hints import transcribe_with_terms
    from skryb.people.terms import load_terms
    from skryb.recording.speech_level import boost_quiet_speech
    from skryb.transcription.engine_factory import build_engine_registry
    from skryb.transcription.engine_registry import EngineSpec
    from skryb.transcription.model_info import get_model_info
    from skryb.transcription.qwen3_asr_prompt import LANGUAGE_NAMES

    paths = AppPaths()
    config = JsonConfigAdapter(paths.config_path)
    config.load()
    registry = build_engine_registry()
    if engine_id not in {descriptor.engine_id for descriptor in registry.all_descriptors()}:
        raise SystemExit(f"This Skryb has no {engine_id!r} engine, which {model} was logged on.")
    configured = config.get("transcription.language", "en")
    language = configured if configured in LANGUAGE_NAMES else "en"
    boost = config.get("audio.boost_quiet_speech", True) is not False
    # The GPU kind the app last probed: it picks a native engine's AMD or NVIDIA build.
    hardware_kind = config.get("hardware.probe.kind")
    info = get_model_info(model)
    engine = registry.build(engine_id, EngineSpec(
        model_name=info.name,
        repo_id=info.repo_id,
        device="cpu" if hardware_kind in (None, "cpu") else "cuda",
        language=language,
        beam_size=1,
        model_dir=paths.models_dir,
        revision=info.revision,
        question_marks=config.get("transcription.question_marks", True) is not False,
        hardware_kind=hardware_kind,
    ))
    engine.load_model()
    terms = load_terms(paths.people_path)
    print(
        f"Engine {engine_id} loaded: language {language}, quiet-speech boost {'on' if boost else 'off'}, "
        f"{len(terms)} people and terms in the names hint"
    )

    def decode(samples: np.ndarray, sample_rate: int) -> str:
        audio = AudioSegment(np.ascontiguousarray(samples, dtype=np.float32), sample_rate)
        heard = boost_quiet_speech(audio) if boost else audio
        return transcribe_with_terms(engine, heard, terms, language).text

    return decode, engine.dispose


# --- Output ---------------------------------------------------------------------


def _print_table(headers: Sequence[str], rows: Sequence[tuple[str, Sequence[str]]]) -> None:
    print(" " * _LABEL_WIDTH + "".join(f"{header:>{_CELL_WIDTH}}" for header in headers))
    for label, cells in rows:
        print(f"{label:<{_LABEL_WIDTH}}" + "".join(f"{cell:>{_CELL_WIDTH}}" for cell in cells))


def _slug(model: str) -> str:
    return re.sub(r"[^\w.-]", "_", model) or "unnamed"


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
