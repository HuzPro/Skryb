"""Turn dictation logs into a correction corpus and a training manifest.

    python scripts/build_correction_dataset.py D:/VoiceToText_logs D:/corpus

Each session directory holds ``session.jsonl`` (what the engine heard) and,
when edit capture was on, ``edits.jsonl`` (what the pasted text became before
Enter). Nothing links the two files, so a delivery is joined to its
transcription by time: the pipeline writes ``delivered`` first and the
``transcription`` event lands a moment later, so each delivery takes the first
transcription whose finish time is nearest to it. Streaming pastes several
times per transcription, so a transcription can own several deliveries.

``dataset.jsonl`` keeps every joined row. ``manifest.jsonl`` keeps only rows
that can train an acoustic model: a reliable replay, sent with Enter, an
untouched or word-level edit, and ``pipeline_delta == 0``. That last condition
is the one that looks over-strict. The manifest's ``text`` is the acoustic
target, so it has to be in the engine's own raw style; a row the dictionary
already rewrote would teach the model to emit the dictionary's output, which
the dictionary would then rewrite again. Rows where the user only fixed case,
spacing or punctuation say nothing about what the model heard, so they are
excluded too.

A row carrying any flag at all is excluded, not only the ones that clear
``reliable``. A caret the replay had to clamp, or a select-all, means the edit
may have reached outside the text we pasted, and a wrong transcript attached to
a real recording is worse than a smaller corpus. ``dataset.jsonl`` keeps every
row, and the flag shares in the summary say what that strictness costs.

Streaming rows are excluded outright. A streaming delivery is a fragment of an
utterance while the transcription covers the whole clip, so this row's texts
are not comparable with its audio at all.

Every WAV is copied into the corpus rather than referenced. ``SessionDiagnostics``
deletes whole session directories past ``diagnostics.retain_sessions``, so a
corpus of references loses its audio within weeks.
"""

import argparse
import json
import re
import shutil
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

DEFAULT_TOLERANCE_SECONDS = 30.0

_Event = dict[str, object]
_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_LABEL_WIDTH = 26


@dataclass(frozen=True)
class _Options:
    logs_dir: Path
    output_dir: Path
    tolerance: timedelta


@dataclass(frozen=True)
class _EditLog:
    deliveries: list[_Event]
    sent: list[_Event]


@dataclass(frozen=True)
class _Span:
    """One pasted transcription and the block it was sent in."""

    block: int
    pasted_text: str
    final_text: str
    process: str
    end_reason: str
    reliable: bool
    flags: tuple[str, ...]
    key_count: int
    paste_count: int


@dataclass(frozen=True)
class _Row:
    session: str
    block: int
    seq: int
    transcription_index: int
    audio_path: str
    audio_seconds: float
    sample_rate: int
    audio_rms: float
    audio_peak: float
    audio_noise_floor: float
    raw_text: str
    pasted_text: str
    final_text: str
    model: str
    engine: str
    device: str
    pipeline_mode: str
    output_mode: str
    process: str
    end_reason: str
    reliable: bool
    flags: tuple[str, ...]
    key_count: int
    paste_count: int
    wer_strict: float
    wer_lexical: float
    cer_lexical: float
    pipeline_delta: float
    edit_kind: str


@dataclass(frozen=True)
class _Corpus:
    rows: list[_Row]
    deliveries: int
    sessions: int
    sessions_with_edits: int


def main() -> None:
    build(_parse_options())


def build(options: _Options) -> None:
    if not options.logs_dir.is_dir():
        raise SystemExit(f"no such logs directory: {options.logs_dir}")
    corpus = _join_sessions(options)
    manifest = [
        _manifest_entry(row)
        for row in corpus.rows
        if _trains_the_acoustic_model(row)
    ]
    _write_jsonl(
        options.output_dir / "dataset.jsonl",
        [asdict(row) for row in corpus.rows],
    )
    _write_jsonl(options.output_dir / "manifest.jsonl", manifest)
    copied = _copy_audio(corpus.rows, options)
    _print_summary(corpus, len(manifest), copied)


def _parse_options() -> _Options:
    parser = argparse.ArgumentParser(
        description="Build a correction corpus from VoiceToText session logs.",
    )
    parser.add_argument(
        "logs_dir",
        type=Path,
        help="directory holding one subdirectory per dictation session",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="directory the corpus and its copied audio are written to",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=DEFAULT_TOLERANCE_SECONDS,
        help="how long after a delivery its transcription may still land",
    )
    arguments = parser.parse_args()
    return _Options(
        logs_dir=arguments.logs_dir,
        output_dir=arguments.output_dir,
        tolerance=timedelta(seconds=arguments.tolerance_seconds),
    )


def _join_sessions(options: _Options) -> _Corpus:
    sessions = sorted(path for path in options.logs_dir.iterdir() if path.is_dir())
    rows: list[_Row] = []
    deliveries = 0
    sessions_with_edits = 0
    for session in sessions:
        edits = _edit_log(session / "edits.jsonl")
        if not edits.deliveries:
            continue
        sessions_with_edits += 1
        deliveries += len(edits.deliveries)
        rows.extend(_session_rows(session, edits, options.tolerance))
    return _Corpus(
        rows=rows,
        deliveries=deliveries,
        sessions=len(sessions),
        sessions_with_edits=sessions_with_edits,
    )


def _session_rows(
    session: Path,
    edits: _EditLog,
    tolerance: timedelta,
) -> list[_Row]:
    transcriptions = _transcription_by_delivery(
        edits.deliveries,
        _of_type(_events(session / "session.jsonl"), "transcription"),
        tolerance,
    )
    spans = _span_by_delivery(edits.sent)
    rows: list[_Row] = []
    for delivery in edits.deliveries:
        seq = _count(delivery, "seq")
        if seq in transcriptions and seq in spans:
            rows.append(
                _row(session.name, delivery, transcriptions[seq], spans[seq])
            )
    return rows


def _row(
    session: str,
    delivery: _Event,
    transcription: _Event,
    span: _Span,
) -> _Row:
    raw_text = _text(transcription, "text")
    lexical_pasted = _lexical(span.pasted_text)
    lexical_final = _lexical(span.final_text)
    return _Row(
        session=session,
        block=span.block,
        seq=_count(delivery, "seq"),
        transcription_index=_count(transcription, "index"),
        audio_path=_corpus_audio_path(session, transcription),
        audio_seconds=_number(transcription, "audio_seconds"),
        sample_rate=_count(transcription, "sample_rate"),
        audio_rms=_number(transcription, "audio_rms"),
        audio_peak=_number(transcription, "audio_peak"),
        audio_noise_floor=_number(transcription, "audio_noise_floor"),
        raw_text=raw_text,
        pasted_text=span.pasted_text,
        final_text=span.final_text,
        model=_text(transcription, "model"),
        engine=_text(transcription, "engine"),
        device=_text(transcription, "device"),
        pipeline_mode=_text(delivery, "pipeline_mode"),
        output_mode=_text(delivery, "output_mode"),
        process=span.process,
        end_reason=span.end_reason,
        reliable=span.reliable,
        flags=span.flags,
        key_count=span.key_count,
        paste_count=span.paste_count,
        wer_strict=_error_rate(span.pasted_text.split(), span.final_text.split()),
        wer_lexical=_error_rate(lexical_pasted.split(), lexical_final.split()),
        cer_lexical=_error_rate(lexical_pasted, lexical_final),
        pipeline_delta=_error_rate(_lexical(raw_text).split(), lexical_pasted.split()),
        edit_kind=_edit_kind(span.pasted_text, span.final_text),
    )


def _corpus_audio_path(session: str, transcription: _Event) -> str:
    recorded = _text(transcription, "audio_path")
    return f"audio/{session}/{Path(recorded).name}" if recorded else ""


def _transcription_by_delivery(
    deliveries: list[_Event],
    transcriptions: list[_Event],
    tolerance: timedelta,
) -> dict[int, _Event]:
    by_time = sorted(transcriptions, key=lambda event: _moment(event, "finished_at"))
    finished = [_moment(event, "finished_at") for event in by_time]
    matched: dict[int, _Event] = {}
    for delivery in deliveries:
        at = _moment(delivery, "at")
        nearest = _nearest_index(finished, at)
        if nearest < len(finished) and abs(finished[nearest] - at) <= tolerance:
            matched[_count(delivery, "seq")] = by_time[nearest]
    return matched


def _nearest_index(finished: list[datetime], at: datetime) -> int:
    """The transcription closest in time, on either side.

    A batch delivery is reported just before its paste, which is after the
    transcription event was written; a streaming delivery is reported during
    the run, before it. Matching only forwards or only backwards would
    silently mislabel one of the two. Transcriptions are serialised and each
    takes at least its own audio duration, so the nearest one is never
    ambiguous.
    """
    after = bisect_left(finished, at)
    if after == 0:
        return 0
    if after >= len(finished):
        return len(finished) - 1
    before = after - 1
    return before if at - finished[before] <= finished[after] - at else after


def _span_by_delivery(sent_events: list[_Event]) -> dict[int, _Span]:
    spans: dict[int, _Span] = {}
    for sent in sent_events:
        entries = _objects(sent, "spans")
        for entry in entries:
            spans[_count(entry, "seq")] = _Span(
                block=_count(sent, "block"),
                pasted_text=_text(entry, "pasted_text"),
                final_text=_text(entry, "final_text"),
                process=_text(sent, "process"),
                end_reason=_text(sent, "reason"),
                reliable=_flag(sent, "reliable"),
                flags=_strings(sent, "flags"),
                key_count=_count(sent, "key_count"),
                paste_count=len(entries),
            )
    return spans


# A transcription the user left alone is the model getting it right, which
# is as much acoustic evidence as one the user corrected. Only cosmetic and
# structural edits are excluded, because neither says what was heard.
_ACOUSTIC_EDIT_KINDS = frozenset({"none", "lexical"})

# A streaming delivery is a fragment of an utterance while its transcription
# covers the whole clip, so its text and its audio do not describe the same
# thing. Verbatim rows are kept: the pipeline is a passthrough there, which is
# exactly the raw style the acoustic target needs.
_STREAMING = "streaming"


def _trains_the_acoustic_model(row: _Row) -> bool:
    return (
        row.reliable
        and not row.flags
        and row.end_reason == "enter"
        and row.edit_kind in _ACOUSTIC_EDIT_KINDS
        and row.pipeline_mode != _STREAMING
        and row.pipeline_delta == 0.0
        and bool(row.audio_path)
    )


def _manifest_entry(row: _Row) -> _Event:
    return {
        "audio_filepath": row.audio_path,
        "duration": row.audio_seconds,
        "text": row.final_text,
    }


def _copy_audio(rows: list[_Row], options: _Options) -> int:
    sources = {
        row.audio_path: options.logs_dir
        / row.session
        / "audio"
        / Path(row.audio_path).name
        for row in rows
        if row.audio_path
    }
    copied = 0
    for relative, source in sources.items():
        if not source.is_file():
            continue
        destination = options.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    return copied


def _edit_kind(pasted: str, final: str) -> str:
    if pasted == final:
        return "none"
    if pasted.casefold() == final.casefold():
        return "case_only"
    if _unpunctuated(pasted) == _unpunctuated(final):
        return "punctuation_only"
    if _lexical(pasted) == _lexical(final):
        return "whitespace_only"
    if _most_words_survive(_lexical(pasted).split(), _lexical(final).split()):
        return "lexical"
    return "restructure"


def _most_words_survive(pasted: list[str], final: list[str]) -> bool:
    longest = max(len(pasted), len(final))
    return longest - _edit_distance(pasted, final) > longest / 2


def _error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference:
        return 1.0 if hypothesis else 0.0
    return _edit_distance(reference, hypothesis) / len(reference)


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    # ponytail: quadratic DP, fine for one dictation; batch a corpus of books
    # through it and you want rapidfuzz instead.
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def _lexical(text: str) -> str:
    return " ".join(_unpunctuated(text).split())


def _unpunctuated(text: str) -> str:
    return _PUNCTUATION.sub("", text.casefold())


def _print_summary(corpus: _Corpus, manifest_rows: int, copied: int) -> None:
    rows = corpus.rows
    kept_as_is = sum(1 for row in rows if row.edit_kind == "none")
    print("Correction dataset summary")
    print("The kept_as_is_rate below is an edit rate, not accuracy. It says how")
    print("much the user bothered to fix, not how much the model got wrong.")
    print()
    _print_value("sessions", corpus.sessions)
    _print_value("sessions with edits", corpus.sessions_with_edits)
    _print_value("deliveries", corpus.deliveries)
    _print_value("rows", len(rows))
    _print_value("deliveries dropped", corpus.deliveries - len(rows))
    _print_value("reliable rows", sum(1 for row in rows if row.reliable))
    _print_value("wav files copied", copied)
    _print_value("manifest rows", manifest_rows)
    _print_value("kept_as_is_rate", f"{_share(kept_as_is, len(rows)):.3f}")
    # Streaming pastes one word at a time, so a single utterance becomes many
    # rows whose text is a fragment. Splitting the counts keeps those from
    # being read as if they were whole dictations.
    _print_counts("pipeline_mode", Counter(row.pipeline_mode for row in rows))
    _print_counts("end_reason", Counter(row.end_reason for row in rows))
    _print_counts("process", Counter(row.process for row in rows))
    _print_counts("edit_kind", Counter(row.edit_kind for row in rows))
    _print_flag_shares(rows)


def _print_flag_shares(rows: list[_Row]) -> None:
    counts = Counter(flag for row in rows for flag in row.flags)
    print("\nflags (share of rows)")
    for flag, count in counts.most_common():
        print(f"  {flag:<{_LABEL_WIDTH}}{_share(count, len(rows)):.3f}")


def _print_counts(title: str, counts: Counter[str]) -> None:
    print(f"\n{title}")
    for name, count in counts.most_common():
        print(f"  {name:<{_LABEL_WIDTH}}{count}")


def _print_value(label: str, value: object) -> None:
    print(f"{label:<{_LABEL_WIDTH}}{value}")


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _edit_log(path: Path) -> _EditLog:
    events = _events(path)
    return _EditLog(
        deliveries=_of_type(events, "delivered"),
        sent=_of_type(events, "sent"),
    )


def _events(path: Path) -> list[_Event]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _of_type(events: list[_Event], name: str) -> list[_Event]:
    return [event for event in events if event.get("event") == name]


def _write_jsonl(path: Path, records: Iterable[_Event]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _moment(event: _Event, key: str) -> datetime:
    return datetime.fromisoformat(_text(event, key))


def _text(event: _Event, key: str) -> str:
    value = event.get(key, "")
    return value if isinstance(value, str) else ""


def _count(event: _Event, key: str) -> int:
    value = event.get(key, 0)
    return value if isinstance(value, int) else 0


def _number(event: _Event, key: str) -> float:
    value = event.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _flag(event: _Event, key: str) -> bool:
    return event.get(key) is True


def _strings(event: _Event, key: str) -> tuple[str, ...]:
    value = event.get(key, [])
    return tuple(cast(list[str], value)) if isinstance(value, list) else ()


def _objects(event: _Event, key: str) -> list[_Event]:
    value = event.get(key, [])
    return cast(list[_Event], value) if isinstance(value, list) else []


if __name__ == "__main__":
    main()
