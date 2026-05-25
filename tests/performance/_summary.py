"""ASCII summary table for benchmark results.

Renders the current-run entries grouped by audio fixture, sorted
(model, engine, device). Numbers are kept short — three decimals for
seconds, one decimal for RTF and VRAM — so the table fits a 120-char
terminal.
"""

from tests.performance._recorder import StoredEntry


_COLUMNS: tuple[tuple[str, int], ...] = (
    ("model", 22),
    ("engine", 16),
    ("device", 6),
    ("v", 4),
    ("load_s", 8),
    ("median_s", 10),
    ("stdev_s", 9),
    ("rtf", 8),
    ("vram_mb", 9),
    ("status", 8),
)


def format_summary_table(entries: list[StoredEntry]) -> str:
    if not entries:
        return "(no benchmark entries recorded this run)"

    grouped: dict[str, list[StoredEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.identity.audio, []).append(entry)

    sections: list[str] = []
    for audio_name in sorted(grouped):
        section_entries = sorted(
            grouped[audio_name],
            key=lambda e: (e.identity.model, e.identity.engine, e.identity.device),
        )
        sections.append(_render_section(audio_name, section_entries))

    return "\n\n".join(sections)


def _render_section(audio_name: str, entries: list[StoredEntry]) -> str:
    header_line = _row(_header_values())
    separator = _row(["-" * w for _, w in _COLUMNS])
    lines = [
        f"### Audio: {audio_name} ({entries[0].audio_duration_seconds:.2f}s)",
        header_line,
        separator,
    ]
    for entry in entries:
        lines.append(_row(_entry_values(entry)))
    return "\n".join(lines)


def _header_values() -> list[str]:
    return [name for name, _ in _COLUMNS]


def _entry_values(entry: StoredEntry) -> list[str]:
    result = entry.result
    identity = entry.identity
    median = result.median_seconds
    stdev = result.stdev_seconds
    load = result.load_seconds
    vram = result.vram_peak_mb
    rtf: float | None = None
    if median is not None and median > 0:
        rtf = entry.audio_duration_seconds / median
    return [
        identity.model,
        identity.engine,
        identity.device,
        str(entry.version),
        _format_float(load, 3),
        _format_float(median, 3),
        _format_float(stdev, 3),
        _format_float(rtf, 2),
        _format_float(vram, 1),
        result.status,
    ]


def _format_float(value: float | None, decimals: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def _row(values: list[str]) -> str:
    cells: list[str] = []
    for value, (_, width) in zip(values, _COLUMNS):
        cells.append(value.ljust(width)[:width])
    return " | ".join(cells)
