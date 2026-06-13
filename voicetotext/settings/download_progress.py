from dataclasses import dataclass


@dataclass(frozen=True)
class DownloadProgressSnapshot:
    """Per-row download state exposed to the model-list view.

    Framework-agnostic so both the Qt and Tk model-row views can render
    it without pulling in a GUI toolkit dependency.
    """

    state: str
    bytes_done: int
    bytes_total: int
    speed_bps: float
    eta_seconds: float
    error_message: str | None


def percent_for(snapshot: DownloadProgressSnapshot) -> int:
    """Clamp a snapshot's byte ratio to an integer 0-100 percentage."""
    if snapshot.bytes_total <= 0:
        return 0
    pct = int((snapshot.bytes_done / snapshot.bytes_total) * 100)
    if pct < 0:
        return 0
    if pct > 100:
        return 100
    return pct
