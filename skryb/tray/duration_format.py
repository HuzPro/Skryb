def format_duration(seconds: float) -> str:
    """Render seconds as ``M:SS`` (under 1h) or ``H:MM:SS`` (over).

    Used by both the recording-cap notification (``tray.app``) and the
    live elapsed-time tooltip (``tray.win32_tray_adapter``) so the two
    surfaces agree on formatting. Always rounds down: the elapsed
    timer should never claim more time has passed than actually has.
    """
    total = int(seconds)
    if total < 0:
        total = 0
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
