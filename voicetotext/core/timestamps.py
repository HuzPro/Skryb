from datetime import UTC, datetime


def utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def utc_now() -> datetime:
    return datetime.now(UTC)
