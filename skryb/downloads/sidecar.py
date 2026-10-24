import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from skryb.downloads.state import ChunkState


_SIDECAR_VERSION = 1
_SIDECAR_TMP_SUFFIX = ".tmp"
_SIDECAR_BAK_SUFFIX = ".bak"


class SidecarCorruptError(Exception):
    """Raised when a sidecar file exists but is unreadable or malformed."""


class SidecarVersionError(Exception):
    """Raised when a sidecar file uses an unsupported schema version."""


@dataclass(frozen=True)
class Sidecar:
    url: str
    etag: str | None
    content_length: int
    chunks: tuple[ChunkState, ...]

    def __init__(
        self,
        url: str,
        etag: str | None,
        content_length: int,
        chunks: list[ChunkState] | tuple[ChunkState, ...],
    ) -> None:
        # Normalize chunks to tuple so the dataclass stays hashable.
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "etag", etag)
        object.__setattr__(self, "content_length", content_length)
        object.__setattr__(self, "chunks", tuple(chunks))

    @property
    def bytes_done(self) -> int:
        return sum(chunk.bytes_done for chunk in self.chunks)

    @property
    def is_complete(self) -> bool:
        return all(chunk.is_complete for chunk in self.chunks)

    def with_chunk_progress(self, index: int, bytes_done: int) -> "Sidecar":
        new_chunks = tuple(
            replace(chunk, bytes_done=bytes_done) if chunk.index == index else chunk
            for chunk in self.chunks
        )
        return Sidecar(
            url=self.url,
            etag=self.etag,
            content_length=self.content_length,
            chunks=new_chunks,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._build_payload()
        self._atomic_write_json(path, payload)

    def _build_payload(self) -> dict[str, object]:
        return {
            "version": _SIDECAR_VERSION,
            "url": self.url,
            "etag": self.etag,
            "content_length": self.content_length,
            "chunks": [
                {
                    "index": c.index,
                    "start": c.start,
                    "end": c.end,
                    "bytes_done": c.bytes_done,
                }
                for c in self.chunks
            ],
        }

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
        tmp = path.with_suffix(path.suffix + _SIDECAR_TMP_SUFFIX)
        try:
            Sidecar._write_and_swap(path=path, tmp=tmp, payload=payload)
        except OSError:
            if tmp.exists():
                tmp.unlink()
            raise

    @staticmethod
    def _write_and_swap(path: Path, tmp: Path, payload: dict[str, object]) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        with open(tmp, "wb") as fd:
            fd.write(data)
            fd.flush()
            os.fsync(fd.fileno())
        if path.exists():
            bak = path.with_suffix(path.suffix + _SIDECAR_BAK_SUFFIX)
            os.replace(path, bak)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "Sidecar":
        if not path.exists():
            raise FileNotFoundError(f"Sidecar not found: {path}")
        payload = cls._load_json_payload(path)
        return cls._from_payload(path, payload)

    @staticmethod
    def _load_json_payload(path: Path) -> dict[str, object]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            backup_payload = Sidecar._read_backup_payload(path)
            if backup_payload is not None:
                return backup_payload
            raise SidecarCorruptError(f"Sidecar {path} is not valid JSON: {exc}")

    @staticmethod
    def _read_backup_payload(path: Path) -> dict[str, object] | None:
        backup = path.with_suffix(path.suffix + _SIDECAR_BAK_SUFFIX)
        if not backup.exists():
            return None
        try:
            return json.loads(backup.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    @classmethod
    def _from_payload(cls, path: Path, payload: dict[str, object]) -> "Sidecar":
        version = payload.get("version")
        if version != _SIDECAR_VERSION:
            raise SidecarVersionError(
                f"Sidecar {path} has version {version}, expected {_SIDECAR_VERSION}"
            )
        try:
            return cls._build_from_fields(payload)
        except (KeyError, TypeError) as exc:
            raise SidecarCorruptError(f"Sidecar {path} missing required field: {exc}")

    @classmethod
    def _build_from_fields(cls, payload: dict[str, object]) -> "Sidecar":
        chunks = tuple(
            ChunkState(
                index=c["index"],
                start=c["start"],
                end=c["end"],
                bytes_done=c["bytes_done"],
            )
            for c in payload["chunks"]
        )
        return cls(
            url=payload["url"],
            etag=payload.get("etag"),
            content_length=payload["content_length"],
            chunks=chunks,
        )
