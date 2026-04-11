import json
from dataclasses import dataclass, replace
from pathlib import Path

from voicetotext.downloads.state import ChunkState


_SIDECAR_VERSION = 1


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
        payload = {
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
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Sidecar":
        if not path.exists():
            raise FileNotFoundError(f"Sidecar not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SidecarCorruptError(f"Sidecar {path} is not valid JSON: {exc}")

        version = payload.get("version")
        if version != _SIDECAR_VERSION:
            raise SidecarVersionError(
                f"Sidecar {path} has version {version}, expected {_SIDECAR_VERSION}"
            )

        try:
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
        except (KeyError, TypeError) as exc:
            raise SidecarCorruptError(f"Sidecar {path} missing required field: {exc}")
