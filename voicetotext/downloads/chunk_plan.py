from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkRange:
    index: int
    start: int  # inclusive
    end: int    # inclusive (HTTP Range semantics)

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def plan_chunks(total_size: int, n_chunks: int) -> list[ChunkRange]:
    """Split a byte range ``[0, total_size)`` into ``n_chunks`` contiguous pieces.

    The last chunk absorbs any remainder. If *total_size* is smaller than
    *n_chunks*, a single chunk covering the whole file is returned.
    """
    if total_size <= 0:
        raise ValueError(f"total_size must be positive, got {total_size}")
    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive, got {n_chunks}")

    effective_chunks = n_chunks if total_size >= n_chunks else 1
    base_size = total_size // effective_chunks

    ranges: list[ChunkRange] = []
    cursor = 0
    for i in range(effective_chunks):
        is_last = i == effective_chunks - 1
        length = total_size - cursor if is_last else base_size
        ranges.append(ChunkRange(index=i, start=cursor, end=cursor + length - 1))
        cursor += length

    return ranges
