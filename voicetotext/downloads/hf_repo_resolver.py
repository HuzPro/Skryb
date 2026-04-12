import fnmatch
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


_IGNORE_PATTERNS = (
    "*.bin",            # PyTorch pickle
    "*.h5",             # TensorFlow
    "*.msgpack",        # Flax
    "*.fp32-*",         # FP32 sharded variants
    "*.index.fp32.json",
    "README*",
    ".gitattributes",
    ".gitignore",
)

_HF_RESOLVE_URL = "https://huggingface.co/{repo_id}/resolve/{sha}/{filename}"


@dataclass(frozen=True)
class ResolvedFile:
    filename: str
    size: int
    url: str


class _RepoSibling(Protocol):
    rfilename: str
    size: int | None


class _RepoInfo(Protocol):
    sha: str
    siblings: Iterable[_RepoSibling]


class HfApiProtocol(Protocol):
    def repo_info(self, repo_id: str, files_metadata: bool) -> _RepoInfo: ...


class HfRepoResolver:
    """Adapter around :class:`huggingface_hub.HfApi` that hands us
    (commit_sha, list-of-files-with-urls) tuples without touching tqdm.
    """

    def __init__(self, api: HfApiProtocol):
        self._api = api

    def resolve(self, repo_id: str) -> tuple[str, list[ResolvedFile]]:
        info = self._api.repo_info(repo_id, files_metadata=True)
        commit_sha = info.sha
        files: list[ResolvedFile] = []
        for sibling in info.siblings:
            filename = sibling.rfilename
            size = sibling.size
            if size is None:
                continue
            if self._is_ignored(filename):
                continue
            files.append(
                ResolvedFile(
                    filename=filename,
                    size=size,
                    url=_HF_RESOLVE_URL.format(
                        repo_id=repo_id, sha=commit_sha, filename=filename
                    ),
                )
            )
        return commit_sha, files

    @staticmethod
    def _is_ignored(filename: str) -> bool:
        return any(fnmatch.fnmatch(filename, pattern) for pattern in _IGNORE_PATTERNS)
