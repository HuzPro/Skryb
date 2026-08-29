import fnmatch
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


_IGNORE_PATTERNS = (
    "pytorch_model.bin",  # PyTorch pickle weights (we ship safetensors instead)
    "*.h5",               # TensorFlow
    "*.msgpack",          # Flax
    "*.fp32-*",           # FP32 sharded variants
    "*.index.fp32.json",
    "README*",
    ".gitattributes",
    ".gitignore",
)

_HF_RESOLVE_URL = "https://huggingface.co/{repo_id}/resolve/{sha}/{filename}"

_FULL_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class RevisionMismatchError(Exception):
    """A pinned commit resolved to a different commit upstream.

    Only a full 40-hex request can be checked this way: a branch or tag
    legitimately resolves to whatever commit it points at today, while a
    commit SHA that comes back as a *different* commit means the request
    was not honoured and the bytes about to be fetched are not the audited
    ones.
    """


@dataclass(frozen=True)
class ResolvedFile:
    filename: str
    size: int
    url: str


@dataclass(frozen=True)
class ResolvedRepo:
    """One repo revision, resolved to a commit and a file list.

    ``requested_revision`` is what the caller asked for (empty means
    "track ``main``"); ``canonical_commit`` is the commit HuggingFace
    resolved that request to, and the commit every file URL points at.
    """
    requested_revision: str
    canonical_commit: str
    files: list[ResolvedFile]


class _RepoSibling(Protocol):
    rfilename: str
    size: int | None


class _RepoInfo(Protocol):
    sha: str
    siblings: Iterable[_RepoSibling]


class HfApiProtocol(Protocol):
    def repo_info(
        self, repo_id: str, revision: str | None, files_metadata: bool,
    ) -> _RepoInfo: ...


class HfRepoResolver:
    """Adapter around :class:`huggingface_hub.HfApi` that hands us a
    :class:`ResolvedRepo` without touching tqdm.
    """

    def __init__(self, api: HfApiProtocol) -> None:
        self._api = api

    def resolve(self, repo_id: str, revision: str = "") -> ResolvedRepo:
        # HuggingFace reads ``None`` as "default branch"; an empty string
        # would select the ``/revision/`` endpoint and fail to resolve.
        info = self._api.repo_info(
            repo_id, revision=revision or None, files_metadata=True,
        )
        canonical_commit = info.sha
        self._assert_pin_honoured(repo_id, revision, canonical_commit)
        return ResolvedRepo(
            requested_revision=revision,
            canonical_commit=canonical_commit,
            files=self._resolved_files(repo_id, canonical_commit, info),
        )

    def _resolved_files(
        self, repo_id: str, commit_sha: str, info: _RepoInfo,
    ) -> list[ResolvedFile]:
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
        return files

    @staticmethod
    def _assert_pin_honoured(
        repo_id: str, revision: str, canonical_commit: str,
    ) -> None:
        if not _FULL_COMMIT_SHA.fullmatch(revision):
            return
        if revision == canonical_commit:
            return
        raise RevisionMismatchError(
            f"{repo_id} pinned to {revision} but resolved to "
            f"{canonical_commit}"
        )

    @staticmethod
    def _is_ignored(filename: str) -> bool:
        return any(fnmatch.fnmatch(filename, pattern) for pattern in _IGNORE_PATTERNS)
