from unittest.mock import MagicMock

import pytest

from skryb.downloads.hf_repo_resolver import (
    HfRepoResolver,
    ResolvedFile,
    RevisionMismatchError,
)


_PINNED_SHA = "de0369c8a68025b7f6e86387b6eb5a3b369787c8"
_PINNED_REPO = "nyralabs/CrisperWhisper2.0_turbo"
_OTHER_SHA = "0000000000000000000000000000000000000000"


def _fake_sibling(name: str, size: int):
    sibling = MagicMock()
    sibling.rfilename = name
    sibling.size = size
    return sibling


def _fake_repo_info(sha: str, files: list):
    info = MagicMock()
    info.sha = sha
    info.siblings = files
    return info


class TestHfRepoResolverListFiles:
    def test_should_return_files_with_sizes_and_urls(self):
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(
            sha="abc123",
            files=[
                _fake_sibling("config.json", 1983),
                _fake_sibling("model.safetensors", 151061672),
            ],
        )

        resolver = HfRepoResolver(api=api)
        resolved = resolver.resolve("nvidia/parakeet-tdt-0.6b-v3")

        assert resolved.canonical_commit == "abc123"
        assert resolved.files == [
            ResolvedFile(
                filename="config.json",
                size=1983,
                url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/abc123/config.json",
            ),
            ResolvedFile(
                filename="model.safetensors",
                size=151061672,
                url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/abc123/model.safetensors",
            ),
        ]

    def test_should_call_repo_info_with_files_metadata(self):
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(sha="x", files=[])

        HfRepoResolver(api=api).resolve("nvidia/parakeet-tdt-0.6b-v3")

        api.repo_info.assert_called_once_with(
            "nvidia/parakeet-tdt-0.6b-v3", revision=None, files_metadata=True
        )


class TestHfRepoResolverRevision:
    """An unpinned model tracks ``main``; a pinned one gets exactly the
    commit it asked for, or an error. HuggingFace reads ``revision=None``
    as "default branch" - an empty string would select the ``/revision/``
    endpoint and resolve nothing.
    """

    def test_should_track_main_when_no_revision_is_requested(self):
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(sha="head", files=[])

        resolved = HfRepoResolver(api=api).resolve("nvidia/parakeet-tdt-0.6b-v3")

        assert api.repo_info.call_args.kwargs["revision"] is None
        assert resolved.requested_revision == ""
        assert resolved.canonical_commit == "head"

    def test_should_request_the_pinned_commit(self):
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(sha=_PINNED_SHA, files=[])

        resolved = HfRepoResolver(api=api).resolve(
            _PINNED_REPO, _PINNED_SHA,
        )

        assert api.repo_info.call_args.kwargs["revision"] == _PINNED_SHA
        assert resolved.requested_revision == _PINNED_SHA
        assert resolved.canonical_commit == _PINNED_SHA

    def test_should_point_file_urls_at_the_pinned_commit(self):
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(
            sha=_PINNED_SHA, files=[_fake_sibling("config.json", 10)],
        )

        resolved = HfRepoResolver(api=api).resolve(
            _PINNED_REPO, _PINNED_SHA,
        )

        assert resolved.files[0].url == (
            f"https://huggingface.co/{_PINNED_REPO}/resolve/"
            f"{_PINNED_SHA}/config.json"
        )

    def test_should_raise_when_pinned_commit_resolves_elsewhere(self):
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(sha=_OTHER_SHA, files=[])

        with pytest.raises(RevisionMismatchError) as exc_info:
            HfRepoResolver(api=api).resolve(_PINNED_REPO, _PINNED_SHA)

        assert _PINNED_SHA in str(exc_info.value)
        assert _OTHER_SHA in str(exc_info.value)

    def test_should_accept_a_branch_resolving_to_its_head_commit(self):
        # Only a full 40-hex request is checkable; a branch legitimately
        # resolves to whatever commit it points at today.
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(sha=_OTHER_SHA, files=[])

        resolved = HfRepoResolver(api=api).resolve("org/repo", "refs/pr/1")

        assert resolved.canonical_commit == _OTHER_SHA


class TestHfRepoResolverIgnorePatterns:
    def test_should_exclude_framework_specific_weights_but_keep_the_rest(self):
        # Only the explicit blocklist (pytorch pickle, TF, Flax) is dropped;
        # everything else - including a bare ``model.bin`` - passes through.
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(
            sha="sha",
            files=[
                _fake_sibling("config.json", 100),
                _fake_sibling("pytorch_model.bin", 200),
                _fake_sibling("tf_model.h5", 300),
                _fake_sibling("flax_model.msgpack", 400),
                _fake_sibling("model.safetensors", 500),
                _fake_sibling("model.bin", 600),
            ],
        )

        files = HfRepoResolver(api=api).resolve("nvidia/parakeet-tdt-0.6b-v3").files

        names = [f.filename for f in files]
        assert names == ["config.json", "model.safetensors", "model.bin"]

    def test_should_exclude_readme_and_gitattributes(self):
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(
            sha="sha",
            files=[
                _fake_sibling("README.md", 10),
                _fake_sibling(".gitattributes", 20),
                _fake_sibling("config.json", 30),
            ],
        )

        files = HfRepoResolver(api=api).resolve("nvidia/parakeet-tdt-0.6b-v3").files

        assert [f.filename for f in files] == ["config.json"]

    def test_should_fetch_only_the_weights_and_notice_from_a_bark_repo(self):
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(
            sha="sha",
            files=[
                _fake_sibling("NOTICE", 603),
                _fake_sibling("bark_frontier.png", 160476),
                _fake_sibling("bark_loader.py", 5820),
                _fake_sibling("bark_mlx.py", 5011),
                _fake_sibling("coreml/ane/BarkEncoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin", 189133632),
                _fake_sibling("coreml/ane/decoder.npz", 11168286),
                _fake_sibling("mlx/config.json", 237042),
                _fake_sibling("model.bark", 163398645),
                _fake_sibling("runtime/transcribe_bark.py", 1817),
            ],
        )

        files = HfRepoResolver(api=api).resolve("ConwayResearch/Underdog-Bark-3.0").files

        assert [f.filename for f in files] == ["NOTICE", "model.bark"]

    def test_should_exclude_fp32_sharded_variants(self):
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(
            sha="sha",
            files=[
                _fake_sibling("model.safetensors", 100),
                _fake_sibling("model.fp32-00001-of-00002.safetensors", 200),
                _fake_sibling("model.safetensors.index.fp32.json", 300),
            ],
        )

        files = HfRepoResolver(api=api).resolve("nvidia/parakeet-tdt-0.6b-v3").files

        assert [f.filename for f in files] == ["model.safetensors"]

    def test_should_skip_files_with_no_size(self):
        api = MagicMock()
        sibling = MagicMock()
        sibling.rfilename = "mysterious"
        sibling.size = None
        api.repo_info.return_value = _fake_repo_info(sha="sha", files=[sibling])

        files = HfRepoResolver(api=api).resolve("nvidia/parakeet-tdt-0.6b-v3").files

        assert files == []
