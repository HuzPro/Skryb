from unittest.mock import MagicMock

from voicetotext.downloads.hf_repo_resolver import HfRepoResolver, ResolvedFile


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
        commit_sha, files = resolver.resolve("openai/whisper-tiny")

        assert commit_sha == "abc123"
        assert files == [
            ResolvedFile(
                filename="config.json",
                size=1983,
                url="https://huggingface.co/openai/whisper-tiny/resolve/abc123/config.json",
            ),
            ResolvedFile(
                filename="model.safetensors",
                size=151061672,
                url="https://huggingface.co/openai/whisper-tiny/resolve/abc123/model.safetensors",
            ),
        ]

    def test_should_call_repo_info_with_files_metadata(self):
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(sha="x", files=[])

        HfRepoResolver(api=api).resolve("openai/whisper-tiny")

        api.repo_info.assert_called_once_with(
            "openai/whisper-tiny", files_metadata=True
        )


class TestHfRepoResolverIgnorePatterns:
    def test_should_exclude_pytorch_bin_but_include_ct2_model_bin(self):
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

        _, files = HfRepoResolver(api=api).resolve("openai/whisper-tiny")

        names = [f.filename for f in files]
        assert names == ["config.json", "model.safetensors", "model.bin"]

    def test_should_include_model_bin_when_resolving_systran_faster_whisper_repo(self):
        api = MagicMock()
        api.repo_info.return_value = _fake_repo_info(
            sha="sha",
            files=[
                _fake_sibling("config.json", 100),
                _fake_sibling("model.bin", 200),
                _fake_sibling("tokenizer.json", 300),
                _fake_sibling("vocabulary.txt", 400),
            ],
        )

        _, files = HfRepoResolver(api=api).resolve("Systran/faster-whisper-tiny")

        names = [f.filename for f in files]
        assert names == [
            "config.json", "model.bin", "tokenizer.json", "vocabulary.txt",
        ]
        assert any(
            f.filename == "model.bin" and f.size == 200 for f in files
        )

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

        _, files = HfRepoResolver(api=api).resolve("openai/whisper-tiny")

        assert [f.filename for f in files] == ["config.json"]

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

        _, files = HfRepoResolver(api=api).resolve("openai/whisper-tiny")

        assert [f.filename for f in files] == ["model.safetensors"]

    def test_should_skip_files_with_no_size(self):
        api = MagicMock()
        sibling = MagicMock()
        sibling.rfilename = "mysterious"
        sibling.size = None
        api.repo_info.return_value = _fake_repo_info(sha="sha", files=[sibling])

        _, files = HfRepoResolver(api=api).resolve("openai/whisper-tiny")

        assert files == []
