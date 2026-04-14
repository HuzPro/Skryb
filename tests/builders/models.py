from voicetotext.transcription.model_info import ModelInfo


class ModelInfoBuilder:
    """Fluent builder for :class:`ModelInfo`."""

    def __init__(self) -> None:
        self._name: str = "base"
        self._size_mb: int = 290
        self._description: str = "Good balance"
        self._repo_id: str = "openai/whisper-base"

    def named(self, name: str) -> "ModelInfoBuilder":
        self._name = name
        return self

    def with_size_mb(self, size_mb: int) -> "ModelInfoBuilder":
        self._size_mb = size_mb
        return self

    def described_as(self, description: str) -> "ModelInfoBuilder":
        self._description = description
        return self

    def from_repo(self, repo_id: str) -> "ModelInfoBuilder":
        self._repo_id = repo_id
        return self

    def build(self) -> ModelInfo:
        return ModelInfo(
            name=self._name,
            size_mb=self._size_mb,
            description=self._description,
            repo_id=self._repo_id,
        )


def a_model_info() -> ModelInfoBuilder:
    return ModelInfoBuilder()
