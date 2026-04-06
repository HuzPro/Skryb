from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_mb: int
    description: str
    repo_id: str = ""

    @property
    def display_size(self) -> str:
        if self.size_mb >= 1000:
            return f"{self.size_mb / 1000:.1f} GB"
        return f"{self.size_mb} MB"


# TODO(deletion): Remove FASTER_WHISPER_MODELS — standardized on transformers engine.
FASTER_WHISPER_MODELS: dict[str, ModelInfo] = {
    "tiny": ModelInfo(
        name="tiny", size_mb=75, description="Fastest, least accurate",
        repo_id="Systran/faster-whisper-tiny",
    ),
    "base": ModelInfo(
        name="base", size_mb=142, description="Good balance (recommended)",
        repo_id="Systran/faster-whisper-base",
    ),
    "small": ModelInfo(
        name="small", size_mb=466, description="More accurate, slower",
        repo_id="Systran/faster-whisper-small",
    ),
    "medium": ModelInfo(
        name="medium", size_mb=1500, description="High accuracy",
        repo_id="Systran/faster-whisper-medium",
    ),
    "large-v3": ModelInfo(
        name="large-v3", size_mb=2900, description="Best accuracy, slowest",
        repo_id="Systran/faster-whisper-large-v3",
    ),
}

TRANSFORMERS_WHISPER_MODELS: dict[str, ModelInfo] = {
    "tiny": ModelInfo(
        name="tiny", size_mb=151, description="Fastest, least accurate",
        repo_id="openai/whisper-tiny",
    ),
    "base": ModelInfo(
        name="base", size_mb=290, description="Good balance (recommended)",
        repo_id="openai/whisper-base",
    ),
    "small": ModelInfo(
        name="small", size_mb=967, description="More accurate, slower",
        repo_id="openai/whisper-small",
    ),
    "medium": ModelInfo(
        name="medium", size_mb=1500, description="High accuracy",
        repo_id="openai/whisper-medium",
    ),
    "large-v3": ModelInfo(
        name="large-v3", size_mb=3090, description="Best accuracy, slowest",
        repo_id="openai/whisper-large-v3",
    ),
}

AVAILABLE_MODELS = TRANSFORMERS_WHISPER_MODELS
