from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_mb: int
    description: str

    @property
    def display_size(self) -> str:
        if self.size_mb >= 1000:
            return f"{self.size_mb / 1000:.1f} GB"
        return f"{self.size_mb} MB"

    @property
    def repo_id(self) -> str:
        return f"Systran/faster-whisper-{self.name}"


AVAILABLE_MODELS = {
    "tiny": ModelInfo(
        name="tiny", size_mb=75, description="Fastest, least accurate"
    ),
    "base": ModelInfo(
        name="base", size_mb=142, description="Good balance (recommended)"
    ),
    "small": ModelInfo(
        name="small", size_mb=466, description="More accurate, slower"
    ),
    "medium": ModelInfo(
        name="medium", size_mb=1500, description="High accuracy"
    ),
    "large-v3": ModelInfo(
        name="large-v3", size_mb=2900, description="Best accuracy, slowest"
    ),
}
