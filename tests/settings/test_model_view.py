import pytest

from voicetotext.settings.model_view import ModelView
from voicetotext.transcription.model_info import ModelInfo


class TestModelViewFromInfo:
    def test_should_copy_name_description_and_display_size(self):
        info = ModelInfo(
            name="base",
            size_mb=290,
            description="Good balance",
            repo_id="openai/whisper-base",
        )

        view = ModelView.from_info(info, installed=True)

        assert view.name == "base"
        assert view.description == "Good balance"
        assert view.display_size == "290 MB"

    def test_should_carry_installed_flag(self):
        info = ModelInfo(name="tiny", size_mb=151, description="Fastest")

        installed_view = ModelView.from_info(info, installed=True)
        uninstalled_view = ModelView.from_info(info, installed=False)

        assert installed_view.installed is True
        assert uninstalled_view.installed is False

    def test_should_format_gigabyte_sizes(self):
        info = ModelInfo(name="large-v3-turbo", size_mb=1620, description="x")

        view = ModelView.from_info(info, installed=False)

        assert view.display_size == "1.6 GB"


class TestModelViewImmutability:
    def test_should_reject_mutation(self):
        view = ModelView(
            name="tiny",
            display_size="151 MB",
            description="Fastest",
            installed=False,
        )

        with pytest.raises(Exception):
            view.installed = True  # type: ignore[misc]
