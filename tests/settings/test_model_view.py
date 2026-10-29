import pytest

from skryb.settings.model_view import ModelView
from skryb.transcription.model_info import ModelInfo


class TestModelViewFromInfo:
    def test_should_copy_name_description_and_display_size(self):
        info = ModelInfo(
            name="parakeet-ultra",
            size_mb=2510,
            description="Best accuracy",
            repo_id="nvidia/parakeet-tdt-0.6b-v3",
        )

        view = ModelView.from_info(info, installed=True)

        assert view.name == "parakeet-ultra"
        assert view.description == "Best accuracy"
        assert view.display_size == "2.5 GB"

    def test_should_carry_installed_flag(self):
        info = ModelInfo(name="example", size_mb=151, description="Fastest")

        installed_view = ModelView.from_info(info, installed=True)
        uninstalled_view = ModelView.from_info(info, installed=False)

        assert installed_view.installed is True
        assert uninstalled_view.installed is False

    def test_should_format_gigabyte_sizes(self):
        info = ModelInfo(name="example", size_mb=1620, description="x")

        view = ModelView.from_info(info, installed=False)

        assert view.display_size == "1.6 GB"


class TestModelViewImmutability:
    def test_should_reject_mutation(self):
        view = ModelView(
            name="example",
            display_size="151 MB",
            description="Fastest",
            installed=False,
        )

        with pytest.raises(Exception):
            view.installed = True  # type: ignore[misc]
