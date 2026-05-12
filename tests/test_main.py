import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from voicetotext.__main__ import (
    _apply_hf_cache_env_overrides,
    _apply_torch_env_overrides,
    _configured_model_installed,
    _show_error,
)


class TestConfiguredModelInstalled:
    def test_should_return_true_when_model_present_on_disk(self):
        config = MagicMock()
        config.get.return_value = "base"
        model_manager = MagicMock()
        model_manager.is_installed.return_value = True

        assert _configured_model_installed(config, model_manager) is True
        model_manager.is_installed.assert_called_once_with("base")

    def test_should_return_false_when_model_missing_from_disk(self):
        config = MagicMock()
        config.get.return_value = "small"
        model_manager = MagicMock()
        model_manager.is_installed.return_value = False

        assert _configured_model_installed(config, model_manager) is False
        model_manager.is_installed.assert_called_once_with("small")

    def test_should_default_to_base_when_config_missing(self):
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: default
        model_manager = MagicMock()
        model_manager.is_installed.return_value = True

        _configured_model_installed(config, model_manager)

        model_manager.is_installed.assert_called_once_with("base")


class TestApplyHfCacheEnvOverrides:
    def test_should_set_hf_hub_cache_to_app_models_dir(self, tmp_path):
        models_dir = tmp_path / "models"
        paths = MagicMock()
        paths.models_dir = models_dir

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HUB_CACHE", None)
            _apply_hf_cache_env_overrides(paths)

            assert os.environ.get("HF_HUB_CACHE") == str(models_dir)

    def test_should_not_override_existing_user_value(self, tmp_path):
        paths = MagicMock()
        paths.models_dir = tmp_path / "models"

        with patch.dict(os.environ, {"HF_HUB_CACHE": "C:\\user\\custom\\hf"}):
            _apply_hf_cache_env_overrides(paths)

            assert os.environ.get("HF_HUB_CACHE") == "C:\\user\\custom\\hf"


class TestApplyTorchEnvOverrides:
    """Enable the AOTriton-based flash / mem-efficient attention path
    on AMD ROCm. Without the experimental flag, ``transformers``'
    default ``sdpa`` attention emits two ``UserWarning``s per attention
    layer per forward pass and falls back to a slower path. Setting
    the flag silences the warnings and lets PyTorch use the faster
    backends. The env var is a no-op on NVIDIA / CPU so we set it
    unconditionally rather than detecting hardware before torch is
    importable.
    """

    def test_should_enable_aotriton_experimental_for_amd_attention(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", None)

            _apply_torch_env_overrides()

            assert (
                os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
                == "1"
            )

    def test_should_respect_user_override_of_aotriton_flag(self):
        with patch.dict(
            os.environ,
            {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0"},
        ):
            _apply_torch_env_overrides()

            assert (
                os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
                == "0"
            )


class TestShowError:
    def test_should_write_message_to_stderr(self):
        captured = io.StringIO()
        mock_widgets = MagicMock()

        with (
            patch("sys.stderr", captured),
            patch.dict(sys.modules, {"PySide6.QtWidgets": mock_widgets}),
        ):
            _show_error("something broke")

        assert "something broke" in captured.getvalue()

    def test_should_write_to_stderr_even_when_qt_unavailable(self):
        captured = io.StringIO()

        with (
            patch("sys.stderr", captured),
            patch.dict(sys.modules, {"PySide6.QtWidgets": None}),
        ):
            _show_error("import failed")

        assert "import failed" in captured.getvalue()
