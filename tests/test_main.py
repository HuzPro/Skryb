import io
import os
import sys
from concurrent.futures import Future
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from voicetotext.__main__ import (
    _apply_hf_cache_env_overrides,
    _apply_torch_env_overrides,
    _configured_model_installed,
    _show_error,
    _try_show_error_dialog,
    main,
    _parse_cli,
    _run,
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


class TestDoctorCli:
    """``voicetotext --doctor`` must print the diagnostic report and
    exit before any Qt or engine code runs, so a user whose engine
    can't load can still get a usable report from the command line.
    """

    def test_should_print_doctor_report_when_doctor_flag_in_argv(self):
        captured = io.StringIO()
        with (
            patch.object(sys, "argv", ["voicetotext", "--doctor"]),
            patch("sys.stdout", captured),
            patch("voicetotext.__main__._run") as mock_run,
            patch(
                "voicetotext.diagnostics.doctor.run_doctor",
                return_value="[Python]\nok\n[Engine backends]\nok",
            ),
            pytest.raises(SystemExit) as exit_info,
        ):
            main()

        output = captured.getvalue()
        assert "[Python]" in output
        assert "[Engine backends]" in output
        assert exit_info.value.code == 0
        mock_run.assert_not_called()

    def test_should_call_run_when_doctor_flag_absent(self, tmp_path):
        prewarm: Future[ModuleType] = Future()
        prewarm.set_result(ModuleType("torch"))
        mutex = MagicMock()
        mutex.acquire.return_value = True
        with (
            patch.object(sys, "argv", ["voicetotext"]),
            patch.dict(os.environ, {"APPDATA": str(tmp_path)}),
            patch("voicetotext.__main__._run") as mock_run,
            patch(
                "voicetotext.core.torch_prewarm.start_torch_prewarm",
                return_value=prewarm,
            ),
            patch(
                "voicetotext.core.single_instance.SingleInstanceMutex",
                return_value=mutex,
            ),
        ):
            main()

        mock_run.assert_called_once()
        mutex.release.assert_called_once()


class TestMaintenanceStartupIsolation:
    @pytest.mark.parametrize(
        ("flag", "target"),
        [
            ("--create-shortcut", "create_shortcut"),
            ("--enable-autostart", "enable_autostart"),
            ("--disable-autostart", "disable_autostart"),
        ],
    )
    def test_should_exit_before_tk_prewarm_mutex_and_run(
        self,
        flag: str,
        target: str,
    ) -> None:
        with (
            patch.object(sys, "argv", ["voicetotext", flag]),
            patch(f"voicetotext.core.dev_install.{target}", return_value=(True, "ok")),
            patch("voicetotext.__main__._run") as run,
            patch("voicetotext.core.torch_prewarm.start_torch_prewarm") as prewarm,
            patch("voicetotext.core.single_instance.SingleInstanceMutex") as mutex,
            patch("tkinter.Tk") as tk,
            pytest.raises(SystemExit) as exit_info,
        ):
            main()

        assert exit_info.value.code == 0
        run.assert_not_called()
        prewarm.assert_not_called()
        mutex.assert_not_called()
        tk.assert_not_called()

    def test_should_reject_multiple_maintenance_flags(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            _parse_cli(["--doctor", "--create-shortcut"])

        assert exit_info.value.code == 2


class TestBindingStartupOrder:
    def test_should_overlap_hardware_independent_setup_before_detection(self) -> None:
        timeline: list[str] = []
        root = MagicMock()
        config = MagicMock()
        config.has_persisted_value.return_value = True
        config.get.return_value = "base"
        manager = MagicMock()
        manager.is_installed.return_value = True
        registry = MagicMock(name="registry")
        hardware = MagicMock(name="hardware")
        app = MagicMock()
        paths = SimpleNamespace(
            config_path=Path("config.json"),
            models_dir=Path("models"),
            is_first_run=lambda: False,
        )
        future: Future[ModuleType] = Future()

        with (
            patch("tkinter.Tk", side_effect=lambda: timeline.append("tk") or root),
            patch(
                "voicetotext.core.json_config_adapter.JsonConfigAdapter",
                side_effect=lambda _path: timeline.append("config-constructed") or config,
            ),
            patch.object(config, "load", side_effect=lambda: timeline.append("config-loaded")),
            patch(
                "voicetotext.downloads.download_manager.DownloadManager",
                side_effect=lambda **_kwargs: timeline.append("downloads") or MagicMock(),
            ),
            patch(
                "voicetotext.transcription.engine_factory.create_model_manager",
                side_effect=lambda *_args: timeline.append("manager") or manager,
            ),
            patch(
                "voicetotext.transcription.engine_factory.build_engine_registry",
                side_effect=lambda: timeline.append("registry") or registry,
            ),
            patch(
                "voicetotext.__main__._consume_torch_prewarm",
                side_effect=lambda _future: timeline.append("consume"),
            ),
            patch(
                "voicetotext.transcription.torch_hardware_detector.TorchHardwareDetector.detect",
                side_effect=lambda: timeline.append("detect") or hardware,
            ) as detect,
            patch(
                "voicetotext.bootstrap.build_app",
                side_effect=lambda *_args, **_kwargs: timeline.append("build-app") or app,
            ) as build_app,
        ):
            _run(paths, future)

        assert timeline == [
            "tk",
            "config-constructed",
            "config-loaded",
            "downloads",
            "manager",
            "registry",
            "consume",
            "detect",
            "build-app",
        ]
        detect.assert_called_once_with()
        assert build_app.call_args.kwargs["registry"] is registry
        assert build_app.call_args.kwargs["hardware"] is hardware
        app.run.assert_called_once_with()


class TestShowError:
    def test_should_write_message_to_stderr(self):
        captured = io.StringIO()

        with (
            patch("sys.stderr", captured),
            patch("voicetotext.__main__._try_show_error_dialog"),
        ):
            _show_error("something broke")

        assert "something broke" in captured.getvalue()

    def test_dialog_should_swallow_errors_when_tk_unavailable(self):
        # Best-effort: a failure constructing the Tk dialog must not
        # propagate out of the crash-reporting path.
        with patch("tkinter.Tk", side_effect=RuntimeError("no display")):
            _try_show_error_dialog("import failed")
