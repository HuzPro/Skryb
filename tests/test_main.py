import io
import os
import sys
from concurrent.futures import Future
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from voicetotext.transcription.license_policy import LicensePolicy

# These reach the real startup path: it builds ``AppPaths`` from ``APPDATA``
# and imports ``voicetotext.bootstrap``, which touches ``ctypes.windll`` at
# module level. Neither exists off Windows.
windows_only = pytest.mark.skipif(
    sys.platform != "win32",
    reason="startup builds AppPaths from APPDATA and imports the Win32 bootstrap",
)
from voicetotext.__main__ import (
    _apply_hf_cache_env_overrides,
    _apply_torch_env_overrides,
    _configured_model_installed,
    _show_error,
    _stderr_is_console,
    _try_show_error_dialog,
    main,
    _parse_cli,
    _run,
)


class TestConfiguredModelInstalled:
    def test_should_return_true_when_model_present_on_disk(self):
        config = MagicMock()
        config.get.return_value = "parakeet-tdt-0.6b-v3"
        model_manager = MagicMock()
        model_manager.is_installed.return_value = True

        assert _configured_model_installed(config, model_manager) is True
        model_manager.is_installed.assert_called_once_with(
            "parakeet-tdt-0.6b-v3",
        )

    def test_should_return_false_when_model_missing_from_disk(self):
        config = MagicMock()
        config.get.return_value = "nemotron-speech-streaming-en-0.6b"
        model_manager = MagicMock()
        model_manager.is_installed.return_value = False

        assert _configured_model_installed(config, model_manager) is False
        model_manager.is_installed.assert_called_once_with(
            "nemotron-speech-streaming-en-0.6b",
        )

    def test_should_default_to_current_parakeet_when_config_missing(self):
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: default
        model_manager = MagicMock()
        model_manager.is_installed.return_value = True

        _configured_model_installed(config, model_manager)

        model_manager.is_installed.assert_called_once_with(
            "parakeet-tdt-0.6b-v3",
        )


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
    on AMD ROCm. The env var is a no-op on NVIDIA / CPU, so it is set
    unconditionally before torch is importable.
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

    @windows_only
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
        paths = mock_run.call_args.args[0]
        assert paths.session_log_path.parent.parent == paths.diagnostics_dir
        assert paths.session_log_path.exists()
        mutex.release.assert_called_once()


class TestMaintenanceStartupIsolation:
    @windows_only
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
    @windows_only
    def test_should_overlap_hardware_independent_setup_before_detection(self) -> None:
        timeline: list[str] = []
        root = MagicMock()
        config = MagicMock()
        config.has_persisted_value.return_value = True
        config.get.return_value = "parakeet-tdt-0.6b-v3"
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
                "voicetotext.transcription.model_info.migrate_removed_models",
                side_effect=lambda _config: timeline.append("models-migrated"),
            ),
            patch(
                "voicetotext.transcription.model_info.migrate_engine_field",
                side_effect=lambda _config: timeline.append("engine-migrated"),
            ),
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
            "models-migrated",
            "engine-migrated",
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


class TestOneLicensePolicyForTheProcess:
    """First-run setup and ``build_app`` each build their own model
    manager. Two policy instances would give them separate acceptance
    state, so a licence accepted during setup would be asked for again -
    or worse, silently missing - once the app proper starts.
    """

    def _run_with_first_time_setup(self, tmp_path: Path) -> tuple:
        config = MagicMock()
        config.has_persisted_value.return_value = True
        config.get.return_value = "parakeet-tdt-0.6b-v3"
        manager = MagicMock()
        manager.is_installed.return_value = True
        paths = SimpleNamespace(
            config_path=Path("config.json"),
            models_dir=tmp_path,
            dictionary_path=tmp_path / "dictionary.json",
            is_first_run=lambda: True,
        )
        future: Future[ModuleType] = Future()

        with (
            patch("tkinter.Tk", return_value=MagicMock()),
            patch(
                "voicetotext.core.json_config_adapter.JsonConfigAdapter",
                return_value=config,
            ),
            patch("voicetotext.transcription.model_info.migrate_removed_models"),
            patch("voicetotext.transcription.model_info.migrate_engine_field"),
            patch("voicetotext.downloads.download_manager.DownloadManager"),
            patch(
                "voicetotext.transcription.engine_factory.create_model_manager",
                return_value=manager,
            ) as create_model_manager,
            patch("voicetotext.transcription.engine_factory.build_engine_registry"),
            patch("voicetotext.__main__._consume_torch_prewarm"),
            patch(
                "voicetotext.transcription.torch_hardware_detector."
                "TorchHardwareDetector.detect",
            ),
            patch("voicetotext.__main__._run_first_time_setup") as first_run_setup,
            patch("voicetotext.bootstrap.build_app") as build_app,
        ):
            _run(paths, future)

        return create_model_manager, first_run_setup, build_app

    @windows_only
    def test_should_hand_the_same_policy_to_every_collaborator(
        self, tmp_path: Path,
    ) -> None:
        create_model_manager, first_run_setup, build_app = (
            self._run_with_first_time_setup(tmp_path)
        )

        policy = create_model_manager.call_args.args[2]
        assert isinstance(policy, LicensePolicy)
        assert first_run_setup.call_args.args[-1] is policy
        assert build_app.call_args.kwargs["license_policy"] is policy


class TestStderrIsConsole:
    def test_should_report_no_console_when_stderr_is_missing(self):
        with patch("sys.stderr", None):
            assert _stderr_is_console() is False

    def test_should_report_a_console_for_an_interactive_stderr(self):
        interactive = MagicMock()
        interactive.isatty.return_value = True

        with patch("sys.stderr", interactive):
            assert _stderr_is_console() is True


class TestShowError:
    def test_should_write_message_to_stderr(self):
        captured = io.StringIO()

        with (
            patch("sys.stderr", captured),
            patch("voicetotext.__main__._try_show_error_dialog"),
        ):
            _show_error("something broke")

        assert "something broke" in captured.getvalue()

    def test_should_still_show_dialog_when_stderr_is_missing(self):
        # pythonw (GUI launches, the Start Menu shortcut) has no stdio.
        with (
            patch("sys.stderr", None),
            patch("voicetotext.__main__._try_show_error_dialog") as dialog,
        ):
            _show_error("something broke")

        dialog.assert_called_once_with("something broke")

    def test_dialog_should_swallow_errors_when_tk_unavailable(self):
        # Best-effort: a failure constructing the Tk dialog must not
        # propagate out of the crash-reporting path.
        with patch("tkinter.Tk", side_effect=RuntimeError("no display")):
            _try_show_error_dialog("import failed")
