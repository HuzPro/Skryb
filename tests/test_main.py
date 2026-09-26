import io
import os
import sys
from collections.abc import Callable
from dataclasses import replace
from concurrent.futures import Future
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.fakes.config import FakeConfig
from voicetotext.transcription.engine_factory import build_engine_registry
from voicetotext.transcription.engine_registry import EngineRegistry
from voicetotext.transcription.license_policy import LicensePolicy
from voicetotext.transcription.model_info import DEFAULT_MODEL
from voicetotext.transcription.cached_hardware_detector import (
    CachedHardwareDetector,
)
from voicetotext.__main__ import (
    _Startup,
    _apply_hf_cache_env_overrides,
    _apply_torch_env_overrides,
    _configured_model_installed,
    _show_error,
    _start_torch_prewarm_if_needed,
    _stderr_is_console,
    _try_show_error_dialog,
    main,
    _parse_cli,
    _run,
)

windows_startup_only = pytest.mark.skipif(
    sys.platform != "win32",
    reason="drives the Windows startup chain: %APPDATA% and ctypes.windll",
)


def _startup(
    registry: EngineRegistry | None = None,
    await_torch_prewarm: Callable[[], None] = lambda: None,
) -> _Startup:
    config = MagicMock()
    config.get.return_value = "parakeet-tdt-0.6b-v3"
    return _Startup(
        config=config,
        model_was_configured=True,
        registry=registry or MagicMock(name="registry"),
        await_torch_prewarm=await_torch_prewarm,
    )


def _torch_free_registry() -> EngineRegistry:
    """The shipped registry with every engine declaring no torch."""
    return EngineRegistry(
        descriptors=[
            replace(descriptor, requires_torch=False)
            for descriptor in build_engine_registry().all_descriptors()
        ],
        builders={},
    )


def _resolved_prewarm() -> Future[ModuleType]:
    prewarm: Future[ModuleType] = Future()
    prewarm.set_result(ModuleType("torch"))
    return prewarm


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


@windows_startup_only
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
        paths = mock_run.call_args.args[0]
        assert paths.session_log_path.parent.parent == paths.diagnostics_dir
        assert paths.session_log_path.exists()
        mutex.release.assert_called_once()


class TestMaintenanceStartupIsolation:
    @windows_startup_only
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


@windows_startup_only
class TestBindingStartupOrder:
    """``_run`` builds everything it can before anything touches the GPU.

    Hardware detection is the expensive step - the probe imports torch -
    so ``_run`` only hands the detector on. Nothing here may call it.
    """

    def test_should_build_the_app_without_ever_probing_hardware(self) -> None:
        timeline: list[str] = []
        root = MagicMock()
        manager = MagicMock()
        manager.is_installed.return_value = True
        app = MagicMock()
        paths = SimpleNamespace(
            config_path=Path("config.json"),
            models_dir=Path("models"),
            is_first_run=lambda: False,
        )
        startup = _startup(
            await_torch_prewarm=lambda: timeline.append("await-prewarm"),
        )

        with (
            patch("tkinter.Tk", side_effect=lambda: timeline.append("tk") or root),
            patch(
                "voicetotext.downloads.download_manager.DownloadManager",
                side_effect=lambda **_kwargs: timeline.append("downloads")
                or MagicMock(),
            ),
            patch(
                "voicetotext.transcription.engine_factory.create_model_manager",
                side_effect=lambda *_args: timeline.append("manager") or manager,
            ),
            patch(
                "voicetotext.transcription.torch_hardware_detector."
                "TorchHardwareDetector.detect",
            ) as detect,
            patch(
                "voicetotext.bootstrap.build_app",
                side_effect=lambda *_args, **_kwargs: timeline.append("build-app")
                or app,
            ) as build_app,
        ):
            _run(paths, startup)

        assert timeline == [
            "tk",
            "downloads",
            "manager",
            "await-prewarm",
            "build-app",
        ]
        detect.assert_not_called()
        assert build_app.call_args.kwargs["registry"] is startup.registry
        assert isinstance(
            build_app.call_args.kwargs["hardware_detector"],
            CachedHardwareDetector,
        )
        app.run.assert_called_once_with()


# No model is installed under this path, so selection falls back to the
# top of the ranking - which is what the gate has to answer about.
_NO_MODELS_DIR = Path("nonexistent-models")


class TestTorchPrewarmGate:
    """Torch costs over a second to import, so a launch that will not use
    it must not start the prewarm at all. The engine descriptor is the
    only thing asked - importing the engine to find out would cost the
    very second the gate is protecting.
    """

    def test_should_prewarm_when_the_selected_engine_needs_torch(self) -> None:
        config = FakeConfig({"transcription.model_size": DEFAULT_MODEL})

        with patch(
            "voicetotext.core.torch_prewarm.start_torch_prewarm",
            return_value=_resolved_prewarm(),
        ) as start:
            wait = _start_torch_prewarm_if_needed(
                config, build_engine_registry(), _NO_MODELS_DIR,
            )
            wait()

        start.assert_called_once_with()

    def test_should_skip_prewarm_when_no_selectable_engine_needs_torch(
        self,
    ) -> None:
        config = FakeConfig({"transcription.model_size": DEFAULT_MODEL})

        with patch(
            "voicetotext.core.torch_prewarm.start_torch_prewarm",
        ) as start:
            wait = _start_torch_prewarm_if_needed(
                config, _torch_free_registry(), _NO_MODELS_DIR,
            )
            wait()

        start.assert_not_called()

    def test_should_prewarm_for_an_explicitly_configured_torch_engine(
        self,
    ) -> None:
        config = FakeConfig(
            {
                "transcription.model_size": DEFAULT_MODEL,
                "transcription.engine": "crisperwhisper",
            }
        )

        with patch(
            "voicetotext.core.torch_prewarm.start_torch_prewarm",
            return_value=_resolved_prewarm(),
        ) as start:
            _start_torch_prewarm_if_needed(
                config, build_engine_registry(), _NO_MODELS_DIR,
            )

        start.assert_called_once_with()

    def test_should_prewarm_when_the_configured_model_is_unknown(self) -> None:
        # A config that predates a catalogue migration must not be able to
        # talk startup out of an import its engine will need.
        config = FakeConfig({"transcription.model_size": "retired-model"})

        with patch(
            "voicetotext.core.torch_prewarm.start_torch_prewarm",
            return_value=_resolved_prewarm(),
        ) as start:
            _start_torch_prewarm_if_needed(
                config, build_engine_registry(), _NO_MODELS_DIR,
            )

        start.assert_called_once_with()


@windows_startup_only
class TestOneLicensePolicyForTheProcess:
    """First-run setup and ``build_app`` each build their own model
    manager. Two policy instances would give them separate acceptance
    state, so a licence accepted during setup would be asked for again -
    or worse, silently missing - once the app proper starts.
    """

    def _run_with_first_time_setup(self, tmp_path: Path) -> tuple:
        manager = MagicMock()
        manager.is_installed.return_value = True
        paths = SimpleNamespace(
            config_path=Path("config.json"),
            models_dir=tmp_path,
            dictionary_path=tmp_path / "dictionary.json",
            is_first_run=lambda: True,
        )

        with (
            patch("tkinter.Tk", return_value=MagicMock()),
            patch("voicetotext.downloads.download_manager.DownloadManager"),
            patch(
                "voicetotext.transcription.engine_factory.create_model_manager",
                return_value=manager,
            ) as create_model_manager,
            patch("voicetotext.__main__._run_first_time_setup") as first_run_setup,
            patch("voicetotext.bootstrap.build_app") as build_app,
        ):
            _run(paths, _startup())

        return create_model_manager, first_run_setup, build_app

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
