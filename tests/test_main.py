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
from skryb.transcription.engine_factory import build_engine_registry
from skryb.transcription.engine_registry import EngineRegistry
from skryb.transcription.license_policy import LicensePolicy
from skryb.transcription.model_info import DEFAULT_MODEL
from skryb.transcription.cached_hardware_detector import (
    CachedHardwareDetector,
)
from skryb.__main__ import (
    _Startup,
    _apply_hf_cache_env_overrides,
    _apply_torch_env_overrides,
    _configured_model_installed,
    _show_error,
    _start_engine_preload,
    _start_torch_prewarm_if_needed,
    _stderr_is_console,
    _take_instance_lock,
    _try_show_error_dialog,
    main,
    _parse_cli,
    _run,
)

windows_startup_only = pytest.mark.skipif(
    sys.platform != "win32",
    reason="drives the Windows startup chain: %APPDATA% and ctypes.windll",
)


@pytest.fixture(autouse=True)
def _app_folders_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``main()`` renames the real VoiceToText folders and writes a session log under %APPDATA%.

    Run on a developer's PC with the real variables, that moved their
    settings and models to Skryb's folders mid-session.
    """
    roaming, local = tmp_path / "Roaming", tmp_path / "Local"
    monkeypatch.setenv("APPDATA", str(roaming))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    return tmp_path


def test_running_main_leaves_the_real_voicetotext_folders_alone(tmp_path: Path) -> None:
    old = tmp_path / "Roaming" / "VoiceToText"
    old.mkdir(parents=True)
    (old / "config.json").write_text("{}", encoding="utf-8")

    with (
        patch.object(sys, "argv", ["skryb", "--doctor"]),
        patch("sys.stdout", io.StringIO()),
        patch("skryb.__main__._run"),
        patch("skryb.diagnostics.doctor.run_doctor", return_value="ok"),
        pytest.raises(SystemExit),
    ):
        main()

    assert os.environ["APPDATA"] == str(tmp_path / "Roaming")
    assert (tmp_path / "Roaming" / "Skryb" / "config.json").exists()


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


class TestStartingTheEnginePreload:
    """Which launches start loading the model before the tray is built."""

    @staticmethod
    def _start(settings: dict[str, object], tmp_path: Path) -> tuple[object, MagicMock]:
        config = FakeConfig({
            "transcription.model_size": "crisperwhisper-2.0-turbo",
            "transcription.engine": "crisperwhisper_ct2",
            "transcription.language": "en",
            "transcription.mode": "batch",
            **settings,
        })
        with patch(
            "skryb.transcription.engine_preload.EnginePreload.start",
        ) as start:
            result = _start_engine_preload(
                config, SimpleNamespace(models_dir=tmp_path),
                build_engine_registry(), MagicMock(), MagicMock(),
            )
        return result, start

    def test_a_ctranslate2_batch_launch_starts_the_preload(self, tmp_path) -> None:
        result, start = self._start({}, tmp_path)

        start.assert_called_once()
        assert result is start.return_value

    def test_streaming_mode_leaves_the_load_to_the_tray(self, tmp_path) -> None:
        result, start = self._start({"transcription.mode": "streaming"}, tmp_path)

        assert result is None
        start.assert_not_called()

    @pytest.mark.parametrize("legacy_language", [None, "auto"])
    def test_a_legacy_language_the_tray_will_migrate_leaves_the_load_to_the_tray(
        self, legacy_language, tmp_path,
    ) -> None:
        result, start = self._start(
            {"transcription.language": legacy_language}, tmp_path,
        )

        assert result is None
        start.assert_not_called()

    def test_a_torch_engine_leaves_the_load_to_the_tray(self, tmp_path) -> None:
        result, start = self._start(
            {
                "transcription.model_size": "parakeet-tdt-0.6b-v3",
                "transcription.engine": "nano_parakeet",
            },
            tmp_path,
        )

        assert result is None
        start.assert_not_called()

    def test_the_preload_builds_the_engine_for_the_active_model(self, tmp_path) -> None:
        with patch(
            "skryb.transcription.engine_factory.create_engine_for_active_model",
        ) as create:
            _result, start = self._start({}, tmp_path)
            build = start.call_args.args[0]
            assert build() is create.return_value

        assert create.call_args.args[0].get("transcription.model_size") == (
            "crisperwhisper-2.0-turbo"
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


@windows_startup_only
class TestDoctorCli:
    """``skryb --doctor`` must print the diagnostic report and
    exit before any Qt or engine code runs, so a user whose engine
    can't load can still get a usable report from the command line.
    """

    def test_should_print_doctor_report_when_doctor_flag_in_argv(self):
        captured = io.StringIO()
        with (
            patch.object(sys, "argv", ["skryb", "--doctor"]),
            patch("sys.stdout", captured),
            patch("skryb.__main__._run") as mock_run,
            patch(
                "skryb.diagnostics.doctor.run_doctor",
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
            patch.object(sys, "argv", ["skryb"]),
            patch.dict(os.environ, {"APPDATA": str(tmp_path)}),
            patch("skryb.__main__._run") as mock_run,
            patch(
                "skryb.core.torch_prewarm.start_torch_prewarm",
                return_value=prewarm,
            ),
            patch(
                "skryb.core.single_instance.SingleInstanceMutex",
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
            patch.object(sys, "argv", ["skryb", flag]),
            patch(f"skryb.core.dev_install.{target}", return_value=(True, "ok")),
            patch("skryb.__main__._run") as run,
            patch("skryb.core.torch_prewarm.start_torch_prewarm") as prewarm,
            patch("skryb.core.single_instance.SingleInstanceMutex") as mutex,
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
        api_server = MagicMock()
        paths = SimpleNamespace(
            config_path=Path("config.json"),
            models_dir=Path("models"),
            is_first_run=lambda: False,
        )
        startup = _startup(
            await_torch_prewarm=lambda: timeline.append("await-prewarm"),
        )
        preload = MagicMock()

        with (
            patch(
                "skryb.__main__._start_engine_preload",
                side_effect=lambda *_args: timeline.append("preload") or preload,
            ),
            patch("tkinter.Tk", side_effect=lambda: timeline.append("tk") or root),
            patch(
                "skryb.__main__._start_api_server",
                side_effect=lambda _bus: timeline.append("api-server") or api_server,
            ),
            patch(
                "skryb.downloads.download_manager.DownloadManager",
                side_effect=lambda **_kwargs: timeline.append("downloads")
                or MagicMock(),
            ),
            patch(
                "skryb.transcription.engine_factory.create_model_manager",
                side_effect=lambda *_args: timeline.append("manager") or manager,
            ),
            patch(
                "skryb.transcription.torch_hardware_detector."
                "TorchHardwareDetector.detect",
            ) as detect,
            patch(
                "skryb.bootstrap.build_app",
                side_effect=lambda *_args, **_kwargs: timeline.append("build-app")
                or app,
            ) as build_app,
            patch("skryb.core.single_instance.LaunchSignal"),
        ):
            _run(paths, startup)

        # The model starts loading before the window and the tray, which
        # are built while it loads.
        assert timeline == [
            "api-server",
            "downloads",
            "manager",
            "await-prewarm",
            "preload",
            "tk",
            "build-app",
        ]
        assert build_app.call_args.kwargs["preload"] is preload
        # A fresh launch looks for newer code once the tray is up.
        assert build_app.call_args.kwargs["check_for_updates"] is True
        assert build_app.call_args.kwargs["second_launches"] is not None
        detect.assert_not_called()
        assert build_app.call_args.kwargs["registry"] is startup.registry
        assert build_app.call_args.kwargs["api_server"] is api_server
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
            "skryb.core.torch_prewarm.start_torch_prewarm",
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
            "skryb.core.torch_prewarm.start_torch_prewarm",
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
            "skryb.core.torch_prewarm.start_torch_prewarm",
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
            "skryb.core.torch_prewarm.start_torch_prewarm",
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
            diagnostics_dir=tmp_path / "diagnostics",
            is_first_run=lambda: True,
        )

        with (
            patch("tkinter.Tk", return_value=MagicMock()),
            patch("skryb.__main__._start_api_server"),
            patch("skryb.downloads.download_manager.DownloadManager"),
            patch(
                "skryb.transcription.engine_factory.create_model_manager",
                return_value=manager,
            ) as create_model_manager,
            patch("skryb.__main__._run_first_time_setup") as first_run_setup,
            patch("skryb.bootstrap.build_app") as build_app,
            patch("skryb.core.single_instance.LaunchSignal"),
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
            patch("skryb.__main__._try_show_error_dialog"),
        ):
            _show_error("something broke")

        assert "something broke" in captured.getvalue()

    def test_should_still_show_dialog_when_stderr_is_missing(self):
        # pythonw (GUI launches, the Start Menu shortcut) has no stdio.
        with (
            patch("sys.stderr", None),
            patch("skryb.__main__._try_show_error_dialog") as dialog,
        ):
            _show_error("something broke")

        dialog.assert_called_once_with("something broke")

    def test_dialog_should_swallow_errors_when_tk_unavailable(self):
        # Best-effort: a failure constructing the Tk dialog must not
        # propagate out of the crash-reporting path.
        with patch("tkinter.Tk", side_effect=RuntimeError("no display")):
            _try_show_error_dialog("import failed")


class TestLaunchingWhileACopyIsRunning:
    """What a launch does when another copy already holds the lock."""

    def _launch(
        self,
        *,
        held: list[bool],
        answered: bool = False,
        ended: bool = False,
        restarting: bool = False,
    ):
        mutex = MagicMock()
        mutex.acquire.side_effect = [not still_held for still_held in held]
        paths = SimpleNamespace(config_path=Path("C:/Users/Zoë/AppData/Roaming/Skryb/config.json"))
        with (
            patch("skryb.core.single_instance.LaunchSignal") as signal,
            patch("skryb.core.single_instance.RunningCopy") as copy,
            patch("skryb.__main__._show_instance_conflict") as conflict,
        ):
            signal.return_value.ask.return_value = answered
            copy.return_value.end.return_value = ended
            took_lock = _take_instance_lock(mutex, paths, restarting=restarting)
        return took_lock, mutex, signal.return_value, copy, conflict

    def test_the_first_copy_takes_the_lock_and_records_itself(self) -> None:
        took_lock, _mutex, signal, copy, conflict = self._launch(held=[False])

        assert took_lock is True
        copy.assert_called_once_with(
            Path("C:/Users/Zoë/AppData/Roaming/Skryb/running.pid"),
        )
        copy.return_value.record.assert_called_once_with()
        signal.ask.assert_not_called()
        conflict.assert_not_called()

    def test_a_second_launch_is_handed_to_the_copy_running_when_it_answers(self) -> None:
        took_lock, _mutex, signal, copy, conflict = self._launch(held=[True], answered=True)

        assert took_lock is False
        signal.ask.assert_called_once()
        copy.return_value.end.assert_not_called()
        conflict.assert_not_called()

    def test_a_copy_that_never_answers_is_ended_and_replaced(self) -> None:
        took_lock, mutex, _signal, copy, conflict = self._launch(
            held=[True, False], answered=False, ended=True,
        )

        assert took_lock is True
        copy.return_value.end.assert_called_once_with()
        copy.return_value.record.assert_called_once_with()
        assert mutex.acquire.call_args.kwargs["wait_seconds"] > 0
        conflict.assert_not_called()

    def test_a_stuck_copy_that_cannot_be_ended_gets_the_conflict_message(self) -> None:
        took_lock, _mutex, _signal, copy, conflict = self._launch(
            held=[True, True], answered=False, ended=False,
        )

        assert took_lock is False
        copy.return_value.record.assert_not_called()
        conflict.assert_called_once_with()

    def test_a_copy_that_exits_on_its_own_while_unanswered_is_replaced(self) -> None:
        took_lock, mutex, _signal, copy, conflict = self._launch(
            held=[True, False], answered=False, ended=False,
        )

        assert took_lock is True
        assert mutex.acquire.call_args.kwargs["wait_seconds"] == 0.0
        copy.return_value.record.assert_called_once_with()
        conflict.assert_not_called()

    def test_an_ended_copy_that_still_holds_the_lock_gets_the_conflict_message(self) -> None:
        took_lock, _mutex, _signal, _copy, conflict = self._launch(
            held=[True, True], answered=False, ended=True,
        )

        assert took_lock is False
        conflict.assert_called_once_with()

    def test_a_restarting_copy_waits_for_the_lock_instead_of_asking(self) -> None:
        took_lock, mutex, signal, _copy, _conflict = self._launch(
            held=[False], restarting=True,
        )

        assert took_lock is True
        assert mutex.acquire.call_args.kwargs["wait_seconds"] >= 10
        signal.ask.assert_not_called()

    def test_the_restart_flag_is_read_from_the_command_line(self) -> None:
        assert _parse_cli(["--restart"]).restart is True
        assert _parse_cli([]).restart is False
