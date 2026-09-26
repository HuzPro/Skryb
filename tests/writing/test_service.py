import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.fakes.writing import FakeChild, FakeLaunch
from voicetotext.core.json_config_adapter import DEFAULT_CONFIG
from voicetotext.output.dictionary_config import DictionaryConfig, LiteralRule, RegexRule
from voicetotext.writing.install import WritingPaths
from voicetotext.writing.learning import Learner
from voicetotext.writing.llama_server import LlamaServer, ServerFiles, ServerState
from voicetotext.writing.service import SERVER_FAILED_NOTICE, WritingService, spellings_from
from voicetotext.writing.settings import WritingSettings
from voicetotext.writing.transforms import BUILTIN_TRANSFORMS, TransformStore


class Rig:
    def __init__(self, tmp_path: Path, children: list[FakeChild], locate_files: bool = True) -> None:
        runtime = tmp_path / "custom" / "llama-server.exe"
        runtime.parent.mkdir(parents=True)
        runtime.write_bytes(b"MZ")
        model = tmp_path / "custom" / "model.gguf"
        model.write_bytes(b"GGUF")
        self.settings = WritingSettings(runtime_path=str(runtime), model_path=str(model))
        if not locate_files:
            self.settings = WritingSettings()
        files = ServerFiles(runtime, model)
        self.launch = FakeLaunch(children, log_line="vk::Device::createBuffer: ErrorOutOfDeviceMemory")
        self.healthy = True
        self.server = LlamaServer(
            locate=lambda: files if locate_files else None,
            log_path=tmp_path / "session" / "writing-server.log",
            warm_up=lambda _endpoint: None,
            launch=self.launch,
            is_healthy=lambda _endpoint: self.healthy,
            read_help=lambda _runtime: "",
            bind=lambda _pid: None,
            backoff=(0.0, 0.0, 0.0),
            readiness_timeout=0.05,
            poll_seconds=0.005,
        )
        self.notices: list[str] = []
        self.opened: list[Path] = []
        self.service = WritingService(
            lambda: self.settings,
            WritingPaths(tmp_path / "managed"),
            self.server,
            Learner(tmp_path / "learning.json", tmp_path / "dictionary.json",
                    lambda: self.settings, lambda: None, "session"),
            TransformStore(tmp_path / "transforms.json"),
            notify=self.notices.append,
            run_in_background=lambda action: action(),
            open_path=self.opened.append,
        )


def _wait_for(condition: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition never became true")
        time.sleep(0.005)


def _crashes(count: int, first_pid: int = 0) -> list[FakeChild]:
    return [FakeChild(pid=first_pid + index, exits_while_loading=1) for index in range(count)]


class TestStatus:
    def test_reads_not_installed_until_the_files_exist(self, tmp_path):
        rig = Rig(tmp_path, [], locate_files=False)

        assert rig.service.status().state == "not_installed"

    def test_reads_ready_with_the_last_cleanup_time(self, tmp_path):
        rig = Rig(tmp_path, [FakeChild(pid=1)])
        rig.service.start()
        _wait_for(rig.server.ready_event.is_set)

        rig.service.timings.record(0.31, 12)

        assert rig.service.status().state == "ready"
        assert rig.service.status().last_cleanup_seconds == pytest.approx(0.31)
        rig.service.dispose()

    def test_a_server_that_gives_up_is_announced_once_per_session(self, tmp_path):
        rig = Rig(tmp_path, [*_crashes(4), *_crashes(4, first_pid=10)])
        rig.healthy = False
        rig.service.start()
        _wait_for(lambda: rig.server.status.state is ServerState.FAILED)
        status = rig.service.status()

        rig.service.retry()
        _wait_for(lambda: len(rig.launch.launched) == 8)
        _wait_for(lambda: rig.server.status.state is ServerState.FAILED)

        assert status.state == "failed"
        assert "ErrorOutOfDeviceMemory" in status.detail
        assert rig.notices == [SERVER_FAILED_NOTICE]

    def test_open_log_opens_the_server_log_only_once_it_exists(self, tmp_path):
        rig = Rig(tmp_path, _crashes(4))
        rig.service.open_log()
        assert rig.opened == []
        rig.healthy = False
        rig.service.start()
        _wait_for(lambda: rig.server.status.state is ServerState.FAILED)

        rig.service.open_log()

        assert rig.opened == [rig.server.log_path]


class TestWhatCleanupGets:
    def test_spellings_are_the_dictionary_names_and_terms_without_duplicates(self):
        rules = (
            LiteralRule(spoken=("kube",), replacement="Kubernetes"),
            LiteralRule(spoken=("k8s",), replacement="Kubernetes"),
            LiteralRule(spoken=("aisha",), replacement="Aïsha", learned=True),
            LiteralRule(spoken=("new line",), replacement="\n"),
            LiteralRule(spoken=("sig",), replacement="Best,\nHuz"),
            LiteralRule(spoken=("pi",), replacement="3.14159"),
            LiteralRule(spoken=("off",), replacement="Disabled", enabled=False),
            LiteralRule(spoken=("essay",), replacement="word " * 20),
            RegexRule(pattern=r"\bteh\b", replacement="the"),
        )

        assert spellings_from(DictionaryConfig(rules=rules)) == ("Kubernetes", "Aïsha")

    def test_a_thousand_rules_still_give_a_short_list(self):
        rules = tuple(
            LiteralRule(spoken=(f"name {index}",), replacement=f"Name{index}")
            for index in range(1000)
        )

        assert len(spellings_from(DictionaryConfig(rules=rules))) == 60

    def test_the_estimate_is_zero_while_cleanup_is_off_or_the_model_is_not_ready(self, tmp_path):
        rig = Rig(tmp_path, [FakeChild(pid=1)])
        assert rig.service.estimated_seconds(10.0) == 0.0

        rig.service.start()
        _wait_for(rig.server.ready_event.is_set)
        assert rig.service.estimated_seconds(40.0) == pytest.approx(1.6)

        rig.settings = WritingSettings(cleanup="off")
        assert rig.service.estimated_seconds(40.0) == 0.0
        rig.service.dispose()

    def test_the_cleanup_stage_reads_the_chosen_transform_when_it_runs(self, tmp_path):
        rig = Rig(tmp_path, [])
        rig.settings = WritingSettings(after_dictation_transform="Concise")

        stage = rig.service.cleanup_stage(DictionaryConfig())
        rig.service.save_transforms({"Concise": "Cut it down to one sentence."})

        assert stage._transform() == "Cut it down to one sentence."

    def test_transforms_start_from_the_built_ins(self, tmp_path):
        assert Rig(tmp_path, []).service.transforms() == BUILTIN_TRANSFORMS


def test_the_config_defaults_match_the_settings_defaults():
    assert DEFAULT_CONFIG["writing"] == asdict(WritingSettings())
