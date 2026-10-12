import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from tests.fakes.transcription import FakeTranscriptionEngine
from voicetotext.transcription.engine_preload import EnginePreload
from voicetotext.transcription.result import TranscriptionResult


def _ready_engine() -> FakeTranscriptionEngine:
    engine = FakeTranscriptionEngine()
    engine.will_return(
        TranscriptionResult(
            text="", language="en", language_probability=0.0,
            transcription_seconds=0.0,
        ),
    )
    return engine


class TestEnginePreload:
    def test_the_engine_is_built_loaded_and_warmed_off_the_calling_thread(self) -> None:
        threads: list[str] = []
        engine = _ready_engine()

        def build() -> FakeTranscriptionEngine:
            threads.append(threading.current_thread().name)
            return engine

        preload = EnginePreload.start(build)

        assert preload.wait() is engine
        assert threads == ["engine-preload"]
        assert engine.load_model_calls == 1
        assert len(engine.transcribe_calls) == 1
        assert preload.warmed

    def test_an_error_building_the_engine_is_raised_to_whoever_waits(self) -> None:
        def build() -> FakeTranscriptionEngine:
            raise PermissionError("licence for 'crisperwhisper-2.0-turbo' not accepted")

        preload = EnginePreload.start(build)

        with pytest.raises(PermissionError, match="not accepted"):
            preload.wait()

    def test_an_error_loading_the_model_is_raised_to_whoever_waits(self) -> None:
        engine = _ready_engine()
        engine.load_model = MagicMock(side_effect=RuntimeError("hipErrorOutOfMemory"))

        preload = EnginePreload.start(lambda: engine)

        with pytest.raises(RuntimeError, match="hipErrorOutOfMemory"):
            preload.wait()

    def test_a_failed_warmup_still_hands_over_the_loaded_engine_unwarmed(self) -> None:
        engine = FakeTranscriptionEngine()
        engine.will_fail(RuntimeError("first kernel compile failed on silence"))

        preload = EnginePreload.start(lambda: engine)

        assert preload.wait() is engine
        assert engine.is_model_loaded()
        assert not preload.warmed

    def test_the_switch_interval_is_shortened_while_loading_and_restored_after(self) -> None:
        before = sys.getswitchinterval()
        seen: list[float] = []
        engine = _ready_engine()
        engine.load_model = lambda: seen.append(sys.getswitchinterval())

        EnginePreload.start(lambda: engine).wait()

        assert seen == [pytest.approx(0.0005)]
        assert sys.getswitchinterval() == pytest.approx(before)

    def test_the_switch_interval_is_restored_even_when_loading_fails(self) -> None:
        before = sys.getswitchinterval()

        def build() -> FakeTranscriptionEngine:
            raise OSError("model.bin is missing")

        preload = EnginePreload.start(build)
        with pytest.raises(OSError):
            preload.wait()

        assert sys.getswitchinterval() == pytest.approx(before)

    def test_a_thousand_threads_waiting_at_once_all_get_the_same_engine(self) -> None:
        engine = _ready_engine()
        release = threading.Event()

        def build() -> FakeTranscriptionEngine:
            release.wait()
            return engine

        preload = EnginePreload.start(build)
        with ThreadPoolExecutor(max_workers=64) as pool:
            waiting = [pool.submit(preload.wait) for _ in range(1000)]
            release.set()
            results = [future.result(timeout=10) for future in waiting]

        assert all(result is engine for result in results)
        assert engine.load_model_calls == 1


@pytest.mark.skipif(sys.platform != "win32", reason="bootstrap imports Win32-only modules")
class TestTakingAPreloadedEngine:
    @pytest.fixture()
    def bootstrap(self):
        from voicetotext import bootstrap

        return bootstrap

    def test_a_warmed_preload_is_used_without_a_second_warmup(self, monkeypatch, bootstrap) -> None:
        warmups = MagicMock()
        monkeypatch.setattr(bootstrap, "_warmup_safely", warmups)
        engine = _ready_engine()
        preload = EnginePreload.start(lambda: engine)

        assert bootstrap._take_preloaded(preload) is engine
        warmups.assert_not_called()

    def test_a_preload_whose_warmup_failed_gets_the_loaders_own_warmup(self, monkeypatch, bootstrap) -> None:
        warmups = MagicMock()
        monkeypatch.setattr(bootstrap, "_warmup_safely", warmups)
        engine = FakeTranscriptionEngine()
        engine.will_fail(RuntimeError("transient"))
        preload = EnginePreload.start(lambda: engine)

        bootstrap._take_preloaded(preload)

        warmups.assert_called_once_with(engine)

    def test_a_gpu_failure_in_the_preload_falls_back_to_the_cpu_like_any_load(
        self, monkeypatch, bootstrap,
    ) -> None:
        cpu_engine = _ready_engine()
        monkeypatch.setattr(bootstrap, "resolve_device", lambda *_args: "auto")
        monkeypatch.setattr(bootstrap, "selectable_engine_ids", lambda *_args: ())
        monkeypatch.setattr(bootstrap, "_load_engine_on_cpu", lambda *_args: cpu_engine)
        monkeypatch.setattr(
            bootstrap, "_load_engine",
            MagicMock(side_effect=AssertionError("preload must replace the first attempt")),
        )
        gpu_engine = _ready_engine()
        gpu_engine.load_model = MagicMock(side_effect=RuntimeError("no HIP device"))
        diagnostics = MagicMock()

        engine = bootstrap._load_engine_with_fallback(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            diagnostics, EnginePreload.start(lambda: gpu_engine),
        )

        assert engine is cpu_engine
        assert diagnostics.engine_loaded.call_args.args[1] == "cpu"
