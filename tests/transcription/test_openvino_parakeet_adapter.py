"""Tests for the OpenVINO (NPU) Parakeet engine.

Everything here runs on the CPU with small randomly initialised modules
and fakes for OpenVINO, so it passes the same on any machine. Real
weights on a real NPU stay a manual check (scripts/check_npu.py).
"""

import importlib.util
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.openvino_parakeet_adapter import (
    BUCKET_FRAMES,
    FIRST_BUCKET_FRAMES,
    CompiledEncoder,
    compile_length,
    split_at_pauses,
)
from skryb.transcription.result import TranscriptionResult
from skryb.transcription.openvino_parakeet_export import MaskedEncoder


_RATE = 16_000
_LONGEST = 30 * _RATE


def _small_encoder() -> torch.nn.Module:
    from nano_parakeet.model import FastConformerEncoder

    torch.manual_seed(7)
    encoder = FastConformerEncoder(d_model=64, n_layers=2, n_heads=4, d_ff=128)
    for layer in encoder.layers:
        layer.conv.batch_norm.running_mean.uniform_(-0.5, 0.5)
        layer.conv.batch_norm.running_var.uniform_(0.5, 2.0)
    return encoder.eval()


class TestPaddedClipsEncodeLikeUnpaddedOnes:
    @pytest.mark.parametrize("length", [9, 200, 201, 203, 208, 401])
    def test_the_real_frames_match_the_unpadded_clip_whatever_the_padding(self, length: int) -> None:
        encoder = _small_encoder()
        features = torch.randn(1, 128, length)
        padded = torch.zeros(1, 128, 401)
        padded[..., :length] = features

        with torch.inference_mode():
            expected, expected_length = encoder(features, torch.tensor([length]))
            encoded, encoded_length = MaskedEncoder(encoder)(padded, torch.tensor([length]))

        assert int(encoded_length) == int(expected_length) == expected.shape[1]
        torch.testing.assert_close(encoded[:, :expected.shape[1]], expected, atol=1e-5, rtol=1e-4)

    def test_without_the_masking_the_padding_leaks_into_the_last_frames(self) -> None:
        encoder = _small_encoder()
        features = torch.randn(1, 128, 203)
        padded = torch.zeros(1, 128, 401)
        padded[..., :203] = features

        with torch.inference_mode():
            expected, length = encoder(features, torch.tensor([203]))
            leaked, _ = encoder(padded, torch.tensor([401]))

        assert not torch.allclose(leaked[:, int(length) - 1], expected[:, -1], atol=1e-3)


class TestSplittingLongAudio:
    def test_audio_up_to_the_limit_stays_whole(self) -> None:
        samples = np.random.default_rng(1).normal(0, 0.1, _LONGEST).astype(np.float32)

        assert [len(p) for p in split_at_pauses(samples, _LONGEST)] == [_LONGEST]

    def test_one_sample_over_the_limit_makes_two_pieces(self) -> None:
        samples = np.random.default_rng(2).normal(0, 0.1, _LONGEST + 1).astype(np.float32)

        pieces = split_at_pauses(samples, _LONGEST)

        assert len(pieces) == 2
        assert np.array_equal(np.concatenate(pieces), samples)

    def test_long_dictation_is_cut_in_its_pauses(self) -> None:
        samples = np.random.default_rng(3).normal(0, 0.1, 70 * _RATE).astype(np.float32)
        samples[27 * _RATE:int(27.3 * _RATE)] = 0.0
        samples[54 * _RATE:int(54.3 * _RATE)] = 0.0

        pieces = split_at_pauses(samples, _LONGEST)
        cuts = np.cumsum([len(p) for p in pieces])[:-1] / _RATE

        assert len(pieces) == 3
        assert 27.0 <= cuts[0] <= 27.3
        assert 54.0 <= cuts[1] <= 54.3
        assert np.array_equal(np.concatenate(pieces), samples)

    def test_seven_minutes_without_a_pause_is_cut_at_the_limit_and_loses_nothing(self) -> None:
        samples = np.random.default_rng(4).normal(0, 0.1, 408 * _RATE).astype(np.float32)

        pieces = split_at_pauses(samples, _LONGEST)

        assert all(0 < len(p) <= _LONGEST for p in pieces)
        assert len(pieces) >= 14
        assert np.array_equal(np.concatenate(pieces), samples)

    def test_empty_audio_has_no_pieces(self) -> None:
        assert split_at_pauses(np.zeros(0, np.float32), _LONGEST) == []


class _FakeCompiled:
    def __init__(self, device: str = "NPU", frames_out_per_in: int = 8, value: float = 1.0) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.device = device
        self.broken: str | None = None
        self._ratio = frames_out_per_in
        self._value = value

    def __call__(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        if self.broken:
            raise RuntimeError(self.broken)
        features, length = inputs
        self.calls.append((features.shape, int(length[0])))
        assert not features[..., int(length[0]):].any(), "padding must be zeros"
        frames = features.shape[-1] // self._ratio + 1
        real = (int(length[0]) - 1) // self._ratio + 1
        return [np.full((1, frames, 4), self._value, np.float32), np.array([real], np.int64)]

    def get_property(self, name: str) -> str | list[str]:
        assert name == "EXECUTION_DEVICES"
        # As the real plugins answer: the NPU's a bare string, the others a list.
        return self.device if self.device == "NPU" else [self.device]


class TestFixedLengths:
    def test_features_are_padded_to_the_shortest_length_that_fits(self) -> None:
        short, long = _FakeCompiled(), _FakeCompiled()
        encoder = CompiledEncoder({FIRST_BUCKET_FRAMES: short, max(BUCKET_FRAMES): long})

        encoded = encoder(np.ones((1, 128, 541), np.float32))

        assert short.calls == [((1, 128, 1024), 541)]
        assert long.calls == []
        assert encoded.shape == (68, 4)

    def test_a_clip_exactly_at_a_length_is_not_padded_further(self) -> None:
        short, long = _FakeCompiled(), _FakeCompiled()
        encoder = CompiledEncoder({FIRST_BUCKET_FRAMES: short, max(BUCKET_FRAMES): long})

        encoder(np.ones((1, 128, 1024), np.float32))

        assert short.calls == [((1, 128, 1024), 1024)]

    def test_a_clip_longer_than_every_length_is_refused(self) -> None:
        encoder = CompiledEncoder({FIRST_BUCKET_FRAMES: _FakeCompiled()})

        with pytest.raises(TranscriptionError):
            encoder(np.ones((1, 128, 1025), np.float32))

    def test_a_device_that_takes_any_length_gets_the_clip_unpadded(self) -> None:
        any_length = _FakeCompiled("CPU")

        CompiledEncoder({0: any_length})(np.ones((1, 128, 77), np.float32))

        assert any_length.calls == [((1, 128, 77), 77)]

    def test_the_device_shown_is_the_one_openvino_says_the_graph_runs_on(self) -> None:
        assert CompiledEncoder({FIRST_BUCKET_FRAMES: _FakeCompiled("NPU")}).device == "NPU"
        assert CompiledEncoder({0: _FakeCompiled("CPU")}).device == "CPU"

    def test_a_length_added_later_takes_the_audio_that_no_longer_has_to_be_cut(self) -> None:
        encoder = CompiledEncoder({FIRST_BUCKET_FRAMES: _FakeCompiled()})
        assert encoder.longest_samples == 1023 * 160, "10.23 s: the most audio whose features fit 1024 frames"

        encoder.add(max(BUCKET_FRAMES), _FakeCompiled())

        assert encoder.longest_samples == 3071 * 160
        assert encoder.length_count == 2


class _FakeModel:
    def __init__(self, shapes: dict | None = None) -> None:
        self.shapes = shapes

    def clone(self) -> "_FakeModel":
        return _FakeModel(self.shapes)

    def reshape(self, shapes: dict) -> None:
        self.shapes = shapes


class _FakeCore:
    """OpenVINO's Core, compiling instantly unless told how long each length takes."""

    def __init__(
        self, behaviour: dict[str, str], *, seconds: float = 0.0, first_seconds: float | None = None,
        rest_gate: threading.Event | None = None,
    ) -> None:
        self._behaviour = behaviour
        self.available_devices = list(behaviour)
        self.compiled: list[tuple[str, int]] = []
        self.configs: list[dict] = []
        self.most_at_once = 0
        self._running = 0
        self._seconds = seconds
        self._first_seconds = seconds if first_seconds is None else first_seconds
        self._rest_gate = rest_gate
        self._lock = threading.Lock()

    def read_model(self, path: str) -> _FakeModel:
        return _FakeModel()

    def compile_model(self, model: _FakeModel, device: str, config: dict) -> _FakeCompiled:
        assert config["CACHE_DIR"].endswith("compiled-npu")
        frames = model.shapes["features"][2] if model.shapes else 0
        first = frames == FIRST_BUCKET_FRAMES
        with self._lock:
            self._running += 1
            self.most_at_once = max(self.most_at_once, self._running) if not first else self.most_at_once
        if not first and self._rest_gate is not None:
            self._rest_gate.wait(10)
        time.sleep(self._first_seconds if first else self._seconds)
        with self._lock:
            self._running -= 1
            self.compiled.append((device, frames))
            self.configs.append(config)
        behaviour = self._behaviour[device]
        if behaviour == "refuse" or behaviour == f"refuse {frames}":
            raise RuntimeError(f"{device} cannot compile this graph")
        return _FakeCompiled(device, value=np.nan if behaviour == "nan" else 1.0)


class _FakeOpenVino:
    def __init__(self, core: _FakeCore) -> None:
        self.core = core

    def Core(self) -> _FakeCore:  # noqa: N802 (OpenVINO's name)
        return self.core


class TestCompilingOneLength:
    def test_the_npu_gets_the_length_baked_in_at_full_clock_and_high_priority(self, tmp_path) -> None:
        core = _FakeCore({"NPU": "ok"})

        compile_length(core, _FakeModel(), tmp_path / "compiled-npu", "NPU", FIRST_BUCKET_FRAMES)

        assert core.compiled == [("NPU", 1024)]
        assert core.configs[0]["NPU_TURBO"] == "YES" and core.configs[0]["MODEL_PRIORITY"] == "HIGH"

    def test_the_cpu_takes_any_length_and_is_left_at_its_own_settings(self, tmp_path) -> None:
        core = _FakeCore({"CPU": "ok"})

        compile_length(core, _FakeModel(), tmp_path / "compiled-npu", "CPU", 0)

        assert core.compiled == [("CPU", 0)]
        assert "NPU_TURBO" not in core.configs[0] and "MODEL_PRIORITY" not in core.configs[0]

    def test_a_device_that_overflows_to_nan_is_refused(self, tmp_path) -> None:
        core = _FakeCore({"NPU": "nan"})

        with pytest.raises(RuntimeError, match="non-finite"):
            compile_length(core, _FakeModel(), tmp_path / "compiled-npu", "NPU", FIRST_BUCKET_FRAMES)

    def test_each_length_reshapes_its_own_copy_of_the_model(self, tmp_path) -> None:
        model = _FakeModel()

        compile_length(_FakeCore({"NPU": "ok"}), model, tmp_path / "compiled-npu", "NPU", max(BUCKET_FRAMES))

        assert model.shapes is None


@pytest.mark.skipif(importlib.util.find_spec("openvino") is None, reason="openvino is not installed")
def test_openvinos_usage_reporting_is_replaced_by_its_no_op_stub() -> None:
    code = (
        "from skryb.transcription.openvino_parakeet_adapter import import_openvino\n"
        "import_openvino()\n"
        "import openvino.tools.ovc.telemetry_utils as telemetry\n"
        "print(telemetry.tm.__name__)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True,
    )

    assert completed.stdout.strip().endswith("telemetry_stub")


class TestUpgradingFrom090:
    def test_a_read_only_cache_folder_is_deleted_whole(self, tmp_path) -> None:
        import os
        import stat

        from skryb.transcription.openvino_parakeet_adapter import delete_cache_directory

        cache = tmp_path / "compiled"
        (cache / "nested").mkdir(parents=True)
        for blob in (cache / "123.blob", cache / "nested" / "456.blob"):
            blob.write_bytes(b"blob")
            os.chmod(blob, stat.S_IREAD)

        delete_cache_directory(cache)

        assert not cache.exists()

    def test_a_missing_cache_folder_is_fine(self, tmp_path) -> None:
        from skryb.transcription.openvino_parakeet_adapter import delete_cache_directory

        delete_cache_directory(tmp_path / "compiled")

    @pytest.mark.skipif(importlib.util.find_spec("openvino") is None, reason="openvino is not installed")
    def test_the_fp16_encoder_and_its_old_blobs_give_way_to_the_int8_encoder(self, tmp_path) -> None:
        import os
        import stat

        from skryb.transcription.openvino_parakeet_adapter import _write_int8_encoder, import_openvino
        from skryb.transcription.openvino_int8_weights import opset

        openvino = import_openvino()
        ops = opset(openvino)
        features = ops.parameter([1, 128, -1], np.float32, name="features")
        weight = np.random.default_rng(0).standard_normal((64, 128)).astype(np.float32)
        out = ops.matmul(ops.transpose(features, ops.constant(np.array([0, 2, 1]))), ops.constant(weight), False, True)
        exported, int8 = tmp_path / "encoder.xml", tmp_path / "encoder-int8.xml"
        openvino.save_model(openvino.Model([out], [features], "tiny"), str(exported))
        (tmp_path / "compiled").mkdir()
        old_blob = tmp_path / "compiled" / "999.blob"
        old_blob.write_bytes(b"0.9.0")
        os.chmod(old_blob, stat.S_IREAD)

        _write_int8_encoder(openvino, exported, int8)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["encoder-int8.bin", "encoder-int8.xml"]


def test_numpys_blas_threads_can_be_lowered_and_put_back() -> None:
    from skryb.transcription.openvino_parakeet_adapter import set_numpy_blas_threads

    before = set_numpy_blas_threads(1)
    if before is None:
        pytest.skip("this NumPy build has no reachable OpenBLAS")
    try:
        assert set_numpy_blas_threads(before) == 1
    finally:
        set_numpy_blas_threads(before)



class _FakeStandIn(TranscriptionEngine):
    """Phonon's CPU engine: loads in a moment, and refuses any use after it is disposed."""

    def __init__(self, events: list[str], load_seconds: float = 0.0) -> None:
        self._events = events
        self._load_seconds = load_seconds
        self.loaded = False
        self.disposed = False

    def load_model(self) -> None:
        self._events.append("stand-in loaded")
        time.sleep(self._load_seconds)
        self.loaded = True

    def is_model_loaded(self) -> bool:
        return self.loaded and not self.disposed

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        assert not self.disposed, "a disposed stand-in was asked to transcribe"
        return TranscriptionResult(text="from the cpu engine", language="en", language_probability=0.0, transcription_seconds=0.0)

    def dispose(self) -> None:
        self.disposed = True


class _BrokenStandIn(_FakeStandIn):
    def load_model(self) -> None:
        raise TranscriptionError("Fermion's CPU engine is not installed")


class _Tokenizer:
    def DecodeIds(self, ids: list[int]) -> str:  # noqa: N802 (SentencePiece's name)
        return "from openvino"


class _Decoder:
    def decode(self, encoded: np.ndarray) -> list[int]:
        return [1, 2, 3]


def _eventually(condition, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.01)


def _speech(seconds: float) -> AudioSegment:
    samples = np.random.default_rng(int(seconds * 10)).normal(0, 0.1, int(seconds * _RATE)).astype(np.float32)
    return AudioSegment(data=samples, sample_rate=_RATE)


@pytest.fixture
def engine_on(tmp_path, monkeypatch):
    """An adapter wired to a fake OpenVINO core and a fake Phonon CPU engine."""
    from skryb.transcription import openvino_parakeet_adapter as adapter

    events: list[str] = []
    made: list[_FakeStandIn] = []
    export_dir = tmp_path / ".openvino"
    export_dir.mkdir()

    def build(core: _FakeCore, *, converted: bool = True, cpu_engine: bool = True, stand_in_seconds: float = 0.0):
        if converted:
            (export_dir / "encoder-int8.xml").write_text("ir")

        def converted_once(self, openvino):
            events.append("conversion started")
            return export_dir / "encoder-int8.xml", export_dir / "decoder.npz"

        def cpu(self):
            if not cpu_engine:
                return None
            made.append(_FakeStandIn(events, stand_in_seconds))
            return made[-1]

        monkeypatch.setattr(adapter, "import_openvino", lambda: _FakeOpenVino(core))
        monkeypatch.setattr(adapter, "_wake_npu", lambda openvino, core, cache_dir: None if "NPU" in core.available_devices else "no NPU device")
        monkeypatch.setattr(adapter.OpenVinoParakeetAdapter, "_export_dir", lambda self: export_dir)
        monkeypatch.setattr(adapter.OpenVinoParakeetAdapter, "_converted_once", converted_once)
        monkeypatch.setattr(adapter.OpenVinoParakeetAdapter, "_cpu_engine", cpu)
        monkeypatch.setattr(adapter, "_load_decoder", lambda openvino, path: _Decoder())
        monkeypatch.setattr(adapter, "bundled_tokenizer", _Tokenizer)
        monkeypatch.setattr(adapter, "set_numpy_blas_threads", lambda count: None)
        engine = adapter.OpenVinoParakeetAdapter(tmp_path, "FermionResearch/Phonon-2.5-English")
        return engine, events, made

    return build


@pytest.fixture
def quick_startup(monkeypatch):
    from skryb.transcription import openvino_parakeet_adapter as adapter

    monkeypatch.setattr(adapter, "NPU_STARTUP_SECONDS", 0.1)


class TestStartingUp:
    def test_a_prepared_model_takes_dictation_on_the_npu_in_under_a_second_without_the_cpu_engine(self, engine_on) -> None:
        engine, events, made = engine_on(_FakeCore({"NPU": "ok", "CPU": "ok"}, seconds=0.05))

        started = time.monotonic()
        engine.load_model()

        assert time.monotonic() - started < 1.0
        assert engine.encoder_device == "NPU" and engine.device_note is None
        assert made == []
        assert engine.transcribe(_speech(3)).text == "from openvino"

    def test_the_other_lengths_load_side_by_side_once_the_first_is_up(self, engine_on) -> None:
        core = _FakeCore({"NPU": "ok", "CPU": "ok"}, seconds=0.1)
        engine, _, _ = engine_on(core)

        engine.load_model()
        _eventually(lambda: len(core.compiled) == len(BUCKET_FRAMES))

        assert core.compiled[0] == ("NPU", FIRST_BUCKET_FRAMES)
        assert sorted(frames for _, frames in core.compiled[1:]) == [512, 1536, 2048, 3072]
        assert core.most_at_once == 4

    def test_while_the_npu_compiles_from_scratch_the_cpu_engine_takes_dictation_then_hands_over(
        self, engine_on, quick_startup,
    ) -> None:
        core = _FakeCore({"NPU": "ok", "CPU": "ok"}, seconds=0.05, first_seconds=0.4)
        engine, _, made = engine_on(core)

        engine.load_model()

        assert engine.encoder_device == "CPU" and "Preparing the NPU" in engine.device_note
        assert engine.transcribe(_speech(3)).text == "from the cpu engine"
        _eventually(lambda: engine.encoder_device == "NPU")
        assert len(core.compiled) == len(BUCKET_FRAMES), "the NPU takes over only once every length is up"
        assert made[0].disposed and engine.device_note is None
        assert engine.transcribe(_speech(3)).text == "from openvino"

    def test_lengths_compiling_from_scratch_go_one_at_a_time_to_leave_the_cores_to_the_cpu_engine(
        self, engine_on, quick_startup,
    ) -> None:
        core = _FakeCore({"NPU": "ok", "CPU": "ok"}, seconds=0.05, first_seconds=0.3)
        engine, _, _ = engine_on(core)

        engine.load_model()
        _eventually(lambda: engine.encoder_device == "NPU")

        assert core.most_at_once == 1

    def test_a_cpu_engine_still_loading_never_holds_up_an_npu_that_is_up_first(self, engine_on, quick_startup) -> None:
        engine, _, made = engine_on(_FakeCore({"NPU": "ok", "CPU": "ok"}, first_seconds=0.3), stand_in_seconds=1.5)

        started = time.monotonic()
        engine.load_model()

        assert time.monotonic() - started < 1.0
        assert engine.encoder_device == "NPU"
        _eventually(lambda: made and made[0].disposed)

    def test_a_model_not_converted_yet_loads_the_cpu_engine_before_the_conversion_starts(self, engine_on) -> None:
        engine, events, _ = engine_on(_FakeCore({"NPU": "ok", "CPU": "ok"}), converted=False)

        engine.load_model()
        _eventually(lambda: "conversion started" in events)

        assert events.index("stand-in loaded") < events.index("conversion started")

    def test_parakeet_ultra_has_no_cpu_engine_here_so_it_waits_for_the_npu(self, engine_on, quick_startup) -> None:
        engine, _, made = engine_on(_FakeCore({"NPU": "ok", "CPU": "ok"}, first_seconds=0.4), cpu_engine=False)

        engine.load_model()

        assert engine.encoder_device == "NPU" and made == []

    def test_a_cpu_engine_that_fails_to_load_still_leaves_the_npu_to_come_up(self, engine_on, quick_startup) -> None:
        engine, _, made = engine_on(_FakeCore({"NPU": "ok", "CPU": "ok"}, first_seconds=0.4))
        engine._cpu_engine = lambda: _BrokenStandIn([])

        engine.load_model()

        assert engine.encoder_device == "NPU"

    def test_disposing_during_the_bring_up_keeps_the_npu_from_going_live_afterwards(self, engine_on, quick_startup) -> None:
        core = _FakeCore({"NPU": "ok", "CPU": "ok"}, seconds=0.05, first_seconds=0.3)
        engine, _, made = engine_on(core)
        engine.load_model()

        engine.dispose()
        _eventually(lambda: len(core.compiled) == len(BUCKET_FRAMES))
        time.sleep(0.05)

        assert not engine.is_model_loaded() and made[0].disposed


class TestTidyingUp:
    def test_once_every_length_is_up_old_compiled_lengths_and_a_cut_short_conversion_are_deleted(
        self, engine_on, tmp_path,
    ) -> None:
        import os
        import stat

        export_dir = tmp_path / ".openvino"
        export_dir.mkdir(exist_ok=True)
        old = export_dir / "compiled-int8"
        old.mkdir()
        blob = old / "1676111272740885876.blob"
        blob.write_bytes(b"0.10 blob")
        os.chmod(blob, stat.S_IREAD)
        for name in ("encoder-int8.partial.bin", "encoder-int8.partial.xml"):
            (export_dir / name).write_bytes(b"staged")
        core = _FakeCore({"NPU": "ok", "CPU": "ok"})
        engine, _, _ = engine_on(core)

        engine.load_model()
        _eventually(lambda: not old.exists() and not list(export_dir.glob("*.partial.*")))

        assert (export_dir / "encoder-int8.xml").exists()


class TestWhenTheNpuWillNotRunIt:
    def test_an_npu_that_refuses_the_model_hands_it_to_the_cpu_and_says_why(self, engine_on) -> None:
        engine, _, _ = engine_on(_FakeCore({"NPU": "refuse", "CPU": "ok"}))

        engine.load_model()

        assert engine.encoder_device == "CPU"
        assert "The NPU could not run the model" in engine.device_note
        assert engine.transcribe(_speech(3)).text == "from openvino"

    def test_an_npu_that_overflows_to_nan_hands_the_model_to_the_cpu(self, engine_on) -> None:
        engine, _, _ = engine_on(_FakeCore({"NPU": "nan", "CPU": "ok"}))

        engine.load_model()
        _eventually(lambda: engine.encoder_device == "CPU")

        assert "non-finite" in engine.device_note
        assert engine.transcribe(_speech(3)).text == "from openvino"

    def test_a_computer_without_an_npu_runs_on_the_cpu_and_says_so(self, engine_on) -> None:
        engine, _, _ = engine_on(_FakeCore({"CPU": "ok"}))

        engine.load_model()

        assert engine.encoder_device == "CPU" and engine.device_note == "No working NPU found, so the CPU runs the model."

    def test_a_length_the_npu_refuses_is_left_out_and_longer_audio_is_cut_to_fit_the_rest(self, engine_on) -> None:
        core = _FakeCore({"NPU": f"refuse {max(BUCKET_FRAMES)}", "CPU": "ok"})
        engine, _, _ = engine_on(core)

        engine.load_model()
        _eventually(lambda: len(core.compiled) == len(BUCKET_FRAMES))

        assert engine.encoder_device == "NPU"
        assert engine.transcribe(_speech(28)).text.startswith("from openvino")

    def test_when_no_device_can_run_it_the_load_fails_with_the_reason(self, engine_on) -> None:
        engine, _, _ = engine_on(_FakeCore({"NPU": "refuse", "CPU": "refuse"}), cpu_engine=False)

        with pytest.raises(RuntimeError, match="CPU cannot compile"):
            engine.load_model()

    def test_when_the_npu_stops_working_mid_session_the_dictation_is_redone_on_the_cpu(self, engine_on) -> None:
        core = _FakeCore({"NPU": "ok", "CPU": "ok"})
        engine, _, _ = engine_on(core)
        engine.load_model()
        # The engine files each length a moment after the core compiles it: wait for the engine, or a length filed
        # after the loop below stays unbroken and the dictation never leaves the NPU.
        _eventually(lambda: engine._model._encoder.length_count == len(BUCKET_FRAMES))
        for compiled in engine._model._encoder._compiled_by_frames.values():
            compiled.broken = "ZE_RESULT_ERROR_DEVICE_LOST"

        result = engine.transcribe(_speech(3))

        assert result.text == "from openvino"
        assert engine.encoder_device == "CPU"
        assert "The NPU stopped working (ZE_RESULT_ERROR_DEVICE_LOST)" in engine.device_note


class TestDictatingWhileItComesUp:
    def test_a_long_dictation_while_only_the_first_length_is_up_is_cut_to_fit_it(self, engine_on) -> None:
        gate = threading.Event()
        core = _FakeCore({"NPU": "ok", "CPU": "ok"}, rest_gate=gate)
        engine, _, _ = engine_on(core)
        engine.load_model()
        first = engine._model._encoder._compiled_by_frames[FIRST_BUCKET_FRAMES]

        engine.transcribe(_speech(25))

        assert first.calls and all(shape[-1] == FIRST_BUCKET_FRAMES for shape, _ in first.calls)
        gate.set()
        _eventually(lambda: engine._model._encoder.length_count == len(BUCKET_FRAMES))
        longest = engine._model._encoder._compiled_by_frames[max(BUCKET_FRAMES)]
        warmed = len(longest.calls)
        engine.transcribe(_speech(25))
        assert len(longest.calls) == warmed + 1, "once every length is up, 25 s goes in one pass"

    def test_a_thousand_dictations_during_the_handover_from_the_cpu_engine_all_come_back(
        self, engine_on, quick_startup,
    ) -> None:
        core = _FakeCore({"NPU": "ok", "CPU": "ok"}, seconds=0.02, first_seconds=0.3)
        engine, _, made = engine_on(core)
        engine.load_model()
        texts: list[str] = []
        failures: list[BaseException] = []

        def dictate(count: int) -> None:
            for index in range(count):
                try:
                    texts.append(engine.transcribe(_speech(0.5 + (index % 7) * 0.25)).text)
                except BaseException as failure:  # noqa: BLE001 - the test reports every one
                    failures.append(failure)

        workers = [threading.Thread(target=dictate, args=(250,)) for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(30)

        _eventually(lambda: engine.encoder_device == "NPU")
        assert failures == []
        assert len(texts) == 1000
        assert set(texts) <= {"from the cpu engine", "from openvino"}
        assert engine.encoder_device == "NPU" and made[0].disposed


def test_a_second_load_during_the_first_conversion_waits_for_it_and_reuses_its_files(tmp_path, monkeypatch) -> None:
    from skryb.transcription import openvino_parakeet_adapter as adapter

    exports, int8_writes, exporting = [], [], threading.Event()

    def slow_export(self, openvino, encoder_path, decoder_path) -> None:
        exports.append(encoder_path)
        exporting.set()
        time.sleep(0.3)
        encoder_path.parent.mkdir(parents=True, exist_ok=True)
        encoder_path.write_text("fp16")
        np.savez(decoder_path, weight=np.zeros(1))

    def write_int8(openvino, exported, ir_path) -> None:
        int8_writes.append(ir_path)
        ir_path.write_text("int8")

    monkeypatch.setattr(adapter, "import_openvino", lambda: _FakeOpenVino(_FakeCore({"NPU": "ok", "CPU": "ok"})))
    monkeypatch.setattr(adapter, "_wake_npu", lambda openvino, core, cache_dir: None if "NPU" in core.available_devices else "no NPU device")
    monkeypatch.setattr(adapter, "loadable_snapshot_dir", lambda model_dir, repo_id: tmp_path)
    monkeypatch.setattr(adapter.OpenVinoParakeetAdapter, "_export", slow_export)
    monkeypatch.setattr(adapter.OpenVinoParakeetAdapter, "_cpu_engine", lambda self: None)
    monkeypatch.setattr(adapter, "_write_int8_encoder", write_int8)
    monkeypatch.setattr(adapter, "_load_decoder", lambda openvino, path: _Decoder())
    monkeypatch.setattr(adapter, "bundled_tokenizer", _Tokenizer)
    monkeypatch.setattr(adapter, "set_numpy_blas_threads", lambda count: None)
    startup, swap = (adapter.OpenVinoParakeetAdapter(tmp_path, "FermionResearch/Phonon-2.5-English") for _ in range(2))

    first = threading.Thread(target=startup.load_model)
    first.start()
    exporting.wait(timeout=5)
    swap.load_model()
    first.join(timeout=5)

    assert startup.is_model_loaded() and swap.is_model_loaded()
    assert len(exports) == 1 and len(int8_writes) == 1
