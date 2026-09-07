import threading
from types import ModuleType
from unittest.mock import MagicMock, patch

from voicetotext.core.torch_prewarm import (
    _warm_gpu_kernels,
    start_torch_prewarm,
)


class TestTorchPrewarm:
    def test_should_import_torch_on_daemon_worker(self) -> None:
        torch_module = ModuleType("torch")
        with patch(
            "voicetotext.core.torch_prewarm.importlib.import_module",
            return_value=torch_module,
        ) as import_module:
            future = start_torch_prewarm()

        assert future.result(timeout=1) is torch_module
        import_module.assert_called_once_with("torch")

    def test_should_surface_import_failure_through_future(self) -> None:
        failure = ImportError("broken torch")
        with patch(
            "voicetotext.core.torch_prewarm.importlib.import_module",
            side_effect=failure,
        ):
            future = start_torch_prewarm()

        assert future.exception(timeout=1) is failure


class TestGpuPrewarm:
    def test_should_warm_gpu_kernels_after_resolving_the_future(self) -> None:
        warmed = threading.Event()
        torch_module = MagicMock()
        torch_module.cuda.is_available.return_value = True
        torch_module.cuda.synchronize.side_effect = warmed.set
        torch_module.randn.return_value.__gt__.return_value = MagicMock()
        with patch(
            "voicetotext.core.torch_prewarm.importlib.import_module",
            return_value=torch_module,
        ):
            future = start_torch_prewarm()

        assert future.result(timeout=1) is torch_module
        assert warmed.wait(timeout=10)
        assert torch_module.isin.called
        assert torch_module.nn.functional.conv1d.called

    def test_should_skip_warming_without_a_gpu(self) -> None:
        torch_module = MagicMock()
        torch_module.cuda.is_available.return_value = False

        _warm_gpu_kernels(torch_module)

        assert not torch_module.empty.called

    def test_should_start_even_when_gpu_warming_fails(self) -> None:
        torch_module = MagicMock()
        torch_module.cuda.is_available.side_effect = RuntimeError("no driver")

        _warm_gpu_kernels(torch_module)
