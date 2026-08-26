from types import ModuleType
from unittest.mock import patch

from voicetotext.core.torch_prewarm import start_torch_prewarm


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
