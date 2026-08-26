import importlib
import threading
from concurrent.futures import Future
from types import ModuleType


def start_torch_prewarm() -> Future[ModuleType]:
    future: Future[ModuleType] = Future()

    def import_torch() -> None:
        try:
            future.set_result(importlib.import_module("torch"))
        except BaseException as exc:
            future.set_exception(exc)

    threading.Thread(
        target=import_torch,
        daemon=True,
        name="torch-prewarm",
    ).start()
    return future
