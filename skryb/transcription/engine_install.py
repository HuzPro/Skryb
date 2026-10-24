from collections.abc import Callable

from skryb.core.exceptions import EngineSwapError


# An installer attaches a freshly-loaded engine to the use case. The
# loader returns one of these instead of a bare engine so the hot-swap
# coordinator stays ignorant of whether the engine is batch or streaming.
EngineInstaller = Callable[[object], None]
_InstallStep = Callable[[object, object], None]


def make_engine_installer(
    engine: object, install: _InstallStep,
) -> EngineInstaller:
    """Wrap *engine* + its *install* step into an idempotent installer.

    The installer disposes *engine* if the use case rejects the swap
    (e.g. a recording is in progress), so a rejected load never leaks a
    loaded model.
    """

    def installer(use_case: object) -> None:
        try:
            install(use_case, engine)
        except EngineSwapError:
            engine.dispose()
            raise

    return installer
