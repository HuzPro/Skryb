import time
from collections.abc import Callable
from pathlib import Path

from tests.fakes.writing import FakeChild, FakeLaunch
from skryb.writing.client import Endpoint
from skryb.writing.llama_server import (
    BACKOFF_SECONDS,
    LlamaServer,
    ServerFiles,
    ServerState,
    ServerStatus,
    launch_flags,
    log_tail,
)

_FULL_HELP = (
    "-fa,   --flash-attn [on|off|auto]  set Flash Attention use\n"
    "-rea,  --reasoning [on|off|auto]   Use reasoning\n"
    "--spec-type none,ngram-simple,ngram-mod\n"
    "--spec-ngram-simple-size-n N\n--spec-ngram-simple-size-m N\n"
)


def _files(tmp_path: Path) -> ServerFiles:
    runtime = tmp_path / "llama-b11200" / "llama-server.exe"
    runtime.parent.mkdir()
    runtime.write_bytes(b"MZ")
    model = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    model.write_bytes(b"GGUF")
    return ServerFiles(runtime, model)


def _server(
    tmp_path: Path,
    launch: FakeLaunch,
    healthy: Callable[[Endpoint], bool] = lambda _endpoint: True,
    backoff: tuple[float, ...] = (0.0, 0.0, 0.0),
    readiness_timeout: float = 5.0,
    locate: Callable[[], ServerFiles | None] | None = None,
    help_text: str = _FULL_HELP,
) -> tuple[LlamaServer, list[ServerStatus], list[int]]:
    files = _files(tmp_path)
    statuses: list[ServerStatus] = []
    bound: list[int] = []
    server = LlamaServer(
        locate=locate or (lambda: files),
        log_path=tmp_path / "session" / "writing-server.log",
        warm_up=lambda _endpoint: None,
        launch=launch,
        is_healthy=healthy,
        read_help=lambda _runtime: help_text,
        bind=bound.append,
        backoff=backoff,
        readiness_timeout=readiness_timeout,
        poll_seconds=0.005,
    )
    server.subscribe(statuses.append)
    return server, statuses, bound


def _wait_for(condition: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition never became true")
        time.sleep(0.005)


class TestSupervision:
    def test_a_healthy_child_becomes_ready_with_a_loopback_endpoint_and_is_tied_to_the_app(
        self, tmp_path,
    ):
        child = FakeChild(pid=101)
        server, _statuses, bound = _server(tmp_path, FakeLaunch([child]))

        server.start()
        _wait_for(server.ready_event.is_set)

        endpoint = server.endpoint()
        assert endpoint is not None and endpoint.base_url.startswith("http://127.0.0.1:")
        assert server.status.state is ServerState.READY
        assert bound == [101]
        server.stop()

    def test_the_supervisor_restarts_a_child_that_exits_after_becoming_ready(self, tmp_path):
        first, second = FakeChild(pid=1), FakeChild(pid=2)
        launch = FakeLaunch([first, second])
        server, _statuses, _bound = _server(tmp_path, launch)
        server.start()
        _wait_for(server.ready_event.is_set)
        first_endpoint = server.endpoint()

        first.exit(3)
        _wait_for(lambda: len(launch.launched) == 2 and server.ready_event.is_set())

        assert server.endpoint() != first_endpoint
        server.stop()

    def test_the_supervisor_backs_off_2_10_60_seconds_then_gives_up_with_the_log_tail(
        self, tmp_path,
    ):
        crashes = [FakeChild(pid=pid, exits_while_loading=1) for pid in range(4)]
        launch = FakeLaunch(crashes, log_line="ggml_vulkan: out of device memory")
        server, statuses, _bound = _server(
            tmp_path, launch, healthy=lambda _endpoint: False, backoff=(0.01, 0.02, 0.03),
        )

        server.start()
        _wait_for(lambda: server.status.state is ServerState.FAILED)

        assert BACKOFF_SECONDS == (2.0, 10.0, 60.0)
        assert len(launch.launched) == 4
        restarts = [s.detail for s in statuses if s.detail.startswith("Restarting")]
        assert [detail.split(":")[0] for detail in restarts] == [
            "Restarting in 0.01 s", "Restarting in 0.02 s", "Restarting in 0.03 s",
        ]
        assert "out of device memory (pid 3)" in server.status.detail
        assert len(server.status.detail.splitlines()) <= 20

    def test_a_child_that_never_becomes_healthy_is_killed_at_the_readiness_timeout(
        self, tmp_path,
    ):
        stuck = FakeChild(pid=7)
        launch = FakeLaunch([stuck])
        server, _statuses, _bound = _server(
            tmp_path, launch, healthy=lambda _endpoint: False, readiness_timeout=0.05,
        )

        server.start()
        _wait_for(lambda: server.status.state is ServerState.FAILED)

        assert stuck.killed
        assert len(launch.launched) == 1
        assert server.endpoint() is None

    def test_stop_kills_the_child_and_forgets_the_endpoint(self, tmp_path):
        child = FakeChild(pid=9)
        server, _statuses, _bound = _server(tmp_path, FakeLaunch([child]))
        server.start()
        _wait_for(server.ready_event.is_set)

        server.stop()

        assert child.killed
        assert server.endpoint() is None
        assert server.status.state is ServerState.STOPPED
        assert not server.ready_event.is_set()

    def test_missing_files_report_not_installed_and_launch_nothing(self, tmp_path):
        launch = FakeLaunch([])
        server, _statuses, _bound = _server(tmp_path, launch, locate=lambda: None)

        server.start()
        _wait_for(lambda: server.status.state is ServerState.NOT_INSTALLED)

        assert launch.launched == []

    def test_start_after_giving_up_launches_a_fresh_supervisor(self, tmp_path):
        crashes = [FakeChild(pid=pid, exits_while_loading=1) for pid in range(4)]
        launch = FakeLaunch([*crashes, FakeChild(pid=50)])
        server, _statuses, _bound = _server(tmp_path, launch, backoff=(0.0, 0.0, 0.0))
        server.start()
        _wait_for(lambda: server.status.state is ServerState.FAILED)

        server.start()
        _wait_for(server.ready_event.is_set)

        assert launch.launched[-1].pid == 50
        server.stop()


    def test_a_retry_the_moment_the_server_gives_up_is_not_lost(self, tmp_path):
        crashes = [FakeChild(pid=pid, exits_while_loading=1) for pid in range(4)]
        launch = FakeLaunch([*crashes, FakeChild(pid=50)])
        server, _statuses, _bound = _server(tmp_path, launch, backoff=(0.0, 0.0, 0.0))
        retried: list[bool] = []

        def retry_once_on_failure(status: ServerStatus) -> None:
            if status.state is ServerState.FAILED and not retried:
                retried.append(True)
                server.start()

        server.subscribe(retry_once_on_failure)
        server.start()
        _wait_for(server.ready_event.is_set)

        assert launch.launched[-1].pid == 50
        server.stop()

    def test_a_retry_the_moment_the_model_is_found_missing_is_not_lost(self, tmp_path):
        (tmp_path / "installed").mkdir()
        files = _files(tmp_path / "installed")
        located = iter([None, files])
        launch = FakeLaunch([FakeChild(pid=7)])
        server, _statuses, _bound = _server(tmp_path, launch, locate=lambda: next(located))
        retried: list[bool] = []

        def retry_once_when_missing(status: ServerStatus) -> None:
            if status.state is ServerState.NOT_INSTALLED and not retried:
                retried.append(True)
                server.start()

        server.subscribe(retry_once_when_missing)
        server.start()
        _wait_for(server.ready_event.is_set)

        assert [child.pid for child in launch.launched] == [7]
        server.stop()


class TestLaunchCommand:
    def test_the_api_key_is_passed_to_the_child_but_never_written_to_the_log(self, tmp_path):
        launch = FakeLaunch([FakeChild(pid=3)])
        server, _statuses, _bound = _server(tmp_path, launch)
        server.start()
        _wait_for(server.ready_event.is_set)
        endpoint = server.endpoint()
        server.stop()

        assert endpoint is not None
        args = launch.args[0]
        assert args[args.index("--api-key") + 1] == endpoint.api_key
        assert endpoint.api_key not in (tmp_path / "session" / "writing-server.log").read_text(
            encoding="utf-8",
        )
        assert args[args.index("--host") + 1] == "127.0.0.1"

    def test_optional_flags_follow_what_the_build_says_it_supports(self):
        full = launch_flags(_FULL_HELP)
        bare = launch_flags("usage: llama-server [options]")

        assert ("--reasoning", "off") == full[full.index("--reasoning"):full.index("--reasoning") + 2]
        assert "ngram-simple" in full and "-fa" in full
        assert "--reasoning" not in bare and "--spec-type" not in bare and "-fa" not in bare
        for flag in ("-c", "8192", "-ngl", "99", "-np", "1", "--no-webui", "--offline"):
            assert flag in bare


def test_the_log_tail_keeps_the_last_twenty_lines_of_a_large_log(tmp_path):
    log = tmp_path / "writing-server.log"
    log.write_text("".join(f"line {index} ✓ ümlaut\n" for index in range(5000)), encoding="utf-8")

    tail = log_tail(log).splitlines()

    assert len(tail) == 20
    assert tail[-1] == "line 4999 ✓ ümlaut"
    assert log_tail(tmp_path / "missing.log") == ""
