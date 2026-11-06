from pathlib import Path
from typing import IO

import pytest

from skryb.translation.gloss import gloss_grammar
from skryb.translation.llama_server import (
    BASE_FLAGS,
    Endpoint,
    LlamaServer,
    ServerError,
    ServerLaunch,
    chat,
    list_devices,
)
from tests.fakes.llama_server import FakeLlamaServer


class Child:
    def __init__(self, pid: int, exits_with: int | None = None) -> None:
        self.pid = pid
        self.returncode = exits_with
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class Starts:
    """Hands out the given children in turn and records how each was started."""

    def __init__(self, *children: Child) -> None:
        self.children = list(children)
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, args: list[str], *, env: dict[str, str], stdout: IO[str], cwd: str) -> Child:
        self.calls.append((args, env))
        stdout.write("model loading\n")
        return self.children.pop(0)


def server(tmp_path: Path, starts: Starts, healthy: bool = True, timeout: float = 5.0) -> tuple[LlamaServer, list[int]]:
    bound: list[int] = []
    launch = ServerLaunch(tmp_path / "llama-server.exe", tmp_path / "model.gguf", ("--device", "ROCm0"), {"PATH": "rocm"})
    return LlamaServer(
        lambda: launch, tmp_path / "logs" / "translation-server.log", start_child=starts, is_healthy=lambda _endpoint: healthy,
        bind=bound.append, readiness_timeout=timeout, poll_seconds=0.001,
    ), bound


def test_nothing_starts_until_the_first_request_asks_for_the_server(tmp_path: Path) -> None:
    starts = Starts(Child(41))
    llama, _bound = server(tmp_path, starts)
    assert starts.calls == []
    llama.endpoint()
    assert len(starts.calls) == 1


def test_a_running_server_is_reused_by_every_later_request(tmp_path: Path) -> None:
    starts = Starts(Child(41))
    llama, _bound = server(tmp_path, starts)
    assert llama.endpoint() == llama.endpoint()
    assert len(starts.calls) == 1


def test_a_server_that_died_is_started_again_by_the_next_request(tmp_path: Path) -> None:
    first, second = Child(41), Child(42)
    starts = Starts(first, second)
    llama, bound = server(tmp_path, starts)
    llama.endpoint()
    first.returncode = 3
    llama.endpoint()
    assert len(starts.calls) == 2 and bound == [41, 42]


def test_the_child_gets_the_model_device_and_loopback_port_but_its_key_only_in_its_environment(tmp_path: Path) -> None:
    starts = Starts(Child(41))
    llama, bound = server(tmp_path, starts)
    endpoint = llama.endpoint()
    args, env = starts.calls[0]
    assert args[:3] == [str(tmp_path / "llama-server.exe"), "-m", str(tmp_path / "model.gguf")]
    assert ["--host", "127.0.0.1", "--port", endpoint.base_url.rsplit(":", 1)[1]] == args[3:7]
    assert args[7:] == [*BASE_FLAGS, "--device", "ROCm0"]
    assert env["LLAMA_API_KEY"] == endpoint.api_key and endpoint.api_key not in " ".join(args)
    assert env["PATH"] == "rocm" and bound == [41]


def test_a_child_that_exits_while_loading_is_reported_with_the_end_of_its_log(tmp_path: Path) -> None:
    llama, _bound = server(tmp_path, Starts(Child(41, exits_with=1)), healthy=False)
    with pytest.raises(ServerError, match="stopped while loading. model loading"):
        llama.endpoint()


def test_a_child_that_never_becomes_healthy_is_stopped_and_reported(tmp_path: Path) -> None:
    child = Child(41)
    llama, _bound = server(tmp_path, Starts(child), healthy=False, timeout=0.05)
    with pytest.raises(ServerError, match="did not load within"):
        llama.endpoint()
    assert child.killed


def test_stopping_kills_the_child_and_the_next_request_starts_a_new_one(tmp_path: Path) -> None:
    first, second = Child(41), Child(42)
    starts = Starts(first, second)
    llama, _bound = server(tmp_path, starts)
    llama.endpoint()
    llama.stop()
    assert first.killed
    llama.endpoint()
    assert len(starts.calls) == 2


def test_a_chat_answer_is_the_assistant_text() -> None:
    with FakeLlamaServer() as fake:
        assert chat(fake.endpoint, "Translate this:\n\nوين؟", 32, None) == "EN: وين؟"
        assert fake.requests[0]["temperature"] == 0 and fake.requests[0]["max_tokens"] == 32


def test_chat_options_go_to_the_template_and_none_are_sent_by_default() -> None:
    with FakeLlamaServer() as fake:
        chat(fake.endpoint, "Translate this:\n\nوين؟", 32, None, options={"reasoning_effort": "low"})
        chat(fake.endpoint, "Translate this:\n\nوين؟", 32, None)
    assert fake.requests[0]["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert "chat_template_kwargs" not in fake.requests[1]


def test_a_chat_answer_cut_short_by_the_token_limit_is_refused() -> None:
    with FakeLlamaServer() as fake:
        fake.finish_reason = "length"
        with pytest.raises(ServerError, match="cut short"):
            chat(fake.endpoint, "Translate this:\n\nوين؟", 32, None)


def test_a_grammar_locked_answer_filed_as_reasoning_is_still_the_answer_but_free_text_reasoning_is_not() -> None:
    grammar = gloss_grammar(["وين", "المفتاح؟"])
    with FakeLlamaServer() as fake:
        fake.answers_as_reasoning = True
        assert chat(fake.endpoint, "Gloss this", 64, grammar) == '[{"w":"وين","g":"gloss of وين"},{"w":"المفتاح؟","g":"gloss of المفتاح؟"}]'
        assert chat(fake.endpoint, "Translate this:\n\nوين؟", 32, None) == ""


def test_a_chat_with_the_wrong_key_or_no_server_is_refused() -> None:
    with FakeLlamaServer() as fake:
        with pytest.raises(ServerError, match="did not answer"):
            chat(Endpoint(fake.endpoint.base_url, "wrong"), "hi", 8, None)
    with pytest.raises(ServerError, match="did not answer"):
        chat(fake.endpoint, "hi", 8, None, timeout=1.0)


def test_devices_are_read_from_the_builds_own_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listing = (
        "Available devices:\n"
        "  ROCm0: AMD Radeon RX 9060 XT (16304 MiB, 16158 MiB free)\n"
        "  ROCm1: AMD Radeon(TM) Graphics (12455 MiB, 12172 MiB free)\n"
        "  OPENVINO0: GGML_OPENVINO_DEVICE=NPU (selected) - Intel(R) AI Boost (18000 MiB, 18000 MiB free)\n"
    )

    class Finished:
        stdout, stderr = listing, ""

    monkeypatch.setattr("skryb.translation.llama_server._run_listing", lambda runtime, env: Finished())
    assert list_devices(tmp_path / "llama-server.exe", {}) == [
        ("ROCm0", "AMD Radeon RX 9060 XT", 16304),
        ("ROCm1", "AMD Radeon(TM) Graphics", 12455),
        ("OPENVINO0", "GGML_OPENVINO_DEVICE=NPU (selected) - Intel(R) AI Boost", 18000),
    ]


def test_a_build_that_lists_no_device_lists_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Finished:
        stdout, stderr = "Available devices:\n  (none)\n", ""

    monkeypatch.setattr("skryb.translation.llama_server._run_listing", lambda runtime, env: Finished())
    assert list_devices(tmp_path / "llama-server.exe", {}) == []
