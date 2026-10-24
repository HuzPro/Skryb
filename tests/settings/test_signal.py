from skryb.settings.signal import Signal


class TestSignal:
    def test_should_invoke_connected_subscriber_with_emitted_args(self) -> None:
        received: list[tuple] = []
        signal = Signal()
        signal.connect(lambda *args: received.append(args))

        signal.emit("parakeet-tdt-0.6b-v3", 42)

        assert received == [("parakeet-tdt-0.6b-v3", 42)]

    def test_should_invoke_every_connected_subscriber(self) -> None:
        first: list = []
        second: list = []
        signal = Signal()
        signal.connect(first.append)
        signal.connect(second.append)

        signal.emit("x")

        assert first == ["x"]
        assert second == ["x"]

    def test_should_be_a_no_op_with_no_subscribers(self) -> None:
        Signal().emit("nothing-listens")
