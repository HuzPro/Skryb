from voicetotext.core.single_instance import SingleInstanceMutex


class FakeMutexApi:
    def __init__(self, error: int = 0) -> None:
        self.error = error
        self.closed: list[int] = []

    def create_mutex(self, name: str) -> int:
        assert name == "Local\\VoiceToText"
        return 42

    def last_error(self) -> int:
        return self.error

    def close_handle(self, handle: int) -> None:
        self.closed.append(handle)


class TestSingleInstanceMutex:
    def test_should_retain_handle_until_release(self) -> None:
        api = FakeMutexApi()
        mutex = SingleInstanceMutex(api)

        assert mutex.acquire() is True
        assert api.closed == []

        mutex.release()
        assert api.closed == [42]

    def test_should_close_conflicting_handle_and_reject_second_instance(self) -> None:
        api = FakeMutexApi(error=183)
        mutex = SingleInstanceMutex(api)

        assert mutex.acquire() is False
        assert api.closed == [42]
