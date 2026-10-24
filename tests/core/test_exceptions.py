"""Tests for the domain exception hierarchy.

Each specialised exception must inherit from ``SkrybError`` so
call sites can catch every domain error with one ``except`` block,
and must carry its message intact.
"""

import pytest

from skryb.core.exceptions import (
    AudioRecorderError,
    ConfigurationError,
    HotkeyError,
    OutputError,
    TranscriptionError,
    SkrybError,
)


_SPECIALISED_EXCEPTIONS = [
    AudioRecorderError,
    TranscriptionError,
    OutputError,
    ConfigurationError,
    HotkeyError,
]


class TestDomainExceptionHierarchy:
    @pytest.mark.parametrize("exception_cls", _SPECIALISED_EXCEPTIONS)
    def test_specialised_exception_is_a_voice_to_text_error(
        self, exception_cls: type[SkrybError]
    ):
        assert issubclass(exception_cls, SkrybError)

    def test_voice_to_text_error_is_a_python_exception(self):
        assert issubclass(SkrybError, Exception)


class TestDomainExceptionMessage:
    @pytest.mark.parametrize("exception_cls", _SPECIALISED_EXCEPTIONS)
    def test_exception_carries_its_message(
        self, exception_cls: type[SkrybError]
    ):
        error = exception_cls("something broke")

        assert str(error) == "something broke"

    def test_exception_can_be_raised_and_caught_as_voice_to_text_error(self):
        with pytest.raises(SkrybError) as info:
            raise AudioRecorderError("mic unavailable")

        assert "mic unavailable" in str(info.value)
