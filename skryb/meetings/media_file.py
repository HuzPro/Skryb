"""Sound from a video or audio file through Windows' own Media Foundation: no FFmpeg, nothing to install.

Microsoft documents MP4, MOV, M4A, M4V, 3GP, WMV, WMA, ASF, AVI, MP3, AAC and WAV ("Supported Media Formats
in Media Foundation"); Windows 11 also reads MKV, WebM and FLAC (checked 2026-10-29), but not standalone OGG
or Opus files. Only the first sound track is decoded, so the video codec never matters.
"""

import ctypes
import uuid
from collections.abc import Callable
from ctypes import POINTER, byref, c_int64, c_uint32, c_ulong, c_void_p
from pathlib import Path

import numpy as np

from skryb.meetings.capture import RATE, Resampler

SUPPORTED_EXTENSIONS = (
    ".mp4", ".mov", ".mkv", ".webm", ".m4a", ".m4v", ".3gp", ".3g2", ".wmv", ".wma", ".asf", ".avi", ".mp3",
    ".aac", ".flac", ".wav",
)

_MF_VERSION = 0x00020070
_FIRST_AUDIO_STREAM = 0xFFFFFFFD
_ALL_STREAMS = 0xFFFFFFFE
_MEDIA_SOURCE = 0xFFFFFFFF
_END_OF_STREAM = 0x2
_STREAM_ERROR = 0x1
_TYPE_CHANGED = 0x20
_VT_UI8 = 21

_MF_MT_MAJOR_TYPE = "48eba18e-f8c9-4687-bf11-0a74c9f96a8f"
_MF_MT_SUBTYPE = "f7e34c9a-42e8-4714-b74b-cb29d72c35e5"
_MF_MT_AUDIO_NUM_CHANNELS = "37e48bf5-645e-4c5b-89de-ada9e29b696a"
_MF_MT_AUDIO_SAMPLES_PER_SECOND = "5faeeae7-0290-4c31-9e8a-c534f68d9dba"
_MF_PD_DURATION = "6c990d33-bb8e-477a-8598-0d5d96fcd88a"
_MF_MEDIA_TYPE_AUDIO = "73647561-0000-0010-8000-00aa00389b71"
_MF_AUDIO_FORMAT_FLOAT = "00000003-0000-0010-8000-00aa00389b71"

# HRESULTs a person can do something about, as sentences.
_NOT_FOUND = (0x80070002, 0x80070003)
_UNSUPPORTED = (0xC00D36C4, 0xC00D36C3, 0xC00D36E6)  # unsupported byte stream type / scheme / format
_NO_SOUND = 0xC00D36B3  # invalid stream number: there is no audio stream
_NO_DECODER = (0xC00D36B4, 0xC00D5212)  # invalid media type, codec not found

# IUnknown, IMFAttributes, IMFMediaType, IMFSourceReader, IMFSample and IMFMediaBuffer vtable slots.
_RELEASE = 2
_GET_UINT32 = 7
_SET_GUID = 24
_SET_STREAM_SELECTION = 4
_GET_CURRENT_MEDIA_TYPE = 6
_SET_CURRENT_MEDIA_TYPE = 7
_READ_SAMPLE = 9
_GET_PRESENTATION_ATTRIBUTE = 12
_CONVERT_TO_CONTIGUOUS_BUFFER = 41
_LOCK = 3
_UNLOCK = 4


class MediaFileError(Exception):
    """A file whose sound can't be read; the message is a sentence to show as is."""


class _Guid(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 16)]


class _PropVariant(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_ushort), ("reserved", ctypes.c_ushort * 3), ("value", ctypes.c_ulonglong),
                ("padding", ctypes.c_ulonglong)]


def _guid(text: str) -> _Guid:
    return _Guid.from_buffer_copy(uuid.UUID(text).bytes_le)


def _call(obj: c_void_p, slot: int, *args: tuple[type, object]) -> None:
    """COM method *slot* of *obj*; a failing HRESULT raises OSError with it as winerror."""
    vtable = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
    prototype = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *(kind for kind, _ in args))
    prototype(vtable[slot])(obj, *(value for _, value in args))


def _release(obj: c_void_p) -> None:
    if obj:
        vtable = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
        ctypes.WINFUNCTYPE(c_ulong, c_void_p)(vtable[_RELEASE])(obj)


def decode_audio(
    path: Path,
    progress: Callable[[float], None] = lambda _fraction: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> np.ndarray:
    """The file's first sound track as 16 kHz mono float32; None-free, empty for a silent stub.

    *progress* hears the share of the file read so far; *cancelled* is asked between blocks and, once
    true, the reading stops and what was read so far comes back.
    """
    ole32 = ctypes.windll.ole32
    initialized = ole32.CoInitializeEx(None, 0) in (0, 1)  # S_OK, S_FALSE; another apartment model is fine too
    mfplat = ctypes.windll.mfplat
    mfplat.MFStartup(_MF_VERSION, 0)
    reader = c_void_p()
    try:
        _open(path, reader)
        rate, channels = _select_float_audio(reader, path)
        return _read_all(reader, rate, channels, _duration(reader), progress, cancelled)
    finally:
        _release(reader)
        mfplat.MFShutdown()
        if initialized:
            ole32.CoUninitialize()


def check(path: Path) -> None:
    """Raise MediaFileError now when Windows can't open *path* or finds no sound in it; reads no audio."""
    ole32 = ctypes.windll.ole32
    initialized = ole32.CoInitializeEx(None, 0) in (0, 1)
    mfplat = ctypes.windll.mfplat
    mfplat.MFStartup(_MF_VERSION, 0)
    reader = c_void_p()
    try:
        _open(path, reader)
        _select_float_audio(reader, path)
    finally:
        _release(reader)
        mfplat.MFShutdown()
        if initialized:
            ole32.CoUninitialize()


def _open(path: Path, reader: c_void_p) -> None:
    create = ctypes.windll.mfreadwrite.MFCreateSourceReaderFromURL
    create.restype = ctypes.HRESULT
    create.argtypes = [ctypes.c_wchar_p, c_void_p, POINTER(c_void_p)]
    try:
        create(str(path), None, byref(reader))
    except OSError as error:
        code = (error.winerror or 0) & 0xFFFFFFFF
        if code in _NOT_FOUND:
            raise MediaFileError(f"{path.name} isn't there any more.") from error
        if code in _UNSUPPORTED:
            raise MediaFileError(_unsupported(path)) from error
        raise MediaFileError(f"Windows couldn't open {path.name} (error 0x{code:08X}).") from error


def _select_float_audio(reader: c_void_p, path: Path) -> tuple[int, int]:
    media_type = c_void_p()
    try:
        _call(reader, _SET_STREAM_SELECTION, (c_uint32, _ALL_STREAMS), (ctypes.c_int, 0))
        _call(reader, _SET_STREAM_SELECTION, (c_uint32, _FIRST_AUDIO_STREAM), (ctypes.c_int, 1))
        mfplat = ctypes.windll.mfplat
        mfplat.MFCreateMediaType(byref(media_type))
        _call(media_type, _SET_GUID, (POINTER(_Guid), byref(_guid(_MF_MT_MAJOR_TYPE))),
              (POINTER(_Guid), byref(_guid(_MF_MEDIA_TYPE_AUDIO))))
        _call(media_type, _SET_GUID, (POINTER(_Guid), byref(_guid(_MF_MT_SUBTYPE))),
              (POINTER(_Guid), byref(_guid(_MF_AUDIO_FORMAT_FLOAT))))
        _call(reader, _SET_CURRENT_MEDIA_TYPE, (c_uint32, _FIRST_AUDIO_STREAM), (c_void_p, None), (c_void_p, media_type))
        return _current_format(reader)
    except OSError as error:
        code = (error.winerror or 0) & 0xFFFFFFFF
        if code == _NO_SOUND:
            raise MediaFileError(f"{path.name} has no sound to transcribe.") from error
        if code in _NO_DECODER:
            raise MediaFileError(f"Windows can't decode the sound in {path.name}. {_tip()}") from error
        raise MediaFileError(f"Windows couldn't read the sound in {path.name} (error 0x{code:08X}).") from error
    finally:
        _release(media_type)


def _current_format(reader: c_void_p) -> tuple[int, int]:
    current = c_void_p()
    try:
        _call(reader, _GET_CURRENT_MEDIA_TYPE, (c_uint32, _FIRST_AUDIO_STREAM), (POINTER(c_void_p), byref(current)))
        rate, channels = c_uint32(), c_uint32()
        _call(current, _GET_UINT32, (POINTER(_Guid), byref(_guid(_MF_MT_AUDIO_SAMPLES_PER_SECOND))),
              (POINTER(c_uint32), byref(rate)))
        _call(current, _GET_UINT32, (POINTER(_Guid), byref(_guid(_MF_MT_AUDIO_NUM_CHANNELS))),
              (POINTER(c_uint32), byref(channels)))
        return int(rate.value), max(1, int(channels.value))
    finally:
        _release(current)


def _duration(reader: c_void_p) -> float:
    """Seconds, or 0 when the file doesn't say."""
    value = _PropVariant()
    try:
        _call(reader, _GET_PRESENTATION_ATTRIBUTE, (c_uint32, _MEDIA_SOURCE),
              (POINTER(_Guid), byref(_guid(_MF_PD_DURATION))), (POINTER(_PropVariant), byref(value)))
    except OSError:
        return 0.0
    return value.value / 1e7 if value.vt == _VT_UI8 else 0.0


def _read_all(
    reader: c_void_p, rate: int, channels: int, duration: float,
    progress: Callable[[float], None], cancelled: Callable[[], bool],
) -> np.ndarray:
    resampler = Resampler(rate)
    pieces: list[np.ndarray] = []
    heard = 0
    while not cancelled():
        flags, timestamp, sample = c_uint32(), c_int64(), c_void_p()
        _call(reader, _READ_SAMPLE, (c_uint32, _FIRST_AUDIO_STREAM), (c_uint32, 0), (POINTER(c_uint32), None),
              (POINTER(c_uint32), byref(flags)), (POINTER(c_int64), byref(timestamp)), (POINTER(c_void_p), byref(sample)))
        try:
            if flags.value & _STREAM_ERROR:
                raise MediaFileError("Windows stopped partway through reading the sound.")
            if flags.value & _TYPE_CHANGED:
                pieces.append(resampler.flush())
                rate, channels = _current_format(reader)
                resampler = Resampler(rate)
            if sample:
                block = _samples(sample, channels)
                heard += len(block) * RATE // rate
                pieces.append(resampler(block))
                if duration:
                    progress(min(1.0, timestamp.value / 1e7 / duration))
            if flags.value & _END_OF_STREAM:
                break
        finally:
            _release(sample)
    pieces.append(resampler.flush())
    audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
    return audio[:heard]


def _samples(sample: c_void_p, channels: int) -> np.ndarray:
    """One block of interleaved float samples, down-mixed to mono."""
    buffer = c_void_p()
    try:
        _call(sample, _CONVERT_TO_CONTIGUOUS_BUFFER, (POINTER(c_void_p), byref(buffer)))
        data, length = c_void_p(), c_uint32()
        _call(buffer, _LOCK, (POINTER(c_void_p), byref(data)), (POINTER(c_uint32), None), (POINTER(c_uint32), byref(length)))
        try:
            raw = ctypes.string_at(data, length.value)
        finally:
            _call(buffer, _UNLOCK)
    finally:
        _release(buffer)
    interleaved = np.frombuffer(raw, np.float32)
    whole = len(interleaved) // channels * channels
    return interleaved[:whole].reshape(-1, channels).mean(axis=1)


def _unsupported(path: Path) -> str:
    return f"Skryb can't read {path.suffix or 'this'} files yet. {_tip()}"


def _tip() -> str:
    return "It reads MP4, MOV, MKV, WebM, M4A, WMV, AVI, MP3, AAC, FLAC and WAV."
