"""The picture Windows Explorer shows for a video, its cover art or a frame, kept beside the meeting made from it.

The Windows Shell makes it, as it does for Explorer: no FFmpeg. A file with no picture of its own, such as most audio
files, gets none, because the Shell's file-type icon is refused.
"""

import ctypes
import logging
import os
from collections.abc import Callable
from ctypes import POINTER, byref, c_int32, c_uint16, c_uint32, c_void_p, wintypes
from pathlib import Path

from PIL import Image

from skryb.meetings.media_file import _call, _Guid, _guid, _release

THUMBNAIL_JPG = "thumbnail.jpg"
_LONG_EDGE = 480
_JPEG_QUALITY = 85
_COINIT_APARTMENTTHREADED = 0x2
_SIIGBF_THUMBNAILONLY = 0x8
_IID_SHELL_ITEM_IMAGE_FACTORY = "bcc18b79-ba16-442f-80c4-8a59c30c463b"
_GET_IMAGE = 3  # IShellItemImageFactory vtable slot, after IUnknown's three
_DIB_RGB_COLORS = 0
_BI_RGB = 0
_log = logging.getLogger(__name__)

# The file's picture, fitted in a square this many pixels wide; raises when the file has none.
Grab = Callable[[Path, int], Image.Image]


class _BitmapInfo(ctypes.Structure):
    """BITMAPINFOHEADER, with room after it for the colour masks GetDIBits may write."""

    _fields_ = [
        ("size", c_uint32), ("width", c_int32), ("height", c_int32), ("planes", c_uint16), ("bit_count", c_uint16),
        ("compression", c_uint32), ("size_image", c_uint32), ("x_pixels_per_metre", c_int32),
        ("y_pixels_per_metre", c_int32), ("colours_used", c_uint32), ("colours_important", c_uint32),
        ("masks", c_uint32 * 3),
    ]


def save_thumbnail(source: Path, folder: Path, grab: Grab) -> None:
    """*source*'s picture as ``thumbnail.jpg`` in *folder*, at most 480 px a side. Never raises: no picture, no file."""
    try:
        _write_thumbnail(source, folder, grab)
    except Exception as exc:
        _log.info("No thumbnail for %s: %s", source.name, exc)


def _write_thumbnail(source: Path, folder: Path, grab: Grab) -> None:
    image = grab(source, _LONG_EDGE).convert("RGB")
    image.thumbnail((_LONG_EDGE, _LONG_EDGE))
    partial = folder / (THUMBNAIL_JPG + ".tmp")
    image.save(partial, "JPEG", quality=_JPEG_QUALITY)
    os.replace(partial, folder / THUMBNAIL_JPG)


def grab_shell_thumbnail(path: Path, size: int) -> Image.Image:
    """The thumbnail Explorer shows for *path*, fitted in *size* px; OSError when the Shell has only an icon."""
    ole32 = ctypes.windll.ole32
    initialized = ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED) in (0, 1)  # S_OK, S_FALSE
    factory, bitmap = c_void_p(), wintypes.HBITMAP()
    try:
        return _ask_the_shell_for_a_thumbnail(path, size, factory, bitmap)
    finally:
        if bitmap:
            ctypes.windll.gdi32.DeleteObject(bitmap)
        _release(factory)
        if initialized:
            ole32.CoUninitialize()


def _ask_the_shell_for_a_thumbnail(path: Path, size: int, factory: c_void_p, bitmap: wintypes.HBITMAP) -> Image.Image:
    create = ctypes.windll.shell32.SHCreateItemFromParsingName
    create.restype = ctypes.HRESULT
    create.argtypes = [ctypes.c_wchar_p, c_void_p, POINTER(_Guid), POINTER(c_void_p)]
    create(str(path), None, byref(_guid(_IID_SHELL_ITEM_IMAGE_FACTORY)), byref(factory))
    _call(factory, _GET_IMAGE, (wintypes.SIZE, wintypes.SIZE(size, size)), (ctypes.c_int, _SIIGBF_THUMBNAILONLY),
          (POINTER(wintypes.HBITMAP), byref(bitmap)))
    return _bitmap_image(bitmap)


def _bitmap_image(bitmap: wintypes.HBITMAP) -> Image.Image:
    user32 = ctypes.windll.user32
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    screen = user32.GetDC(None)
    try:
        return _read_pixels(screen, bitmap)
    finally:
        user32.ReleaseDC(None, screen)


def _read_pixels(screen: int, bitmap: wintypes.HBITMAP) -> Image.Image:
    """The bitmap as top-down 32-bit rows, whatever its own layout."""
    get_bits = ctypes.windll.gdi32.GetDIBits
    get_bits.argtypes = [wintypes.HDC, wintypes.HBITMAP, c_uint32, c_uint32, c_void_p, POINTER(_BitmapInfo), c_uint32]
    info = _BitmapInfo(size=ctypes.sizeof(_BitmapInfo) - ctypes.sizeof(c_uint32 * 3))
    if not get_bits(screen, bitmap, 0, 0, None, byref(info), _DIB_RGB_COLORS):
        raise OSError("Windows didn't say how big the thumbnail is.")
    width, height = info.width, abs(info.height)
    info.height, info.bit_count, info.compression = -height, 32, _BI_RGB
    pixels = ctypes.create_string_buffer(width * height * 4)
    if get_bits(screen, bitmap, 0, height, pixels, byref(info), _DIB_RGB_COLORS) != height:
        raise OSError("Windows didn't hand over the thumbnail's pixels.")
    return Image.frombuffer("RGB", (width, height), pixels.raw, "raw", "BGRX", 0, 1)
