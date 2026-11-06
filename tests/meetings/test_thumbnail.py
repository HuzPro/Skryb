"""The picture kept from the video a meeting was made from, with a fake Shell: no Windows needed."""

from pathlib import Path

import pytest
from PIL import Image

from skryb.meetings.thumbnail import THUMBNAIL_JPG, save_thumbnail
from tests.meetings.fakes import no_thumbnail, video_frame


def _folder(tmp_path: Path) -> Path:
    folder = tmp_path / "Meetings" / "2026-10-29 14-05 Weekly sync – ہفتہ وار"
    folder.mkdir(parents=True)
    return folder


@pytest.mark.parametrize(("size", "saved"), [
    ((1920, 1080), (480, 270)),
    ((1080, 1920), (270, 480)),
    ((481, 481), (480, 480)),
    ((480, 270), (480, 270)),
    ((200, 112), (200, 112)),
    ((4000, 10), (480, 1)),
    ((1, 1), (1, 1)),
])
def test_a_video_s_picture_is_kept_as_a_jpeg_no_more_than_480_pixels_on_its_long_edge(
    tmp_path: Path, size: tuple[int, int], saved: tuple[int, int],
) -> None:
    folder = _folder(tmp_path)

    save_thumbnail(tmp_path / "clip.mp4", folder, lambda _path, _size: Image.new("RGB", size, (176, 112, 64)))

    with Image.open(folder / THUMBNAIL_JPG) as kept:
        assert kept.format == "JPEG" and kept.size == saved
    assert [p.name for p in folder.iterdir()] == [THUMBNAIL_JPG]


@pytest.mark.parametrize("mode", ["RGBA", "P", "L", "CMYK"])
def test_cover_art_in_any_colour_mode_is_kept_as_an_rgb_jpeg(tmp_path: Path, mode: str) -> None:
    folder = _folder(tmp_path)

    save_thumbnail(tmp_path / "talk.mkv", folder, lambda _path, _size: Image.new(mode, (600, 600)))

    with Image.open(folder / THUMBNAIL_JPG) as kept:
        assert kept.mode == "RGB" and kept.size == (480, 480)


def test_the_shell_is_asked_for_the_source_file_at_480_pixels(tmp_path: Path) -> None:
    asked: list[tuple[Path, int]] = []
    source = tmp_path / "Ελληνικά – 会议 ✓.mov"

    save_thumbnail(source, _folder(tmp_path), lambda path, size: asked.append((path, size)) or video_frame(path, size))

    assert asked == [(source, 480)]


def test_a_file_the_shell_has_only_an_icon_for_gets_no_thumbnail_and_no_error(tmp_path: Path) -> None:
    folder = _folder(tmp_path)

    save_thumbnail(tmp_path / "standup.mp3", folder, no_thumbnail)

    assert list(folder.iterdir()) == []


@pytest.mark.parametrize("failure", [OSError("0x8004B200"), RuntimeError("the Shell crashed"), ValueError(""), MemoryError()])
def test_any_failure_while_making_the_thumbnail_is_logged_and_swallowed(
    tmp_path: Path, app_caplog: pytest.LogCaptureFixture, failure: Exception,
) -> None:
    folder = _folder(tmp_path)

    def broken(_path: Path, _size: int) -> Image.Image:
        raise failure

    save_thumbnail(tmp_path / "Q3 planning.mp4", folder, broken)

    assert list(folder.iterdir()) == []
    assert "No thumbnail for Q3 planning.mp4" in app_caplog.text


def test_a_meeting_folder_that_is_gone_gets_no_thumbnail_and_no_error(tmp_path: Path) -> None:
    gone = tmp_path / "Meetings" / "deleted meanwhile"

    save_thumbnail(tmp_path / "clip.mp4", gone, video_frame)

    assert not gone.exists()


def test_a_second_thumbnail_for_the_same_meeting_replaces_the_first(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    save_thumbnail(tmp_path / "clip.mp4", folder, lambda _path, _size: Image.new("RGB", (1920, 1080), "white"))

    save_thumbnail(tmp_path / "clip.mp4", folder, lambda _path, _size: Image.new("RGB", (640, 640), "black"))

    with Image.open(folder / THUMBNAIL_JPG) as kept:
        assert kept.size == (480, 480) and kept.getpixel((10, 10)) == (0, 0, 0)
