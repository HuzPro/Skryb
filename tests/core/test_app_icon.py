import shutil
from pathlib import Path

import pytest
from PIL import Image

from skryb.core import app_icon
from skryb.core.app_icon import ICO_SIZES, app_icon_bytes, paint_app_icon
from skryb.core.brand_mark import EMERALD_DEEP, EMERALD_LIGHT, TILE_GRADIENTS, render_mark, svg

_SETUPS_ICON = Path(__file__).resolve().parents[2] / "brand" / "skryb.ico"


@pytest.fixture
def shell_told(monkeypatch) -> list[None]:
    """Stands in for SHChangeNotify, so no test ever pokes the real Explorer."""
    calls: list[None] = []
    monkeypatch.setattr(app_icon, "_tell_shell_icons_changed", lambda: calls.append(None))
    return calls


@pytest.fixture
def install(tmp_path) -> Path:
    """A copy as Skryb-Setup lays it out, in a folder with spaces and accents; returns ``{app}``."""
    app = tmp_path / "Prógrams Ünïcode" / "Skryb"
    (app / "python").mkdir(parents=True)
    (app / "unins000.exe").write_bytes(b"MZ")
    shutil.copyfile(_SETUPS_ICON, app / "skryb.ico")
    return app


def _nearest_light_colour(pixel: tuple[int, ...]) -> str:
    return min(
        TILE_GRADIENTS,
        key=lambda name: sum((a - b) ** 2 for a, b in zip(pixel[:3], TILE_GRADIENTS[name][0])),
    )


class TestPaintingTheInstalledIcon:
    @pytest.mark.parametrize("colourway", list(TILE_GRADIENTS))
    def test_every_size_is_painted_on_the_chosen_colourways_tile(self, install, shell_told, colourway):
        assert paint_app_icon(colourway, install / "python")

        with Image.open(install / "skryb.ico") as icon:
            assert icon.info["sizes"] == {(size, size) for size in ICO_SIZES}
            light = TILE_GRADIENTS[colourway][0]
            top_left = icon.convert("RGBA").getpixel((24, 24))
            assert top_left[3] == 255
            assert all(abs(got - want) <= 12 for got, want in zip(top_left[:3], light)), (top_left, light)
            for size in ICO_SIZES:
                icon.size = (size, size)
                corner = round(size * 0.12)
                assert _nearest_light_colour(icon.convert("RGBA").getpixel((corner, corner))) == colourway, size
        assert shell_told == [None]

    def test_a_source_checkout_never_has_its_icon_painted(self, tmp_path, shell_told):
        checkout = tmp_path / "Skryb"
        (checkout / ".venv").mkdir(parents=True)
        (checkout / "brand").mkdir()
        shutil.copyfile(_SETUPS_ICON, checkout / "brand" / "skryb.ico")
        before = sorted(checkout.rglob("*"))

        assert not paint_app_icon("plum", checkout / ".venv")

        assert sorted(checkout.rglob("*")) == before
        assert (checkout / "brand" / "skryb.ico").read_bytes() == _SETUPS_ICON.read_bytes()
        assert shell_told == []

    def test_painting_the_same_colourway_again_leaves_the_file_and_explorer_alone(self, install, shell_told):
        assert paint_app_icon("terracotta", install / "python")
        written = (install / "skryb.ico").stat().st_mtime_ns

        for _ in range(3):
            assert not paint_app_icon("terracotta", install / "python")

        assert (install / "skryb.ico").stat().st_mtime_ns == written
        assert shell_told == [None]

    def test_switching_back_and_forth_repaints_each_time(self, install, shell_told):
        for colourway in ("ink", "graphite", "ink", "green"):
            assert paint_app_icon(colourway, install / "python")
            assert (install / "skryb.ico").read_bytes() == app_icon_bytes(colourway)

        assert len(shell_told) == 4
        assert not (install / "skryb.ico.tmp").exists()

    def test_an_icon_another_program_holds_open_is_logged_and_left_as_it_was(
        self, install, shell_told, monkeypatch, app_caplog,
    ):
        def locked(source, target):
            raise PermissionError(32, "The process cannot access the file because it is being used by another process")

        monkeypatch.setattr(app_icon.os, "replace", locked)

        assert not paint_app_icon("plum", install / "python")

        assert (install / "skryb.ico").read_bytes() == _SETUPS_ICON.read_bytes()
        assert "Could not paint the app icon plum" in app_caplog.text
        assert shell_told == []

    def test_an_icon_path_that_cannot_be_written_does_not_raise(self, install, shell_told):
        (install / "skryb.ico").unlink()
        (install / "skryb.ico").mkdir()

        assert not paint_app_icon("plum", install / "python")

        assert (install / "skryb.ico").is_dir()
        assert shell_told == []

    @pytest.mark.parametrize("colourway", ["teal", "", "Green", "plum ", "émeraude"])
    def test_an_unknown_colourway_is_painted_green(self, install, shell_told, colourway):
        assert paint_app_icon(colourway, install / "python")

        assert (install / "skryb.ico").read_bytes() == app_icon_bytes("green")


class TestTheTrayMark:
    @pytest.mark.parametrize("size", [16, 24, 32, 48, 256])
    def test_the_mark_is_drawn_on_the_emerald_tile_unless_asked_otherwise(self, size):
        emerald = render_mark(size, gradient=(EMERALD_LIGHT, EMERALD_DEEP))

        assert render_mark(size).tobytes() == emerald.tobytes()
        assert render_mark(size, gradient=TILE_GRADIENTS["plum"]).tobytes() != emerald.tobytes()

    def test_green_is_the_brand_emerald_and_the_svg_still_uses_it(self):
        assert TILE_GRADIENTS["green"] == ((52, 211, 153), (4, 120, 87)) == (EMERALD_LIGHT, EMERALD_DEEP)
        assert 'stop-color="#34D399"' in svg() and 'stop-color="#047857"' in svg()
        assert 'stop-color="#E09AD2"' in svg(gradient=TILE_GRADIENTS["plum"])
