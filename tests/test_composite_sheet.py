"""Unit tests for utils.composite_sheet — the free half of the sheet chain.

Fixture-driven, zero paid spend. The load-bearing assertion is that tile 3
(back, bottom-right) comes through the deface pass pixel-identical while tiles
0-2 have their head bands changed: the back tile keeps its head deliberately,
and a feather bleeding across the seam would silently blur it.
"""

import os
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.composite_sheet import (  # noqa: E402
    DEFACE_TILES,
    POSE_CX_FRAC,
    build_character_sheet,
    character_sheet_description,
    compose_character_sheet,
    deface_rotation_grid,
    resolve_grid,
)

TILE = 256
GRID_SIDE = TILE * 2  # 512x512, gutterless 2x2


def _make_grid(side: int = GRID_SIDE) -> Image.Image:
    """A 2x2 grid whose head bands carry high-frequency detail.

    Flat fills would survive a blur unchanged, so each tile gets concentric
    stripes in its head band — that way "the blur did something" is detectable.
    """
    img = Image.new("RGB", (side, side), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    tile = side // 2
    fills = [(220, 40, 40), (40, 220, 40), (40, 40, 220), (240, 200, 60)]
    for idx in range(4):
        col, row = idx % 2, idx // 2
        left, top = col * tile, row * tile
        draw.rectangle([left, top, left + tile - 1, top + tile - 1], fill=fills[idx])
        # Stripes across the head band region (top ~30% of the tile).
        for y in range(top, top + int(tile * 0.32), 3):
            draw.line([(left, y), (left + tile - 1, y)], fill=(10, 10, 10))
    return img


def _make_portrait(w: int = 400, h: int = 560) -> Image.Image:
    img = Image.new("RGB", (w, h), (120, 118, 110))
    draw = ImageDraw.Draw(img)
    draw.ellipse([w * 0.3, h * 0.1, w * 0.7, h * 0.45], fill=(230, 190, 160))
    return img


def _tile_box(idx: int, tile_w: int, tile_h: int, gutter: int):
    col, row = idx % 2, idx // 2
    left = col * (tile_w + gutter)
    top = row * (tile_h + gutter)
    return (left, top, left + tile_w, top + tile_h)


def _tile_bytes(img: Image.Image, idx: int, geom) -> bytes:
    gutter, tile_w, tile_h = geom
    return img.crop(_tile_box(idx, tile_w, tile_h, gutter)).tobytes()


class TestResolveGrid(unittest.TestCase):
    def test_gutterless_square(self):
        self.assertEqual(resolve_grid(512, 512, 2, 2), (0, 256, 256))

    def test_gutter_override_honoured(self):
        gutter, tw, th = resolve_grid(1000, 1000, 2, 2, gutter_override=20)
        self.assertEqual(gutter, 20)
        self.assertEqual((tw, th), (490, 490))
        # Tiles plus gutter must fit inside the image.
        self.assertLessEqual(tw * 2 + gutter, 1000)

    def test_tiles_never_exceed_bounds(self):
        """The fallback path must not produce tiles wider than the image."""
        for w, h in ((513, 511), (1001, 999), (777, 333)):
            gutter, tw, th = resolve_grid(w, h, 2, 2)
            self.assertLessEqual(tw * 2 + gutter, w, f"{w}x{h}")
            self.assertLessEqual(th * 2 + gutter, h, f"{w}x{h}")

    def test_out_of_range_override_falls_through_to_search(self):
        self.assertEqual(resolve_grid(512, 512, 2, 2, gutter_override=500).gutter, 0)


class TestDeface(unittest.TestCase):
    def setUp(self):
        self.grid = _make_grid()
        self.defaced = deface_rotation_grid(self.grid)
        self.geom = resolve_grid(GRID_SIDE, GRID_SIDE, 2, 2)

    def test_output_dimensions_preserved(self):
        self.assertEqual(self.defaced.size, self.grid.size)

    def test_back_tile_is_pixel_identical(self):
        """Tile 3 keeps its head. A feather bleeding across the seam from the
        side tile would blur it, so the hard rect erase is what this asserts."""
        self.assertEqual(
            _tile_bytes(self.defaced, 3, self.geom),
            _tile_bytes(self.grid, 3, self.geom),
        )

    def test_face_tiles_head_bands_changed(self):
        for idx in DEFACE_TILES:
            self.assertNotEqual(
                _tile_bytes(self.defaced, idx, self.geom),
                _tile_bytes(self.grid, idx, self.geom),
                f"tile {idx} head band should have been blurred",
            )

    def test_head_band_confined_to_top_of_tile(self):
        """Only the head band is suppressed — the body is the whole point of the
        grid, so the ellipse must not reach past roughly the top half."""
        gutter, tile_w, tile_h = self.geom
        for idx in DEFACE_TILES:
            left, top, right, _ = _tile_box(idx, tile_w, tile_h, gutter)
            band = (left, top + int(tile_h * 0.5), right, top + int(tile_h * 0.85))
            self.assertEqual(
                self.defaced.crop(band).tobytes(),
                self.grid.crop(band).tobytes(),
                f"tile {idx} mid-body should be untouched",
            )

    def test_front_tile_bottom_bleeds_from_side_tile_seam(self):
        """Documents a KNOWN ported behaviour, not an aspiration.

        The back tile is the only one with a hard rect erase, so on a gutterless
        grid the side tile's head ellipse (which starts ~2% of a tile height above
        its own top edge) feathers UP into the bottom of the front tile. Tile 1
        escapes it only because tile 3 below it is erased. If this assertion ever
        flips, the erase geometry changed — check it against a real sheet.
        """
        gutter, tile_w, tile_h = self.geom
        left, top, right, _ = _tile_box(0, tile_w, tile_h, gutter)
        band = (left, top + int(tile_h * 0.9), right, top + tile_h)
        self.assertNotEqual(
            self.defaced.crop(band).tobytes(),
            self.grid.crop(band).tobytes(),
        )
        # Tile 1's equivalent band is clean, courtesy of the back-tile erase.
        left1, top1, right1, _ = _tile_box(1, tile_w, tile_h, gutter)
        band1 = (left1, top1 + int(tile_h * 0.9), right1, top1 + tile_h)
        self.assertEqual(
            self.defaced.crop(band1).tobytes(),
            self.grid.crop(band1).tobytes(),
        )

    def test_zero_sigma_is_a_no_op(self):
        untouched = deface_rotation_grid(
            self.grid, blur_sigma=0, feather_sigma=0,
        )
        self.assertEqual(untouched.tobytes(), self.grid.convert("RGB").tobytes())

    def test_non_square_grid_supported(self):
        grid = _make_grid().resize((640, 480))
        defaced = deface_rotation_grid(grid)
        geom = resolve_grid(640, 480, 2, 2)
        self.assertEqual(defaced.size, (640, 480))
        self.assertEqual(_tile_bytes(defaced, 3, geom), _tile_bytes(grid, 3, geom))

    def test_short_pose_cx_frac_raises(self):
        with self.assertRaises(ValueError):
            deface_rotation_grid(self.grid, pose_cx_frac=(0.5,))

    def test_pose_cx_defaults_match_ported_values(self):
        self.assertEqual(POSE_CX_FRAC, (0.50, 0.50, 0.48))


class TestCompose(unittest.TestCase):
    def setUp(self):
        self.portrait = _make_portrait()
        self.grid = deface_rotation_grid(_make_grid())

    def test_long_edge_respected(self):
        sheet = compose_character_sheet(self.portrait, self.grid, long_edge=1024)
        self.assertEqual(max(sheet.size), 1024)

    def test_output_is_landscape(self):
        sheet = compose_character_sheet(self.portrait, self.grid)
        self.assertGreater(sheet.width, sheet.height)

    def test_aspect_is_driven_by_grid_aspect(self):
        """The sheet width is computed from the panels, so a wider grid means a
        wider sheet. A square 2x2 grid lands near 1.72:1 at the default anchor
        aspect — the ported geometry, not an independently chosen ratio."""
        square = compose_character_sheet(self.portrait, self.grid)
        self.assertAlmostEqual(square.width / square.height, 1.72, delta=0.05)

        wide_grid = self.grid.resize((1024, 512))
        wide = compose_character_sheet(self.portrait, wide_grid)
        self.assertGreater(wide.width / wide.height, square.width / square.height)

    def test_gutter_colour_sampled_from_portrait(self):
        """The gutter reads as one continuous backdrop rather than black bars."""
        sheet = compose_character_sheet(self.portrait, self.grid, long_edge=1024)
        corner = sheet.getpixel((1, 1))
        expected = self.portrait.getpixel((1, 1))
        for got, want in zip(corner, expected):
            self.assertLess(abs(got - want), 12, f"{corner} vs {expected}")

    def test_wide_grid_still_scales_to_long_edge(self):
        wide = self.grid.resize((1600, 400))
        sheet = compose_character_sheet(self.portrait, wide, long_edge=2048)
        self.assertEqual(max(sheet.size), 2048)

    def test_anchor_aspect_changes_width(self):
        narrow = compose_character_sheet(
            self.portrait, self.grid, long_edge=2048, anchor_aspect=0.4,
        )
        wide = compose_character_sheet(
            self.portrait, self.grid, long_edge=2048, anchor_aspect=1.2,
        )
        # Both are normalized to long_edge, so a wider anchor shows up as a
        # shorter sheet rather than a wider one.
        self.assertNotEqual(narrow.size, wide.size)


class TestBuildCharacterSheet(unittest.TestCase):
    def test_writes_sheet_and_returns_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchor_path = os.path.join(tmp, "anchor.png")
            grid_path = os.path.join(tmp, "rotation_grid.png")
            out_path = os.path.join(tmp, "nested", "sheet.png")
            _make_portrait().save(anchor_path)
            _make_grid().save(grid_path)

            returned = build_character_sheet(anchor_path, grid_path, out_path)

            self.assertEqual(returned, out_path)
            self.assertTrue(os.path.exists(out_path))
            with Image.open(out_path) as sheet:
                self.assertEqual(max(sheet.size), 2048)
                self.assertGreater(sheet.width, sheet.height)

    def test_source_files_not_mutated(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchor_path = os.path.join(tmp, "anchor.png")
            grid_path = os.path.join(tmp, "rotation_grid.png")
            _make_portrait().save(anchor_path)
            _make_grid().save(grid_path)
            before = os.path.getsize(grid_path)
            build_character_sheet(anchor_path, grid_path, os.path.join(tmp, "s.png"))
            self.assertEqual(os.path.getsize(grid_path), before)


class TestRegistryDescription(unittest.TestCase):
    def test_states_the_sheet_information_design(self):
        desc = character_sheet_description("Marella Lewis")
        self.assertIn("Marella Lewis", desc)
        self.assertIn("identity anchor", desc)
        # The model has no other way to know the right panel's suppressed faces
        # are deliberate.
        self.assertIn("identity from the anchor only", desc)


if __name__ == "__main__":
    unittest.main()
