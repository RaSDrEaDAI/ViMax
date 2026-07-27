"""Composite character sheet — the deterministic (free, no paid call) half of
the composite-sheet chain. Ported from destack-app ``lib/n3/composite_sheet.ts``.

The sheet is a COMPOSED deliverable with a fixed information design:

  - LEFT (identity anchor): one large close-up/medium portrait. The ONLY place
    identity lives on the sheet.
  - RIGHT (body reference): a 2x2 full-figure rotation grid (front / 3-4 / side
    / back) with faces SUPPRESSED on the three face-showing tiles (back keeps
    its head). Small low-res faces in rotation tiles confuse identity lock — so
    the image model reads body/outfit/silhouette from the grid and identity
    ONLY from the anchor.

Two pure Image->Image functions, both fixture-testable, zero paid spend:

  - :func:`deface_rotation_grid`: blur the head band of the 3 face tiles
    (feathered ellipse), leave the back tile pixel-for-pixel untouched.
  - :func:`compose_character_sheet`: anchor left + defaced grid right -> one PNG.

Tile geometry lives in :func:`resolve_grid` — the ONE source for the math. Tiles
are read in reading order:
  index 0 = top-left = FRONT, 1 = top-right = 3/4, 2 = bottom-left = SIDE,
  3 = bottom-right = BACK (exempt from defacing).

PIL rather than sharp: already a dependency (via the Google adapter and moviepy)
and the operations needed are a Gaussian blur, an alpha mask, and a paste.
"""

from typing import NamedTuple, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter

# ── Tile geometry ────────────────────────────────────────────────────────────


class GridResolution(NamedTuple):
    gutter: int
    tile_width: int
    tile_height: int


def resolve_grid(
    image_width: int,
    image_height: int,
    cols: int,
    rows: int,
    gutter_override: Optional[int] = None,
) -> GridResolution:
    """Resolve tile geometry for a ``cols`` x ``rows`` grid image.

    With no override, searches gutter sizes 0..100 for one that divides the
    image into whole-pixel tiles, then verifies the tiles actually fit inside
    the image bounds. Falls back to a gutterless floor division, which can never
    exceed the bounds. Ported verbatim from destack-app ``resolveGrid`` /
    ``calculateGutter`` — do NOT fork this math.
    """
    if gutter_override is not None and 0 <= gutter_override <= 100:
        return GridResolution(
            gutter=gutter_override,
            tile_width=(image_width - (cols - 1) * gutter_override) // cols,
            tile_height=(image_height - (rows - 1) * gutter_override) // rows,
        )

    for gutter in range(101):
        tile_w = (image_width - (cols - 1) * gutter) / cols
        tile_h = (image_height - (rows - 1) * gutter) / rows
        # Whole-pixel tiles, within a tight tolerance.
        if abs(tile_w - int(tile_w)) < 0.01 and abs(tile_h - int(tile_h)) < 0.01:
            tw, th = int(tile_w), int(tile_h)
            total_w = tw * cols + gutter * (cols - 1)
            total_h = th * rows + gutter * (rows - 1)
            if total_w <= image_width and total_h <= image_height:
                return GridResolution(gutter=gutter, tile_width=tw, tile_height=th)

    # No perfect gutter — floor without one so tiles never exceed the bounds.
    return GridResolution(
        gutter=0,
        tile_width=image_width // cols,
        tile_height=image_height // rows,
    )


# ── Deface ───────────────────────────────────────────────────────────────────

# Tiles to deface: 0=front (TL), 1=3/4 (TR), 2=side (BL). 3=back (BR) exempt.
DEFACE_TILES = (0, 1, 2)

# Per-pose horizontal center offset (fraction of tile width). Front centered;
# 3/4 and side heads sit slightly off-center as the figure turns. The generous
# ellipse width covers the head either way — these just bias the center.
POSE_CX_FRAC = (0.50, 0.50, 0.48)


def deface_rotation_grid(
    grid_img: Image.Image,
    *,
    blur_sigma: float = 22,
    feather_sigma: float = 14,
    head_band_height_frac: float = 0.17,
    head_band_width_frac: float = 0.42,
    head_band_center_frac: float = 0.15,
    pose_cx_frac: Sequence[float] = POSE_CX_FRAC,
) -> Image.Image:
    """Blur the head region of the front / 3-4 / side tiles of a 2x2 grid.

    Programmatic suppression (a fixed feathered ellipse per tile) — NOT face
    detection (v1) and NOT prompted headless anatomy (models distort bodies
    fighting it). The back tile's pixels are never composited over, so they are
    identical to the source through the round-trip.
    """
    w, h = grid_img.size
    if not w or not h:
        raise ValueError("deface_rotation_grid: grid image has no dimensions")
    if len(pose_cx_frac) < len(DEFACE_TILES):
        raise ValueError(
            f"deface_rotation_grid: pose_cx_frac needs at least "
            f"{len(DEFACE_TILES)} entries, got {len(pose_cx_frac)}"
        )

    gutter, tile_w, tile_h = resolve_grid(w, h, 2, 2)

    # White feathered ellipses on a black canvas -> the alpha mask.
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for idx in DEFACE_TILES:
        col, row = idx % 2, idx // 2
        left = col * (tile_w + gutter)
        top = row * (tile_h + gutter)
        cx = left + tile_w * pose_cx_frac[idx]
        cy = top + tile_h * head_band_center_frac
        rx = tile_w * head_band_width_frac
        ry = tile_h * head_band_height_frac
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)

    mask = mask.filter(ImageFilter.GaussianBlur(feather_sigma))

    # GUARANTEE the back tile (BR) is untouched: erase its whole rectangle from
    # the mask so the side-tile ellipse's feather can never bleed across the
    # seam. This is what makes the back tile byte-stable through the round-trip.
    #
    # KNOWN, ported as-is: the back tile is the ONLY tile with this protection.
    # On a gutterless grid the SIDE tile's head ellipse sits ~2% of a tile height
    # above its own top edge, so its feather bleeds UP into the bottom ~12% of
    # the FRONT tile (feet / lower legs). Tile 1 escapes this only because tile 3
    # below it is erased. Real grids usually carry a gutter, which limits it.
    # Left matching destack-app rather than silently widening the erase, since
    # that changes sheet geometry an operator should judge on a real sheet.
    back_left = tile_w + gutter
    back_top = tile_h + gutter
    ImageDraw.Draw(mask).rectangle([back_left, back_top, w, h], fill=0)

    # Blur the whole grid, then keep it ONLY inside the feathered ellipses. The
    # back tile has no ellipse -> mask is 0 there -> pixels identical to source.
    source = grid_img.convert("RGB")
    blurred = source.filter(ImageFilter.GaussianBlur(blur_sigma))
    return Image.composite(blurred, source, mask)


# ── Composite ────────────────────────────────────────────────────────────────

# Internal working height — both panels fill it; the canvas WIDTH is computed
# from the panels (landscape), then the whole sheet is scaled so its long edge
# equals long_edge.
_WORKING_HEIGHT = 1600


def compose_character_sheet(
    portrait_img: Image.Image,
    defaced_grid_img: Image.Image,
    *,
    long_edge: int = 2048,
    anchor_aspect: float = 0.72,
) -> Image.Image:
    """Compose the anchor portrait (left) + defaced rotation grid (right).

    BOTH panels fill the FULL sheet height: the grid is scaled to the anchor's
    height (aspect preserved) rather than contained in a fixed right panel — so
    the four-pose grid is the SAME height as the anchor and there is NO vertical
    negative space above/below it. Because the grid keeps its aspect at full
    height, the canvas WIDTH is computed from the two panels: the sheet is always
    landscape, and wider the wider the grid. A square 2x2 grid with the default
    ``anchor_aspect`` lands near 1.72:1. The finished sheet is then scaled so its
    long edge equals ``long_edge``.

    The gutter color is sampled from the portrait's top-left corner (a studio
    backdrop fills the corners) so the sheet reads as one continuous background.
    """
    height = _WORKING_HEIGHT
    gutter = max(8, round(height * 0.01))
    panel_h = height - gutter * 2
    if panel_h <= 0:
        raise ValueError("compose_character_sheet: degenerate layout")

    bg = _sample_backdrop(portrait_img)

    # Grid: scale to the FULL panel height, aspect preserved. Its width then
    # drives the sheet width.
    gw, gh = defaced_grid_img.size
    grid_h = panel_h
    grid_w = max(1, round(gw * (grid_h / gh)))

    # Anchor: a full-height portrait panel, cover-cropped, biased to the top so
    # the face stays in frame.
    anchor_h = panel_h
    anchor_w = max(1, round(anchor_h * anchor_aspect))

    width = gutter * 3 + anchor_w + grid_w

    anchor = _resize_cover_top(portrait_img.convert("RGB"), anchor_w, anchor_h)
    # grid_w/grid_h preserve the grid aspect, so this does not distort it.
    grid = defaced_grid_img.convert("RGB").resize(
        (grid_w, grid_h), Image.LANCZOS,
    )

    sheet = Image.new("RGB", (width, height), bg)
    sheet.paste(anchor, (gutter, gutter))
    sheet.paste(grid, (gutter + anchor_w + gutter, gutter))

    # Scale so the long edge (the width — landscape) equals long_edge.
    long_now = max(width, height)
    if long_now == long_edge:
        return sheet
    f = long_edge / long_now
    return sheet.resize(
        (max(1, round(width * f)), max(1, round(height * f))), Image.LANCZOS,
    )


# ── Path-level convenience ───────────────────────────────────────────────────


def build_character_sheet(
    anchor_path: str,
    grid_path: str,
    out_path: str,
    *,
    long_edge: int = 2048,
    anchor_aspect: float = 0.72,
) -> str:
    """Deface the grid, compose it with the anchor, write the sheet PNG.

    The one place the deface -> compose -> write sequence lives, so the pipeline
    and ``scripts/seed_character_portraits.py`` cannot drift apart on it.
    Returns ``out_path``.
    """
    import os

    with Image.open(anchor_path) as anchor, Image.open(grid_path) as grid:
        defaced = deface_rotation_grid(grid)
        sheet = compose_character_sheet(
            anchor, defaced, long_edge=long_edge, anchor_aspect=anchor_aspect,
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path)
    return out_path


def character_sheet_description(identifier: str) -> str:
    """The registry description for a composite sheet.

    Spells out the sheet's information design in the prompt itself, because the
    image model has no other way to know that the right panel's suppressed faces
    are deliberate and that identity must come from the left panel alone.
    """
    return (
        f"Character sheet for {identifier}: left panel is the identity anchor "
        f"(face + wardrobe), right panel is a body rotation reference (front, "
        f"3/4, side, back); identity from the anchor only."
    )


def _sample_backdrop(img: Image.Image) -> Tuple[int, int, int]:
    """Sample the flat backdrop color from a portrait's top-left corner."""
    w = min(24, img.width) or 1
    h = min(24, img.height) or 1
    patch = img.convert("RGB").crop((0, 0, w, h)).resize((1, 1), Image.BOX)
    return patch.getpixel((0, 0))


def _resize_cover_top(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize to cover target_w x target_h, cropping from the top.

    Top-biased rather than center-cropped: on a portrait, a center crop is what
    cuts the face off, and the face is the whole point of the anchor panel.
    """
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    scaled = img.resize(
        (max(1, round(src_w * scale)), max(1, round(src_h * scale))), Image.LANCZOS,
    )
    left = max(0, (scaled.width - target_w) // 2)
    return scaled.crop((left, 0, left + target_w, target_h))
