"""Post-production: trim Veo's tail frames + apply brand overlay + emit
multi-aspect deliverables (16:9, 9:16 letterbox, 9:16 centercut, 1:1).

Per-project usage:
    uv run python scripts/overlay_branding.py --project marella_victory_kdd

Looks for:
    projects/<project>/working_dir/final_video.mp4   (raw pipeline output)
    projects/<project>/assets/logo.png               (top-right brand mark)

Writes:
    projects/<project>/working_dir/branded/branded_16x9.mp4
    projects/<project>/working_dir/branded/branded_9x16.mp4
    projects/<project>/working_dir/branded/branded_9x16_centercut.mp4
    projects/<project>/working_dir/branded/branded_1x1.mp4

Defaults match the original Marella overlay; tweak the constants block for
other brands.
"""

import argparse
import os
import platform
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips

REPO_ROOT = Path(__file__).resolve().parent.parent

# Trim the tail of Veo outputs (last frame can be corrupt).
TRIM_FRAMES = 2

# End card duration (seconds).
END_CARD_DURATION = 2.0

# Brand-specific palette + copy. Override per-project via --url / --tagline.
BRAND_BLUE = (12, 86, 168)
END_CARD_BG = (235, 235, 235)
END_CARD_TEXT = (12, 86, 168)


def _font(size: int):
    """Best-effort font lookup that works on Windows + Linux/Mac."""
    candidates = []
    if platform.system() == "Windows":
        candidates += [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _make_lower_third(width: int, url: str) -> Image.Image:
    h = max(60, int(width * 0.06))
    img = Image.new("RGBA", (width, h), BRAND_BLUE + (220,))
    draw = ImageDraw.Draw(img)
    font = _font(int(h * 0.55))
    draw.text((int(width * 0.04), int(h * 0.18)), url, fill=(255, 255, 255), font=font)
    return img


def _make_end_card(width: int, height: int, headline: str, url: str) -> Image.Image:
    img = Image.new("RGB", (width, height), END_CARD_BG)
    draw = ImageDraw.Draw(img)
    f1 = _font(int(height * 0.10))
    f2 = _font(int(height * 0.07))
    bbox1 = draw.textbbox((0, 0), headline, font=f1)
    bbox2 = draw.textbbox((0, 0), url, font=f2)
    x1 = (width - (bbox1[2] - bbox1[0])) // 2
    x2 = (width - (bbox2[2] - bbox2[0])) // 2
    y1 = int(height * 0.38)
    y2 = int(height * 0.58)
    draw.text((x1, y1), headline, fill=END_CARD_TEXT, font=f1)
    draw.text((x2, y2), url, fill=END_CARD_TEXT, font=f2)
    return img


def _load_logo(logo_path: Path, height: int) -> Image.Image | None:
    if not logo_path.exists():
        return None
    logo = Image.open(logo_path).convert("RGBA")
    ratio = height / logo.height
    return logo.resize((int(logo.width * ratio), height), Image.LANCZOS)


def _composite_landscape(base_clip, logo_img: Image.Image | None, lower_third: Image.Image, end_card: Image.Image, end_card_dur: float):
    layers = [base_clip]
    w, h = base_clip.w, base_clip.h
    if logo_img is not None:
        from numpy import asarray
        logo_clip = ImageClip(asarray(logo_img)).with_duration(base_clip.duration)
        logo_clip = logo_clip.with_position((w - logo_img.width - int(w * 0.02), int(h * 0.03)))
        layers.append(logo_clip)
    from numpy import asarray
    lt_clip = ImageClip(asarray(lower_third)).with_duration(base_clip.duration)
    lt_clip = lt_clip.with_position(("center", h - lower_third.height - int(h * 0.04)))
    layers.append(lt_clip)
    body = CompositeVideoClip(layers, size=(w, h))
    end_clip = ImageClip(asarray(end_card)).with_duration(end_card_dur)
    end_clip = end_clip.resized((w, h))
    return concatenate_videoclips([body, end_clip])


def _to_aspect(clip, target_w: int, target_h: int, mode: str = "letterbox"):
    src_w, src_h = clip.w, clip.h
    src_ar = src_w / src_h
    tgt_ar = target_w / target_h
    if mode == "letterbox":
        if src_ar > tgt_ar:
            scale = target_w / src_w
        else:
            scale = target_h / src_h
        scaled = clip.resized(scale)
        return CompositeVideoClip(
            [scaled.with_position(("center", "center"))],
            size=(target_w, target_h),
        )
    elif mode == "centercut":
        if src_ar > tgt_ar:
            scale = target_h / src_h
        else:
            scale = target_w / src_w
        scaled = clip.resized(scale)
        return CompositeVideoClip(
            [scaled.with_position(("center", "center"))],
            size=(target_w, target_h),
        )
    raise ValueError(f"Unknown mode {mode!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Project subdirectory under projects/")
    parser.add_argument("--url", default="VICTORYKDDCLEANING.COM", help="Lower-third URL")
    parser.add_argument("--headline", default="Victory KDD Cleaning", help="End card headline")
    args = parser.parse_args()

    project_dir = REPO_ROOT / "projects" / args.project
    working = project_dir / "working_dir"
    final_path = working / "final_video.mp4"
    if not final_path.exists():
        raise SystemExit(f"No final_video.mp4 at {final_path}. Run render.py first.")

    branded_dir = working / "branded"
    branded_dir.mkdir(parents=True, exist_ok=True)

    raw = VideoFileClip(str(final_path))
    fps = raw.fps or 24
    trimmed_dur = max(0.1, raw.duration - TRIM_FRAMES / fps)
    trimmed = raw.subclipped(0, trimmed_dur)

    logo = _load_logo(project_dir / "assets" / "logo.png", height=int(trimmed.h * 0.10))
    lower_third = _make_lower_third(trimmed.w, args.url)
    end_card_landscape = _make_end_card(trimmed.w, trimmed.h, args.headline, args.url)

    landscape = _composite_landscape(trimmed, logo, lower_third, end_card_landscape, END_CARD_DURATION)
    landscape.write_videofile(str(branded_dir / "branded_16x9.mp4"), codec="libx264", preset="medium", audio_codec="aac")

    # Aspect variants. We composite over the landscape (with overlays already
    # baked in) so the lower-third + logo stay in proportion.
    portrait_letter = _to_aspect(landscape, 1080, 1920, mode="letterbox")
    portrait_letter.write_videofile(str(branded_dir / "branded_9x16.mp4"), codec="libx264", preset="medium", audio_codec="aac")

    portrait_cut = _to_aspect(landscape, 1080, 1920, mode="centercut")
    portrait_cut.write_videofile(str(branded_dir / "branded_9x16_centercut.mp4"), codec="libx264", preset="medium", audio_codec="aac")

    square = _to_aspect(landscape, 1080, 1080, mode="centercut")
    square.write_videofile(str(branded_dir / "branded_1x1.mp4"), codec="libx264", preset="medium", audio_codec="aac")

    print(f"Wrote branded variants to {branded_dir}")


if __name__ == "__main__":
    main()
