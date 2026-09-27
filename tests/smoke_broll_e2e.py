"""End-to-end smoke test: generate one B-roll keyframe + one Wan 2.2 video clip,
talking only to local ComfyUI. No LLM, no fal, no API keys required.

This validates:
  - ComfyUI reachable on 127.0.0.1:8189
  - Flux1-dev t2i workflow round-trips and saves a real PNG
  - Wan 2.2 i2v workflow accepts the keyframe and returns an mp4

Outputs:
  .test/broll_e2e/keyframe.png
  .test/broll_e2e/clip.mp4
"""

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.comfyui_client import ComfyUIClient
from tools.image_generator_comfyui import ImageGeneratorComfyUI
from tools.video_generator_comfyui import VideoGeneratorComfyUI


KEYFRAME_PROMPT = (
    "A wide cinematic drone shot of a Southwest Florida beach at golden hour, "
    "calm turquoise water, palm fronds in the foreground, soft warm light, "
    "photorealistic, sharp focus."
)

VIDEO_PROMPT = (
    "Slow forward dolly along the shoreline. The camera glides smoothly forward, "
    "palm leaves drift gently, small waves lap the sand."
)


async def main():
    client = ComfyUIClient()
    if not await client.ping():
        raise SystemExit("ComfyUI not reachable on 127.0.0.1:8189. Start it first.")

    out_dir = REPO_ROOT / ".test" / "broll_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/2: Flux1-dev t2i keyframe...")
    img_gen = ImageGeneratorComfyUI()
    img = await img_gen.generate_single_image(
        prompt=KEYFRAME_PROMPT,
        reference_image_paths=[],
        size="1280x720",
    )
    keyframe_path = out_dir / "keyframe.png"
    img.save(str(keyframe_path))
    print(f"  -> {keyframe_path}")

    print("Step 2/2: Wan 2.2 i2v from the keyframe...")
    vid_gen = VideoGeneratorComfyUI()
    video = await vid_gen.generate_single_video(
        prompt=VIDEO_PROMPT,
        reference_image_paths=[str(keyframe_path)],
        duration=5,
        aspect_ratio="16:9",
    )
    clip_path = out_dir / "clip.mp4"
    video.save(str(clip_path))
    print(f"  -> {clip_path}")
    print("DONE.")


if __name__ == "__main__":
    asyncio.run(main())
