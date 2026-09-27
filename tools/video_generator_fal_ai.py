"""fal.ai video generator (Veo 3 Fast by default).

Routes:
  - 0 reference images -> ``t2v_model``  (default fal-ai/veo3/fast)
  - 1 reference image  -> ``ff2v_model`` (default fal-ai/veo3/fast/image-to-video)
  - 2 reference images -> ``flf2v_model`` (first+last frame; only set if your model supports it)
"""

import os
import logging
import asyncio
from typing import List, Optional

# NOTE: ``fal_client`` is imported lazily inside the methods that call it (see
# ``_require_fal_client``). fal.ai is a paid, optional escalation backend; the
# default local rail (ComfyUI/LTX) must import and instantiate this class
# without fal-client installed. Only an actual fal API call requires the dep.

from interfaces.video_output import VideoOutput
from utils.rate_limiter import RateLimiter


def _require_fal_client():
    """Import fal_client on demand, with an actionable error if it's missing.

    Keeps the module importable (and this class instantiable) on the local
    rail where fal-client is intentionally not installed.
    """
    try:
        import fal_client  # noqa: PLC0415 (deliberately lazy)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "fal.ai backend called but 'fal-client' is not installed. "
            "This is the paid escalation backend. Either install it "
            "(`pip install fal-client` + set FAL_KEY), or keep to the local "
            "rail (default_route: local with all escalate_on_* gates off)."
        ) from exc
    return fal_client


class VideoGeneratorFalAI:
    def __init__(
        self,
        api_key: str,
        t2v_model: str = "fal-ai/veo3/fast",
        ff2v_model: str = "fal-ai/veo3/fast/image-to-video",
        flf2v_model: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        os.environ["FAL_KEY"] = api_key
        self.t2v_model = t2v_model
        self.ff2v_model = ff2v_model
        self.flf2v_model = flf2v_model
        self.rate_limiter = rate_limiter

    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str],
        aspect_ratio: str = "16:9",
        duration: int = 8,
        resolution: str = "1080p",
        **kwargs,
    ) -> VideoOutput:
        fal_client = _require_fal_client()
        if len(reference_image_paths) == 0:
            model = self.t2v_model
            image_url = None
            last_image_url = None
        elif len(reference_image_paths) == 1:
            model = self.ff2v_model
            image_url = await fal_client.upload_file_async(reference_image_paths[0])
            last_image_url = None
        elif len(reference_image_paths) == 2:
            if not self.flf2v_model:
                raise ValueError(
                    "Two reference images supplied but flf2v_model is not configured. "
                    "Set flf2v_model in the YAML config or supply only the first frame."
                )
            model = self.flf2v_model
            image_url = await fal_client.upload_file_async(reference_image_paths[0])
            last_image_url = await fal_client.upload_file_async(reference_image_paths[1])
        else:
            raise ValueError("reference_image_paths must contain 0, 1, or 2 images.")

        # Veo's fal endpoint expects duration as a string literal with unit:
        # valid values are '4s', '6s', '8s'. Older code passed str(int) like '8'
        # which fal rejects with literal_error. Clamp to the nearest allowed
        # value and append the 's' suffix.
        _allowed = [4, 6, 8]
        _dur = int(duration) if duration else 8
        _dur = min(_allowed, key=lambda x: abs(x - _dur))
        arguments = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": f"{_dur}s",
            "resolution": resolution,
        }
        if image_url:
            arguments["image_url"] = image_url
        if last_image_url:
            arguments["last_image_url"] = last_image_url

        logging.info(f"Calling {model} (refs={len(reference_image_paths)}) to generate video...")

        if self.rate_limiter:
            await self.rate_limiter.acquire()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await fal_client.subscribe_async(
                    model,
                    arguments=arguments,
                    with_logs=False,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 10 * (2 ** attempt)
                    logging.warning(f"fal {model} error: {e}. Retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    raise

        video = result.get("video") or {}
        video_url = video.get("url")
        if not video_url:
            raise ValueError(f"No video URL returned by {model}: {result}")

        return VideoOutput(fmt="url", ext="mp4", data=video_url)
