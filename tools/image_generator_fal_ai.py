"""fal.ai image generator (nano-banana for t2i and edit).

Routes:
  - 0 reference images -> ``fal-ai/nano-banana``        (text-to-image)
  - 1+ reference images -> ``fal-ai/nano-banana/edit``  (multi-ref edit)

Both model IDs are configurable via ``t2i_model`` / ``i2i_model``.

Reference images are uploaded to fal's CDN via ``fal_client.upload_file_async``
and the returned URLs are passed in ``image_urls``.
"""

import os
import logging
import asyncio
from typing import List, Optional

# NOTE: ``fal_client`` is imported lazily inside the methods that call it (see
# ``_require_fal_client``). fal.ai is a paid, optional escalation backend; the
# default local rail (ComfyUI/LTX) must import and instantiate this class
# without fal-client installed. Only an actual fal API call requires the dep.

from interfaces.image_output import ImageOutput
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


class ImageGeneratorFalAI:
    def __init__(
        self,
        api_key: str,
        t2i_model: str = "fal-ai/nano-banana",
        i2i_model: str = "fal-ai/nano-banana/edit",
        rate_limiter: Optional[RateLimiter] = None,
        client_timeout: float = 600.0,
    ):
        os.environ["FAL_KEY"] = api_key
        self.t2i_model = t2i_model
        self.i2i_model = i2i_model
        self.rate_limiter = rate_limiter
        self.client_timeout = client_timeout

    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = None,
        aspect_ratio: Optional[str] = "16:9",
        **kwargs,
    ) -> ImageOutput:
        reference_image_paths = reference_image_paths or []
        fal_client = _require_fal_client()

        if len(reference_image_paths) == 0:
            model = self.t2i_model
            arguments = {
                "prompt": prompt,
                "num_images": 1,
                "output_format": "png",
            }
            if aspect_ratio:
                arguments["aspect_ratio"] = aspect_ratio
        else:
            model = self.i2i_model
            image_urls = await asyncio.gather(
                *(fal_client.upload_file_async(p) for p in reference_image_paths)
            )
            arguments = {
                "prompt": prompt,
                "image_urls": list(image_urls),
                "num_images": 1,
                "output_format": "png",
            }

        logging.info(f"Calling {model} (t2i={len(reference_image_paths) == 0}) to generate image...")

        if self.rate_limiter:
            await self.rate_limiter.acquire()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # fal's subscribe_async waits indefinitely without client_timeout.
                # A timeout is NOT retried: the retry would resubmit (and re-bill)
                # a job that may still be running on fal's side.
                result = await fal_client.subscribe_async(
                    model,
                    arguments=arguments,
                    with_logs=False,
                    client_timeout=self.client_timeout,
                )
                break
            except TimeoutError:
                raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)
                    logging.warning(f"fal {model} error: {e}. Retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    raise

        images = result.get("images") or []
        if not images:
            raise ValueError(f"No image returned by {model}: {result}")

        image_url = images[0]["url"]
        return ImageOutput(fmt="url", ext="png", data=image_url)
