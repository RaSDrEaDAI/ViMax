"""Vision model client for ViMax agents that need image understanding.

Wraps fal.ai's `fal-ai/any-llm` endpoint with a vision-capable model
(default: Gemini 2.5 Flash). Used by CompositionPlanner to look at reference
boards / character portraits and factor them into bbox layouts.

Why this exists
---------------
GLM-5.2 (the text reasoning model for ViMax) is text-only on z.ai's
coding-plan endpoint. For vision tasks (looking at reference images, describing
boards from the orchestrator), we use a separate vision-capable model routed
through fal.ai — the same key already wired for Veo 3 video escalation.

Cost
----
Gemini 2.5 Flash via fal: ~$0.0003-0.001 per call (depending on image size).
Typical use: 1-3 vision calls per multi-char frame = ~$0.003 per frame on top
of the local Ideogram generation (free).

Schema
------
fal-ai/any-llm takes a flat JSON body:
    {
      "model": "google/gemini-2.5-flash",
      "prompt": "<text prompt>",
      "image_url": "<single image URL>"  # optional
    }

For multiple images, chain calls or use a single composite prompt with the
primary image. Fal's any-llm endpoint is single-image per call.

Returns:
    {"output": "<text response>", "reasoning": null, "partial": false, "error": null}

Alternative models on the same endpoint (verified available):
  - google/gemini-2.5-flash       (default; fast, cheap, accurate)
  - google/gemini-2.5-flash-lite  (cheaper, slightly less accurate)
  - google/gemini-2.0-flash-001   (older Gemini 2.0)
  - openai/gpt-4o                 (vision-capable via fal, but more expensive)
  - openai/gpt-4o-mini            (cheaper but fal wrapper rejects images)
"""

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import List, Optional

import aiohttp


class VisionModelClient:
    """Async client for vision LLM calls via fal-ai/any-llm.

    Args:
        api_key: fal.ai API key. If None, reads from FAL_API_KEY env var.
        model: Vision model ID on fal-ai/any-llm. Default: gemini-2.5-flash.
        endpoint: fal.run URL. Default: https://fal.run/fal-ai/any-llm.
        timeout: Request timeout in seconds. Default: 60.
        max_requests_per_minute: Optional rate limit.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.5-flash",
        endpoint: str = "https://fal.run/fal-ai/any-llm",
        timeout: int = 60,
        max_requests_per_minute: Optional[int] = None,
    ):
        self.api_key = api_key or os.environ.get("FAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "VisionModelClient: no api_key provided and FAL_API_KEY env var not set."
            )
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self._min_interval = 60.0 / max_requests_per_minute if max_requests_per_minute else 0
        self._last_call = 0.0

    async def describe_image(
        self,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,
        prompt: str = "Describe this image in detail. Note the subject, clothing, pose, environment, lighting, and composition.",
    ) -> str:
        """Describe a single image.

        Args:
            image_url: HTTP(S) URL of the image. Preferred.
            image_path: Local file path. Will be base64-encoded as a data URL.
                        (Note: fal-ai/any-llm may not accept data URLs for all
                        models — prefer image_url with a hosted image.)
            prompt: What to ask about the image.

        Returns:
            Model's text response.
        """
        if not image_url and not image_path:
            raise ValueError("VisionModelClient.describe_image: provide image_url or image_path.")

        # Resolve image input
        if image_url:
            payload_image = image_url
        else:
            # Encode local file as data URL (may not work with all fal models)
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            mime = f"image/{path.suffix.lstrip('.').lower() or 'png'}"
            data = path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            payload_image = f"data:{mime};base64,{b64}"

        # Rate limit
        if self._min_interval > 0:
            now = asyncio.get_event_loop().time()
            wait = self._last_call + self._min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = asyncio.get_event_loop().time()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "image_url": payload_image,
        }

        logging.info(f"VisionModelClient: model={self.model} image={'url' if image_url else 'path'}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                headers={
                    "Authorization": f"Key {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"VisionModelClient: fal returned {resp.status}: {body[:500]}"
                    )
                data = await resp.json()
                if data.get("error"):
                    raise RuntimeError(f"VisionModelClient: model error: {data['error']}")
                return data.get("output") or ""

    async def describe_images(
        self,
        image_urls: List[str],
        prompt: str = "Describe this image in detail. Note the subject, clothing, pose, environment, lighting, and composition.",
    ) -> List[str]:
        """Describe multiple images concurrently (subject to rate limit).

        Returns one description per input URL, in order.
        """
        tasks = [self.describe_image(image_url=url, prompt=prompt) for url in image_urls]
        return await asyncio.gather(*tasks)

    async def plan_with_references(
        self,
        base_prompt: str,
        reference_image_urls: List[str],
        per_image_prompt: str = "Describe this reference image. Note character identity, outfit, environment, and any composition cues.",
    ) -> str:
        """Look at reference images, then answer a planning question.

        Useful for CompositionPlanner: describe each board, then use those
        descriptions as context for the layout planning prompt.

        Args:
            base_prompt: The planning question to answer after seeing images.
            reference_image_urls: Images to look at first.
            per_image_prompt: What to extract from each image.

        Returns:
            The model's answer to base_prompt, informed by the image descriptions.
        """
        if not reference_image_urls:
            # No images — just answer the prompt directly (text-only).
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Key {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.model, "prompt": base_prompt},
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    data = await resp.json()
                    return data.get("output") or ""

        # Describe each image
        descriptions = await self.describe_images(reference_image_urls, prompt=per_image_prompt)

        # Build composite prompt with descriptions inline
        composite = base_prompt + "\n\nReference image descriptions:\n"
        for i, desc in enumerate(descriptions, 1):
            composite += f"\n[Image {i}]: {desc}\n"

        # Ask the planning question (no image this time, just text)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                headers={
                    "Authorization": f"Key {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "prompt": composite},
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                data = await resp.json()
                return data.get("output") or ""
