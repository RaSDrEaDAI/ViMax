"""ComfyUI image generator.

Routes:
  - 0 reference images   -> ``t2i_workflow``  (default: workflows/flux1_dev_t2i_lora.json)
  - 1+ reference images  -> ``i2i_workflow``  (default: workflows/flux1_kontext_i2i.json,
    which is single-reference; if more refs are given the first is used and the rest
    are dropped with a warning. For multi-ref composition route via ImageGeneratorRouter.)

Workflow contract (named via ``_meta.title`` so we don't depend on node IDs):
  - ``Positive Prompt`` (CLIPTextEncode-style):  inputs.text   -> prompt
  - ``Latent Size``    (EmptyLatentImage / similar): inputs.width, inputs.height -> size
  - ``Sampler``        (KSampler-style):           inputs.seed -> seed (int)
  - ``Reference Image`` (LoadImage):               inputs.image -> uploaded filename (i2i only)
  - ``Save Image``     (SaveImage):                inputs.filename_prefix -> "vimax_<run>"
"""

import asyncio
import io
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import List, Optional

from PIL import Image

from interfaces.image_output import ImageOutput
from utils.rate_limiter import RateLimiter

from .comfyui_client import ComfyUIClient, load_workflow, set_node_input, find_node_by_title


_DEFAULT_T2I = "workflows/flux1_dev_t2i_lora.json"
_DEFAULT_I2I = "workflows/flux1_kontext_i2i.json"


def _parse_size(kwargs_size: Optional[str], aspect_ratio: Optional[str]) -> tuple:
    """Resolve (width, height). Prefers explicit kwargs.size like '1600x900'."""
    if kwargs_size and "x" in kwargs_size.lower():
        w, h = kwargs_size.lower().split("x", 1)
        return int(w), int(h)
    ar = (aspect_ratio or "16:9").strip()
    table = {
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "1:1": (1024, 1024),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
        "21:9": (1536, 640),
    }
    return table.get(ar, (1280, 720))


class ImageGeneratorComfyUI:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8189",
        t2i_workflow: str = _DEFAULT_T2I,
        i2i_workflow: str = _DEFAULT_I2I,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.client = ComfyUIClient(base_url=base_url)
        self.t2i_workflow_path = t2i_workflow
        self.i2i_workflow_path = i2i_workflow
        self.rate_limiter = rate_limiter

    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = None,
        aspect_ratio: Optional[str] = "16:9",
        **kwargs,
    ) -> ImageOutput:
        reference_image_paths = reference_image_paths or []
        width, height = _parse_size(kwargs.get("size"), aspect_ratio)

        # If a structured json_prompt is supplied (e.g. Ideogram V4 layout from
        # CompositionPlanner), we inject it into the "JSON Prompt" titled node
        # and skip the text-prompt path.
        json_prompt_str = kwargs.get("json_prompt")

        if len(reference_image_paths) == 0:
            workflow = load_workflow(self.t2i_workflow_path)
            mode = "t2i"
        else:
            workflow = load_workflow(self.i2i_workflow_path)
            mode = "i2i"
            if len(reference_image_paths) > 1:
                # Some i2i workflows accept multiple refs (Qwen Edit 2511, etc.)
                # via "Reference Image" (1st), "Reference Image 2" (2nd), etc.
                # Try to set each; if the workflow only has "Reference Image",
                # fall back to single-ref with a warning.
                refs_set = 0
                first_upload = None
                for i, ref_path in enumerate(reference_image_paths):
                    title = "Reference Image" if i == 0 else f"Reference Image {i+1}"
                    if find_node_by_title(workflow, title) is not None:
                        uploaded = await self.client.upload_image(ref_path)
                        if first_upload is None:
                            first_upload = uploaded
                        set_node_input(workflow, title, "image", uploaded)
                        refs_set += 1
                    else:
                        break
                if refs_set < len(reference_image_paths):
                    logging.warning(
                        f"ComfyUI i2i workflow accepted {refs_set} of "
                        f"{len(reference_image_paths)} refs; rest dropped."
                    )
                # CRITICAL: clear any unfilled "Reference Image N" slots. Some
                # workflows (e.g. qwen_edit_2511_multiref.json) ship with 3
                # LoadImage nodes hard-coded to placeholder filenames like
                # "reference_3.png" that don't exist in ComfyUI's input dir.
                # If we only provide 2 refs, the 3rd slot still triggers
                # LoadImage's custom_validation_failed. Fix: duplicate the
                # first uploaded ref into every remaining "Reference Image N"
                # slot — the extra input is harmless to models that ignore it,
                # and it satisfies the validator.
                if first_upload is not None:
                    n = refs_set + 1
                    while True:
                        title = f"Reference Image {n}" if n > 1 else "Reference Image"
                        if find_node_by_title(workflow, title) is None:
                            break
                        set_node_input(workflow, title, "image", first_upload)
                        logging.info(f"Filled unused ref slot '{title}' with first image to pass validation")
                        n += 1
            else:
                uploaded = await self.client.upload_image(reference_image_paths[0])
                set_node_input(workflow, "Reference Image", "image", uploaded)
                # Same clearing logic for the single-ref case — workflow may
                # still have "Reference Image 2" / "Reference Image 3" slots
                # with bad defaults.
                n = 2
                while True:
                    title = f"Reference Image {n}"
                    if find_node_by_title(workflow, title) is None:
                        break
                    set_node_input(workflow, title, "image", uploaded)
                    logging.info(f"Filled unused ref slot '{title}' with first image to pass validation")
                    n += 1

        # Prompt injection. Field name differs by node class:
        #   CLIPTextEncode          -> "text"
        #   IdeogramV4 (API node)   -> "prompt" (json_prompt on same node)
        #   Ideogram4PromptBuilderKJ (local) -> "import_json" (built string -> downstream CLIPTextEncode)
        #   TextEncodeQwenImageEditPlus -> "prompt"
        positive_node_id = find_node_by_title(workflow, "Positive Prompt")
        if positive_node_id is not None:
            inputs = workflow[positive_node_id]["inputs"]
            if json_prompt_str:
                # Ideogram (either API node or local builder).
                if "import_json" in inputs:
                    # Local Ideogram4PromptBuilderKJ: inject JSON, output is built downstream.
                    inputs["import_json"] = json_prompt_str
                elif "json_prompt" in inputs:
                    # IdeogramV4 API node: JSON overrides text prompt.
                    inputs["json_prompt"] = json_prompt_str
                    inputs["prompt"] = prompt[:200]
                else:
                    inputs["text"] = prompt[:5000]
                mode += "+json"
            else:
                # Standard text prompt.
                prompt_field = "prompt" if "prompt" in inputs else "text"
                inputs[prompt_field] = prompt[:5000]
        else:
            set_node_input(workflow, "Positive Prompt", "text", prompt)

        if find_node_by_title(workflow, "Latent Size") is not None:
            set_node_input(workflow, "Latent Size", "width", width)
            set_node_input(workflow, "Latent Size", "height", height)

        sampler_id = find_node_by_title(workflow, "Sampler")
        if sampler_id is not None:
            seed = random.randint(0, 2**31 - 1)
            inputs = workflow[sampler_id]["inputs"]
            for key in ("seed", "noise_seed"):
                if key in inputs:
                    inputs[key] = seed
                    break

        if self.rate_limiter:
            await self.rate_limiter.acquire()

        logging.info(f"ComfyUI image {mode}: prompt='{prompt[:80]}...' size={width}x{height}")
        prompt_id = await self.client.queue_prompt(workflow)
        history = await self.client.wait_for_completion(prompt_id)
        outputs = ComfyUIClient.collect_outputs(history)

        images = [o for o in outputs if o[2] == "output" and o[0].lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        if not images:
            raise RuntimeError(f"ComfyUI returned no image outputs for prompt {prompt_id}: {outputs}")
        filename, subfolder, out_type = images[0]
        data = await self.client.fetch_output(filename, subfolder=subfolder, out_type=out_type)

        ext = Path(filename).suffix.lstrip(".") or "png"
        image = Image.open(io.BytesIO(data))
        image.load()
        return ImageOutput(fmt="pil", ext=ext, data=image)
