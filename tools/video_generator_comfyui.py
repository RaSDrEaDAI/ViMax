"""ComfyUI video generator (Wan 2.2 i2v by default).

Routes:
  - 0 reference images -> not supported in v1 (Wan 2.2 needs an i2v keyframe).
                          Configure i2v_workflow only; if you want pure t2v,
                          add t2v_workflow and we'll fall back.
  - 1 reference image  -> ``i2v_workflow``
  - 2 reference images -> not supported in v1 (no first/last-frame Wan workflow yet).

Workflow contract (named via ``_meta.title``):
  - ``Positive Prompt``  (CLIPTextEncode-style):           inputs.text
  - ``Negative Prompt``  (CLIPTextEncode-style, optional): inputs.text
  - ``Reference Image``  (LoadImage):                       inputs.image
  - ``Sampler``          (KSampler-style):                  inputs.seed
  - ``Video Size``       (custom resize/empty-latent node, optional): inputs.width, inputs.height
  - ``Frame Count``      (custom node, optional):           inputs.length / inputs.frames
"""

import asyncio
import logging
import random
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .comfyui_client import (
    ComfyUIClient,
    load_workflow,
    find_node_by_title,
    find_node_by_titles,
    set_node_input,
    set_node_input_by_titles,
    I2V_REF_TITLES,
    FLF2V_FIRST_TITLES,
    FLF2V_LAST_TITLES,
    PROMPT_TITLES,
)

from interfaces.video_output import VideoOutput
from utils.rate_limiter import RateLimiter


_DEFAULT_I2V = "workflows/wan22_i2v.json"


def _parse_size(kwargs_size: Optional[str], aspect_ratio: Optional[str]) -> tuple:
    if kwargs_size and "x" in kwargs_size.lower():
        w, h = kwargs_size.lower().split("x", 1)
        return int(w), int(h)
    ar = (aspect_ratio or "16:9").strip()
    table = {
        "16:9": (832, 480),
        "9:16": (480, 832),
        "1:1": (640, 640),
    }
    return table.get(ar, (832, 480))


class VideoGeneratorComfyUI:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8189",
        i2v_workflow: str = _DEFAULT_I2V,
        t2v_workflow: Optional[str] = None,
        flf2v_workflow: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.client = ComfyUIClient(base_url=base_url)
        self.i2v_workflow_path = i2v_workflow
        self.t2v_workflow_path = t2v_workflow
        self.flf2v_workflow_path = flf2v_workflow
        self.rate_limiter = rate_limiter

    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str],
        aspect_ratio: str = "16:9",
        duration: int = 5,
        resolution: str = "720p",
        **kwargs,
    ) -> VideoOutput:
        if len(reference_image_paths) == 0:
            if not self.t2v_workflow_path:
                raise ValueError("VideoGeneratorComfyUI: no t2v_workflow configured; supply a first frame.")
            workflow = load_workflow(self.t2v_workflow_path)
            uploaded = None
        elif len(reference_image_paths) == 1:
            workflow = load_workflow(self.i2v_workflow_path)
            uploaded = await self.client.upload_image(reference_image_paths[0])
            set_node_input_by_titles(workflow, I2V_REF_TITLES, "image", uploaded)
        elif len(reference_image_paths) == 2:
            if self.flf2v_workflow_path:
                workflow = load_workflow(self.flf2v_workflow_path)
                uploaded_first = await self.client.upload_image(reference_image_paths[0])
                uploaded_last = await self.client.upload_image(reference_image_paths[1])
                set_node_input_by_titles(workflow, FLF2V_FIRST_TITLES, "image", uploaded_first)
                set_node_input_by_titles(workflow, FLF2V_LAST_TITLES, "image", uploaded_last)
            else:
                # Graceful degradation: no flf2v workflow registered.
                # Fall back to single-frame i2v using the first frame only,
                # log loudly so the loss of the designed transition is visible.
                # This unblocks end-to-end runs while flf2v wiring is pending.
                # When an flf2v workflow is later registered, this branch is
                # never taken and the shot gets its first→last transition back.
                logging.warning(
                    "VideoGeneratorComfyUI: 2 reference frames supplied but no "
                    "flf2v_workflow configured. Degrading to single-frame i2v "
                    "(using first_frame only, dropping last_frame). Shot will "
                    "lose its designed first→last transition. Register an "
                    "flf2v_workflow to restore full fidelity."
                )
                workflow = load_workflow(self.i2v_workflow_path)
                uploaded = await self.client.upload_image(reference_image_paths[0])
                set_node_input_by_titles(workflow, I2V_REF_TITLES, "image", uploaded)
        else:
            raise ValueError("reference_image_paths must contain 0, 1, or 2 images.")

        set_node_input_by_titles(workflow, PROMPT_TITLES, "text", prompt)

        sampler_id = find_node_by_title(workflow, "Sampler")
        if sampler_id is not None:
            seed = random.randint(0, 2**31 - 1)
            inputs = workflow[sampler_id]["inputs"]
            for key in ("seed", "noise_seed"):
                if key in inputs:
                    inputs[key] = seed
                    break

        # Wan 2.2 default 16fps; convert duration seconds -> frame count
        fps = int(kwargs.get("fps") or 16)
        frames = max(1, int(duration * fps))

        size_node_id = find_node_by_title(workflow, "Video Size")
        if size_node_id is not None:
            width, height = _parse_size(kwargs.get("size"), aspect_ratio)
            inputs = workflow[size_node_id]["inputs"]
            inputs["width"] = width
            inputs["height"] = height
            for key in ("length", "frames", "frame_count", "num_frames", "video_frames"):
                if key in inputs:
                    inputs[key] = frames
                    break

        frame_node_id = find_node_by_title(workflow, "Frame Count")
        if frame_node_id is not None and frame_node_id != size_node_id:
            inputs = workflow[frame_node_id]["inputs"]
            for key in ("length", "frames", "frame_count", "num_frames", "video_frames"):
                if key in inputs:
                    inputs[key] = frames
                    break

        if self.rate_limiter:
            await self.rate_limiter.acquire()

        logging.info(f"ComfyUI video i2v: prompt='{prompt[:80]}...' duration={duration}s ({frames} frames)")
        prompt_id = await self.client.queue_prompt(workflow)
        history = await self.client.wait_for_completion(prompt_id)
        outputs = ComfyUIClient.collect_outputs(history)

        videos = [o for o in outputs if o[0].lower().endswith((".mp4", ".webm", ".mov", ".gif"))]
        if not videos:
            raise RuntimeError(f"ComfyUI returned no video outputs for prompt {prompt_id}: {outputs}")
        filename, subfolder, out_type = videos[0]
        data = await self.client.fetch_output(filename, subfolder=subfolder, out_type=out_type)

        ext = Path(filename).suffix.lstrip(".") or "mp4"
        return VideoOutput(fmt="bytes", ext=ext, data=data)
