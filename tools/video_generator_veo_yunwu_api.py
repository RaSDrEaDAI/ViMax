import logging
from typing import List, Optional
from PIL import Image
import asyncio
import aiohttp
from interfaces.video_output import VideoOutput
from utils.image import image_path_to_b64


class VideoGeneratorVeoYunwuAPI:
    def __init__(
        self,
        api_key: str,
        t2v_model: str = "veo3.1-fast",  # text to video
        ff2v_model: str = "veo3.1-fast",   # first frame to video
        flf2v_model: str = "veo2-fast-frames",  # first and last frame to video
        max_create_attempts: int = 3,
        poll_interval: int = 2,
        max_poll_attempts: int = 300,
    ):
        """
        all models:
            veo2
            veo2-fast
            veo2-fast-frames
            veo2-fast-components
            veo2-pro
            veo3
            veo3-fast
            veo3-pro
            veo3-pro-frames
            veo3-fast-frames
            veo3-frames

        NOTE: veo3 does not support first and last frame to video generation.
        """
        self.base_url = "https://yunwu.ai"
        self.api_key = api_key
        self.t2v_model = t2v_model
        self.ff2v_model = ff2v_model
        self.flf2v_model = flf2v_model
        self.max_create_attempts = max_create_attempts
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts

    async def generate_single_video(
        self,
        prompt: str = "",
        reference_image_paths: List[Image.Image] = [],
        aspect_ratio: str = "16:9",
        **kwargs,
    ) -> VideoOutput:
        if len(reference_image_paths) == 0:
            model = self.t2v_model
        elif len(reference_image_paths) == 1:
            model = self.ff2v_model
        elif len(reference_image_paths) == 2:
            model = self.flf2v_model
        else:
            raise ValueError("The number of reference images must be no more than 2")

        logging.info(f"Calling {model} to generate video...")

        # 1. Create video generation task
        payload = {
            "prompt": prompt,
            "model": model,
            "images": [image_path_to_b64(image_path, mime=True) for image_path in reference_image_paths],
            "enhance_prompt": True,
        }
        # only veo3 supports aspect ratio setting
        if model.startswith("veo3"):
            payload["aspect_ratio"] = aspect_ratio

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


        url = f"https://yunwu.ai/v1/video/create"
        task_id = None
        last_error = None
        for attempt in range(1, self.max_create_attempts + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload) as response:
                        response_json = await response.json()
                        http_status = response.status
                logging.debug(f"Response: {response_json}")
            except Exception as e:
                last_error = e
                logging.error(f"Error occurred while creating video generation task (attempt {attempt}/{self.max_create_attempts}): {e}")
                if attempt < self.max_create_attempts:
                    await asyncio.sleep(attempt)
                continue

            if http_status >= 400:
                message = f"Video generation task creation failed with HTTP {http_status}: {response_json}"
                if http_status < 500:
                    raise RuntimeError(message)
                last_error = RuntimeError(message)
                logging.error(f"{message} (attempt {attempt}/{self.max_create_attempts})")
                if attempt < self.max_create_attempts:
                    await asyncio.sleep(attempt)
                continue

            task_id = response_json.get("id")
            if not task_id:
                raise RuntimeError(f"Video generation task creation returned no task id: {response_json}")
            logging.info(f"Video generation task created successfully. Task ID: {task_id}")
            break
        if task_id is None:
            raise RuntimeError(f"Failed to create video generation task after {self.max_create_attempts} attempts.") from last_error


        # 2. Query the video generation task until the video generation is completed
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }

        attempts = 0
        consecutive_errors = 0
        while True:
            if attempts >= self.max_poll_attempts:
                raise TimeoutError(f"Video generation task {task_id} did not complete after {attempts} polls.")
            attempts += 1

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/v1/video/query?id={task_id}", headers=headers) as response:
                        payload = await response.json()
                        http_status = response.status
                logging.debug(f"Response: {payload}")
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    raise RuntimeError(f"Querying video generation task {task_id} failed {consecutive_errors} times in a row.") from e
                logging.error(f"Error occurred while querying video generation task: {e}. Retrying in {self.poll_interval} seconds...")
                await asyncio.sleep(self.poll_interval)
                continue
            consecutive_errors = 0

            if http_status >= 400:
                raise RuntimeError(f"Querying video generation task {task_id} failed with HTTP {http_status}: {payload}")

            status = payload.get("status")
            if status == "completed":
                logging.info(f"Video generation completed successfully")
                video_url = payload["video_url"]
                return VideoOutput(fmt="url", ext="mp4", data=video_url)
            elif status == "failed":
                # Used to `break` out of the loop and fall off the end of the
                # function, returning None to a caller that then crashed far
                # from the cause.
                logging.error(f"Video generation failed: \n{payload}")
                raise RuntimeError(f"Video generation task {task_id} failed: {payload}")
            else:
                logging.info(f"Video generation status: {status}, waiting {self.poll_interval} seconds...")
                await asyncio.sleep(self.poll_interval)
