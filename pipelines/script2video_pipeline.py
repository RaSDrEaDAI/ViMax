import os
import re
import shutil
import json
import logging
import asyncio
import time
from typing import Optional, Dict, List, Tuple, Literal, Union
from urllib.parse import urlparse
from moviepy import VideoFileClip, concatenate_videoclips
from PIL import Image
from agents import *
import yaml
from interfaces import *
from langchain.chat_models import init_chat_model
from tools.render_backend import RenderBackend, _load_dotenv, _substitute_env_vars
from utils.provider_presets import resolve_chat_model_config
from utils.composite_sheet import build_character_sheet, character_sheet_description
from utils.image import download_image
from utils.reference_slots import (
    assemble_indexed_prompt,
    build_reference_slots,
    resolve_character_sheet,
    resolve_environment_plate,
)


def _seedbed_slug(url: str) -> str:
    """Filesystem-safe slug from a seedbed URL's basename.

    Keeps the ingested filename recognizable against the source URL, which is
    what makes a seedbed directory readable when an operator is checking which
    asset landed where.

    Parses the path rather than splitting the raw string: a URL with no path
    ("https://host/") would otherwise yield a slice of the HOSTNAME as the
    filename, which reads as a real asset name and isn't one.
    """
    path = urlparse(url).path
    stem = os.path.splitext(os.path.basename(path.rstrip("/")))[0]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return slug[:48] or "asset"

class Script2VideoPipeline:

    # events
    character_portrait_events = {}
    shot_desc_events = {}
    frame_events = {}


    def __init__(
        self,
        chat_model: str,
        image_generator,
        video_generator,
        working_dir: str,
        sheet_image_generator=None,
        vision_model=None,
    ):

        self.chat_model = chat_model
        self.image_generator = image_generator
        self.video_generator = video_generator
        # Sheet-class assets (character portraits, rotation grids, environment
        # plates) can render on a different backend than shot keyframes — they
        # are the identity and location authorities the keyframes reference, not
        # keyframes themselves, so a free local backend is usually the right
        # place for them. Defaults to the keyframe backend, which is the
        # pre-existing single-backend behaviour.
        self.sheet_image_generator = sheet_image_generator or image_generator
        # Optional. Used to describe seedbed assets at ingest. Absence is
        # recorded as description: null, never faked.
        self.vision_model = vision_model

        self.character_extractor = CharacterExtractor(chat_model=self.chat_model)
        self.character_portraits_generator = CharacterPortraitsGenerator(image_generator=self.sheet_image_generator)
        self.environment_extractor = EnvironmentExtractor(chat_model=self.chat_model)
        self.environment_plate_generator = EnvironmentPlateGenerator(image_generator=self.sheet_image_generator)
        self.storyboard_artist = StoryboardArtist(chat_model=self.chat_model)
        self.camera_image_generator = CameraImageGenerator(chat_model=self.chat_model, image_generator=self.image_generator, video_generator=self.video_generator)
        self.reference_image_selector = ReferenceImageSelector(chat_model=self.chat_model)

        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)



    @classmethod
    def init_from_config(
        cls,
        config_path: str,
        working_dir_override: Optional[str] = None,
    ):
        """Build a pipeline from a YAML config.

        Args:
            config_path: Path to the YAML config file.
            working_dir_override: If set, overrides ``config["working_dir"]``.
                Used by the orchestrator bridge for per-job isolation so
                concurrent triggers don't collide on the same working dir.
                The directory is created (exist_ok=True) in __init__.
        """
        # Source ViMax's .env into os.environ BEFORE anything else — chat_model
        # and other secrets are referenced as ${VAR} in the YAML and must be
        # available for substitution. _load_dotenv does NOT override existing
        # env vars (env wins over file), so the orchestrator bridge can still
        # inject keys via the subprocess environment.
        _load_dotenv()

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Apply ${VAR} substitution to chat_model init_args BEFORE init_chat_model.
        # The chat_model is constructed here, NOT via _instantiate, so the
        # substitution that runs inside RenderBackend.from_config never touches
        # it. Without this, init_chat_model receives the literal "${ZAI_API_KEY}"
        # string as api_key and z.ai returns "401 token expired or incorrect".
        chat_model_args = _substitute_env_vars(config["chat_model"]["init_args"])
        chat_model_args = resolve_chat_model_config(chat_model_args)
        chat_model = init_chat_model(**chat_model_args)
        backend = RenderBackend.from_config(
            config,
            chat_model=chat_model,
            working_dir_override=working_dir_override,
        )

        pipeline = cls(
            chat_model=chat_model,
            image_generator=backend.image_generator,
            video_generator=backend.video_generator,
            working_dir=working_dir_override or config["working_dir"],
            sheet_image_generator=backend.sheet_image_generator,
        )

        # If the chat model is text-only (e.g. z.ai Coding Plan endpoint),
        # disable the multimodal vision pass in the reference image selector.
        if config.get("chat_model", {}).get("text_only"):
            pipeline.reference_image_selector.text_only = True

        return pipeline

    async def __call__(
        self,
        script: str,
        user_requirement: str,
        style: str,
        characters: List[CharacterInScene] = None,
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
        seed_assets: Optional[List[Union[SeedAsset, str, Dict]]] = None,
        reference_image_urls: Optional[List[str]] = None,
    ):
        # Seedbed intake runs FIRST, before any generation: a pinned asset has to
        # be on disk before the frame path decides whether to render that frame.
        #
        # reference_image_urls is the legacy name for this seam. It was plumbed
        # through both pipeline signatures and never read in the body, so
        # orchestrators sending seedbed URLs observed no effect at all. Bare URLs
        # coerce to role='reference', so existing callers keep working.
        if reference_image_urls and not seed_assets:
            seed_assets = reference_image_urls
        seedbed_registry = await self.ingest_seed_assets(seed_assets)

        if characters is None:
            characters = await self.extract_characters(script=script)

            # characters_path = os.path.join(self.working_dir, "characters.json")
            # if os.path.exists(characters_path):
            #     with open(characters_path, "r", encoding="utf-8") as f:
            #         characters = [CharacterInScene.model_validate(c) for c in json.load(f)]
            #     print(f"🚀 Loaded {len(characters)} characters from existing file.")
            # else:
            #     print(f"🔍 Extracting characters from script...")
            #     characters = await self.extract_characters(script=script)
            #     with open(characters_path, "w", encoding="utf-8") as f:
            #         json.dump([c.model_dump() for c in characters], f, ensure_ascii=False, indent=4)
            #     print(f"☑️ Extracted {len(characters)} characters from script and saved to {characters_path}.")

        if character_portraits_registry is None:
            character_portraits_registry_path = os.path.join(self.working_dir, "character_portraits_registry.json")
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, "r", encoding="utf-8") as f:
                    character_portraits_registry = json.load(f)
                print(f"🚀 Loaded {len(character_portraits_registry)} character portraits from existing file.")
            else:
                print(f"🔍 Generating character portraits...")
                character_portraits_registry = await self.generate_character_portraits(
                    characters=characters,
                    character_portraits_registry=None,
                    style=style,
                )

                with open(character_portraits_registry_path, "w", encoding="utf-8") as f:
                    json.dump(character_portraits_registry, f, ensure_ascii=False, indent=4)
                print(f"☑️ Generated {len(character_portraits_registry)} character portraits and saved to {character_portraits_registry_path}.")



        # Environments run BEFORE the storyboard: the storyboard prompt needs
        # the location list in order to assign each shot an env_idx.
        environments = await self.extract_environments(script=script)
        environment_registry = await self.generate_environment_plates(
            environments=environments,
            style=style,
        )

        # design shots
        storyboard = await self.design_storyboard(
            script=script,
            characters=characters,
            user_requirement=user_requirement,
            environments=environments,
        )

        # decompose visual descriptions of shots
        shot_descriptions = await self.decompose_visual_descriptions(
            shot_brief_descriptions=storyboard,
            characters=characters,
        )

        # construct camera tree
        camera_tree = await self.construct_camera_tree(
            shot_descriptions=shot_descriptions,
        )

        # Pins land AFTER the shot list exists (shot_idx has to be range-checked
        # against it) and BEFORE any frame generation, so the existing
        # skip-if-exists checks adopt the pinned PNG instead of spending on it.
        self.apply_seedbed_pins(
            seedbed_registry=seedbed_registry,
            shot_descriptions=shot_descriptions,
        )

        priority_shot_idxs = [camera.parent_cam_idx for camera in camera_tree if camera.parent_cam_idx is not None]
        tasks = [
            self.generate_frames_for_single_camera(
                camera=camera,
                shot_descriptions=shot_descriptions,
                characters=characters,
                character_portraits_registry=character_portraits_registry,
                priority_shot_idxs=priority_shot_idxs,
                environments=environments,
                environment_registry=environment_registry,
                seedbed_registry=seedbed_registry,
            )
            for camera in camera_tree
        ]

        video_tasks = [
            self.generate_video_for_single_shot(
                shot_description=shot_description,
            )
            for shot_description in shot_descriptions
        ]
        tasks.extend(video_tasks)
        await asyncio.gather(*tasks)

        final_video_path = os.path.join(self.working_dir, "final_video.mp4")
        if os.path.exists(final_video_path):
            print(f"🚀 Skipped concatenating videos, already exists.")
        else:
            print(f"🎬 Starting concatenating videos...")
            video_clips = [
                VideoFileClip(os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "video.mp4"))
                for shot_description in shot_descriptions
            ]
            final_video = concatenate_videoclips(video_clips)
            final_video.write_videofile(final_video_path, codec="libx264", preset="medium")
            print(f"☑️ Concatenated videos, saved to {final_video_path}.")

        return final_video_path


    async def generate_frames_for_single_camera(
        self,
        camera: Camera,
        shot_descriptions: List[ShotDescription],
        characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        priority_shot_idxs: List[int],
        environments: Optional[List[EnvironmentInScene]] = None,
        environment_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
        seedbed_registry: Optional[List[Dict]] = None,
    ):
        environments = environments or []
        environment_registry = environment_registry or {}

        # 1. generate the first_frame of the first shot of the camera
        first_shot_idx = camera.active_shot_idxs[0]
        first_shot_ff_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", "first_frame.png")

        if os.path.exists(first_shot_ff_path):
            print(f"🚀 Skipped generating first_frame for shot {first_shot_idx}, already exists.")
            self.frame_events[first_shot_idx]["first_frame"].set()

        else:
            print(f"🖼️ Starting first_frame generation for shot {first_shot_idx}...")
            continuity_anchor = None

            # generate the first_frame based on the shot_description.ff_desc
            if camera.parent_shot_idx is not None:
                # generate the first_frame based on the transition video
                parent_shot_idx = camera.parent_shot_idx
                await self.frame_events[parent_shot_idx]["first_frame"].wait()
                parent_shot_ff_path = os.path.join(self.working_dir, "shots", f"{parent_shot_idx}", "first_frame.png")
                transition_video_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", f"transition_video_from_shot_{parent_shot_idx}.mp4")

                if os.path.exists(transition_video_path):
                    print(f"🚀 Skipped generating transition video for shot {first_shot_idx} from shot {parent_shot_idx}, already exists.")
                else:
                    print(f"🖼️ Starting transition video generation for shot {first_shot_idx} from shot {parent_shot_idx}...")
                    transition_video_output = await self.camera_image_generator.generate_transition_video(
                        first_shot_visual_desc=shot_descriptions[parent_shot_idx].visual_desc,
                        second_shot_visual_desc=shot_descriptions[first_shot_idx].visual_desc,
                        first_shot_ff_path=parent_shot_ff_path,
                    )
                    transition_video_output.save(transition_video_path)
                    print(f"☑️ Generated transition video for shot {first_shot_idx} from shot {parent_shot_idx}, saved to {transition_video_path}.")

                new_camera_image_path = os.path.join(self.working_dir, "shots", f"{first_shot_idx}", f"new_camera_{camera.idx}.png")
                if os.path.exists(new_camera_image_path):
                    print(f"🚀 Skipped generating new camera image for shot {first_shot_idx}, already exists.")
                else:
                    print(f"🖼️ Starting new camera image generation for shot {first_shot_idx}...")
                    new_camera_image = self.camera_image_generator.get_new_camera_image(transition_video_path)
                    new_camera_image.save(new_camera_image_path)
                    print(f"☑️ Generated new camera image for shot {first_shot_idx} (not completed), saved to {new_camera_image_path}.")

                # Set OUTSIDE the skip-if-exists branch. Previously this was
                # appended only when the image had just been generated, so a
                # re-run over an existing new_camera_*.png silently lost the
                # camera anchor and re-rendered the frame without its
                # composition reference.
                continuity_anchor = (
                    new_camera_image_path,
                    f"continuity anchor for this camera — the composition and "
                    f"background are correct and must be preserved exactly. Some "
                    f"elements are wrong and must be replaced: "
                    f"{camera.missing_info}. Replace the characters in this image "
                    f"with the characters from their character sheets. Do not "
                    f"change the background.",
                )


            # 如果子镜头缺少信息，则需要选择参考图像生成
            if camera.parent_shot_idx is None or camera.missing_info is not None:
                await self._generate_keyframe(
                    shot_idx=first_shot_idx,
                    frame_type="first_frame",
                    out_path=first_shot_ff_path,
                    frame_desc=shot_descriptions[first_shot_idx].ff_desc,
                    visible_characters=[
                        characters[idx]
                        for idx in shot_descriptions[first_shot_idx].ff_vis_char_idxs
                    ],
                    character_portraits_registry=character_portraits_registry,
                    environments=environments,
                    environment_registry=environment_registry,
                    env_idx=shot_descriptions[first_shot_idx].env_idx,
                    continuity_anchor=continuity_anchor,
                    seedbed_registry=seedbed_registry,
                    variation_type=shot_descriptions[first_shot_idx].variation_type,
                )
                self.frame_events[first_shot_idx]["first_frame"].set()
            else:
                shutil.copy(new_camera_image_path, first_shot_ff_path)
                self.frame_events[first_shot_idx]["first_frame"].set()
                print(f"☑️ Generated first_frame for shot {first_shot_idx}, saved to {first_shot_ff_path}.")


        # 2. generate the following frames of the camera
        priority_tasks = []
        normal_tasks = []

        # The camera's own first frame is the continuity anchor for every other
        # frame this camera shoots — that is what holds the composition steady
        # across the shots sharing it.
        camera_anchor = (
            first_shot_ff_path,
            f"continuity anchor — the first frame of this camera. Match its "
            f"framing, background, and lighting. It shows: "
            f"{shot_descriptions[first_shot_idx].ff_desc}",
        )

        def _frame_task(shot_idx: int, frame_type: str):
            shot = shot_descriptions[shot_idx]
            frame_desc = shot.ff_desc if frame_type == "first_frame" else shot.lf_desc
            vis_idxs = (
                shot.ff_vis_char_idxs if frame_type == "first_frame"
                else shot.lf_vis_char_idxs
            )
            return self.generate_frame_for_single_shot(
                shot_idx=shot_idx,
                frame_type=frame_type,
                camera_anchor=camera_anchor,
                frame_desc=frame_desc,
                visible_characters=[characters[idx] for idx in vis_idxs],
                character_portraits_registry=character_portraits_registry,
                environments=environments,
                environment_registry=environment_registry,
                env_idx=shot.env_idx,
                seedbed_registry=seedbed_registry,
                variation_type=shot.variation_type,
            )

        if shot_descriptions[first_shot_idx].variation_type in ["medium", "large"]:
            normal_tasks.append(_frame_task(first_shot_idx, "last_frame"))

        for shot_idx in camera.active_shot_idxs[1:]:
            first_frame_task = _frame_task(shot_idx, "first_frame")
            if shot_idx in priority_shot_idxs:
                priority_tasks.append(first_frame_task)
            else:
                normal_tasks.append(first_frame_task)

            if shot_descriptions[shot_idx].variation_type in ["medium", "large"]:
                normal_tasks.append(_frame_task(shot_idx, "last_frame"))


        await asyncio.gather(*priority_tasks)
        await asyncio.gather(*normal_tasks)



    async def generate_video_for_single_shot(
        self,
        shot_description: ShotDescription,
    ):
        video_path = os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "video.mp4")
        if os.path.exists(video_path):
            print(f"🚀 Skipped generating video for shot {shot_description.idx}, already exists.")
        else:
            await self.frame_events[shot_description.idx]["first_frame"].wait()
            if shot_description.variation_type in ["medium", "large"]:
                await self.frame_events[shot_description.idx]["last_frame"].wait()

            frame_paths = []
            frame_paths.append(os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "first_frame.png"))
            if shot_description.variation_type in ["medium", "large"]:
                frame_paths.append(os.path.join(self.working_dir, "shots", f"{shot_description.idx}", "last_frame.png"))

            print(f"🎬 Starting video generation for shot {shot_description.idx}...")
            video_output = await self.video_generator.generate_single_video(
                prompt=shot_description.motion_desc + "\n" + shot_description.audio_desc,
                reference_image_paths=frame_paths,
                # Structured metadata for smart routing (local LTX vs paid API):
                # - variation_type drives large-variation escalation
                # - audio_desc drives lipsync escalation
                # - motion_desc drives complex-motion heuristic fallback
                variation_type=shot_description.variation_type,
                audio_desc=shot_description.audio_desc,
                motion_desc=shot_description.motion_desc,
            )
            video_output.save(video_path)
            print(f"☑️ Generated video for shot {shot_description.idx}, saved to {video_path}.")

    async def generate_frame_for_single_shot(
        self,
        shot_idx: int,
        frame_type: Literal["first_frame", "last_frame"],
        camera_anchor: Tuple[str, str],
        frame_desc: str,
        visible_characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        environments: Optional[List[EnvironmentInScene]] = None,
        environment_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
        env_idx: Optional[int] = None,
        seedbed_registry: Optional[List[Dict]] = None,
        variation_type: Optional[str] = None,
    ) -> str:
        frame_image_path = os.path.join(self.working_dir, "shots", f"{shot_idx}", f"{frame_type}.png")

        if os.path.exists(frame_image_path):
            print(f"🚀 Skipped generating {frame_type} for shot {shot_idx}, already exists.")
        else:
            await self._generate_keyframe(
                shot_idx=shot_idx,
                frame_type=frame_type,
                out_path=frame_image_path,
                frame_desc=frame_desc,
                visible_characters=visible_characters,
                character_portraits_registry=character_portraits_registry,
                environments=environments or [],
                environment_registry=environment_registry or {},
                env_idx=env_idx,
                continuity_anchor=camera_anchor,
                seedbed_registry=seedbed_registry,
                variation_type=variation_type,
            )

        self.frame_events[shot_idx][frame_type].set()
        return frame_image_path


    async def _generate_keyframe(
        self,
        *,
        shot_idx: int,
        frame_type: str,
        out_path: str,
        frame_desc: str,
        visible_characters: List[CharacterInScene],
        character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
        environments: List[EnvironmentInScene],
        environment_registry: Dict[str, Dict[str, Dict[str, str]]],
        env_idx: Optional[int],
        continuity_anchor: Optional[Tuple[str, str]],
        seedbed_registry: Optional[List[Dict]],
        variation_type: Optional[str],
    ) -> None:
        """Render one keyframe through the deterministic slot budget.

        The single keyframe call site. References are budgeted by role priority
        rather than chosen by an LLM: character sheets, then the location plate,
        then the continuity anchor are ALWAYS in, because they are the identity,
        location, and continuity authorities — a frame that silently loses one
        renders an invented face or an invented room. Only the seedbed reference
        pool is model-selected, and only from that pool.
        """
        print(f"🖼️ Starting {frame_type} generation for shot {shot_idx}...")

        # Own the directory rather than relying on the decomposition stage having
        # created it earlier in the run. This writer is now also reachable
        # directly (pins, reruns), and an implicit ordering dependency between
        # two unrelated stages is exactly the kind of thing that breaks later.
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # Slot 1: character sheets. Fails loud when a character has no sheet.
        character_sheets = [
            resolve_character_sheet(character_portraits_registry, c.identifier_in_scene)
            for c in visible_characters
        ]

        # Slot 2: location plate. Fails loud on missing / out-of-range env_idx.
        environment_plate = resolve_environment_plate(
            environment_registry, environments, env_idx, shot_idx,
        )

        # Slot 4: seedbed references. The selector picks from this pool ONLY; it
        # no longer chooses among the authorities above.
        seedbed_references, selector_text_prompt = await self._select_seedbed_references(
            shot_idx=shot_idx,
            frame_type=frame_type,
            frame_desc=frame_desc,
            seedbed_registry=seedbed_registry,
        )

        budget = build_reference_slots(
            character_sheets=character_sheets,
            environment_plate=environment_plate,
            continuity_anchor=continuity_anchor,
            seedbed_references=seedbed_references,
        )
        for notice in budget.notices:
            # Printed, never swallowed: silent truncation reads as full coverage.
            print(f"⚠️  shot {shot_idx} {frame_type}: {notice}")

        frame_instruction = frame_desc
        if selector_text_prompt:
            frame_instruction = f"{frame_desc}\n\n{selector_text_prompt}"
        prompt = assemble_indexed_prompt(budget.slots, frame_instruction)

        frame_image: ImageOutput = await self.image_generator.generate_single_image(
            prompt=prompt,
            reference_image_paths=budget.paths,
            # Explicit, and the ONLY geometry argument. The old size="1600x900"
            # was swallowed by **kwargs and never reached any backend. Keyframes
            # stay 16:9 to match the video contract.
            aspect_ratio="16:9",
            # Structured metadata for smart routing (multi-model router):
            # - visible_characters drives multi-char -> Ideogram bbox routing
            # - frame_desc is the input to CompositionPlanner
            # - shot_notes gives the planner framing context
            visible_characters=visible_characters,
            frame_desc=frame_desc,
            shot_notes=f"Shot {shot_idx}, {frame_type} frame. Variation: {variation_type or 'unknown'}.",
        )
        frame_image.save(out_path)

        # Forensics: persist what was ACTUALLY dispatched, not a pre-call draft.
        # Backends that don't report sent_input still get the slot record, so
        # there is always a file saying which references produced this frame.
        sent_input_path = os.path.join(
            self.working_dir, "shots", f"{shot_idx}", f"{frame_type}_nbpro_input.json",
        )
        record = {
            "sent_input": getattr(frame_image, "sent_input", None),
            "slots": [
                {"role": s.role, "path": s.path, "text": s.text}
                for s in budget.slots
            ],
            "degrade_notices": budget.notices,
        }
        with open(sent_input_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=4, default=str)

        print(f"☑️ Generated {frame_type} for shot {shot_idx}, saved to {out_path}.")


    async def _select_seedbed_references(
        self,
        *,
        shot_idx: int,
        frame_type: str,
        frame_desc: str,
        seedbed_registry: Optional[List[Dict]],
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Pick seedbed `reference` assets for this frame from that pool alone.

        Returns ``(pairs, text_prompt)``. With an empty pool this is a no-op and
        no LLM call fires — worth noting because the old path ran the selector
        unconditionally, including when there was nothing to choose between.
        """
        pool = seedbed_reference_pool(seedbed_registry)
        if not pool:
            return [], ""

        selector_output_path = os.path.join(
            self.working_dir, "shots", f"{shot_idx}", f"{frame_type}_selector_output.json",
        )
        if os.path.exists(selector_output_path):
            with open(selector_output_path, "r", encoding="utf-8") as f:
                selector_output = json.load(f)
            print(f"🚀 Loaded existing seedbed reference selection for {frame_type} of shot {shot_idx}.")
        else:
            print(f"🔍 Selecting seedbed references for {frame_type} of shot {shot_idx}...")
            selector_output = await self.reference_image_selector.select_reference_images_and_generate_prompt(
                available_image_path_and_text_pairs=pool,
                frame_description=frame_desc,
            )
            os.makedirs(os.path.dirname(selector_output_path), exist_ok=True)
            with open(selector_output_path, "w", encoding="utf-8") as f:
                json.dump(selector_output, f, ensure_ascii=False, indent=4)

        pairs = [
            (item[0], item[1])
            for item in selector_output.get("reference_image_path_and_text_pairs", [])
        ]
        return pairs, selector_output.get("text_prompt", "")


    async def construct_camera_tree(
        self,
        shot_descriptions: List[ShotDescription],
    ):
        camera_tree_path = os.path.join(self.working_dir, "camera_tree.json")

        if os.path.exists(camera_tree_path):
            with open(camera_tree_path, "r", encoding="utf-8") as f:
                camera_tree = json.load(f)
            camera_tree = [Camera.model_validate(camera) for camera in camera_tree]
            print(f"🚀 Loaded {len(camera_tree)} cameras from existing file.")
            return camera_tree

        cameras: List[Camera] = []
        for shot_description in shot_descriptions:
            if shot_description.cam_idx not in [camera.idx for camera in cameras]:
                cameras.append(Camera(idx=shot_description.cam_idx, active_shot_idxs=[shot_description.idx]))
            else:
                cameras[shot_description.cam_idx].active_shot_idxs.append(shot_description.idx)

        camera_tree = await self.camera_image_generator.construct_camera_tree(cameras=cameras, shot_descs=shot_descriptions)
        with open(camera_tree_path, "w", encoding="utf-8") as f:
            json.dump([camera.model_dump() for camera in camera_tree], f, ensure_ascii=False, indent=4)
        print(f"✅ Constructed camera tree and saved to {camera_tree_path}.")
        return camera_tree




    async def extract_characters(
        self,
        script: str,
    ):
        save_path = os.path.join(self.working_dir, "characters.json")

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
            characters = [CharacterInScene.model_validate(character) for character in characters]
            print(f"🚀 Loaded {len(characters)} characters from existing file.")
        else:
            characters = await self.character_extractor.extract_characters(script)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([character.model_dump() for character in characters], f, ensure_ascii=False, indent=4)
            print(f"✅ Extracted {len(characters)} characters from script and saved to {save_path}.")

        for character in characters:
            self.character_portrait_events[character.idx] = asyncio.Event()

        return characters


    async def generate_character_portraits(
        self,
        characters: List[CharacterInScene],
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]],
        style: str,
    ):
        character_portraits_registry_path = os.path.join(self.working_dir, "character_portraits_registry.json")
        if character_portraits_registry is None:
            if os.path.exists(character_portraits_registry_path):
                with open(character_portraits_registry_path, 'r', encoding='utf-8') as f:
                    character_portraits_registry = json.load(f)
            else:
                character_portraits_registry = {}


        tasks = [
            self.generate_portraits_for_single_character(character, style)
            for character in characters
            if character.identifier_in_scene not in character_portraits_registry
        ]
        if tasks:
            for future in asyncio.as_completed(tasks):
                character_portraits_registry.update(await future)
                with open(character_portraits_registry_path, 'w', encoding='utf-8') as f:
                    json.dump(character_portraits_registry, f, ensure_ascii=False, indent=4)

            print(f"✅ Completed character portrait generation for {len(characters)} characters.")
        else:
            print("🚀 All characters already have portraits, skipping portrait generation.")
        return character_portraits_registry


    async def generate_portraits_for_single_character(
        self,
        character: CharacterInScene,
        style: str,
    ):
        """Generate the composite character sheet for one character.

        Chain: front portrait (identity master, t2i) -> identity anchor (i2i on
        front) -> 2x2 rotation grid (i2i on anchor) -> deface + compose. Three
        paid/rendered calls plus one free local composite, replacing the old
        front/side/back triple.

        The separate side and back portrait calls are NOT invoked here any more.
        ``CharacterPortraitsGenerator.generate_side_portrait`` /
        ``generate_back_portrait`` remain in the codebase for the legacy
        registry shape and for callers that still want individual views.

        Skip-if-exists at every step, like every other stage — a re-run after a
        failure resumes rather than re-spending.
        """
        character_dir = os.path.join(self.working_dir, "character_portraits", f"{character.idx}_{character.identifier_in_scene}")
        os.makedirs(character_dir, exist_ok=True)

        front_portrait_path = os.path.join(character_dir, "front.png")
        if os.path.exists(front_portrait_path):
            print(f"🚀 Skipped front portrait for {character.identifier_in_scene}, already exists.")
        else:
            front_portrait_output = await self.character_portraits_generator.generate_front_portrait(character, style)
            front_portrait_output.save(front_portrait_path)

        anchor_path = os.path.join(character_dir, "anchor.png")
        if os.path.exists(anchor_path):
            print(f"🚀 Skipped identity anchor for {character.identifier_in_scene}, already exists.")
        else:
            print(f"🖼️ Generating identity anchor for {character.identifier_in_scene}...")
            anchor_output = await self.character_portraits_generator.generate_identity_anchor(character, front_portrait_path)
            anchor_output.save(anchor_path)

        grid_path = os.path.join(character_dir, "rotation_grid.png")
        if os.path.exists(grid_path):
            print(f"🚀 Skipped rotation grid for {character.identifier_in_scene}, already exists.")
        else:
            print(f"🖼️ Generating rotation grid for {character.identifier_in_scene}...")
            grid_output = await self.character_portraits_generator.generate_rotation_grid(character, anchor_path)
            grid_output.save(grid_path)

        sheet_path = os.path.join(character_dir, "sheet.png")
        if os.path.exists(sheet_path):
            print(f"🚀 Skipped character sheet for {character.identifier_in_scene}, already exists.")
        else:
            print(f"🧩 Composing character sheet for {character.identifier_in_scene}...")
            build_character_sheet(
                anchor_path=anchor_path,
                grid_path=grid_path,
                out_path=sheet_path,
            )
            print(f"☑️ Composed character sheet, saved to {sheet_path}.")

        self.character_portrait_events[character.idx].set()

        print(f"☑️ Completed character portrait generation for {character.identifier_in_scene}.")

        return {
            character.identifier_in_scene: {
                "sheet": {
                    "path": sheet_path,
                    "description": character_sheet_description(character.identifier_in_scene),
                },
                "front": {
                    "path": front_portrait_path,
                    "description": f"A front view portrait of {character.identifier_in_scene}.",
                },
            }
        }



    async def ingest_seed_assets(
        self,
        seed_assets: Optional[List[Union[SeedAsset, str, Dict]]],
    ) -> List[Dict]:
        """Download seedbed assets and write ``seedbed_registry.json``.

        Runs before any generation. Validation (a pin without a shot_idx) raises
        here rather than at frame time, so a malformed job fails before spending
        anything.
        """
        assets = coerce_seed_assets(seed_assets)
        registry_path = os.path.join(self.working_dir, "seedbed_registry.json")

        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            print(f"🚀 Loaded {len(registry)} seedbed assets from existing file.")
            return registry

        if not assets:
            return []

        seedbed_dir = os.path.join(self.working_dir, "seedbed")
        os.makedirs(seedbed_dir, exist_ok=True)

        registry: List[Dict] = []
        for n, asset in enumerate(assets):
            slug = _seedbed_slug(asset.url)
            path = os.path.join(seedbed_dir, f"{n}_{slug}.png")
            if os.path.exists(path):
                print(f"🚀 Skipped downloading seedbed asset {n}, already exists.")
            else:
                print(f"⬇️ Downloading seedbed asset {n} ({asset.role}) from {asset.url}...")
                download_image(asset.url, path)

            registry.append({
                "path": path,
                "url": asset.url,
                "role": asset.role,
                "shot_idx": asset.shot_idx,
                "hint": asset.hint,
                "note": asset.note,
                # Filled below when a vision model is configured. Absence is
                # recorded as null, never faked.
                "description": None,
            })

        await self._describe_seedbed_assets(registry)

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=4)
        print(f"☑️ Ingested {len(registry)} seedbed assets, registry at {registry_path}.")

        return registry

    async def _describe_seedbed_assets(self, registry: List[Dict]) -> None:
        """Fill each entry's ``description`` via the configured vision model.

        Non-fatal but recorded: a vision failure leaves ``description: null`` and
        the pool text says "no description available" rather than inventing one.
        A confidently wrong description of a reference image is worse than an
        acknowledged missing one.
        """
        if self.vision_model is None:
            print("ℹ️ No vision_model configured — seedbed descriptions left null.")
            return

        for entry in registry:
            try:
                entry["description"] = await self.vision_model.describe_image(
                    image_path=entry["path"],
                    prompt=(
                        "Describe this reference image for an image-generation "
                        "prompt. State the setting, materials, palette, lighting "
                        "direction, and any distinctive props. Do not speculate "
                        "about story or intent."
                    ),
                )
            except Exception as e:  # noqa: BLE001
                logging.warning(
                    "Seedbed vision description failed for %s: %s", entry["path"], e,
                )
                entry["description"] = None

    def apply_seedbed_pins(
        self,
        seedbed_registry: Optional[List[Dict]],
        shot_descriptions: List[ShotDescription],
    ) -> None:
        """Copy pinned assets into place as shot first frames.

        The existing skip-if-exists checks then adopt them, so no image call
        fires for a pinned frame. The provenance file is what distinguishes a
        deliberate pin from a stale artifact left by an interrupted run.

        v1 pins FIRST frames only. Last-frame pinning is out of scope: it changes
        flf2v pairing semantics, which needs its own design.
        """
        if not seedbed_registry:
            return

        for n, entry in enumerate(seedbed_registry):
            if entry.get("role") != "pin":
                continue

            shot_idx = entry.get("shot_idx")
            if not 0 <= shot_idx < len(shot_descriptions):
                raise RuntimeError(
                    f"Seedbed asset {n} pins shot_idx={shot_idx}, but the "
                    f"storyboard has {len(shot_descriptions)} shot(s) "
                    f"(valid: 0..{len(shot_descriptions) - 1}). Fix the job's "
                    f"seed_assets — no guessing which shot was meant."
                )

            shot_dir = os.path.join(self.working_dir, "shots", f"{shot_idx}")
            os.makedirs(shot_dir, exist_ok=True)
            ff_path = os.path.join(shot_dir, "first_frame.png")
            provenance_path = os.path.join(shot_dir, "first_frame_source.json")

            if os.path.exists(ff_path) and os.path.exists(provenance_path):
                print(f"🚀 Shot {shot_idx} first_frame already pinned, skipping.")
                continue

            shutil.copy(entry["path"], ff_path)
            with open(provenance_path, "w", encoding="utf-8") as f:
                json.dump({
                    "source": "seedbed",
                    "url": entry["url"],
                    "registry_idx": n,
                }, f, ensure_ascii=False, indent=4)
            print(
                f"📌 Pinned seedbed asset {n} as first_frame of shot {shot_idx} "
                f"(no image generation will fire for this frame)."
            )

    async def extract_environments(
        self,
        script: str,
    ) -> List[EnvironmentInScene]:
        save_path = os.path.join(self.working_dir, "environments.json")

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                environments = [EnvironmentInScene.model_validate(e) for e in json.load(f)]
            print(f"🚀 Loaded {len(environments)} environments from existing file.")
        else:
            print(f"🔍 Extracting environments from script...")
            environments = await self.environment_extractor.extract_environments(script)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([e.model_dump() for e in environments], f, ensure_ascii=False, indent=4)
            print(f"✅ Extracted {len(environments)} environments and saved to {save_path}.")

        if not environments:
            raise RuntimeError(
                f"No environments extracted from the script. Every shot needs a "
                f"location plate to reference, so there is nothing to render "
                f"against. Check {save_path} and the script content."
            )

        return environments

    async def generate_environment_plates(
        self,
        environments: List[EnvironmentInScene],
        style: str,
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Render one master plate per environment. Skip-if-exists per plate.

        Registry is keyed by slugline, mirroring the portraits registry shape
        (keyed by identifier_in_scene) so both read the same way at the call site.
        """
        registry_path = os.path.join(self.working_dir, "environment_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            print(f"🚀 Loaded {len(registry)} environment plates from existing file.")
        else:
            registry = {}

        for environment in environments:
            if environment.slugline in registry:
                continue

            plate_dir = os.path.join(
                self.working_dir, "environment_plates",
                f"{environment.idx}_{environment.slug}",
            )
            os.makedirs(plate_dir, exist_ok=True)
            plate_path = os.path.join(plate_dir, "plate.png")

            if os.path.exists(plate_path):
                print(f"🚀 Skipped plate for {environment.slugline}, already exists.")
            else:
                print(f"🏞️ Generating location plate for {environment.slugline}...")
                plate_output = await self.environment_plate_generator.generate_plate(
                    environment=environment,
                    style=style,
                )
                plate_output.save(plate_path)
                print(f"☑️ Generated location plate, saved to {plate_path}.")

            registry[environment.slugline] = {
                "plate": {
                    "path": plate_path,
                    "description": environment_plate_description(environment),
                },
            }
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, ensure_ascii=False, indent=4)

        return registry

    async def design_storyboard(
        self,
        script: str,
        characters: List[CharacterInScene],
        user_requirement: str,
        environments: Optional[List[EnvironmentInScene]] = None,
    ):
        storyboard_path = os.path.join(self.working_dir, "storyboard.json")
        if os.path.exists(storyboard_path):
            with open(storyboard_path, 'r', encoding='utf-8') as f:
                storyboard = json.load(f)
            storyboard = [ShotBriefDescription.model_validate(shot) for shot in storyboard]
            print(f"🚀 Loaded {len(storyboard)} shot brief descriptions from existing file.")
        else:
            print(f"🔍 Designing storyboard...")
            storyboard = await self.storyboard_artist.design_storyboard(
                script=script,
                characters=characters,
                user_requirement=user_requirement,
                retry_timeout=150,
                environments=environments,
            )
            with open(storyboard_path, 'w', encoding='utf-8') as f:
                json.dump([shot.model_dump() for shot in storyboard], f, ensure_ascii=False, indent=4)
            print(f"✅ Designed storyboard and saved to {storyboard_path}.")

        for shot_brief_description in storyboard:
            self.shot_desc_events[shot_brief_description.idx] = asyncio.Event()

        return storyboard



    async def decompose_visual_descriptions(
        self,
        shot_brief_descriptions: List[ShotBriefDescription],
        characters: List[CharacterInScene],
    ):
        tasks = [
            self.decompose_visual_description_for_single_shot_brief_description(shot_brief_description, characters)
            for shot_brief_description in shot_brief_descriptions
        ]

        shot_descriptions = await asyncio.gather(*tasks)
        return shot_descriptions


    async def decompose_visual_description_for_single_shot_brief_description(
        self,
        shot_brief_description: ShotBriefDescription,
        characters: List[CharacterInScene],
    ):
        shot_description_path = os.path.join(self.working_dir, "shots", f"{shot_brief_description.idx}", "shot_description.json")
        os.makedirs(os.path.dirname(shot_description_path), exist_ok=True)

        if os.path.exists(shot_description_path):
            with open(shot_description_path, 'r', encoding='utf-8') as f:
                shot_description = ShotDescription.model_validate(json.load(f))
            print(f"🚀 Loaded shot {shot_brief_description.idx} description from existing file.")
        else:
            shot_description = await self.storyboard_artist.decompose_visual_description(
                shot_brief_desc=shot_brief_description,
                characters=characters,
                retry_timeout=120,
            )
            with open(shot_description_path, 'w', encoding='utf-8') as f:
                json.dump(shot_description.model_dump(), f, ensure_ascii=False, indent=4)
            print(f"✅ Decomposed visual description for shot {shot_brief_description.idx} and saved to {shot_description_path}.")

        self.shot_desc_events[shot_brief_description.idx].set()

        if shot_description.variation_type in ["medium", "large"]:
            self.frame_events[shot_brief_description.idx] = {
                "first_frame": asyncio.Event(),
                "last_frame": asyncio.Event(),
            }
        else:
            self.frame_events[shot_brief_description.idx] = {
                "first_frame": asyncio.Event(),
            }

        return shot_description