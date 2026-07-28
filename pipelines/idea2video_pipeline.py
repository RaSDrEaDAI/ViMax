import os
import logging
from agents import Screenwriter, CharacterExtractor, CharacterPortraitsGenerator
from pipelines.script2video_pipeline import Script2VideoPipeline
from interfaces import CharacterInScene, SeedAsset
from typing import List, Dict, Optional, Union
import asyncio
import json
from moviepy import VideoFileClip, concatenate_videoclips
import yaml
from langchain.chat_models import init_chat_model
from tools.render_backend import RenderBackend, _load_dotenv, _substitute_env_vars
from utils.provider_presets import resolve_chat_model_config
from utils.composite_sheet import build_character_sheet, character_sheet_description
from utils.video import concatenate_shot_videos


class Idea2VideoPipeline:
    def __init__(
        self,
        chat_model: str,
        image_generator: str,
        video_generator: str,
        working_dir: str,
        sheet_image_generator=None,
    ):
        self.chat_model = chat_model
        self.image_generator = image_generator
        self.video_generator = video_generator
        # Sheet-class assets (portraits, rotation grids, plates) can render on a
        # separate backend from keyframes. Defaults to the keyframe backend.
        self.sheet_image_generator = sheet_image_generator or image_generator
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)

        self.screenwriter = Screenwriter(chat_model=self.chat_model)
        self.character_extractor = CharacterExtractor(
            chat_model=self.chat_model)
        self.character_portraits_generator = CharacterPortraitsGenerator(
            image_generator=self.sheet_image_generator)

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
        """
        # Source ViMax's .env into os.environ BEFORE anything else — chat_model
        # and other secrets are referenced as ${VAR} in the YAML and must be
        # available for substitution. _load_dotenv does NOT override existing
        # env vars (env wins over file).
        _load_dotenv()

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Apply ${VAR} substitution to chat_model init_args BEFORE init_chat_model.
        # Without this, init_chat_model receives the literal "${ZAI_API_KEY}"
        # string as api_key and z.ai returns "401 token expired or incorrect".
        chat_model_args = _substitute_env_vars(config["chat_model"]["init_args"])
        chat_model_args = resolve_chat_model_config(chat_model_args)
        chat_model = init_chat_model(**chat_model_args)
        backend = RenderBackend.from_config(
            config,
            chat_model=chat_model,
            working_dir_override=working_dir_override,
        )

        return cls(
            chat_model=chat_model,
            image_generator=backend.image_generator,
            video_generator=backend.video_generator,
            working_dir=working_dir_override or config["working_dir"],
            sheet_image_generator=backend.sheet_image_generator,
        )

    async def extract_characters(
        self,
        story: str,
    ):
        save_path = os.path.join(self.working_dir, "characters.json")

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                characters = json.load(f)
            characters = [CharacterInScene.model_validate(
                character) for character in characters]
            print(f"🚀 Loaded {len(characters)} characters from existing file.")
        else:
            characters = await self.character_extractor.extract_characters(story)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([character.model_dump()
                          for character in characters], f, ensure_ascii=False, indent=4)
            print(
                f"✅ Extracted {len(characters)} characters from story and saved to {save_path}.")

        return characters

    async def generate_character_portraits(
        self,
        characters: List[CharacterInScene],
        character_portraits_registry: Optional[Dict[str, Dict[str, Dict[str, str]]]],
        style: str,
    ):
        character_portraits_registry_path = os.path.join(
            self.working_dir, "character_portraits_registry.json")
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
                    json.dump(character_portraits_registry,
                              f, ensure_ascii=False, indent=4)

            print(
                f"✅ Completed character portrait generation for {len(characters)} characters.")
        else:
            print(
                "🚀 All characters already have portraits, skipping portrait generation.")

        return character_portraits_registry

    async def develop_story(
        self,
        idea: str,
        user_requirement: str,
    ):
        save_path = os.path.join(self.working_dir, "story.txt")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                story = f.read()
            print(f"🚀 Loaded story from existing file.")
        else:
            print("🧠 Developing story...")
            story = await self.screenwriter.develop_story(idea=idea, user_requirement=user_requirement)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(story)
            print(f"✅ Developed story and saved to {save_path}.")

        return story

    async def write_script_based_on_story(
        self,
        story: str,
        user_requirement: str,
    ):
        save_path = os.path.join(self.working_dir, "script.json")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                script = json.load(f)
            print(f"🚀 Loaded script from existing file.")
        else:
            print("🧠 Writing script based on story...")
            script = await self.screenwriter.write_script_based_on_story(story=story, user_requirement=user_requirement)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(script, f, ensure_ascii=False, indent=4)
            print(f"✅ Written script based on story and saved to {save_path}.")
        return script

    async def generate_portraits_for_single_character(
        self,
        character: CharacterInScene,
        style: str,
    ):
        character_dir = os.path.join(
            self.working_dir, "character_portraits", f"{character.idx}_{character.identifier_in_scene}")
        os.makedirs(character_dir, exist_ok=True)

        # Same composite-sheet chain as Script2VideoPipeline. It has to match:
        # this registry is handed straight to Script2VideoPipeline, whose keyframe
        # path requires a `sheet` entry and fails loud without one. A 3-view
        # registry here would break every idea2video run at the first frame.
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

        print(
            f"☑️ Completed character portrait generation for {character.identifier_in_scene}.")

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

    async def __call__(
        self,
        idea: str,
        user_requirement: str,
        style: str,
        seed_assets: Optional[List[Union[SeedAsset, str, Dict]]] = None,
        reference_image_urls: Optional[List[str]] = None,
    ):
        # Legacy alias for the seedbed seam. Bare URLs coerce to
        # role='reference', so existing orchestrator scripts keep working.
        if reference_image_urls and not seed_assets:
            seed_assets = reference_image_urls

        story = await self.develop_story(idea=idea, user_requirement=user_requirement)

        characters = await self.extract_characters(story=story)

        character_portraits_registry = await self.generate_character_portraits(
            characters=characters,
            character_portraits_registry=None,
            style=style,
        )

        scene_scripts = await self.write_script_based_on_story(story=story, user_requirement=user_requirement)

        all_video_paths = []

        for idx, scene_script in enumerate(scene_scripts):
            scene_working_dir = os.path.join(self.working_dir, f"scene_{idx}")
            os.makedirs(scene_working_dir, exist_ok=True)
            script2video_pipeline = Script2VideoPipeline(
                chat_model=self.chat_model,
                image_generator=self.image_generator,
                video_generator=self.video_generator,
                working_dir=scene_working_dir,
                sheet_image_generator=self.sheet_image_generator,
            )
            final_video_path = await script2video_pipeline(
                script=scene_script,
                user_requirement=user_requirement,
                style=style,
                characters=characters,
                character_portraits_registry=character_portraits_registry,
                seed_assets=seed_assets,
            )
            all_video_paths.append(final_video_path)

        final_video_path = os.path.join(self.working_dir, "final_video.mp4")
        if os.path.exists(final_video_path):
            print(f"🚀 Skipped concatenating videos, already exists.")
        else:
            print(f"🎬 Starting concatenating videos...")
            # This used to build its clip list with a comprehension variable also
            # named final_video_path, which reads like the output path is being
            # clobbered before it is written. Comprehension scope meant it never
            # was, but nobody should have to prove that to read the line.
            concatenate_shot_videos(all_video_paths, final_video_path)
            print(f"☑️ Concatenated videos, saved to {final_video_path}.")
        return final_video_path
