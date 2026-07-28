import logging
import os
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from tenacity import retry, stop_after_attempt
from interfaces import CharacterInScene, ImageOutput
from langchain_core.messages import HumanMessage, SystemMessage
from utils.retry import after_func



prompt_template_front = \
"""
Generate a full-body, front-view portrait of character {identifier} based on the following description, with a pure white background. The character should be centered in the image, occupying most of the frame. Gazing straight ahead. Standing with arms relaxed at sides. Natural expression.
Features: {features}
Style: {style}
"""

prompt_template_side = \
"""
Generate a full-body, side-view portrait of character {identifier} based on the provided front-view portrait, with a pure white background. The character should be centered in the image, occupying most of the frame. Facing left. Standing with arms relaxed at sides.
"""

prompt_template_back = \
"""
Generate a full-body, back-view portrait of character {identifier} based on the provided front-view portrait, with a pure white background. The character should be centered in the image, occupying most of the frame. No facial features should be visible.
"""


# ── Composite-sheet chain prompts ────────────────────────────────────────────
# Grammar from references/ai_model_context_v8.csv L1164 (Nano Banana Pro —
# Character Sheet Creation). Both calls are single-reference i2i: the CSV is
# explicit that ONE reference image is the identity anchor, and its caps note
# that mixing sheet types in one generation compounds drift. So the anchor and
# the rotation grid are two separate calls, each with one reference.

prompt_template_identity_anchor = \
"""
Generate a single-panel character identity anchor for {identifier} (identity locked from Image 1).
Sheet type: detail close-up. Layout: ONE panel, no grid, no labels.
Panel: head-and-shoulders close-up, front 0 degrees, gazing straight into the lens, neutral expression, wardrobe visible from the chest up.
Face fills the frame: sharp focus on the eyes, brows, nose bridge, mouth, jawline, hairline, and skin texture. This panel is the ONLY place identity is read from, so facial detail is the priority.
Background: plain neutral studio white, no shadows below, no environment elements.
Lighting: flat diffused studio light, even from all sides, no directional shadows on the subject.
ONE reference image is the identity anchor: preserve the face, hair, and wardrobe of Image 1 exactly. Change only the framing (full body to close-up).
"""

prompt_template_rotation_grid = \
"""
Generate a character turnaround sheet for {identifier} (identity locked from Image 1).
Sheet type: turnaround. Layout: 2x2 grid, four panels, read in reading order, no text labels.
Per-panel spec:
  Panel 1 (top-left): full-figure standing view, front 0 degrees.
  Panel 2 (top-right): full-figure standing view, 3/4 turn 45 degrees.
  Panel 3 (bottom-left): full-figure standing view, profile 90 degrees.
  Panel 4 (bottom-right): full-figure standing view, back 180 degrees, no facial features visible.
Pose is identical in every panel: standing upright, arms relaxed at the sides, feet together. Only the camera angle changes.
Scale: the subject occupies an identical frame height in every panel, constant at roughly 80 percent of panel height, feet on the same baseline.
Background: plain neutral studio white in every panel, no shadows below, no environment elements.
Lighting: flat diffused studio light, even from all sides, no directional shadows on the subject.
ONE reference image is the identity anchor: the wardrobe, silhouette, proportions, and colours of Image 1 must carry to all four panels unchanged.
"""


def describe_features(character: CharacterInScene) -> str:
    """Join a character's static and dynamic features for a portrait prompt.

    Both fields are ``Optional[str]`` and the extractor prompt explicitly tells
    the model to leave them null when the script doesn't describe them (and when
    the character isn't visible at all). String-concatenating them therefore
    raised ``TypeError: can only concatenate str (not "NoneType") to str`` — and
    because the input never changes between attempts, tenacity retried it three
    times and failed identically each time. It killed four orchestrator runs at
    ~45s (.working_dir/orchestrator/{47e37ece,79be644b,ac428304,fc617145}).

    Omits absent fields rather than writing "None" into the prompt: the literal
    word None in a generation prompt is a real instruction to the image model.
    """
    parts = []
    if character.static_features:
        parts.append(f"(static) {character.static_features}")
    if character.dynamic_features:
        parts.append(f"(dynamic) {character.dynamic_features}")
    if not parts:
        # Nothing described at all. Say so plainly instead of sending an empty
        # Features: line, which reads as a truncated prompt.
        return "not described in the script; design plausible, distinctive features"
    return "; ".join(parts)


class CharacterPortraitsGenerator:
    def __init__(
        self,
        image_generator,
    ):
        self.image_generator = image_generator


    @retry(stop=stop_after_attempt(3), after=after_func, reraise=True)
    async def generate_front_portrait(
        self,
        character: CharacterInScene,
        style: str,
    ) -> ImageOutput:
        features = describe_features(character)
        prompt = prompt_template_front.format(
            identifier=character.identifier_in_scene,
            features=features,
            style=style,
        )
        image_output = await self.image_generator.generate_single_image(
            prompt=prompt,
            # Pass character as visible_characters so the router knows this is a
            # single-character portrait and routes to Qwen 2512 (best identity fidelity)
            # when no LoRA is configured. LoRA-configured Flux still wins via
            # the lora_name heuristic in the router.
            visible_characters=[character],
            style=style,
        )
        return image_output

    @retry(stop=stop_after_attempt(3), after=after_func, reraise=True)
    async def generate_side_portrait(
        self,
        character: CharacterInScene,
        front_image_path: str,
    ) -> ImageOutput:
        prompt = prompt_template_side.format(
            identifier=character.identifier_in_scene,
        )
        image_output = await self.image_generator.generate_single_image(
            prompt=prompt,
            reference_image_paths=[front_image_path],
            # size="1024x1024",
        )
        return image_output


    @retry(stop=stop_after_attempt(3), after=after_func, reraise=True)
    async def generate_back_portrait(
        self,
        character: CharacterInScene,
        front_image_path: str,
    ) -> ImageOutput:
        prompt = prompt_template_back.format(
            identifier=character.identifier_in_scene,
        )
        image_output = await self.image_generator.generate_single_image(
            prompt=prompt,
            reference_image_paths=[front_image_path],
            # size="512x512",
        )
        return image_output

    # ── Composite-sheet chain ────────────────────────────────────────────────
    # front portrait (identity master, t2i)
    #   -> identity anchor  (i2i on front,  3:4 — the close-up that locks identity)
    #   -> rotation grid    (i2i on anchor, 1:1 — so the anchor's wardrobe carries)
    #   -> deface + compose (utils.composite_sheet, free)
    #
    # The grid is conditioned on the ANCHOR rather than the front portrait
    # deliberately: the anchor is what every downstream keyframe reads identity
    # from, so the outfit on the grid must match the anchor's, not a third
    # independent interpretation of the character description.

    @retry(stop=stop_after_attempt(3), after=after_func, reraise=True)
    async def generate_identity_anchor(
        self,
        character: CharacterInScene,
        front_image_path: str,
    ) -> ImageOutput:
        """Close-up identity anchor: the left panel of the composite sheet."""
        prompt = prompt_template_identity_anchor.format(
            identifier=character.identifier_in_scene,
        )
        return await self.image_generator.generate_single_image(
            prompt=prompt,
            reference_image_paths=[front_image_path],
            aspect_ratio="3:4",
            visible_characters=[character],
        )

    @retry(stop=stop_after_attempt(3), after=after_func, reraise=True)
    async def generate_rotation_grid(
        self,
        character: CharacterInScene,
        anchor_image_path: str,
    ) -> ImageOutput:
        """2x2 body rotation grid: the right panel, before defacing.

        Four views in one call (front / 3-4 / side / back), inside the CSV's
        turnaround cap of <= 5 views. This replaces the separate side and back
        portrait calls on the sheet path — one generation keeps the four views
        mutually consistent, where three independent calls did not.
        """
        prompt = prompt_template_rotation_grid.format(
            identifier=character.identifier_in_scene,
        )
        return await self.image_generator.generate_single_image(
            prompt=prompt,
            reference_image_paths=[anchor_image_path],
            aspect_ratio="1:1",
            visible_characters=[character],
        )