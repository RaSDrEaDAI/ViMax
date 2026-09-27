"""CompositionPlanner agent.

Converts a ViMax frame description + visible-character list into an Ideogram V4
``json_prompt`` with explicit per-character bbox layout.

Why this exists
---------------
ViMax's StoryboardArtist emits a frame description (``ff_desc`` / ``lf_desc``)
and a list of visible character indices. That gets flattened into a text string
and sent to a single image model. For multi-character frames this produces
identity merge and composition drift.

Ideogram V4 with ``json_prompt`` solves this: each character gets its own
``element`` with a ``desc`` and a ``bbox`` (normalized 0-1000, row-first).
The model respects the spatial contract and renders each element independently.

This agent is the bridge: frame metadata in, Ideogram JSON layout out.

Cost
----
One LLM call per multi-character frame (~$0.001 on GLM-4.6). Only invoked
when the ImageGeneratorRouter selects the Ideogram backend (2+ characters).

Schema
------
Output matches the Ideogram V4 ``json_prompt`` contract:

    {
      "high_level_description": "1-2 sentence overall image description",
      "style_description": {"aesthetics": "..."},
      "compositional_deconstruction": {
        "background": "...",
        "elements": [
          {"type": "obj", "desc": "...", "bbox": [y_min, x_min, y_max, x_max]},
          {"type": "text", "text": "...", "desc": "...", "bbox": [...]}  // optional
        ]
      }
    }

bbox is [y_min, x_min, y_max, x_max], integers 0-1000.
"""

import asyncio
import logging
from typing import Any, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt

from interfaces.character import CharacterInScene


# ---------------------------------------------------------------------------
# Output schema (matches Ideogram V4 json_prompt contract)
# ---------------------------------------------------------------------------

class IdeogramElement(BaseModel):
    type: str = Field(
        description="Element type: 'obj' for characters/objects/props, 'text' for rendered text.",
    )
    desc: str = Field(
        description=(
            "Detailed description of this element. For characters include: "
            "identity, clothing, pose, expression, orientation. "
            "For objects include: material, color, position."
        ),
    )
    bbox: List[int] = Field(
        description=(
            "Bounding box as [y_min, x_min, y_max, x_max], integers 0-1000. "
            "Row-first ordering (y before x). 1000x1000 grid stretched to aspect ratio."
        ),
        examples=[[100, 50, 900, 450], [100, 550, 900, 950]],
    )
    # Optional text-element fields (ignored for 'obj' type)
    text: Optional[str] = Field(
        default=None,
        description="Literal text to render. Only used when type='text'.",
    )

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, v):
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 integers: [y_min, x_min, y_max, x_max]")
        for n in v:
            if not isinstance(n, int) or n < 0 or n > 1000:
                raise ValueError(f"bbox values must be integers 0-1000, got {n}")
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError(
                f"bbox invalid: y_min ({v[0]}) must be < y_max ({v[2]}), "
                f"x_min ({v[1]}) must be < x_max ({v[3]})"
            )
        return v


class CompositionalDeconstruction(BaseModel):
    background: str = Field(
        description="Description of the background, environment, and setting.",
    )
    elements: List[IdeogramElement] = Field(
        description="List of elements (characters, objects, text) with their bbox positions.",
    )


class StyleDescription(BaseModel):
    aesthetics: str = Field(
        description=(
            "Visual style description: medium (photoreal / illustration / anime), "
            "tone, color palette, lighting, depth of field."
        ),
        examples=[
            "cinematic photoreal, warm golden hour light, shallow DOF, film grain",
            "flat editorial illustration, bold outlines, limited palette, high contrast",
        ],
    )


class IdeogramJsonPrompt(BaseModel):
    """Complete Ideogram V4 json_prompt structure."""

    high_level_description: str = Field(
        description="1-2 sentence overall image description covering the action and setting.",
    )
    style_description: StyleDescription
    compositional_deconstruction: CompositionalDeconstruction


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a composition planner for Ideogram V4, an image model that accepts \
structured JSON layouts with per-element bounding boxes.

Your job: take a frame description and a cast list, then emit a JSON layout \
that places each character and key object in its own bbox region.

BBOX RULES (critical):
- bbox format: [y_min, x_min, y_max, x_max] — ROW-FIRST, not xywh
- All values are integers in [0, 1000]
- The grid is always 1000x1000 regardless of output aspect ratio
- The model stretches the grid to the chosen aspect, so plan for the canvas shape
- y=0 is TOP, y=1000 is BOTTOM (screen coordinates, not cartesian)
- x=0 is LEFT, x=1000 is RIGHT
- bbox is ADVISORY, not a hard mask. Use ranges 100-900 for predictable results
- Characters should NOT overlap. Leave at least 50 units of padding between bboxes
- A character's bbox should leave room for their whole body / pose, not just their face

LAYOUT HEURISTICS:
- 1 character: centered [100, 250, 900, 750] or per shot framing
- 2 characters: left/right halves with center gap
  e.g. A=[100, 50, 900, 450], B=[100, 550, 900, 950]
- 3 characters: left/center/right thirds
  e.g. A=[150, 50, 850, 350], B=[100, 350, 900, 650], C=[150, 650, 850, 950]
- 4+ characters: 2x2 grid or staggered; keep each bbox >= 300 units wide
- Character on higher ground / background: smaller bbox, higher y_min
- Character closer to camera: larger bbox, lower y_min

ELEMENT DESC RULES:
- For characters: include identity, body type, clothing, pose, expression, gaze direction
- Pull static features (face, build) from the cast list verbatim
- Pull dynamic features (clothing, accessories) from the cast list verbatim
- Add pose / action details from the frame description
- Do NOT merge characters. Each gets their own element with their own desc.

BACKGROUND:
- Pull environment from the frame description
- Include time of day, lighting source, architectural / natural features
- Background fills the full canvas; characters compose ON TOP

STYLE:
- Derive from the project's style string (passed in)
- Default: "cinematic photoreal, warm natural light, shallow DOF"

TEXT ELEMENTS (optional):
- Only add 'text' elements if the frame explicitly requires rendered text
- Include the literal text and style notes (font, weight, color)
"""


_HUMAN_PROMPT = """\
Frame description:
<FRAME>
{frame_desc}
</FRAME>

Visible characters (in order of appearance in the scene):
<CHARACTERS>
{characters_str}
</CHARACTERS>

Project visual style:
<STYLE>
{style}
</STYLE>

Shot framing notes:
<SHOT>
{shot_notes}
</SHOT>

Output the Ideogram V4 json_prompt layout. Place each visible character in \
their own bbox element with full identity + pose description. Derive the \
background from the frame description. Match the project visual style.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CompositionPlanner:
    """LLM agent that converts frame metadata → Ideogram V4 json_prompt.

    Optionally uses a vision model to look at reference images (boards from
    the orchestrator) and factor their composition into the bbox layout.
    """

    def __init__(
        self,
        chat_model: BaseChatModel,
        vision_model: Any = None,
    ):
        self.chat_model = chat_model
        self.vision_model = vision_model

    @retry(stop=stop_after_attempt(3))
    async def plan_composition(
        self,
        frame_desc: str,
        visible_characters: List[CharacterInScene],
        style: str,
        shot_notes: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        retry_timeout: int = 60,
    ) -> IdeogramJsonPrompt:
        """Plan the Ideogram V4 JSON layout for a frame.

        Args:
            frame_desc: The ff_desc or lf_desc from StoryboardArtist.
            visible_characters: Characters visible in this frame (from ff_vis_char_idxs).
            style: Project visual style string (e.g. "cinematic photoreal, warm tones").
            shot_notes: Optional framing hints (shot type, camera angle, composition notes).
            reference_image_urls: Optional board/reference image URLs. When provided
                AND a vision_model is configured, the planner will look at each image
                and incorporate its composition cues into the layout. Ignored if no
                vision_model is configured (boards become style-string mood anchors).

        Returns:
            IdeogramJsonPrompt ready to serialize and pass to the IdeogramV4 node.
        """
        if not visible_characters:
            raise ValueError(
                "CompositionPlanner is for multi-character frames; "
                "no visible characters were provided."
            )

        # If we have a vision model and reference images, describe them first
        # and fold the descriptions into the frame_desc.
        enriched_frame_desc = frame_desc
        if self.vision_model and reference_image_urls:
            try:
                descriptions = await self.vision_model.describe_images(
                    reference_image_urls,
                    prompt=(
                        "Describe this reference image for a film storyboard composition. "
                        "Note: subject positions, camera angle, lighting direction, color "
                        "palette, and any composition cues (rule of thirds, leading lines, "
                        "depth). Be concise (2-3 sentences)."
                    ),
                )
                desc_block = "\n".join(
                    f"[Reference {i+1}]: {d}" for i, d in enumerate(descriptions)
                )
                enriched_frame_desc = (
                    f"{frame_desc}\n\n"
                    f"Reference image composition cues (mirror these where appropriate):\n"
                    f"{desc_block}"
                )
                logging.info(
                    f"CompositionPlanner: enriched frame_desc with "
                    f"{len(descriptions)} vision descriptions."
                )
            except Exception as e:
                logging.warning(
                    f"CompositionPlanner: vision_model call failed ({e}); "
                    f"falling back to text-only planning."
                )

        parser = PydanticOutputParser(pydantic_object=IdeogramJsonPrompt)

        characters_str = "\n".join(
            f"Character {i}: {char.identifier_in_scene}\n"
            f"  static features: {char.static_features}\n"
            f"  dynamic features: {char.dynamic_features}"
            for i, char in enumerate(visible_characters)
        )

        shot_notes_str = (shot_notes or "").strip() or "No additional framing notes."

        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])

        structured_model = self.chat_model.with_structured_output(IdeogramJsonPrompt)
        chain = prompt | structured_model

        logging.info(
            f"CompositionPlanner: planning layout for "
            f"{len(visible_characters)} characters, frame_desc='{frame_desc[:80]}...'"
        )

        result: IdeogramJsonPrompt = await asyncio.wait_for(
            chain.ainvoke({
                "frame_desc": enriched_frame_desc.strip(),
                "characters_str": characters_str,
                "style": style.strip(),
                "shot_notes": shot_notes_str,
                "format_instructions": parser.get_format_instructions(),
            }),
            timeout=retry_timeout,
        )

        # Sanity: ensure each visible character is represented as an element.
        obj_elements = [e for e in result.compositional_deconstruction.elements if e.type == "obj"]
        if len(obj_elements) < len(visible_characters):
            logging.warning(
                f"CompositionPlanner: produced {len(obj_elements)} obj elements "
                f"for {len(visible_characters)} visible characters; "
                f"some characters may not have their own bbox."
            )

        return result

    def to_json_prompt_str(self, layout: IdeogramJsonPrompt) -> str:
        """Serialize to the JSON string the IdeogramV4 node expects."""
        import json
        return json.dumps(layout.model_dump(exclude_none=True), separators=(",", ":"))
