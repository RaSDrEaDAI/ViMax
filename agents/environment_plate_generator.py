"""Render one master location plate per environment.

The plate is the location authority: every keyframe set in that location carries
it as a reference so architecture, palette, and lighting direction stay fixed
across shots. It is deliberately EMPTY of characters — a plate with a person in
it would inject a stray identity into every frame that references it, competing
with the character sheets.

Prompt grammar from ``references/ai_model_context_v8.csv`` L419 (Nano Banana Pro
— World Building): wide establishing view, concrete material/lighting vocabulary,
explicit scale statement, style appended.
"""

from tenacity import retry, stop_after_attempt

from interfaces import EnvironmentInScene, ImageOutput
from utils.retry import after_func


prompt_template_environment_plate = \
"""
Generate a location plate for {slugline}.
Framing: wide establishing view of the full environment, eye-level, deep focus, the architecture legible from wall to wall.
Scale: the space reads at natural human scale — doorways, seating, and counters sized as if a person could walk in, though nobody is present.
Location: {description}
Preserve throughout: architecture, materials, palette, lighting direction, time of day, haze.
Critical: the frame is EMPTY. No people, no characters, no figures, no faces, no hands, no silhouettes, no reflections of people, and no animals. Nothing animate anywhere in the image.
Style: {style}
"""


class EnvironmentPlateGenerator:
    def __init__(
        self,
        image_generator,
    ):
        # Renders through the pipeline's ``sheet_image_generator`` — plates are
        # reference authorities rather than delivered keyframes, so they belong
        # on the free local backend by default.
        self.image_generator = image_generator

    @retry(stop=stop_after_attempt(3), after=after_func, reraise=True)
    async def generate_plate(
        self,
        environment: EnvironmentInScene,
        style: str,
    ) -> ImageOutput:
        prompt = prompt_template_environment_plate.format(
            slugline=environment.slugline,
            description=environment.description,
            style=style,
        )
        return await self.image_generator.generate_single_image(
            prompt=prompt,
            reference_image_paths=[],
            # 16:9 to match the keyframe and video contract: a plate at a
            # different aspect would have the model reframe rather than match
            # the composition it is meant to anchor.
            aspect_ratio="16:9",
            # No visible_characters — this is what routes the local router to its
            # environment backend rather than a character/portrait model.
            visible_characters=[],
            style=style,
        )


def environment_plate_description(environment: EnvironmentInScene) -> str:
    """The registry description for a location plate.

    Says what the referencing frame should take FROM the plate, not just what
    the plate is — otherwise the model treats it as loose inspiration and
    re-invents the room.
    """
    return (
        f"Location plate: {environment.slugline}. {environment.description} "
        f"Match its architecture, materials, palette, and lighting direction."
    )
