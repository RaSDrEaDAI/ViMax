"""Extract the distinct locations a script plays out in.

Environments become first-class objects with one master plate each, so every
shot set in the same location conditions on the same image. Before this, the
location existed only as prose inside each shot's frame description — which is
why two shots in "the same" coffee shop could render two different coffee shops.

Same ``with_structured_output`` pattern as ``CharacterExtractor``: the model is
constrained to the schema at the tool-call layer, so there is no free-text JSON
parse to fail.
"""

import json
import logging
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt

from interfaces import EnvironmentInScene, normalize_slugline
from utils.retry import after_func


system_prompt_template_extract_environments = \
"""
[Role]
You are a top-tier movie script analysis expert and location scout.

[Task]
Your task is to analyze the provided script and extract every DISTINCT physical location the action plays out in.

[Input]
You will receive a script enclosed within <SCRIPT> and </SCRIPT>.

[Output]
Return the structured list of environments. Populate every field for each environment.
Index them from 0 in the order they first appear in the script.

[Guidelines]
- Ensure that the language of all output values (not including keys) matches that used in the script.
- Write each slugline in standard screenplay form: INT. or EXT., the location name, a dash, and the time of day. For example "INT. COFFEE SHOP - NIGHT" or "EXT. PARK - DAY".
- ONE entry per distinct location. If the action returns to a location it has already visited, do NOT emit a second entry for it. A location at a different time of day (day vs night) IS a distinct entry, because the lighting differs.
- Most short scripts have exactly ONE location. Do not invent additional locations to pad the list. Only split a location when the script genuinely moves somewhere else.
- The description must describe ONLY the setting: architecture, materials, surfaces, furniture, props in place, palette, light sources, direction and quality of light, time of day, weather, atmosphere, haze.
- The description must NOT contain any character, any person, any body part, or any action. This description is used to render an empty location plate; anything animate in it becomes an error in the plate.
- The description should be detailed and concretely visualizable. Use specific materials and colours (e.g. "mottled red brick", "brushed stainless steel", "warm tungsten from a low pendant lamp") rather than abstract terms like "cozy" or "atmospheric".
- Describe the light explicitly: where it comes from, its colour, its hardness, and which direction the shadows fall. Every shot in this location will be lit to match the plate.
- If the script is vague about the setting, design plausible concrete details from context so the location is vivid and complete.
"""

human_prompt_template_extract_environments = \
"""
<SCRIPT>
{script}
</SCRIPT>
"""


class ExtractEnvironmentsResponse(BaseModel):
    environments: List[EnvironmentInScene] = Field(
        ..., description="A list of distinct environments extracted from the script."
    )

    @field_validator("environments", mode="before")
    @classmethod
    def _unwrap_stringified_payload(cls, v):
        # Same defect as ExtractCharactersResponse: the model sometimes
        # tool-calls with this field set to the WHOLE response JSON as a string
        # instead of the list itself — deterministically, so every retry failed
        # identically. Unwrap instead of failing.
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, dict) and "environments" in v:
            v = v["environments"]
        return v


class EnvironmentExtractor:
    def __init__(
        self,
        chat_model,
    ):
        self.chat_model = chat_model

    @retry(
        stop=stop_after_attempt(3),
        after=after_func,
    )
    async def extract_environments(self, script: str) -> List[EnvironmentInScene]:
        structured_model = self.chat_model.with_structured_output(ExtractEnvironmentsResponse)

        messages = [
            SystemMessage(content=system_prompt_template_extract_environments),
            HumanMessage(content=human_prompt_template_extract_environments.format(script=script)),
        ]

        response: ExtractEnvironmentsResponse = await structured_model.ainvoke(messages)

        return _dedupe_and_reindex(response.environments)


def _dedupe_and_reindex(
    environments: List[EnvironmentInScene],
) -> List[EnvironmentInScene]:
    """Collapse duplicate sluglines and renumber idx contiguously from 0.

    Dedupe by NORMALIZED slugline: the model is told one entry per location but
    "INT. Coffee Shop - Night" and "INT COFFEE SHOP - NIGHT" still slip through
    as two, and two master plates for one location defeats the whole point.

    Reindexing is not cosmetic — the storyboard assigns shots to environments by
    index, so the indices handed to it must be contiguous and must match the
    saved list exactly.
    """
    seen = {}
    for env in environments:
        key = normalize_slugline(env.slugline)
        if key in seen:
            logging.info(
                "EnvironmentExtractor: collapsed duplicate slugline %r into %r",
                env.slugline, seen[key].slugline,
            )
            continue
        seen[key] = env

    deduped = []
    for idx, env in enumerate(seen.values()):
        deduped.append(env.model_copy(update={"idx": idx}))
    return deduped
