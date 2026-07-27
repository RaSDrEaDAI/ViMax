import re

from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict
from PIL import Image



class EnvironmentInScene(BaseModel):
    idx: int = Field(
        default=0,
        description="The index of the environment in the scene, starting from 0. Shots reference environments by this index.",
        examples=[0, 1, 2],
    )
    slugline: str = Field(
        description="The slugline of the scene, indicating the location and time of day",
        examples=[
            "INT. COFFEE SHOP - NIGHT",
            "EXT. PARK - DAY",
        ]
    )
    description: str = Field(
        description="A detailed description of the environment in the specific scene. Don't describe any characters or actions here, just the setting.",
        examples=[
            "The warm yellow light glowed against the mottled brick wall, while raindrops streaked the glass window with blurred neon reflections. Among the empty booths sat a lone half-finished iced latte—its foam collapsed, a faint lipstick mark on the rim. beads of condensation gleamed on the stainless steel espresso machine, and the record player's turntable rotated slowly in the shadows. A patch of wet floor shimmered with hazy reflected light.",
        ]
    )

    def __str__(self):
        s = f"{self.slugline} -- {self.description}"
        return s

    @property
    def slug(self) -> str:
        """Filesystem-safe form of the slugline, for the plate directory name."""
        return normalize_slugline(self.slugline).replace(" ", "_")[:60] or "location"


def normalize_slugline(slugline: str) -> str:
    """Normalize a slugline for dedupe and for path building.

    Uppercased, punctuation collapsed to spaces, whitespace squeezed. Sluglines
    arrive from an LLM, so "INT. Coffee Shop - Night" and "INT COFFEE SHOP -
    NIGHT" are the same location and must not become two master plates.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", slugline or "")
    return re.sub(r"\s+", " ", cleaned).strip().upper()


