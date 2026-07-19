"""Model prompting guide loader.

Reads references/ai_model_context_v8.csv (214KB, 42 models) and exposes
per-model prompt structure, presets, and reference rows (FACS, Camera
Grammar, Lighting Vocabulary) for injection into ViMax agent prompts.

The CSV is the single source of truth for how to prompt each model. Agents
that build prompts for image/video generation should call load_model_guide()
to get the relevant row and append its `content` (the how-to-prompt guide)
and `prompt_presets` (fill-in-the-blank templates) to their system prompt.
"""

import csv
import os
from functools import lru_cache
from typing import Dict, List, Optional


_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "references",
    "ai_model_context_v8.csv",
)


@lru_cache(maxsize=1)
def load_all_rows() -> List[Dict[str, str]]:
    """Load and cache all rows from the prompting guide CSV."""
    if not os.path.exists(_CSV_PATH):
        return []
    with open(_CSV_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_model_guide(model_name: str) -> Optional[Dict[str, str]]:
    """Find the prompting guide row for a model by fuzzy name match.

    Args:
        model_name: Model identifier (e.g. "LTX 2.3", "ltx23", "WAN 2.2").

    Returns:
        The matching CSV row as a dict, or None if not found.
    """
    rows = load_all_rows()
    if not rows:
        return None
    needle = model_name.lower().strip()
    # Exact title match first
    for r in rows:
        if r.get("title", "").lower() == needle:
            return r
    # Substring match (handles "ltx23" matching "LTX 2.3")
    for r in rows:
        title = r.get("title", "").lower()
        # Normalize: strip spaces, dashes, dots for comparison
        norm_title = title.replace(" ", "").replace("-", "").replace(".", "")
        norm_needle = needle.replace(" ", "").replace("-", "").replace(".", "")
        if norm_needle in norm_title:
            return r
    return None


def load_reference_guide(ref_name: str) -> Optional[Dict[str, str]]:
    """Find a reference row (FACS, Camera Grammar, Lighting Vocabulary).

    Args:
        ref_name: e.g. "FACS", "Camera Grammar", "Lighting Vocabulary".
    """
    rows = load_all_rows()
    if not rows:
        return None
    needle = ref_name.lower()
    for r in rows:
        if r.get("category") == "reference":
            title = r.get("title", "").lower()
            tags = r.get("tags", "").lower()
            if needle in title or needle in tags:
                return r
    return None


def build_video_prompt_context(
    model_name: str = "LTX 2.3",
    include_references: bool = True,
) -> str:
    """Build a system-prompt context block for a video model.

    Combines:
    - The model's `content` (how-to-prompt guide)
    - The model's `prompt_presets` (fill-in-the-blank templates)
    - Optional reference rows (FACS, Camera Grammar, Lighting Vocabulary)

    Returns a string suitable for appending to an agent's system prompt.
    Returns empty string if the model guide is not found (fail-soft —
    agents still work, just without the prompting discipline).
    """
    guide = load_model_guide(model_name)
    if not guide:
        return ""

    parts = []
    content = (guide.get("content") or "").strip()
    if content:
        parts.append(f"# {guide['title']} — Prompting Guide\n\n{content}")

    presets = (guide.get("prompt_presets") or "").strip()
    if presets:
        parts.append(f"\n\n# {guide['title']} — Prompt Templates\n\n{presets}")

    if include_references:
        companions = (guide.get("companions") or "").strip()
        if companions:
            for comp_name in [c.strip() for c in companions.split(",") if c.strip()]:
                ref = load_reference_guide(comp_name)
                if ref:
                    ref_content = (ref.get("content") or "").strip()
                    if ref_content:
                        parts.append(
                            f"\n\n# {ref['title']}\n\n{ref_content}"
                        )

    return "\n".join(parts)


def build_image_prompt_context(
    model_name: str = "Flux 2 Pro",
    include_references: bool = True,
) -> str:
    """Build a system-prompt context block for an image model.

    Same structure as build_video_prompt_context but for image-gen models.
    """
    return build_video_prompt_context(model_name=model_name, include_references=include_references)
