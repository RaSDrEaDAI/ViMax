"""Seedbed asset intake schema.

A seedbed asset is an image supplied by the orchestrator alongside the job, in
one of two roles:

  - ``pin``: this image IS the keyframe for an assigned shot. Nothing is
    generated for that frame — the asset is copied in and adopted.
  - ``reference``: this image is conditioning input. It joins the keyframe
    reference pool and the selector decides which frames use it.

The distinction is the whole point of the schema. Before it, seedbed URLs arrived
through a parameter (``reference_image_urls``) that was plumbed through both
pipeline signatures and never read — so an orchestrator could send images and
observe no effect whatsoever.
"""

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class SeedAsset(BaseModel):
    url: str = Field(
        description="The URL of the seedbed image.",
    )
    role: Literal["pin", "reference"] = Field(
        default="reference",
        description=(
            "'pin' means this image IS the first frame of shot_idx — nothing is "
            "generated for that frame. 'reference' means it joins the keyframe "
            "reference pool as conditioning input."
        ),
    )
    shot_idx: Optional[int] = Field(
        default=None,
        description="Required when role == 'pin'. Which shot this image is the first frame of.",
    )
    hint: Optional[Literal["character", "location", "style"]] = Field(
        default=None,
        description="What the image is a reference FOR. Prefixes its pool text.",
    )
    note: Optional[str] = Field(
        default=None,
        description="Operator note. Used as the pool text when no vision description is available.",
    )

    @model_validator(mode="after")
    def _pin_requires_shot_idx(self):
        # Raised at ingest, deliberately. There is no defensible way to guess
        # which shot an image belongs to, and guessing wrong means pinning a
        # deliberate keyframe onto the wrong shot — a failure that looks like a
        # creative choice rather than a bug.
        if self.role == "pin" and self.shot_idx is None:
            raise ValueError(
                "SeedAsset with role='pin' requires shot_idx. There is no way "
                "to infer which shot an image is the keyframe for. Either set "
                "shot_idx or use role='reference'."
            )
        if self.role == "pin" and self.shot_idx < 0:
            raise ValueError(
                f"SeedAsset shot_idx must be >= 0, got {self.shot_idx}."
            )
        return self


def coerce_seed_assets(
    seed_assets: Optional[list],
) -> list:
    """Normalize a mixed list of URLs / dicts / SeedAssets into SeedAssets.

    Back-compat: a bare URL string coerces to ``role='reference'``, so
    orchestrator scripts that pass a plain list of URLs (the shape
    ``reference_image_urls`` took) keep working unmodified. 'reference' is the
    right default because it is the non-destructive role — a mis-defaulted pin
    would overwrite a shot's keyframe.
    """
    if not seed_assets:
        return []

    coerced = []
    for item in seed_assets:
        if isinstance(item, SeedAsset):
            coerced.append(item)
        elif isinstance(item, str):
            coerced.append(SeedAsset(url=item, role="reference"))
        elif isinstance(item, dict):
            coerced.append(SeedAsset.model_validate(item))
        else:
            raise TypeError(
                f"Cannot coerce {type(item).__name__} to SeedAsset: {item!r}. "
                f"Pass a URL string, a dict, or a SeedAsset."
            )
    return coerced


# ── Reference pool ───────────────────────────────────────────────────────────

_HINT_PREFIX = {
    "character": "Character reference",
    "location": "Location reference",
    "style": "Style reference",
}


def seed_asset_pool_text(entry: dict) -> str:
    """Build the pool text for one ingested seedbed reference.

    Prefers the vision description, falls back to the operator note, and says so
    plainly when there is neither — an unlabelled reference the model has to
    guess at is worth flagging in the prompt rather than passing off as
    described.
    """
    body = entry.get("description") or entry.get("note")
    prefix = _HINT_PREFIX.get(entry.get("hint") or "", "Reference")
    if not body:
        return f"{prefix} (no description available)"
    return f"{prefix}: {body}"


def seedbed_reference_pool(seedbed_registry: Optional[list]) -> list:
    """The ``(path, text)`` pairs for every ``reference``-role seedbed asset.

    Pins are excluded: a pinned image is already a frame, so offering it back as
    a reference would have a shot conditioning on itself.
    """
    if not seedbed_registry:
        return []
    return [
        (entry["path"], seed_asset_pool_text(entry))
        for entry in seedbed_registry
        if entry.get("role") == "reference" and entry.get("path")
    ]
