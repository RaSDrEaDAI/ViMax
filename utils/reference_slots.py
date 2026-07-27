"""Deterministic reference-slot budgeting for keyframe generation.

Replaces pure-LLM reference selection on the keyframe path. Which references a
frame gets is now a fixed priority order, not a model's judgement call:

  1. Character sheets for the frame's visible characters (max 3)
  2. The location plate for the shot's environment
  3. The camera continuity anchor
  4. Seedbed ``reference`` assets, chosen by the selector from that pool only

Why deterministic: the character sheets, the location plate, and the continuity
anchor are exactly the things that must NEVER be dropped — they are the identity,
location, and continuity authorities. Letting a model pick among them meant a
frame could silently lose its character sheet and render an invented face. So
those are always in, and the only thing left to choose is which seedbed
references fill whatever slots remain.

Caps come from ``references/ai_model_context_v8.csv`` L419 (NB Pro World
Building), verbatim: "<= 5-6 references per call; <= 3 distinct characters before
identity-merge risk climbs. Exactly ONE composition delta per generation."

Overflow drops from the TAIL and always emits a notice. Silent truncation reads
as "everything was included" when it wasn't — the operator has to be able to see
that a 4th character or a seedbed reference didn't make it into a frame.

Lives in utils/ rather than inline in the pipeline so the budget is unit-testable
without standing up a pipeline, a registry, and a chat model.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

# NB Pro takes 5-6 references before quality collapses. 6 is the ceiling.
SLOT_BUDGET = 6

# Beyond 3 distinct characters in one generation, identity-merge risk climbs:
# the model starts averaging faces together.
MAX_CHARACTERS = 3

SlotRole = Literal[
    "character_sheet",
    "environment_plate",
    "continuity_anchor",
    "seedbed_reference",
]


@dataclass(frozen=True)
class ReferenceSlot:
    path: str
    text: str
    role: SlotRole


@dataclass
class SlotBudgetResult:
    slots: List[ReferenceSlot]
    # Human-readable degrade notices. The caller prints these; they are never
    # swallowed.
    notices: List[str]

    @property
    def paths(self) -> List[str]:
        return [s.path for s in self.slots]


def build_reference_slots(
    *,
    character_sheets: Sequence[Tuple[str, str]] = (),
    environment_plate: Optional[Tuple[str, str]] = None,
    continuity_anchor: Optional[Tuple[str, str]] = None,
    seedbed_references: Sequence[Tuple[str, str]] = (),
    budget: int = SLOT_BUDGET,
    max_characters: int = MAX_CHARACTERS,
) -> SlotBudgetResult:
    """Assemble reference slots in priority order, dropping the tail on overflow.

    Each input is a ``(path, text)`` pair; ``character_sheets`` and
    ``seedbed_references`` are ordered lists whose own tails drop first.
    """
    notices: List[str] = []

    # Character cap applies BEFORE the slot budget: dropping a 4th character is
    # about identity merge, not about running out of slots, so it happens even
    # when slots are free.
    sheets = list(character_sheets)
    if len(sheets) > max_characters:
        dropped = sheets[max_characters:]
        sheets = sheets[:max_characters]
        notices.append(
            f"DEGRADED: {len(dropped)} character sheet(s) dropped — "
            f"{max_characters} distinct characters is the NB Pro identity-merge "
            f"cap. Dropped: {', '.join(text for _, text in dropped)}"
        )

    candidates: List[ReferenceSlot] = []
    for path, text in sheets:
        candidates.append(ReferenceSlot(path, text, "character_sheet"))
    if environment_plate is not None:
        candidates.append(ReferenceSlot(*environment_plate, "environment_plate"))
    if continuity_anchor is not None:
        candidates.append(ReferenceSlot(*continuity_anchor, "continuity_anchor"))
    for path, text in seedbed_references:
        candidates.append(ReferenceSlot(path, text, "seedbed_reference"))

    if len(candidates) > budget:
        dropped = candidates[budget:]
        candidates = candidates[:budget]
        by_role: Dict[str, int] = {}
        for slot in dropped:
            by_role[slot.role] = by_role.get(slot.role, 0) + 1
        summary = ", ".join(f"{n} {role}" for role, n in by_role.items())
        notices.append(
            f"DEGRADED: {len(dropped)} reference(s) dropped to fit the "
            f"{budget}-slot budget ({summary}). Lowest-priority references go "
            f"first: seedbed, then continuity anchor, then location plate."
        )

    return SlotBudgetResult(slots=candidates, notices=notices)


# ── Prompt assembly ──────────────────────────────────────────────────────────

_ROLE_PREFIX = {
    "character_sheet": "character sheet",
    "environment_plate": "location plate",
    "continuity_anchor": "continuity anchor",
    "seedbed_reference": "reference",
}


def assemble_indexed_prompt(
    slots: Sequence[ReferenceSlot],
    frame_instruction: str,
) -> str:
    """Build the NB Pro World Building indexed prompt.

    1-BASED indexing, matching the CSV skeleton ("Image 1: location..."). The old
    path emitted 0-based labels, which fights the grammar the model was trained
    on and reads as an off-by-one to anyone comparing prompt to reference list.

    Emits the mandatory distinct-identity clause whenever two or more character
    sheets are present — the CSV marks it mandatory for multi-character
    generations, and without it the model averages faces together.
    """
    lines = [f"Image {i}: {slot.text}" for i, slot in enumerate(slots, start=1)]

    sheet_positions = [
        (i, slot) for i, slot in enumerate(slots, start=1)
        if slot.role == "character_sheet"
    ]
    if len(sheet_positions) >= 2:
        refs = "; ".join(
            f"the character in Image {i}: preserve all of its Image {i} "
            f"features exactly"
            for i, _ in sheet_positions
        )
        lines.append(
            "\nCritical: the characters must be visually distinct and "
            "individually identifiable. Do not merge, blend, or average "
            f"identities. {refs}."
        )

    prefix = "\n".join(lines)
    # One composition delta per generation, per the CSV cap. ff_desc / lf_desc
    # each describe a single static frame, so this holds naturally.
    return f"{prefix}\n\n{frame_instruction}"


# ── Registry resolution (fail loud) ──────────────────────────────────────────


def resolve_character_sheet(
    character_portraits_registry: Dict[str, Dict[str, Dict[str, str]]],
    identifier_in_scene: str,
) -> Tuple[str, str]:
    """Resolve one character's composite sheet from the registry.

    Fails loud when there is no ``sheet`` entry. There is deliberately NO
    fallback to the legacy front/side/back views: a frame that quietly swapped
    the composite sheet for a front portrait would render with measurably worse
    identity lock, and it would do so invisibly, on some frames and not others —
    which is precisely the drift the sheet exists to eliminate.
    """
    entry = character_portraits_registry.get(identifier_in_scene)
    if entry is None:
        raise KeyError(
            f"Character {identifier_in_scene!r} is not in the character "
            f"portraits registry. Known: "
            f"{sorted(character_portraits_registry)}"
        )

    sheet = entry.get("sheet")
    if not sheet or not sheet.get("path"):
        raise RuntimeError(
            f"Character {identifier_in_scene!r} has no `sheet` entry in the "
            f"character portraits registry (found: {sorted(entry)}). The "
            f"keyframe path requires a composite sheet and will not fall back "
            f"to individual portrait views. Either delete "
            f"character_portraits_registry.json to regenerate the sheet chain "
            f"(front -> anchor -> rotation grid -> compose), or register an "
            f"existing sheet with:\n"
            f"  uv run python scripts/seed_character_portraits.py "
            f"--project <project> --identifier \"{identifier_in_scene}\" "
            f"--sheet <path>"
        )
    return sheet["path"], sheet["description"]


def resolve_environment_plate(
    environment_registry: Dict[str, Dict[str, Dict[str, str]]],
    environments: Sequence,
    env_idx: Optional[int],
    shot_idx: int,
) -> Tuple[str, str]:
    """Resolve the location plate for a shot's ``env_idx``.

    Fails loud on a missing or out-of-range index. No "closest match" guessing:
    silently rendering a shot against the wrong room is worse than stopping,
    because it looks like a creative choice rather than a bug.
    """
    if env_idx is None:
        raise RuntimeError(
            f"Shot {shot_idx} has no env_idx. Every shot must be assigned to "
            f"exactly one environment by the storyboard. Delete storyboard.json "
            f"and the per-shot shot_description.json files to re-run the "
            f"storyboard with environment assignment."
        )
    if not 0 <= env_idx < len(environments):
        raise RuntimeError(
            f"Shot {shot_idx} has env_idx={env_idx}, which is out of range for "
            f"{len(environments)} extracted environment(s) (valid: 0.."
            f"{len(environments) - 1}). No closest-match guessing — fix the "
            f"storyboard assignment."
        )

    environment = environments[env_idx]
    entry = environment_registry.get(environment.slugline)
    if not entry or not entry.get("plate", {}).get("path"):
        raise RuntimeError(
            f"Environment {environment.slugline!r} (idx {env_idx}, shot "
            f"{shot_idx}) has no plate in the environment registry. Known: "
            f"{sorted(environment_registry)}"
        )
    plate = entry["plate"]
    return plate["path"], plate["description"]
