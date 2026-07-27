"""Unit tests for the deterministic keyframe reference-slot budget.

Covers the two failure modes the budget exists to prevent — silently dropping an
identity/location/continuity authority, and silently exceeding the NB Pro caps —
plus the fail-loud registry resolution.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from interfaces import EnvironmentInScene  # noqa: E402
from utils.reference_slots import (  # noqa: E402
    MAX_CHARACTERS,
    SLOT_BUDGET,
    assemble_indexed_prompt,
    build_reference_slots,
    resolve_character_sheet,
    resolve_environment_plate,
)

SHEET = lambda n: (f"/w/char{n}/sheet.png", f"character sheet for Char{n}")  # noqa: E731
PLATE = ("/w/env/plate.png", "location plate for INT. SHOP - NIGHT")
ANCHOR = ("/w/shots/0/first_frame.png", "continuity anchor for this camera")
SEED = lambda n: (f"/w/seedbed/{n}.png", f"Location reference: seed {n}")  # noqa: E731


class TestPriorityOrder(unittest.TestCase):
    def test_roles_appear_in_priority_order(self):
        result = build_reference_slots(
            character_sheets=[SHEET(0)],
            environment_plate=PLATE,
            continuity_anchor=ANCHOR,
            seedbed_references=[SEED(0)],
        )
        self.assertEqual(
            [s.role for s in result.slots],
            ["character_sheet", "environment_plate", "continuity_anchor",
             "seedbed_reference"],
        )
        self.assertEqual(result.notices, [])

    def test_optional_inputs_omitted_cleanly(self):
        result = build_reference_slots(character_sheets=[SHEET(0)])
        self.assertEqual([s.role for s in result.slots], ["character_sheet"])

    def test_empty_inputs_produce_no_slots(self):
        result = build_reference_slots()
        self.assertEqual(result.slots, [])
        self.assertEqual(result.notices, [])

    def test_paths_helper_matches_slot_order(self):
        result = build_reference_slots(
            character_sheets=[SHEET(0), SHEET(1)],
            environment_plate=PLATE,
        )
        self.assertEqual(
            result.paths,
            [SHEET(0)[0], SHEET(1)[0], PLATE[0]],
        )


class TestCharacterCap(unittest.TestCase):
    def test_fourth_character_dropped_with_notice(self):
        result = build_reference_slots(
            character_sheets=[SHEET(0), SHEET(1), SHEET(2), SHEET(3)],
        )
        sheets = [s for s in result.slots if s.role == "character_sheet"]
        self.assertEqual(len(sheets), MAX_CHARACTERS)
        self.assertEqual(len(result.notices), 1)
        self.assertIn("identity-merge", result.notices[0])
        self.assertIn("Char3", result.notices[0])

    def test_character_cap_applies_even_when_slots_are_free(self):
        """Dropping a 4th character is about identity merge, not slot pressure —
        4 sheets fit in 6 slots but must still be capped at 3."""
        result = build_reference_slots(
            character_sheets=[SHEET(0), SHEET(1), SHEET(2), SHEET(3)],
        )
        self.assertLess(len(result.slots), SLOT_BUDGET)
        self.assertTrue(result.notices)

    def test_three_characters_pass_without_notice(self):
        result = build_reference_slots(
            character_sheets=[SHEET(0), SHEET(1), SHEET(2)],
        )
        self.assertEqual(len(result.slots), 3)
        self.assertEqual(result.notices, [])


class TestSlotBudget(unittest.TestCase):
    def test_overflow_drops_from_the_tail(self):
        result = build_reference_slots(
            character_sheets=[SHEET(0), SHEET(1), SHEET(2)],
            environment_plate=PLATE,
            continuity_anchor=ANCHOR,
            seedbed_references=[SEED(0), SEED(1), SEED(2)],
        )
        self.assertEqual(len(result.slots), SLOT_BUDGET)
        # 3 sheets + plate + anchor = 5, so exactly one seedbed ref survives.
        seedbed = [s for s in result.slots if s.role == "seedbed_reference"]
        self.assertEqual(len(seedbed), 1)
        self.assertEqual(seedbed[0].path, SEED(0)[0])

    def test_authorities_never_dropped_under_seedbed_pressure(self):
        """The point of the whole module: a frame must not lose its character
        sheet, its location plate, or its continuity anchor to a seedbed ref."""
        result = build_reference_slots(
            character_sheets=[SHEET(0), SHEET(1)],
            environment_plate=PLATE,
            continuity_anchor=ANCHOR,
            seedbed_references=[SEED(i) for i in range(20)],
        )
        roles = [s.role for s in result.slots]
        self.assertEqual(roles.count("character_sheet"), 2)
        self.assertEqual(roles.count("environment_plate"), 1)
        self.assertEqual(roles.count("continuity_anchor"), 1)
        self.assertEqual(len(result.slots), SLOT_BUDGET)

    def test_overflow_notice_is_emitted_and_names_the_count(self):
        result = build_reference_slots(
            character_sheets=[SHEET(0)],
            environment_plate=PLATE,
            continuity_anchor=ANCHOR,
            seedbed_references=[SEED(i) for i in range(5)],
        )
        self.assertEqual(len(result.notices), 1)
        self.assertIn("DEGRADED", result.notices[0])
        self.assertIn("2 reference(s) dropped", result.notices[0])

    def test_exactly_at_budget_emits_no_notice(self):
        result = build_reference_slots(
            character_sheets=[SHEET(0), SHEET(1), SHEET(2)],
            environment_plate=PLATE,
            continuity_anchor=ANCHOR,
            seedbed_references=[SEED(0)],
        )
        self.assertEqual(len(result.slots), SLOT_BUDGET)
        self.assertEqual(result.notices, [])

    def test_both_caps_can_fire_together(self):
        result = build_reference_slots(
            character_sheets=[SHEET(i) for i in range(5)],
            environment_plate=PLATE,
            continuity_anchor=ANCHOR,
            seedbed_references=[SEED(i) for i in range(4)],
        )
        self.assertEqual(len(result.notices), 2)
        self.assertEqual(len(result.slots), SLOT_BUDGET)

    def test_custom_budget_respected(self):
        result = build_reference_slots(
            character_sheets=[SHEET(0)],
            environment_plate=PLATE,
            continuity_anchor=ANCHOR,
            budget=2,
        )
        self.assertEqual(len(result.slots), 2)
        self.assertTrue(result.notices)


class TestPromptAssembly(unittest.TestCase):
    def test_indexing_is_one_based(self):
        result = build_reference_slots(
            character_sheets=[SHEET(0)], environment_plate=PLATE,
        )
        prompt = assemble_indexed_prompt(result.slots, "Wide shot of the room.")
        self.assertIn("Image 1: character sheet for Char0", prompt)
        self.assertIn("Image 2: location plate", prompt)
        self.assertNotIn("Image 0:", prompt)

    def test_frame_instruction_comes_last(self):
        result = build_reference_slots(character_sheets=[SHEET(0)])
        prompt = assemble_indexed_prompt(result.slots, "THE FRAME INSTRUCTION")
        self.assertTrue(prompt.rstrip().endswith("THE FRAME INSTRUCTION"))

    def test_distinct_identity_clause_on_multi_character(self):
        """Mandatory per the CSV — without it the model averages faces."""
        result = build_reference_slots(
            character_sheets=[SHEET(0), SHEET(1)], environment_plate=PLATE,
        )
        prompt = assemble_indexed_prompt(result.slots, "Two people talk.")
        self.assertIn("visually distinct and individually identifiable", prompt)
        self.assertIn("Do not merge, blend, or average identities", prompt)

    def test_no_distinct_identity_clause_on_single_character(self):
        result = build_reference_slots(
            character_sheets=[SHEET(0)], environment_plate=PLATE,
        )
        prompt = assemble_indexed_prompt(result.slots, "One person.")
        self.assertNotIn("Do not merge", prompt)

    def test_distinct_identity_clause_cites_actual_slot_indices(self):
        """The clause must reference the sheets' real 1-based positions, not
        their positions among sheets — an off-by-one here points the model at
        the location plate."""
        result = build_reference_slots(
            character_sheets=[SHEET(0), SHEET(1)],
            environment_plate=PLATE,
            continuity_anchor=ANCHOR,
        )
        prompt = assemble_indexed_prompt(result.slots, "x")
        self.assertIn("Image 1: preserve all of its Image 1 features", prompt)
        self.assertIn("Image 2: preserve all of its Image 2 features", prompt)
        self.assertNotIn("Image 3: preserve", prompt)

    def test_no_slots_still_produces_the_instruction(self):
        prompt = assemble_indexed_prompt([], "Just render this.")
        self.assertIn("Just render this.", prompt)


class TestResolveCharacterSheet(unittest.TestCase):
    def test_returns_path_and_description(self):
        registry = {"Marella": {"sheet": {"path": "/w/s.png", "description": "d"}}}
        self.assertEqual(resolve_character_sheet(registry, "Marella"), ("/w/s.png", "d"))

    def test_unknown_character_raises_keyerror(self):
        with self.assertRaises(KeyError):
            resolve_character_sheet({"Bob": {"sheet": {"path": "p"}}}, "Marella")

    def test_legacy_three_view_registry_fails_loud(self):
        """No silent fallback to `front`. A frame that quietly swapped the sheet
        for a front portrait would degrade identity invisibly."""
        legacy = {"Marella": {
            "front": {"path": "/w/front.png", "description": "front"},
            "side": {"path": "/w/side.png", "description": "side"},
            "back": {"path": "/w/back.png", "description": "back"},
        }}
        with self.assertRaises(RuntimeError) as ctx:
            resolve_character_sheet(legacy, "Marella")
        msg = str(ctx.exception)
        self.assertIn("no `sheet` entry", msg)
        # The error has to tell the operator how to fix it.
        self.assertIn("--sheet", msg)
        self.assertIn("seed_character_portraits.py", msg)

    def test_sheet_entry_without_path_fails_loud(self):
        with self.assertRaises(RuntimeError):
            resolve_character_sheet({"M": {"sheet": {"description": "d"}}}, "M")


class TestResolveEnvironmentPlate(unittest.TestCase):
    def setUp(self):
        self.environments = [
            EnvironmentInScene(idx=0, slugline="INT. SHOP - NIGHT", description="d0"),
            EnvironmentInScene(idx=1, slugline="EXT. PARK - DAY", description="d1"),
        ]
        self.registry = {
            "INT. SHOP - NIGHT": {"plate": {"path": "/w/p0.png", "description": "pd0"}},
            "EXT. PARK - DAY": {"plate": {"path": "/w/p1.png", "description": "pd1"}},
        }

    def test_resolves_by_index(self):
        self.assertEqual(
            resolve_environment_plate(self.registry, self.environments, 1, 3),
            ("/w/p1.png", "pd1"),
        )

    def test_missing_env_idx_fails_loud_naming_the_shot(self):
        with self.assertRaises(RuntimeError) as ctx:
            resolve_environment_plate(self.registry, self.environments, None, 7)
        self.assertIn("Shot 7 has no env_idx", str(ctx.exception))

    def test_out_of_range_env_idx_fails_loud_without_guessing(self):
        with self.assertRaises(RuntimeError) as ctx:
            resolve_environment_plate(self.registry, self.environments, 5, 2)
        msg = str(ctx.exception)
        self.assertIn("out of range", msg)
        self.assertIn("No closest-match guessing", msg)

    def test_negative_env_idx_rejected(self):
        with self.assertRaises(RuntimeError):
            resolve_environment_plate(self.registry, self.environments, -1, 0)

    def test_environment_missing_from_registry_fails_loud(self):
        with self.assertRaises(RuntimeError) as ctx:
            resolve_environment_plate({}, self.environments, 0, 0)
        self.assertIn("no plate in the environment registry", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
