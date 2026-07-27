"""Unit tests for the environment stage: dedupe, slug, env_idx plumbing.

The dedupe test is load-bearing: two master plates for one location defeats the
entire reason environments became first-class objects.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.environment_extractor import _dedupe_and_reindex  # noqa: E402
from agents.environment_plate_generator import (  # noqa: E402
    environment_plate_description,
    prompt_template_environment_plate,
)
from interfaces import (  # noqa: E402
    EnvironmentInScene,
    ShotBriefDescription,
    ShotDescription,
    normalize_slugline,
)


def _env(slugline, description="d", idx=0):
    return EnvironmentInScene(idx=idx, slugline=slugline, description=description)


class TestNormalizeSlugline(unittest.TestCase):
    def test_case_and_punctuation_collapsed(self):
        self.assertEqual(
            normalize_slugline("INT. Coffee Shop - Night"),
            normalize_slugline("INT COFFEE SHOP  -  NIGHT"),
        )

    def test_distinct_locations_stay_distinct(self):
        self.assertNotEqual(
            normalize_slugline("INT. COFFEE SHOP - NIGHT"),
            normalize_slugline("INT. COFFEE SHOP - DAY"),
        )

    def test_int_and_ext_stay_distinct(self):
        self.assertNotEqual(
            normalize_slugline("INT. PARK - DAY"),
            normalize_slugline("EXT. PARK - DAY"),
        )

    def test_empty_input_safe(self):
        self.assertEqual(normalize_slugline(""), "")
        self.assertEqual(normalize_slugline(None), "")


class TestDedupeAndReindex(unittest.TestCase):
    def test_punctuation_variant_duplicates_collapsed(self):
        envs = _dedupe_and_reindex([
            _env("INT. COFFEE SHOP - NIGHT", idx=0),
            _env("INT Coffee Shop - night", idx=1),
        ])
        self.assertEqual(len(envs), 1)
        # First spelling wins — it is the one the script used first.
        self.assertEqual(envs[0].slugline, "INT. COFFEE SHOP - NIGHT")

    def test_indices_renumbered_contiguously(self):
        """The storyboard assigns shots by index, so indices must be contiguous
        and must match the saved list exactly."""
        envs = _dedupe_and_reindex([
            _env("A - DAY", idx=7),
            _env("A - Day", idx=8),
            _env("B - NIGHT", idx=9),
        ])
        self.assertEqual([e.idx for e in envs], [0, 1])
        self.assertEqual([e.slugline for e in envs], ["A - DAY", "B - NIGHT"])

    def test_distinct_locations_preserved_in_order(self):
        envs = _dedupe_and_reindex([
            _env("EXT. PARK - DAY"), _env("INT. CAR - DAY"), _env("INT. HOUSE - NIGHT"),
        ])
        self.assertEqual([e.idx for e in envs], [0, 1, 2])

    def test_empty_input(self):
        self.assertEqual(_dedupe_and_reindex([]), [])


class TestEnvironmentSlug(unittest.TestCase):
    def test_slug_is_filesystem_safe(self):
        slug = _env("INT. COFFEE SHOP - NIGHT").slug
        self.assertEqual(slug, "INT_COFFEE_SHOP_NIGHT")
        for bad in '/\\:*?"<>|':
            self.assertNotIn(bad, slug)

    def test_slug_truncated(self):
        self.assertLessEqual(len(_env("INT. " + "X" * 200 + " - DAY").slug), 60)

    def test_degenerate_slugline_gets_a_fallback(self):
        self.assertEqual(_env("...").slug, "location")


class TestPlatePrompt(unittest.TestCase):
    def test_prompt_forbids_characters(self):
        """A plate with a person in it injects a stray identity into every frame
        that references it, competing with the character sheets."""
        prompt = prompt_template_environment_plate.format(
            slugline="INT. SHOP - NIGHT", description="a shop", style="cinematic",
        )
        lowered = prompt.lower()
        for forbidden in ("no people", "no characters", "no faces", "nothing animate"):
            self.assertIn(forbidden, lowered)

    def test_prompt_requests_a_wide_establishing_view(self):
        prompt = prompt_template_environment_plate.format(
            slugline="s", description="d", style="st",
        )
        self.assertIn("wide establishing view", prompt)

    def test_style_is_appended(self):
        prompt = prompt_template_environment_plate.format(
            slugline="s", description="d", style="MY_STYLE_TOKEN",
        )
        self.assertIn("MY_STYLE_TOKEN", prompt)

    def test_registry_description_states_what_to_match(self):
        desc = environment_plate_description(
            _env("INT. COFFEE SHOP - NIGHT", "warm brick and neon"),
        )
        self.assertIn("INT. COFFEE SHOP - NIGHT", desc)
        self.assertIn("warm brick and neon", desc)
        # Says what to take FROM the plate, so it isn't treated as loose
        # inspiration and the room re-invented.
        self.assertIn("Match its architecture", desc)


class TestEnvIdxPlumbing(unittest.TestCase):
    def _brief(self, **kw):
        base = dict(
            idx=0, is_last=True, cam_idx=0, visual_desc="v", audio_desc="a",
        )
        base.update(kw)
        return ShotBriefDescription(**base)

    def test_env_idx_defaults_to_none(self):
        self.assertIsNone(self._brief().env_idx)

    def test_env_idx_accepted_on_brief(self):
        self.assertEqual(self._brief(env_idx=2).env_idx, 2)

    def test_legacy_storyboard_json_still_loads(self):
        """Storyboards written before env_idx existed must still validate — the
        frame path fails loud by shot number instead, which an operator can act
        on. A required field would fail pydantic on a file they can't connect to
        a missing assignment."""
        legacy = {
            "idx": 0, "is_last": False, "cam_idx": 0,
            "visual_desc": "v", "audio_desc": "a",
        }
        shot = ShotBriefDescription.model_validate(legacy)
        self.assertIsNone(shot.env_idx)

    def test_shot_description_carries_env_idx(self):
        shot = ShotDescription(
            idx=0, is_last=True, cam_idx=0, env_idx=1, visual_desc="v",
            variation_type="small", variation_reason="r",
            ff_desc="ff", lf_desc="lf", motion_desc="m", audio_desc="a",
        )
        self.assertEqual(shot.env_idx, 1)
        self.assertEqual(
            ShotDescription.model_validate(shot.model_dump()).env_idx, 1,
        )


class TestStoryboardEnvAutofill(unittest.TestCase):
    """Single-environment scripts are the common case and the model sometimes
    omits env_idx on them. Filling the ONLY possible value is not a guess."""

    def test_autofill_only_when_exactly_one_environment(self):
        import asyncio
        from agents.storyboard_artist import StoryboardArtist

        shots = [
            ShotBriefDescription(idx=0, is_last=False, cam_idx=0,
                                 visual_desc="v", audio_desc="a"),
            ShotBriefDescription(idx=1, is_last=True, cam_idx=0, env_idx=0,
                                 visual_desc="v", audio_desc="a"),
        ]

        class _StubResponse:
            storyboard = shots

        class _StubChain:
            async def ainvoke(self, *a, **kw):
                return _StubResponse()

        artist = StoryboardArtist(chat_model=None)
        artist.chat_model = _FakeModel(_StubChain())

        result = asyncio.run(artist.design_storyboard(
            script="s", characters=[], environments=[_env("INT. A - DAY")],
        ))
        self.assertEqual([s.env_idx for s in result], [0, 0])

    def test_no_autofill_with_multiple_environments(self):
        import asyncio
        from agents.storyboard_artist import StoryboardArtist

        shots = [ShotBriefDescription(idx=0, is_last=True, cam_idx=0,
                                      visual_desc="v", audio_desc="a")]

        class _StubResponse:
            storyboard = shots

        class _StubChain:
            async def ainvoke(self, *a, **kw):
                return _StubResponse()

        artist = StoryboardArtist(chat_model=None)
        artist.chat_model = _FakeModel(_StubChain())

        result = asyncio.run(artist.design_storyboard(
            script="s", characters=[],
            environments=[_env("A - DAY", idx=0), _env("B - DAY", idx=1)],
        ))
        # Left None so the frame path rejects it by name rather than picking one.
        self.assertIsNone(result[0].env_idx)


class _FakeModel:
    """Stands in for a chat model so `model | parser` yields our stub chain."""

    def __init__(self, chain):
        self._chain = chain

    def __or__(self, _other):
        return self._chain


if __name__ == "__main__":
    unittest.main()
