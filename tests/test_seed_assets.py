"""Unit tests for seedbed intake: SeedAsset validation, coercion, and the pool.

The pin-without-shot_idx raise and the bare-URL coercion are the two behaviours
the rest of the seedbed path depends on: the first stops a job from pinning an
image onto a guessed shot, the second is what keeps existing orchestrator
scripts (which pass a plain list of URLs) working unmodified.
"""

import os
import sys
import unittest

from pydantic import ValidationError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from interfaces import (  # noqa: E402
    SeedAsset,
    coerce_seed_assets,
    seed_asset_pool_text,
    seedbed_reference_pool,
)

URL = "https://example.test/seedbed/tx_cooking_01.jpg"


class TestSeedAssetValidation(unittest.TestCase):
    def test_default_role_is_reference(self):
        """'reference' is the non-destructive default — a mis-defaulted pin
        would overwrite a shot's keyframe."""
        self.assertEqual(SeedAsset(url=URL).role, "reference")

    def test_pin_without_shot_idx_raises(self):
        with self.assertRaises(ValidationError) as ctx:
            SeedAsset(url=URL, role="pin")
        self.assertIn("requires shot_idx", str(ctx.exception))

    def test_pin_with_shot_idx_is_valid(self):
        asset = SeedAsset(url=URL, role="pin", shot_idx=3)
        self.assertEqual(asset.shot_idx, 3)

    def test_pin_with_negative_shot_idx_raises(self):
        with self.assertRaises(ValidationError):
            SeedAsset(url=URL, role="pin", shot_idx=-1)

    def test_reference_without_shot_idx_is_fine(self):
        self.assertIsNone(SeedAsset(url=URL, role="reference").shot_idx)

    def test_invalid_role_rejected(self):
        with self.assertRaises(ValidationError):
            SeedAsset(url=URL, role="keyframe")

    def test_invalid_hint_rejected(self):
        with self.assertRaises(ValidationError):
            SeedAsset(url=URL, hint="vibe")

    def test_round_trips_through_json(self):
        asset = SeedAsset(url=URL, role="pin", shot_idx=2, hint="location", note="n")
        self.assertEqual(SeedAsset.model_validate(asset.model_dump()), asset)


class TestCoercion(unittest.TestCase):
    def test_bare_url_string_coerces_to_reference(self):
        [asset] = coerce_seed_assets([URL])
        self.assertEqual(asset.url, URL)
        self.assertEqual(asset.role, "reference")
        self.assertIsNone(asset.shot_idx)

    def test_run_fog_soup_style_url_list_still_works(self):
        """The exact shape run_fog_soup.py passes must keep working."""
        urls = [f"https://example.test/a{i}.jpg" for i in range(3)]
        assets = coerce_seed_assets(urls)
        self.assertEqual(len(assets), 3)
        self.assertTrue(all(a.role == "reference" for a in assets))

    def test_dict_coerces_and_validates(self):
        [asset] = coerce_seed_assets([{"url": URL, "role": "pin", "shot_idx": 1}])
        self.assertEqual(asset.role, "pin")
        self.assertEqual(asset.shot_idx, 1)

    def test_dict_pin_without_shot_idx_still_raises(self):
        with self.assertRaises(ValidationError):
            coerce_seed_assets([{"url": URL, "role": "pin"}])

    def test_seed_asset_passes_through(self):
        original = SeedAsset(url=URL)
        [asset] = coerce_seed_assets([original])
        self.assertIs(asset, original)

    def test_mixed_list_supported(self):
        assets = coerce_seed_assets([
            URL,
            {"url": "https://example.test/b.jpg", "role": "pin", "shot_idx": 0},
            SeedAsset(url="https://example.test/c.jpg", hint="style"),
        ])
        self.assertEqual([a.role for a in assets], ["reference", "pin", "reference"])

    def test_none_and_empty_produce_empty_list(self):
        self.assertEqual(coerce_seed_assets(None), [])
        self.assertEqual(coerce_seed_assets([]), [])

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            coerce_seed_assets([42])


class TestPoolText(unittest.TestCase):
    def test_hint_prefixes_the_text(self):
        self.assertTrue(seed_asset_pool_text(
            {"hint": "location", "description": "a kitchen"},
        ).startswith("Location reference: "))
        self.assertTrue(seed_asset_pool_text(
            {"hint": "character", "description": "a host"},
        ).startswith("Character reference: "))
        self.assertTrue(seed_asset_pool_text(
            {"hint": "style", "description": "grainy"},
        ).startswith("Style reference: "))

    def test_no_hint_falls_back_to_generic_prefix(self):
        self.assertTrue(
            seed_asset_pool_text({"description": "d"}).startswith("Reference: "),
        )

    def test_description_preferred_over_note(self):
        text = seed_asset_pool_text({"description": "vision text", "note": "op note"})
        self.assertIn("vision text", text)
        self.assertNotIn("op note", text)

    def test_note_used_when_no_description(self):
        text = seed_asset_pool_text({"description": None, "note": "op note"})
        self.assertIn("op note", text)

    def test_missing_both_says_so_rather_than_inventing(self):
        """Absence is recorded, never faked — a confidently wrong description of
        a reference image is worse than an acknowledged missing one."""
        text = seed_asset_pool_text({"description": None, "note": None})
        self.assertIn("no description available", text)


class TestReferencePool(unittest.TestCase):
    REGISTRY = [
        {"path": "/w/seedbed/0_a.png", "role": "reference", "hint": "location",
         "description": "a wood-panelled kitchen", "note": None},
        {"path": "/w/seedbed/1_b.png", "role": "pin", "shot_idx": 3,
         "hint": "location", "description": "overhead", "note": None},
        {"path": "/w/seedbed/2_c.png", "role": "reference", "hint": None,
         "description": None, "note": "the title card"},
    ]

    def test_pins_excluded_from_the_pool(self):
        """A pinned image IS a frame — offering it back as a reference would
        have that shot conditioning on itself."""
        pool = seedbed_reference_pool(self.REGISTRY)
        self.assertEqual(len(pool), 2)
        self.assertNotIn("/w/seedbed/1_b.png", [p for p, _ in pool])

    def test_pairs_are_path_and_text(self):
        pool = seedbed_reference_pool(self.REGISTRY)
        self.assertEqual(pool[0][0], "/w/seedbed/0_a.png")
        self.assertEqual(pool[0][1], "Location reference: a wood-panelled kitchen")
        self.assertEqual(pool[1][1], "Reference: the title card")

    def test_entries_without_a_path_skipped(self):
        pool = seedbed_reference_pool([{"role": "reference", "path": None}])
        self.assertEqual(pool, [])

    def test_empty_registry_returns_empty(self):
        self.assertEqual(seedbed_reference_pool(None), [])
        self.assertEqual(seedbed_reference_pool([]), [])

    def test_all_pins_returns_empty_pool(self):
        pool = seedbed_reference_pool([
            {"path": "/w/a.png", "role": "pin", "shot_idx": 0},
        ])
        self.assertEqual(pool, [])


class TestSeedbedSlug(unittest.TestCase):
    """The ingest filename must stay recognizable against the source URL."""

    def setUp(self):
        from pipelines.script2video_pipeline import _seedbed_slug
        self.slug = _seedbed_slug

    def test_basename_becomes_the_slug(self):
        self.assertEqual(self.slug(URL), "tx_cooking_01")

    def test_query_string_stripped(self):
        self.assertEqual(self.slug(URL + "?token=abc&v=2"), "tx_cooking_01")

    def test_punctuation_collapsed(self):
        self.assertEqual(
            self.slug("https://x.test/My%20Photo (final).JPG"),
            "my_20photo_final",
        )

    def test_trailing_slash_and_empty_basename_handled(self):
        self.assertEqual(self.slug("https://x.test/"), "asset")

    def test_long_names_truncated(self):
        self.assertLessEqual(len(self.slug("https://x.test/" + "a" * 200 + ".png")), 48)


if __name__ == "__main__":
    unittest.main()
