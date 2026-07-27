"""Pin smoke test (free, no network, no image backend).

Asserts the whole point of a pin: the seedbed PNG becomes the shot's first frame,
provenance is written, and NO image generation call fires for that frame. Also
covers the sent_input forensics record and the degrade notice reaching stdout.
"""

import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The pipeline's progress prints use emoji. On a Windows console defaulting to
# cp1252 those raise UnicodeEncodeError, which would fail these tests for a
# reason that has nothing to do with what they assert.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from interfaces import (  # noqa: E402
    CharacterInScene, EnvironmentInScene, ImageOutput, ShotDescription,
)
from pipelines.script2video_pipeline import Script2VideoPipeline  # noqa: E402
from utils.composite_sheet import character_sheet_description  # noqa: E402


class _CountingImageGenerator:
    """Records every generate call. Returns a real 1x1 PNG so .save() works."""

    def __init__(self):
        self.calls = []

    async def generate_single_image(self, prompt, reference_image_paths=None, **kwargs):
        self.calls.append({
            "prompt": prompt,
            "reference_image_paths": list(reference_image_paths or []),
            "kwargs": kwargs,
        })
        return ImageOutput(
            fmt="pil", ext="png", data=Image.new("RGB", (1, 1), (1, 2, 3)),
            sent_input={"prompt": prompt, "image_urls": list(reference_image_paths or [])},
        )


class _NoopVideoGenerator:
    async def generate_single_video(self, prompt, reference_image_paths=None, **kwargs):
        raise AssertionError("video generation should not fire in this test")


def _pipeline(working_dir, image_generator=None):
    pipeline = Script2VideoPipeline(
        chat_model=None,
        image_generator=image_generator or _CountingImageGenerator(),
        video_generator=_NoopVideoGenerator(),
        working_dir=working_dir,
    )
    # Class-level event dicts are shared between instances; clear them so tests
    # don't inherit another test's events.
    pipeline.frame_events = {}
    return pipeline


def _shot(idx, env_idx=0, variation="small"):
    return ShotDescription(
        idx=idx, is_last=False, cam_idx=0, env_idx=env_idx, visual_desc="v",
        variation_type=variation, variation_reason="r",
        ff_desc=f"first frame of shot {idx}", lf_desc=f"last frame of shot {idx}",
        motion_desc="m", audio_desc="a",
    )


def _write_png(path, colour=(200, 100, 50)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (8, 8), colour).save(path)
    return path


class TestApplySeedbedPins(unittest.TestCase):
    def test_pin_adopted_with_provenance_and_no_image_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _CountingImageGenerator()
            pipeline = _pipeline(tmp, gen)
            src = _write_png(os.path.join(tmp, "seedbed", "0_tx_cooking.png"))
            registry = [{
                "path": src, "url": "https://x.test/tx_cooking.jpg", "role": "pin",
                "shot_idx": 1, "hint": "location", "note": None, "description": None,
            }]

            pipeline.apply_seedbed_pins(registry, [_shot(0), _shot(1), _shot(2)])

            ff = os.path.join(tmp, "shots", "1", "first_frame.png")
            self.assertTrue(os.path.exists(ff), "pinned PNG must land as first_frame")
            with Image.open(ff) as img:
                self.assertEqual(img.getpixel((0, 0)), (200, 100, 50))

            with open(os.path.join(tmp, "shots", "1", "first_frame_source.json")) as f:
                provenance = json.load(f)
            self.assertEqual(provenance, {
                "source": "seedbed",
                "url": "https://x.test/tx_cooking.jpg",
                "registry_idx": 0,
            })

            # The load-bearing assertion: pinning spends nothing.
            self.assertEqual(gen.calls, [])

    def test_pinned_frame_is_skipped_by_frame_generation(self):
        """The existing skip-if-exists check is what adopts the pin, so no image
        call may fire for that frame afterwards."""
        with tempfile.TemporaryDirectory() as tmp:
            gen = _CountingImageGenerator()
            pipeline = _pipeline(tmp, gen)
            src = _write_png(os.path.join(tmp, "seedbed", "0_a.png"))
            registry = [{
                "path": src, "url": "https://x.test/a.jpg", "role": "pin",
                "shot_idx": 0, "hint": None, "note": None, "description": None,
            }]
            pipeline.apply_seedbed_pins(registry, [_shot(0)])
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}

            asyncio.run(pipeline.generate_frame_for_single_shot(
                shot_idx=0,
                frame_type="first_frame",
                camera_anchor=("/w/anchor.png", "anchor"),
                frame_desc="ff",
                visible_characters=[],
                character_portraits_registry={},
                environments=[],
                environment_registry={},
                env_idx=0,
            ))
            self.assertEqual(gen.calls, [])
            self.assertTrue(pipeline.frame_events[0]["first_frame"].is_set())

    def test_pin_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            src = _write_png(os.path.join(tmp, "seedbed", "0_a.png"))
            registry = [{
                "path": src, "url": "u", "role": "pin", "shot_idx": 0,
                "hint": None, "note": None, "description": None,
            }]
            pipeline.apply_seedbed_pins(registry, [_shot(0)])
            out = io.StringIO()
            with redirect_stdout(out):
                pipeline.apply_seedbed_pins(registry, [_shot(0)])
            self.assertIn("already pinned", out.getvalue())

    def test_out_of_range_shot_idx_fails_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            src = _write_png(os.path.join(tmp, "seedbed", "0_a.png"))
            registry = [{
                "path": src, "url": "u", "role": "pin", "shot_idx": 9,
                "hint": None, "note": None, "description": None,
            }]
            with self.assertRaises(RuntimeError) as ctx:
                pipeline.apply_seedbed_pins(registry, [_shot(0), _shot(1)])
            msg = str(ctx.exception)
            self.assertIn("shot_idx=9", msg)
            self.assertIn("no guessing", msg)

    def test_reference_role_is_not_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            src = _write_png(os.path.join(tmp, "seedbed", "0_a.png"))
            pipeline.apply_seedbed_pins([{
                "path": src, "url": "u", "role": "reference", "shot_idx": None,
                "hint": None, "note": None, "description": None,
            }], [_shot(0)])
            self.assertFalse(
                os.path.exists(os.path.join(tmp, "shots", "0", "first_frame.png")),
            )

    def test_empty_registry_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            _pipeline(tmp).apply_seedbed_pins(None, [_shot(0)])
            self.assertFalse(os.path.exists(os.path.join(tmp, "shots")))


class TestKeyframeGeneration(unittest.TestCase):
    """The generated call: 16:9 sent explicitly, 1-based prompt, forensics file."""

    def setUp(self):
        self.registry = {
            "Marella": {"sheet": {
                "path": "/w/char0/sheet.png",
                "description": character_sheet_description("Marella"),
            }},
            "Bob": {"sheet": {
                "path": "/w/char1/sheet.png",
                "description": character_sheet_description("Bob"),
            }},
        }
        self.environments = [
            EnvironmentInScene(idx=0, slugline="INT. SHOP - NIGHT", description="d"),
        ]
        self.env_registry = {
            "INT. SHOP - NIGHT": {"plate": {
                "path": "/w/env/plate.png", "description": "Location plate: shop",
            }},
        }

    def _characters(self, *names):
        return [
            CharacterInScene(
                idx=i, identifier_in_scene=n, is_visible=True,
                static_features="s", dynamic_features="d",
            )
            for i, n in enumerate(names)
        ]

    def _run(self, tmp, gen, characters, seedbed_registry=None):
        pipeline = _pipeline(tmp, gen)
        pipeline.frame_events[0] = {"first_frame": asyncio.Event()}
        out = io.StringIO()
        with redirect_stdout(out):
            asyncio.run(pipeline.generate_frame_for_single_shot(
                shot_idx=0,
                frame_type="first_frame",
                camera_anchor=("/w/shots/0/ff.png", "continuity anchor"),
                frame_desc="Wide shot of the shop.",
                visible_characters=characters,
                character_portraits_registry=self.registry,
                environments=self.environments,
                environment_registry=self.env_registry,
                env_idx=0,
                seedbed_registry=seedbed_registry,
                variation_type="small",
            ))
        return out.getvalue()

    def test_aspect_ratio_sent_and_legacy_size_gone(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _CountingImageGenerator()
            self._run(tmp, gen, self._characters("Marella"))
            kwargs = gen.calls[0]["kwargs"]
            self.assertEqual(kwargs["aspect_ratio"], "16:9")
            self.assertNotIn("size", kwargs)

    def test_routing_metadata_still_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _CountingImageGenerator()
            self._run(tmp, gen, self._characters("Marella"))
            kwargs = gen.calls[0]["kwargs"]
            self.assertIn("visible_characters", kwargs)
            self.assertEqual(kwargs["frame_desc"], "Wide shot of the shop.")
            self.assertIn("Shot 0, first_frame frame", kwargs["shot_notes"])

    def test_references_are_authorities_in_priority_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _CountingImageGenerator()
            self._run(tmp, gen, self._characters("Marella", "Bob"))
            self.assertEqual(gen.calls[0]["reference_image_paths"], [
                "/w/char0/sheet.png",
                "/w/char1/sheet.png",
                "/w/env/plate.png",
                "/w/shots/0/ff.png",
            ])

    def test_prompt_is_one_based_and_ends_with_the_frame_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _CountingImageGenerator()
            self._run(tmp, gen, self._characters("Marella"))
            prompt = gen.calls[0]["prompt"]
            self.assertIn("Image 1: Character sheet for Marella", prompt)
            self.assertNotIn("Image 0:", prompt)
            self.assertTrue(prompt.rstrip().endswith("Wide shot of the shop."))

    def test_forensics_file_records_sent_input_and_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _CountingImageGenerator()
            self._run(tmp, gen, self._characters("Marella"))
            path = os.path.join(tmp, "shots", "0", "first_frame_nbpro_input.json")
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
            self.assertEqual(
                record["sent_input"]["image_urls"],
                gen.calls[0]["reference_image_paths"],
            )
            self.assertEqual(
                [s["role"] for s in record["slots"]],
                ["character_sheet", "environment_plate", "continuity_anchor"],
            )
            self.assertEqual(record["degrade_notices"], [])

    def test_legacy_three_view_registry_fails_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _CountingImageGenerator()
            pipeline = _pipeline(tmp, gen)
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}
            legacy = {"Marella": {"front": {"path": "/w/f.png", "description": "f"}}}
            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(pipeline.generate_frame_for_single_shot(
                    shot_idx=0, frame_type="first_frame",
                    camera_anchor=("/w/a.png", "anchor"), frame_desc="ff",
                    visible_characters=self._characters("Marella"),
                    character_portraits_registry=legacy,
                    environments=self.environments,
                    environment_registry=self.env_registry, env_idx=0,
                ))
            self.assertIn("no `sheet` entry", str(ctx.exception))
            self.assertEqual(gen.calls, [], "must fail before spending")

    def test_fourth_character_drop_notice_reaches_stdout(self):
        """The notice must be visible to an operator, not just returned."""
        with tempfile.TemporaryDirectory() as tmp:
            gen = _CountingImageGenerator()
            registry = {
                f"C{i}": {"sheet": {"path": f"/w/c{i}.png", "description": f"sheet {i}"}}
                for i in range(4)
            }
            pipeline = _pipeline(tmp, gen)
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}
            out = io.StringIO()
            with redirect_stdout(out):
                asyncio.run(pipeline.generate_frame_for_single_shot(
                    shot_idx=0, frame_type="first_frame",
                    camera_anchor=("/w/a.png", "anchor"), frame_desc="ff",
                    visible_characters=self._characters("C0", "C1", "C2", "C3"),
                    character_portraits_registry=registry,
                    environments=self.environments,
                    environment_registry=self.env_registry, env_idx=0,
                ))
            printed = out.getvalue()
            self.assertIn("DEGRADED", printed)
            self.assertIn("identity-merge", printed)
            self.assertEqual(len(gen.calls[0]["reference_image_paths"]), 5)

    def test_missing_env_idx_fails_loud_before_spending(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _CountingImageGenerator()
            pipeline = _pipeline(tmp, gen)
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}
            with self.assertRaises(RuntimeError):
                asyncio.run(pipeline.generate_frame_for_single_shot(
                    shot_idx=0, frame_type="first_frame",
                    camera_anchor=("/w/a.png", "anchor"), frame_desc="ff",
                    visible_characters=self._characters("Marella"),
                    character_portraits_registry=self.registry,
                    environments=self.environments,
                    environment_registry=self.env_registry, env_idx=None,
                ))
            self.assertEqual(gen.calls, [])

    def test_no_selector_call_when_seedbed_pool_is_empty(self):
        """The old path ran the selector unconditionally, including when there
        was nothing to choose between."""
        with tempfile.TemporaryDirectory() as tmp:
            gen = _CountingImageGenerator()
            pipeline = _pipeline(tmp, gen)
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}

            class _ExplodingSelector:
                async def select_reference_images_and_generate_prompt(self, **kw):
                    raise AssertionError("selector must not run on an empty pool")

            pipeline.reference_image_selector = _ExplodingSelector()
            with redirect_stdout(io.StringIO()):
                asyncio.run(pipeline.generate_frame_for_single_shot(
                    shot_idx=0, frame_type="first_frame",
                    camera_anchor=("/w/a.png", "anchor"), frame_desc="ff",
                    visible_characters=self._characters("Marella"),
                    character_portraits_registry=self.registry,
                    environments=self.environments,
                    environment_registry=self.env_registry, env_idx=0,
                    seedbed_registry=[],
                ))
            self.assertEqual(len(gen.calls), 1)

    def test_seedbed_references_fill_remaining_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _CountingImageGenerator()
            pipeline = _pipeline(tmp, gen)
            pipeline.frame_events[0] = {"first_frame": asyncio.Event()}

            class _TakeAllSelector:
                async def select_reference_images_and_generate_prompt(
                    self, available_image_path_and_text_pairs, frame_description,
                ):
                    return {
                        "reference_image_path_and_text_pairs":
                            [list(p) for p in available_image_path_and_text_pairs],
                        "text_prompt": "SELECTOR_GUIDANCE",
                    }

            pipeline.reference_image_selector = _TakeAllSelector()
            seedbed = [{
                "path": "/w/seedbed/0_set.png", "role": "reference",
                "hint": "location", "description": "wood-panelled kitchen",
                "note": None, "url": "u",
            }]
            with redirect_stdout(io.StringIO()):
                asyncio.run(pipeline.generate_frame_for_single_shot(
                    shot_idx=0, frame_type="first_frame",
                    camera_anchor=("/w/a.png", "anchor"), frame_desc="ff",
                    visible_characters=self._characters("Marella"),
                    character_portraits_registry=self.registry,
                    environments=self.environments,
                    environment_registry=self.env_registry, env_idx=0,
                    seedbed_registry=seedbed,
                ))
            call = gen.calls[0]
            self.assertIn("/w/seedbed/0_set.png", call["reference_image_paths"])
            self.assertIn("Location reference: wood-panelled kitchen", call["prompt"])
            self.assertIn("SELECTOR_GUIDANCE", call["prompt"])


class TestIngestSeedAssets(unittest.TestCase):
    def test_bare_urls_ingested_as_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            downloaded = []

            def _fake_download(url, save_path):
                downloaded.append((url, save_path))
                _write_png(save_path)

            import pipelines.script2video_pipeline as mod
            original = mod.download_image
            mod.download_image = _fake_download
            try:
                with redirect_stdout(io.StringIO()):
                    registry = asyncio.run(pipeline.ingest_seed_assets([
                        "https://x.test/seedbed/tx_cooking_set.jpg",
                        "https://x.test/seedbed/tx_cooking_title.jpg",
                    ]))
            finally:
                mod.download_image = original

            self.assertEqual(len(registry), 2)
            self.assertTrue(all(e["role"] == "reference" for e in registry))
            self.assertTrue(all(e["description"] is None for e in registry))
            self.assertTrue(registry[0]["path"].endswith("0_tx_cooking_set.png"))
            self.assertEqual(len(downloaded), 2)

            with open(os.path.join(tmp, "seedbed_registry.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f), registry)

    def test_pin_without_shot_idx_raises_at_ingest(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            with self.assertRaises(Exception) as ctx:
                asyncio.run(pipeline.ingest_seed_assets([
                    {"url": "https://x.test/a.jpg", "role": "pin"},
                ]))
            self.assertIn("requires shot_idx", str(ctx.exception))

    def test_no_assets_produces_empty_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            self.assertEqual(asyncio.run(pipeline.ingest_seed_assets(None)), [])
            self.assertFalse(
                os.path.exists(os.path.join(tmp, "seedbed_registry.json")),
            )


if __name__ == "__main__":
    unittest.main()
