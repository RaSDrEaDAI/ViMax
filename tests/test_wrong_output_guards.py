"""Regression tests for silent wrong-output bugs in the script2video render path.

Ported from upstream hkuds/vimax df1480d (tests/test_wrong_output_guards.py)
and adapted to this fork. None of these bugs raised — they rendered the wrong
thing — so each test pins the observable choice, not just "no exception".
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from agents.reference_image_selector import (
    RefImageIndicesAndTextPrompt,
    ReferenceImageSelector,
    select_pairs_by_indices,
)
from agents.storyboard_artist import validate_char_idxs
from interfaces import CharacterInScene
from interfaces.camera import Camera
from interfaces.shot_description import ShotDescription
from pipelines.script2video_pipeline import (
    Script2VideoPipeline,
    _collect_priority_shot_idxs,
    _group_shots_into_cameras,
)
from tests.fakes import FakeChatModel
from utils.text import safe_path_component


def _shot(idx, cam_idx, variation_type="small", ff_chars=None, lf_chars=None):
    return ShotDescription(
        idx=idx,
        is_last=False,
        cam_idx=cam_idx,
        visual_desc=f"shot {idx}",
        variation_type=variation_type,
        variation_reason="r",
        ff_desc=f"first frame {idx}",
        ff_vis_char_idxs=ff_chars or [],
        lf_desc=f"last frame {idx}",
        lf_vis_char_idxs=lf_chars or [],
        motion_desc="m",
        audio_desc="a",
    )


class TestCameraGrouping(unittest.TestCase):
    def _by_idx(self, shots):
        return {c.idx: c.active_shot_idxs for c in _group_shots_into_cameras(shots)}

    def test_out_of_order_camera_indices_group_correctly(self):
        # Shot 0 uses camera 1, shot 1 camera 0, shot 2 camera 1 again. The old
        # positional lookup appended shot 2 to cameras[1] — which was camera 0.
        shots = [_shot(0, cam_idx=1), _shot(1, cam_idx=0), _shot(2, cam_idx=1)]
        self.assertEqual(self._by_idx(shots), {1: [0, 2], 0: [1]})

    def test_brief_example_2_0_1(self):
        shots = [_shot(0, cam_idx=2), _shot(1, cam_idx=0), _shot(2, cam_idx=1)]
        self.assertEqual(self._by_idx(shots), {2: [0], 0: [1], 1: [2]})

    def test_sparse_camera_indices_do_not_index_error(self):
        # Old code: cameras[5] on a 1-element list -> IndexError.
        shots = [_shot(0, cam_idx=5), _shot(1, cam_idx=5)]
        self.assertEqual(self._by_idx(shots), {5: [0, 1]})

    def test_in_order_indices_unchanged(self):
        # Happy path (every recorded run): identical to the old grouping.
        shots = [_shot(0, 0), _shot(1, 1), _shot(2, 0), _shot(3, 2), _shot(4, 1)]
        cams = _group_shots_into_cameras(shots)
        self.assertEqual([c.idx for c in cams], [0, 1, 2])
        self.assertEqual([c.active_shot_idxs for c in cams], [[0, 2], [1, 4], [3]])


class TestPriorityShotIdxs(unittest.TestCase):
    def test_priorities_are_shot_indices_not_camera_indices(self):
        # Camera 2 depends on shot 7 of camera 0: shot 7 must be prioritized.
        camera_tree = [
            Camera(idx=0, active_shot_idxs=[7, 8]),
            Camera(idx=2, active_shot_idxs=[9], parent_cam_idx=0, parent_shot_idx=7),
        ]
        self.assertEqual(_collect_priority_shot_idxs(camera_tree), [7])

    def test_roots_contribute_nothing(self):
        self.assertEqual(_collect_priority_shot_idxs([Camera(idx=0, active_shot_idxs=[0])]), [])


def _pipeline(working_dir):
    return Script2VideoPipeline(
        chat_model=MagicMock(),
        image_generator=MagicMock(),
        video_generator=MagicMock(),
        working_dir=working_dir,
    )


class _Stop(Exception):
    pass


class TestEventDictsAreInstanceState(unittest.IsolatedAsyncioTestCase):
    def test_two_pipelines_do_not_share_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = _pipeline(os.path.join(tmp, "a"))
            p2 = _pipeline(os.path.join(tmp, "b"))
            p1.frame_events[0] = {"first_frame": asyncio.Event()}
            p1.shot_desc_events[0] = asyncio.Event()
            p1.character_portrait_events[0] = asyncio.Event()
            self.assertEqual(p2.frame_events, {})
            self.assertEqual(p2.shot_desc_events, {})
            self.assertEqual(p2.character_portrait_events, {})

    def test_no_class_level_mutable_event_dicts(self):
        for name in ("frame_events", "shot_desc_events", "character_portrait_events"):
            self.assertNotIsInstance(
                Script2VideoPipeline.__dict__.get(name), dict,
                f"{name} must not be shared class state",
            )

    async def test_each_render_starts_with_empty_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            seen = []

            async def record_then_stop(seed_assets):
                seen.append((dict(pipeline.frame_events), dict(pipeline.shot_desc_events),
                             dict(pipeline.character_portrait_events)))
                raise _Stop()

            pipeline.ingest_seed_assets = record_then_stop
            for _ in range(2):
                # Leftovers from a previous render on the same instance.
                pipeline.frame_events[3] = {"first_frame": asyncio.Event()}
                pipeline.shot_desc_events[3] = asyncio.Event()
                pipeline.character_portrait_events[3] = asyncio.Event()
                with self.assertRaises(_Stop):
                    await pipeline(script="s", user_requirement="u", style="st")
            self.assertEqual(seen, [({}, {}, {}), ({}, {}, {})])


class TestResumeEqualsFresh(unittest.IsolatedAsyncioTestCase):
    """A4 (already fixed in this fork): a resumed camera must hand the keyframe
    the same new-camera continuity anchor a fresh run does."""

    async def _anchor_for(self, resume: bool):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            shots = [_shot(0, cam_idx=0), _shot(1, cam_idx=1)]
            camera = Camera(
                idx=1, active_shot_idxs=[1],
                parent_cam_idx=0, parent_shot_idx=0,
                missing_info="wrong background",
            )
            parent_done = asyncio.Event()
            parent_done.set()
            pipeline.frame_events = {0: {"first_frame": parent_done}, 1: {"first_frame": asyncio.Event()}}

            shot_dir = os.path.join(tmp, "shots", "1")
            os.makedirs(shot_dir, exist_ok=True)
            new_camera_path = os.path.join(shot_dir, "new_camera_1.png")
            open(os.path.join(shot_dir, "transition_video_from_shot_0.mp4"), "wb").close()
            if resume:
                open(new_camera_path, "wb").close()
            else:
                extracted = MagicMock()
                extracted.save.side_effect = lambda path: open(path, "wb").close()
                pipeline.camera_image_generator.get_new_camera_image = MagicMock(return_value=extracted)

            keyframe = AsyncMock()
            pipeline._generate_keyframe = keyframe
            await pipeline.generate_frames_for_single_camera(
                camera=camera,
                shot_descriptions=shots,
                characters=[],
                character_portraits_registry={},
                priority_shot_idxs=[],
            )
            keyframe.assert_awaited_once()
            anchor = keyframe.await_args.kwargs["continuity_anchor"]
            self.assertIsNotNone(anchor, "the new-camera composition reference was dropped")
            return os.path.relpath(anchor[0], tmp), anchor[1]

    async def test_resumed_camera_offers_new_camera_reference(self):
        path, _ = await self._anchor_for(resume=True)
        self.assertEqual(path, os.path.join("shots", "1", "new_camera_1.png"))

    async def test_resume_matches_fresh(self):
        self.assertEqual(await self._anchor_for(resume=True), await self._anchor_for(resume=False))


class TestCharIdxValidation(unittest.TestCase):
    def test_valid_indices_pass(self):
        validate_char_idxs([0, 1], 2, "ff_vis_char_idxs")
        validate_char_idxs([], 0, "ff_vis_char_idxs")

    def test_out_of_range_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[2\]"):
            validate_char_idxs([0, 2], 2, "ff_vis_char_idxs")

    def test_negative_rejected(self):
        with self.assertRaisesRegex(ValueError, "lf_vis_char_idxs"):
            validate_char_idxs([-1], 2, "lf_vis_char_idxs")


class TestResumedShotDescriptionIsValidated(unittest.IsolatedAsyncioTestCase):
    async def test_negative_char_idx_on_disk_raises(self):
        from interfaces import ShotBriefDescription
        import json
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp)
            pipeline.shot_desc_events[0] = asyncio.Event()
            path = os.path.join(tmp, "shots", "0", "shot_description.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_shot(0, 0, ff_chars=[-1]).model_dump(), f)
            brief = ShotBriefDescription(
                idx=0, is_last=True, cam_idx=0, visual_desc="v", audio_desc="a",
            )
            character = CharacterInScene(idx=0, identifier_in_scene="Mira", is_visible=True)
            with self.assertRaisesRegex(ValueError, "ff_vis_char_idxs"):
                await pipeline.decompose_visual_description_for_single_shot_brief_description(brief, [character])


class TestReferenceSelectorIndices(unittest.IsolatedAsyncioTestCase):
    def test_valid_selection(self):
        pairs = [("a.png", "a"), ("b.png", "b")]
        self.assertEqual(select_pairs_by_indices(pairs, [1]), [("b.png", "b")])

    def test_negative_index_rejected(self):
        with self.assertRaises(ValueError):
            select_pairs_by_indices([("a.png", "a")], [-1])

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            select_pairs_by_indices([("a.png", "a")], [3])

    async def test_selector_re_asks_then_fails_loud(self):
        # text_only path (the fork's third index site): [-1] used to return the
        # LAST image silently. Now each bad answer re-asks, bounded at 3.
        bad = RefImageIndicesAndTextPrompt(ref_image_indices=[-1], text_prompt="p")
        model = FakeChatModel([bad])
        selector = ReferenceImageSelector(chat_model=model, text_only=True)
        with self.assertRaises(Exception) as ctx:
            await selector.select_reference_images_and_generate_prompt(
                available_image_path_and_text_pairs=[("a.png", "a"), ("b.png", "b")],
                frame_description="f",
            )
        self.assertEqual(model.calls, 3)
        self.assertIn("out of range", repr(ctx.exception.last_attempt.exception()))

    async def test_selector_recovers_on_good_second_answer(self):
        bad = RefImageIndicesAndTextPrompt(ref_image_indices=[5], text_prompt="p")
        good = RefImageIndicesAndTextPrompt(ref_image_indices=[1], text_prompt="p2")
        selector = ReferenceImageSelector(chat_model=FakeChatModel([bad, good]), text_only=True)
        out = await selector.select_reference_images_and_generate_prompt(
            available_image_path_and_text_pairs=[("a.png", "a"), ("b.png", "b")],
            frame_description="f",
        )
        self.assertEqual(out["reference_image_path_and_text_pairs"], [("b.png", "b")])


class TestSafePathComponent(unittest.TestCase):
    def test_clean_names_unchanged(self):
        for name in ("Alice", "Bob_2", "Marella Lewis", "Dr. Chen", "Jean-Luc"):
            self.assertEqual(safe_path_component(name), name)

    def test_recorded_run_names_unchanged(self):
        # Real character dirs from .working_dir — must keep resolving on resume.
        for name in ("The Wednesday Chef (Marion)", "Contestant Three (Marla)", "O'Brien"):
            self.assertEqual(safe_path_component(name), name)

    def test_unicode_names_preserved(self):
        self.assertEqual(safe_path_component("李雷"), "李雷")

    def test_path_separators_removed(self):
        for name in ("a/b", "a\\b", "../Mira/Chen", "C:\\x"):
            cleaned = safe_path_component(name)
            self.assertNotIn("/", cleaned)
            self.assertNotIn("\\", cleaned)
            self.assertNotIn(":", cleaned)
        self.assertEqual(safe_path_component("../Mira/Chen"), "_Mira_Chen")

    def test_traversal_neutralized(self):
        cleaned = safe_path_component("../../etc/passwd")
        self.assertNotIn("/", cleaned)
        self.assertFalse(cleaned.startswith("."))

    def test_empty_becomes_placeholder(self):
        self.assertEqual(safe_path_component(""), "unnamed")
        self.assertEqual(safe_path_component("..."), "unnamed")


if __name__ == "__main__":
    unittest.main()
