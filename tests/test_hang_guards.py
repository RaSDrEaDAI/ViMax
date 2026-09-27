"""Regression tests for unbounded retry/polling loops and LLM-trusted graphs.

Ported from upstream hkuds/vimax c061793 (tests/test_hang_guards.py) and
adapted to this fork. The bugs under test previously looped forever. Each
fake succeeds after N calls, so buggy code produces a fast assertion failure
(extra calls or a missing exception) rather than hanging the suite; the fixed
code must give up before the fake ever succeeds.
"""

import json
import os
import tempfile
import time
import unittest

from agents.camera_image_generator import (
    CameraImageGenerator,
    CameraParentItem,
    CameraTreeResponse,
    _validate_camera_tree,
)
from interfaces.camera import Camera
from interfaces.shot_description import ShotDescription
from tests.fakes import FakeChatModel


def _camera(idx, parent=None):
    # Each camera films exactly the shot with its own index, so a valid parent
    # link is (parent_cam_idx=p, parent_shot_idx=p).
    return Camera(
        idx=idx,
        active_shot_idxs=[idx],
        parent_cam_idx=parent,
        parent_shot_idx=parent,
    )


def _shot(idx, cam_idx):
    return ShotDescription(
        idx=idx, is_last=False, cam_idx=cam_idx, visual_desc=f"shot {idx}",
        variation_type="small", variation_reason="r",
        ff_desc="ff", ff_vis_char_idxs=[], lf_desc="lf", lf_vis_char_idxs=[],
        motion_desc="m", audio_desc="a",
    )


def _item(parent_cam_idx=None, parent_shot_idx=None):
    return CameraParentItem(
        parent_cam_idx=parent_cam_idx, parent_shot_idx=parent_shot_idx, reason="r",
    )


class TestCameraTreeValidation(unittest.TestCase):
    def test_valid_chain_passes(self):
        _validate_camera_tree([_camera(0), _camera(1, parent=0), _camera(2, parent=1)])

    def test_single_root_passes(self):
        _validate_camera_tree([_camera(0)])

    def test_two_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "[Cc]ycle"):
            _validate_camera_tree([_camera(0, parent=1), _camera(1, parent=0)])

    def test_three_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "[Cc]ycle"):
            _validate_camera_tree([_camera(0, parent=2), _camera(1, parent=0), _camera(2, parent=1)])

    def test_cycle_off_the_root_is_rejected(self):
        # Root is fine; cameras 1 and 2 point at each other.
        with self.assertRaisesRegex(ValueError, "[Cc]ycle"):
            _validate_camera_tree([_camera(0), _camera(1, parent=2), _camera(2, parent=1)])

    def test_self_parent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "itself"):
            _validate_camera_tree([_camera(0, parent=0)])

    def test_unknown_parent_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown parent"):
            _validate_camera_tree([_camera(0), _camera(1, parent=7)])

    # -- fork-only checks: the wait keys on parent_shot_idx ------------------

    def test_parent_shot_not_filmed_by_parent_is_rejected(self):
        # Camera 1 names camera 0 but depends on its OWN shot 1: it would wait
        # on an event only it can set.
        cams = [_camera(0), Camera(idx=1, active_shot_idxs=[1], parent_cam_idx=0, parent_shot_idx=1)]
        with self.assertRaisesRegex(ValueError, "does not film"):
            _validate_camera_tree(cams)

    def test_half_specified_parent_is_rejected(self):
        cams = [_camera(0), Camera(idx=1, active_shot_idxs=[1], parent_cam_idx=None, parent_shot_idx=0)]
        with self.assertRaisesRegex(ValueError, "both"):
            _validate_camera_tree(cams)

    def test_duplicate_camera_idx_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _validate_camera_tree([_camera(0), _camera(0)])


class TestConstructCameraTree(unittest.IsolatedAsyncioTestCase):
    def _generator(self, responses):
        model = FakeChatModel(responses)
        return CameraImageGenerator(chat_model=model, image_generator=None, video_generator=None), model

    def _cameras(self):
        return [Camera(idx=0, active_shot_idxs=[0]), Camera(idx=1, active_shot_idxs=[1])]

    async def test_length_mismatch_raises_after_bounded_retries(self):
        bad = CameraTreeResponse(camera_parent_items=[_item()])  # 1 item for 2 cameras
        gen, model = self._generator([bad])
        with self.assertRaisesRegex(ValueError, "1 items for 2 cameras"):
            await gen.construct_camera_tree(cameras=self._cameras(), shot_descs=[_shot(0, 0), _shot(1, 1)])
        self.assertEqual(model.calls, 3, "retry must be bounded at 3 attempts")

    async def test_cyclic_answer_is_re_asked(self):
        cyclic = CameraTreeResponse(camera_parent_items=[_item(1, 1), _item(0, 0)])
        good = CameraTreeResponse(camera_parent_items=[_item(), _item(0, 0)])
        gen, model = self._generator([cyclic, good])
        cams = await gen.construct_camera_tree(cameras=self._cameras(), shot_descs=[_shot(0, 0), _shot(1, 1)])
        self.assertEqual(model.calls, 2)
        self.assertEqual([c.parent_cam_idx for c in cams], [None, 0])

    async def test_persistent_cycle_fails_loud_not_hangs(self):
        cyclic = CameraTreeResponse(camera_parent_items=[_item(1, 1), _item(0, 0)])
        gen, model = self._generator([cyclic])
        with self.assertRaisesRegex(ValueError, "[Cc]ycle"):
            await gen.construct_camera_tree(cameras=self._cameras(), shot_descs=[_shot(0, 0), _shot(1, 1)])
        self.assertEqual(model.calls, 3)


class TestResumedCameraTreeIsValidated(unittest.IsolatedAsyncioTestCase):
    """A hand-crafted cyclic camera_tree.json on disk must fail in seconds —
    the resume path used to load it unchecked and deadlock frame generation."""

    async def test_cyclic_tree_on_disk_raises(self):
        from pipelines.script2video_pipeline import Script2VideoPipeline

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "camera_tree.json"), "w", encoding="utf-8") as f:
                json.dump([_camera(0, parent=1).model_dump(), _camera(1, parent=0).model_dump()], f)
            pipeline = Script2VideoPipeline(
                chat_model=None, image_generator=None, video_generator=None, working_dir=tmp,
            )
            start = time.monotonic()
            with self.assertRaisesRegex(ValueError, "[Cc]ycle"):
                await pipeline.construct_camera_tree(shot_descriptions=[_shot(0, 0), _shot(1, 1)])
            self.assertLess(time.monotonic() - start, 5)


if __name__ == "__main__":
    unittest.main()
