"""Regression tests for silent wrong-output bugs in the script2video render path.

Ported from upstream hkuds/vimax df1480d (tests/test_wrong_output_guards.py)
and adapted to this fork. None of these bugs raised — they rendered the wrong
thing — so each test pins the observable choice, not just "no exception".
"""

import unittest

from interfaces.camera import Camera
from interfaces.shot_description import ShotDescription
from pipelines.script2video_pipeline import (
    _collect_priority_shot_idxs,
    _group_shots_into_cameras,
)


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


if __name__ == "__main__":
    unittest.main()
