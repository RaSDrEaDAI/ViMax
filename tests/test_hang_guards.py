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
from unittest.mock import AsyncMock, MagicMock, patch

import requests

from agents.camera_image_generator import (
    CameraImageGenerator,
    CameraParentItem,
    CameraTreeResponse,
    _validate_camera_tree,
)
from interfaces.camera import Camera
from interfaces.shot_description import ShotDescription
from tests.fakes import FakeChatModel
from utils.image import download_image
from utils.video import download_video


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


# ---------------------------------------------------------------------------
# Fence 5 — bounded downloads (B2)
# ---------------------------------------------------------------------------

def _no_sleep(fn):
    retrying = getattr(fn, "retry", None)
    if retrying is not None:
        retrying.sleep = lambda seconds: None


def _http_error(status):
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.HTTPError(str(status), response=MagicMock(status_code=status))
    return resp


class TestDownloadRetries(unittest.TestCase):
    def setUp(self):
        _no_sleep(download_image)
        _no_sleep(download_video)
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = os.path.join(self.tmp.name, "out.bin")

    def tearDown(self):
        self.tmp.cleanup()

    def _counting(self, script):
        calls = {"n": 0}

        def fake_get(url, **kwargs):
            calls["n"] += 1
            item = script[min(calls["n"] - 1, len(script) - 1)]
            if isinstance(item, BaseException):
                raise item
            return item
        return fake_get, calls

    def test_image_gives_up_on_persistent_network_error(self):
        get, calls = self._counting([requests.ConnectionError("refused")] * 9 + [MagicMock()])
        with patch("utils.image.requests.get", side_effect=get):
            with self.assertRaises(requests.ConnectionError):
                download_image("http://example.com/a.png", self.dest)
        self.assertEqual(calls["n"], 3, "retry must be bounded, not retry-until-success")

    def test_expired_signed_url_fails_after_one_attempt(self):
        get, calls = self._counting([_http_error(403), _http_error(403), MagicMock()])
        with patch("utils.image.requests.get", side_effect=get):
            with self.assertRaises(requests.HTTPError):
                download_image("http://example.com/expired.png", self.dest)
        self.assertEqual(calls["n"], 1, "4xx responses must fail fast, not be retried")

    def test_503_retries_then_raises_after_three(self):
        get, calls = self._counting([_http_error(503)] * 5 + [MagicMock()])
        with patch("utils.video.requests.get", side_effect=get):
            with self.assertRaises(requests.HTTPError):
                download_video("http://example.com/a.mp4", self.dest)
        self.assertEqual(calls["n"], 3)

    def test_video_gives_up_on_persistent_network_error(self):
        get, calls = self._counting([requests.ConnectionError("refused")] * 9 + [MagicMock()])
        with patch("utils.video.requests.get", side_effect=get):
            with self.assertRaises(requests.ConnectionError):
                download_video("http://example.com/a.mp4", self.dest)
        self.assertEqual(calls["n"], 3)

    def test_downloads_set_a_timeout(self):
        for module, fn in (("utils.image", download_image), ("utils.video", download_video)):
            captured = {}

            def record_get(url, **kwargs):
                captured.update(kwargs)
                resp = MagicMock()
                resp.iter_content.return_value = [b"x"]
                return resp

            with patch(f"{module}.requests.get", side_effect=record_get):
                fn("http://example.com/a", self.dest)
            self.assertIsNotNone(captured.get("timeout"), f"{module}: requests.get must not wait forever")


class TestNoBareRetry(unittest.TestCase):
    def test_script_planner_retry_is_bounded(self):
        from agents.script_planner import ScriptPlanner
        stop = ScriptPlanner.plan_script.retry.stop
        self.assertIsNot(type(stop).__name__, "_stop_never", "plan_script must not retry forever")
        self.assertEqual(getattr(stop, "max_attempt_number", None), 3)


# ---------------------------------------------------------------------------
# Fence 5 — bounded generator create/poll loops (B3)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self.payload

    async def text(self):
        return json.dumps(self.payload)


class _FakeSession:
    """Shared across ClientSession() calls: returns each scripted item in turn
    (repeating the last). An item that is an exception is raised."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def _next(self):
        item = self.scripted[min(self.calls, len(self.scripted) - 1)]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        return _FakeResponse(*item)

    def post(self, url, **kwargs):
        return self._next()

    def get(self, url, **kwargs):
        return self._next()


def _patch_http(module, session):
    return patch(f"{module}.aiohttp.ClientSession", new=session), \
        patch(f"{module}.asyncio.sleep", new=AsyncMock())


class TestSeedanceClientBounds(unittest.IsolatedAsyncioTestCase):
    MOD = "tools.video_generator_doubao_seedance_yunwu_api"

    def _gen(self, **kw):
        from tools.video_generator_doubao_seedance_yunwu_api import VideoGeneratorDoubaoSeedanceYunwuAPI
        return VideoGeneratorDoubaoSeedanceYunwuAPI(api_key="k", **kw)

    async def test_create_auth_error_fails_fast_with_body(self):
        session = _FakeSession([({"error": "invalid api key"}, 401), ({"id": "task-1"}, 200)])
        a, b = _patch_http(self.MOD, session)
        with a, b:
            with self.assertRaisesRegex(RuntimeError, "401.*invalid api key"):
                await self._gen().create_video_generation_task("a prompt", [])
        self.assertEqual(session.calls, 1, "4xx must not be retried")

    async def test_create_5xx_retries_then_raises(self):
        session = _FakeSession([({"error": "busy"}, 503)] * 5 + [({"id": "t"}, 200)])
        a, b = _patch_http(self.MOD, session)
        with a, b:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                await self._gen().create_video_generation_task("a prompt", [])
        self.assertEqual(session.calls, 3)

    async def test_poll_exceeding_max_attempts_times_out(self):
        session = _FakeSession([({"status": "queued"}, 200)])
        a, b = _patch_http(self.MOD, session)
        with a, b:
            with self.assertRaises(TimeoutError):
                await self._gen(max_poll_attempts=3).query_video_generation_task("task-1")
        self.assertEqual(session.calls, 3)

    async def test_poll_consecutive_errors_raise(self):
        session = _FakeSession([OSError("reset")] * 10 + [({"status": "succeeded"}, 200)])
        a, b = _patch_http(self.MOD, session)
        with a, b:
            with self.assertRaisesRegex(RuntimeError, "5 times in a row"):
                await self._gen().query_video_generation_task("task-1")
        self.assertEqual(session.calls, 5)

    async def test_poll_happy_path_returns_url(self):
        session = _FakeSession([({"status": "running"}, 200),
                                ({"status": "succeeded", "content": {"video_url": "u"}}, 200)])
        a, b = _patch_http(self.MOD, session)
        with a, b:
            self.assertEqual(await self._gen().query_video_generation_task("t"), "u")


class TestVeoYunwuClientBounds(unittest.IsolatedAsyncioTestCase):
    MOD = "tools.video_generator_veo_yunwu_api"

    def _gen(self, **kw):
        from tools.video_generator_veo_yunwu_api import VideoGeneratorVeoYunwuAPI
        return VideoGeneratorVeoYunwuAPI(api_key="k", **kw)

    async def test_failed_task_raises_instead_of_returning_none(self):
        session = _FakeSession([({"id": "t1"}, 200), ({"status": "failed", "detail": "nsfw"}, 200)])
        a, b = _patch_http(self.MOD, session)
        with a, b:
            with self.assertRaisesRegex(RuntimeError, "failed"):
                await self._gen().generate_single_video(prompt="p", reference_image_paths=[])

    async def test_create_auth_error_fails_fast(self):
        session = _FakeSession([({"error": "bad key"}, 401), ({"id": "t1"}, 200)])
        a, b = _patch_http(self.MOD, session)
        with a, b:
            with self.assertRaises(RuntimeError):
                await self._gen().generate_single_video(prompt="p", reference_image_paths=[])
        self.assertEqual(session.calls, 1)

    async def test_poll_is_bounded(self):
        session = _FakeSession([({"id": "t1"}, 200), ({"status": "processing"}, 200)])
        a, b = _patch_http(self.MOD, session)
        with a, b:
            with self.assertRaises(TimeoutError):
                await self._gen(max_poll_attempts=4).generate_single_video(prompt="p", reference_image_paths=[])
        self.assertEqual(session.calls, 1 + 4)


class TestVeoGoogleClientBounds(unittest.IsolatedAsyncioTestCase):
    async def test_poll_is_bounded(self):
        from tools.video_generator_veo_google_api import VideoGeneratorVeoGoogleAPI
        gen = VideoGeneratorVeoGoogleAPI(api_key="k", max_poll_attempts=3)
        never_done = MagicMock(done=False)
        gen.client = MagicMock()
        gen.client.models.generate_videos.return_value = never_done
        gen.client.operations.get.return_value = never_done
        with patch("tools.video_generator_veo_google_api.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(TimeoutError):
                await gen.generate_single_video(prompt="p", reference_image_paths=[])
        self.assertEqual(gen.client.operations.get.call_count, 3)


class TestComfyUIPollBounds(unittest.IsolatedAsyncioTestCase):
    MOD = "tools.comfyui_client"

    def _client(self, **kw):
        from tools.comfyui_client import ComfyUIClient
        return ComfyUIClient(base_url="http://127.0.0.1:1", client_id="t", **kw)

    async def test_persistent_non_200_raises_instead_of_hanging(self):
        # Before: `continue` skipped the deadline check, so this looped forever.
        session = _FakeSession([({"error": "unauthorized"}, 401)])
        a, b = _patch_http(self.MOD, session)
        with a, b:
            with self.assertRaisesRegex(RuntimeError, "401"):
                await self._client(max_consecutive_poll_http_errors=4).wait_for_completion("p1")
        self.assertEqual(session.calls, 4)

    async def test_intermittent_proxy_401_is_tolerated(self):
        done = {"p1": {"status": {"completed": True}, "outputs": {}}}
        session = _FakeSession([({"error": "unauthorized"}, 401), ({}, 200), (done, 200)])
        a, b = _patch_http(self.MOD, session)
        with a, b:
            entry = await self._client().wait_for_completion("p1")
        self.assertTrue(entry["status"]["completed"])

    async def test_non_200_honours_overall_deadline(self):
        session = _FakeSession([({"error": "bad gateway"}, 502)])
        a, b = _patch_http(self.MOD, session)
        with a, b:
            with self.assertRaises(TimeoutError):
                await self._client(request_timeout=6.0).wait_for_completion("p1", poll_interval=2.0)
        self.assertEqual(session.calls, 3)


class TestFalClientTimeouts(unittest.IsolatedAsyncioTestCase):
    """fal's subscribe_async waits indefinitely unless given client_timeout;
    a timeout must surface after ONE submission (a retry would re-bill)."""

    def _fake_fal(self, raises=None):
        fal = MagicMock()
        fal.upload_file_async = AsyncMock(return_value="https://fal.media/x.png")
        fal.subscribe_async = AsyncMock(
            side_effect=raises,
            return_value={"video": {"url": "https://fal.media/v.mp4"}, "images": [{"url": "https://fal.media/i.png"}]},
        )
        return fal

    async def test_video_client_passes_timeout_and_does_not_retry_it(self):
        from tools.video_generator_fal_ai import VideoGeneratorFalAI
        fal = self._fake_fal(raises=TimeoutError("fal queue stuck"))
        with patch.dict(os.environ, {}, clear=False), \
             patch("tools.video_generator_fal_ai._require_fal_client", return_value=fal), \
             patch("tools.video_generator_fal_ai.asyncio.sleep", new=AsyncMock()):
            gen = VideoGeneratorFalAI(api_key="k", client_timeout=42)
            with self.assertRaises(TimeoutError):
                await gen.generate_single_video(prompt="p", reference_image_paths=[])
        self.assertEqual(fal.subscribe_async.await_count, 1)
        self.assertEqual(fal.subscribe_async.await_args.kwargs["client_timeout"], 42)

    async def test_image_client_passes_timeout_and_does_not_retry_it(self):
        from tools.image_generator_fal_ai import ImageGeneratorFalAI
        fal = self._fake_fal(raises=TimeoutError("fal queue stuck"))
        with patch.dict(os.environ, {}, clear=False), \
             patch("tools.image_generator_fal_ai._require_fal_client", return_value=fal), \
             patch("tools.image_generator_fal_ai.asyncio.sleep", new=AsyncMock()):
            gen = ImageGeneratorFalAI(api_key="k", client_timeout=7)
            with self.assertRaises(TimeoutError):
                await gen.generate_single_image(prompt="p", reference_image_paths=[])
        self.assertEqual(fal.subscribe_async.await_count, 1)
        self.assertEqual(fal.subscribe_async.await_args.kwargs["client_timeout"], 7)

    def test_nb_pro_has_a_default_timeout(self):
        from tools.image_generator_nb_pro_fal import ImageGeneratorNanoBananaProFalAI
        with patch.dict(os.environ, {}, clear=False):
            self.assertIsNotNone(ImageGeneratorNanoBananaProFalAI(api_key="k").client_timeout)


# ---------------------------------------------------------------------------
# Fence 5 — is_last extraction caps
# ---------------------------------------------------------------------------

class TestEventExtractionCap(unittest.TestCase):
    def test_extraction_aborts_when_model_never_emits_is_last(self):
        from agents.event_extractor import EventExtractor
        from interfaces.event import Event
        extractor = object.__new__(EventExtractor)
        calls = {"n": 0}

        def never_last(novel_text, extracted_events):
            calls["n"] += 1
            if calls["n"] > 200:
                raise AssertionError("loop was not capped")
            return Event(
                index=len(extracted_events),
                is_last=False,
                description="an event",
                process_chain=["something happens"],
            )

        extractor.extract_next_event = never_last
        with self.assertRaisesRegex(RuntimeError, "maximum of 50"):
            extractor("some novel text")
        self.assertEqual(calls["n"], 50)


if __name__ == "__main__":
    unittest.main()
