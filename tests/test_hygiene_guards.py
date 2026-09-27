"""Regression tests for rate-limiter lock behaviour and media-handle cleanup.

Ported from upstream hkuds/vimax 333b5f4 (tests/test_hygiene_guards.py) and
adapted to this fork: the concat helper here is utils.video.
concatenate_shot_videos (logger=None, lazy moviepy import), and the packaging /
MiniMax-template / suite-isolation checks upstream bundled in the same file are
out of scope for this port.
"""

import asyncio
import os
import tempfile
import unittest
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from utils.rate_limiter import RateLimiter
from utils.video import concatenate_shot_videos

_real_sleep = asyncio.sleep  # captured before any test patches asyncio.sleep


class _FakeClock:
    """time.time + asyncio.sleep stand-ins: sleeping advances the clock
    instantly and records how long each caller asked to wait."""

    def __init__(self, start=1_000_000.0):
        self.now = start
        self.sleeps = []

    def time(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        await _real_sleep(0)  # yield, like a real sleep would


class TestRateLimiterLocking(unittest.IsolatedAsyncioTestCase):
    async def test_waiting_acquirer_does_not_hold_the_lock(self):
        limiter = RateLimiter(max_requests_per_minute=1)
        await limiter.acquire()  # consume the only slot in the window
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0.05)  # waiter is now waiting for the window to free
        try:
            try:
                await asyncio.wait_for(limiter.lock.acquire(), timeout=0.25)
                limiter.lock.release()
            except asyncio.TimeoutError:
                self.fail("rate limiter sleeps while holding its lock, blocking every other caller")
        finally:
            waiter.cancel()
            with suppress(asyncio.CancelledError):
                await waiter

    async def test_second_acquirer_is_not_serialized_behind_a_long_wait(self):
        # Fake clock: caller A must wait out a 24h daily window. Before the fix
        # A slept INSIDE the lock, so B could not even run its check until A
        # woke. Now B's check completes while A is still asleep.
        clock = _FakeClock()
        limiter = RateLimiter(max_requests_per_minute=None, max_requests_per_day=1)
        limiter.request_times = [clock.now - 10]  # daily budget already spent
        a_asleep = asyncio.Event()
        release_a = asyncio.Event()

        async def a_sleep(seconds):
            clock.sleeps.append(seconds)
            a_asleep.set()
            await release_a.wait()
            clock.now += seconds

        with patch("utils.rate_limiter.time.time", side_effect=clock.time), \
             patch("utils.rate_limiter.asyncio.sleep", side_effect=a_sleep):
            a = asyncio.create_task(limiter.acquire())
            await asyncio.wait_for(a_asleep.wait(), timeout=1)
            # B only needs the lock to decide it must wait too; that decision
            # must not be blocked by A's 24h sleep.
            await asyncio.wait_for(limiter.lock.acquire(), timeout=0.25)
            limiter.lock.release()
            release_a.set()
            await asyncio.wait_for(a, timeout=1)
        self.assertAlmostEqual(clock.sleeps[0], 86400 - 10, delta=1)

    async def test_daily_limit_checked_before_per_minute(self):
        clock = _FakeClock()
        limiter = RateLimiter(max_requests_per_minute=10, max_requests_per_day=2)
        limiter.request_times = [clock.now - 100, clock.now - 50]
        with patch("utils.rate_limiter.time.time", side_effect=clock.time), \
             patch("utils.rate_limiter.asyncio.sleep", side_effect=clock.sleep):
            await limiter.acquire()
        self.assertAlmostEqual(clock.sleeps[0], 86400 - 100, delta=1)

    async def test_limits_rechecked_after_waking(self):
        # Two callers race for the one slot freed by a window: only one gets
        # it; the other must go back to sleep instead of over-admitting.
        clock = _FakeClock()
        limiter = RateLimiter(max_requests_per_minute=1)
        limiter.request_times = [clock.now]
        with patch("utils.rate_limiter.time.time", side_effect=clock.time), \
             patch("utils.rate_limiter.asyncio.sleep", side_effect=clock.sleep):
            await asyncio.gather(limiter.acquire(), limiter.acquire())
        in_any_minute = [t for t in limiter.request_times if clock.now - t < 60]
        self.assertLessEqual(len(in_any_minute), 1, f"over-admitted: {limiter.request_times}")

    async def test_min_delay_smoothing_still_applies(self):
        limiter = RateLimiter(max_requests_per_minute=600)  # min delay 0.1s
        loop = asyncio.get_running_loop()
        start = loop.time()
        await limiter.acquire()
        await limiter.acquire()
        self.assertGreaterEqual(loop.time() - start, 0.08)

    async def test_disabled_limiter_returns_immediately(self):
        limiter = RateLimiter()
        with patch("utils.rate_limiter.asyncio.sleep", new=AsyncMock()) as sleep:
            for _ in range(5):
                await limiter.acquire()
        sleep.assert_not_awaited()
        self.assertEqual(limiter.request_times, [])


class TestVideoConcatenationCleanup(unittest.TestCase):
    def _run(self, clips, final, **kw):
        with patch("moviepy.VideoFileClip", side_effect=clips), \
             patch("moviepy.concatenate_videoclips", return_value=final):
            return concatenate_shot_videos([f"{i}.mp4" for i in range(len(clips))], "out.mp4", **kw)

    def test_all_readers_closed_on_success(self):
        clips = [MagicMock(), MagicMock()]
        final = MagicMock()
        self._run(clips, final)
        final.write_videofile.assert_called_once()
        self.assertIsNone(final.write_videofile.call_args.kwargs["logger"])
        final.close.assert_called_once()
        for clip in clips:
            clip.close.assert_called_once()

    def test_all_readers_closed_when_write_fails(self):
        clips = [MagicMock(), MagicMock()]
        final = MagicMock()
        final.write_videofile.side_effect = OSError("disk full")
        with self.assertRaises(OSError):
            self._run(clips, final)
        final.close.assert_called_once()
        for clip in clips:
            clip.close.assert_called_once()

    def test_earlier_readers_closed_when_a_later_open_fails(self):
        # The old list comprehension ran OUTSIDE the try: clip 0 leaked.
        first = MagicMock()
        with self.assertRaises(OSError):
            self._run([first, OSError("corrupt mp4")], MagicMock())
        first.close.assert_called_once()


class TestNewCameraImageClosesClips(unittest.TestCase):
    def _generator(self):
        from agents.camera_image_generator import CameraImageGenerator
        return CameraImageGenerator(chat_model=None, image_generator=None, video_generator=None)

    def _clip(self):
        clip = MagicMock(duration=2.0, fps=24)
        clip.__enter__.return_value = clip
        clip.get_frame.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
        return clip

    def _run(self, second_scene_exists):
        clip = self._clip()
        with tempfile.TemporaryDirectory() as tmp:
            transition = os.path.join(tmp, "transition_video_from_shot_0.mp4")
            open(transition, "wb").close()
            if second_scene_exists:
                os.makedirs(os.path.join(tmp, "cache"))
                open(os.path.join(tmp, "cache", "transition_video_from_shot_0-Scene-002.mp4"), "wb").close()
            with patch("agents.camera_image_generator.open_video"), \
                 patch("agents.camera_image_generator.SceneManager"), \
                 patch("agents.camera_image_generator.split_video_ffmpeg"), \
                 patch("agents.camera_image_generator.VideoFileClip", return_value=clip):
                self._generator().get_new_camera_image(transition)
        return clip

    def test_second_scene_clip_closed(self):
        self._run(second_scene_exists=True).__exit__.assert_called_once()

    def test_transition_clip_closed(self):
        self._run(second_scene_exists=False).__exit__.assert_called_once()


class TestNanobananaClosesReferenceImages(unittest.IsolatedAsyncioTestCase):
    async def _assert_closed_on_failure(self, module, cls_name):
        mod = __import__(module, fromlist=[cls_name])
        gen = object.__new__(getattr(mod, cls_name))
        gen.model = "m"
        gen.rate_limiter = None
        gen.client = MagicMock()
        gen.client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("upstream 500"))
        opened = []

        def fake_open(path):
            img = MagicMock()
            opened.append(img)
            return img

        with patch(f"{module}.Image.open", side_effect=fake_open):
            with self.assertRaises(Exception):
                await gen.generate_single_image(prompt="p", reference_image_paths=["a.png", "b.png"])
        self.assertTrue(opened)
        for img in opened:
            img.close.assert_called_once()

    async def test_google_client(self):
        await self._assert_closed_on_failure("tools.image_generator_nanobanana_google_api", "ImageGeneratorNanobananaGoogleAPI")

    async def test_yunwu_client(self):
        await self._assert_closed_on_failure("tools.image_generator_nanobanana_yunwu_api", "ImageGeneratorNanobananaYunwuAPI")


if __name__ == "__main__":
    unittest.main()
