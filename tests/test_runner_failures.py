"""Regression tests for two failures found in the orchestrator runner logs.

Both are reproduced from real `_result.json` records under
.working_dir/orchestrator/, and neither had any coverage before.

1. OSError [Errno 22] on the final concat (run b368d288, died at 3720.9s with
   every shot's video.mp4 already written and final_video: null). moviepy's
   default logger="bar" flushes sys.stdout; under the bridge stdout is a pipe and
   that flush fails on Windows.

2. TypeError: can only concatenate str (not "NoneType") to str at
   character_portraits_generator.py:49 (runs 47e37ece, 79be644b, ac428304,
   fc617145, all dead at ~45s). Visible non-human "characters" the extractor
   emits — DIAL, The Board, KNOB — legitimately have no wardrobe, so
   dynamic_features is null.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.character_portraits_generator import describe_features  # noqa: E402
from interfaces import CharacterInScene  # noqa: E402
from utils.video import concatenate_shot_videos  # noqa: E402


def _character(static=None, dynamic=None, name="Somebody", visible=True):
    return CharacterInScene(
        idx=0, identifier_in_scene=name, is_visible=visible,
        static_features=static, dynamic_features=dynamic,
    )


class TestDescribeFeatures(unittest.TestCase):
    """The exact shapes that appeared in the four failing runs' characters.json."""

    def test_both_present(self):
        got = describe_features(_character(static="tall, grey beard", dynamic="knit sweater"))
        self.assertEqual(got, "(static) tall, grey beard; (dynamic) knit sweater")

    def test_null_dynamic_does_not_raise(self):
        """DIAL / The Board: visible, described, but no wardrobe. This is the
        exact input that killed four runs."""
        got = describe_features(_character(static="a brass dial, worn edges", dynamic=None))
        self.assertEqual(got, "(static) a brass dial, worn edges")

    def test_null_static_does_not_raise(self):
        got = describe_features(_character(static=None, dynamic="red scarf"))
        self.assertEqual(got, "(dynamic) red scarf")

    def test_both_null_gets_an_explicit_instruction(self):
        got = describe_features(_character(static=None, dynamic=None))
        self.assertIn("not described", got)
        # The literal word "None" in a generation prompt is a real instruction
        # to the image model, so it must never be interpolated.
        self.assertNotIn("None", got)

    def test_empty_strings_treated_as_absent(self):
        got = describe_features(_character(static="", dynamic=""))
        self.assertIn("not described", got)

    def test_never_emits_the_word_none_for_any_combination(self):
        for static in (None, "", "described"):
            for dynamic in (None, "", "described"):
                got = describe_features(_character(static=static, dynamic=dynamic))
                self.assertNotIn("None", got, f"static={static!r} dynamic={dynamic!r}")
                self.assertTrue(got.strip(), "must never be blank")

    def test_front_portrait_prompt_builds_with_null_dynamic(self):
        """End to end through the prompt template — the original crash site."""
        import asyncio

        from agents.character_portraits_generator import CharacterPortraitsGenerator

        captured = {}

        class _Gen:
            async def generate_single_image(self, prompt, **kwargs):
                captured["prompt"] = prompt
                return MagicMock()

        gen = CharacterPortraitsGenerator(image_generator=_Gen())
        asyncio.run(gen.generate_front_portrait(
            _character(static="a brass dial", dynamic=None, name="DIAL"),
            style="painterly",
        ))
        self.assertIn("DIAL", captured["prompt"])
        self.assertIn("a brass dial", captured["prompt"])
        self.assertNotIn("None", captured["prompt"])


class TestConcatenateShotVideos(unittest.TestCase):
    """moviepy is patched at its source module so no encoding happens."""

    def _run(self, paths=("/w/0.mp4", "/w/1.mp4"), out="/w/final.mp4"):
        clips, final = [], MagicMock()

        def _clip(path):
            c = MagicMock(name=f"clip:{path}")
            c.path = path
            clips.append(c)
            return c

        with patch("moviepy.VideoFileClip", side_effect=_clip) as vfc, \
             patch("moviepy.concatenate_videoclips", return_value=final) as cat:
            result = concatenate_shot_videos(list(paths), out)
        return result, clips, final, vfc, cat

    def test_logger_is_none(self):
        """The fix. logger defaults to "bar", whose tqdm bar flushes sys.stdout
        and raises OSError [Errno 22] against the bridge's pipe on Windows."""
        _, _, final, _, _ = self._run()
        final.write_videofile.assert_called_once()
        self.assertIsNone(final.write_videofile.call_args.kwargs["logger"])

    def test_writes_to_the_requested_path(self):
        result, _, final, _, _ = self._run(out="/w/final.mp4")
        self.assertEqual(result, "/w/final.mp4")
        self.assertEqual(final.write_videofile.call_args.args[0], "/w/final.mp4")

    def test_codec_and_preset_match_moviepy_effective_defaults(self):
        """Named explicitly, not changed — codec=None infers libx264 for .mp4 and
        preset is already "medium", so output is unchanged from before the fix."""
        _, _, final, _, _ = self._run()
        kwargs = final.write_videofile.call_args.kwargs
        self.assertEqual(kwargs["codec"], "libx264")
        self.assertEqual(kwargs["preset"], "medium")

    def test_clips_opened_in_order(self):
        _, clips, _, _, cat = self._run(paths=("/w/0.mp4", "/w/1.mp4", "/w/2.mp4"))
        self.assertEqual([c.path for c in clips], ["/w/0.mp4", "/w/1.mp4", "/w/2.mp4"])
        self.assertEqual(cat.call_args.args[0], clips)

    def test_clips_closed_on_success(self):
        _, clips, _, _, _ = self._run()
        for c in clips:
            c.close.assert_called_once()

    def test_clips_closed_even_when_the_write_fails(self):
        """On Windows a leaked ffmpeg reader blocks the retry from overwriting
        the files it needs, turning one failure into a stuck working dir."""
        clips = []

        def _clip(path):
            c = MagicMock()
            clips.append(c)
            return c

        final = MagicMock()
        final.write_videofile.side_effect = OSError(22, "Invalid argument")

        with patch("moviepy.VideoFileClip", side_effect=_clip), \
             patch("moviepy.concatenate_videoclips", return_value=final):
            with self.assertRaises(OSError):
                concatenate_shot_videos(["/w/0.mp4", "/w/1.mp4"], "/w/final.mp4")

        self.assertEqual(len(clips), 2)
        for c in clips:
            c.close.assert_called_once()

    def test_close_failure_does_not_mask_the_real_error(self):
        clips = []

        def _clip(path):
            c = MagicMock()
            c.close.side_effect = RuntimeError("reader already gone")
            clips.append(c)
            return c

        final = MagicMock()
        final.write_videofile.side_effect = OSError(22, "Invalid argument")

        with patch("moviepy.VideoFileClip", side_effect=_clip), \
             patch("moviepy.concatenate_videoclips", return_value=final):
            with self.assertRaises(OSError) as ctx:
                concatenate_shot_videos(["/w/0.mp4"], "/w/final.mp4")
        self.assertEqual(ctx.exception.errno, 22)


class TestBothPipelinesUseTheHelper(unittest.TestCase):
    """Neither pipeline may call write_videofile directly again — the whole point
    of the helper is that logger=None cannot regress in one and not the other."""

    PIPELINES = ("pipelines/script2video_pipeline.py", "pipelines/idea2video_pipeline.py")

    def test_no_direct_write_videofile_calls(self):
        for rel in self.PIPELINES:
            with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
                source = f.read()
            self.assertNotIn("write_videofile", source, rel)

    def test_both_call_the_helper(self):
        for rel in self.PIPELINES:
            with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
                source = f.read()
            self.assertIn("concatenate_shot_videos(", source, rel)


if __name__ == "__main__":
    unittest.main()
