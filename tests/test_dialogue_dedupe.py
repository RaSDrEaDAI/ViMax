"""Tests for the doubled-VO guard.

Real strings from .working_dir/smoke_nbpro_paid (the doubling case) and from
.working_dir/studio/a7368326 (the studio render, which never doubled and is the
convention being ported).
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.dialogue import (  # noqa: E402
    build_video_prompt,
    find_duplicated_lines,
    strip_duplicated_speech,
)

# --- verbatim from the smoke run (the bug) ---------------------------------

SHOT1_MOTION = (
    'Static camera. The woman with short black hair, wearing a bright yellow rain '
    'slicker, half-turns her head over her shoulder to glance toward the off-frame '
    'left while holding a wooden spoon, and speaks in a calm, caring tone: '
    '"Soup\'s nearly done. You should eat before the next bell."'
)
SHOT1_AUDIO = (
    "[Speaker] Mira (Gentle, caring): Soup's nearly done. You should eat before "
    "the next bell."
)

SHOT2_MOTION = (
    'Static camera. The man with the full grey beard and cream-colored knit sweater '
    'keeps his eyes fixed on the lantern, his fingers turning a small screw. A hand '
    'in a bright yellow sleeve (Mira) enters from the left of the frame and sets a '
    'steaming bowl of soup down on the wooden table in front of him. His eyes flick '
    'briefly toward the bowl. He says: "The light comes first. It always has." then '
    'a woman replies off-screen: "Then eat fast."'
)
SHOT2_AUDIO = (
    "[Speaker] Tomas (Gruff, resolute): The light comes first. It always has. "
    "[Sound Effect] A ceramic bowl set down on wood. "
    "[Speaker] Mira (Firm, affectionate): Then eat fast."
)

SHOT0_MOTION = (
    "Static camera. On the left, the woman with short black hair slowly stirs the "
    "pot with a wooden spoon, steam rising softly from the pot."
)
SHOT0_AUDIO = (
    "[Sound Effect] Storm wind howling faintly outside, rain tapping against the "
    "round window, soup bubbling gently on the stove."
)

# --- verbatim from the studio render (the convention) ----------------------

STUDIO_MOTION = (
    "Static camera with persistent CRT scan-line roll and VHS chroma bleed. The "
    "radiant woman (feathered 80s hair, mint-green leotard, coral sweatband) snaps "
    "her head to the lens with a radiant smile, lips parted mid-count, then lowers "
    "her raised arms to extend them forward."
)
STUDIO_AUDIO = (
    "[Music] Bright synth-and-drum-machine aerobics track, slightly muffled as if "
    "through an old broadcast. [Speaker] Denise (breathless, bright, urgent "
    'encouragement): "—and reach! You\'re forgiving them, you\'re forgiving them, GOOD!"'
)


class TestFindDuplicatedLines(unittest.TestCase):
    def test_detects_the_smoke_shot1_duplicate(self):
        found = find_duplicated_lines(SHOT1_MOTION, SHOT1_AUDIO)
        self.assertEqual(len(found), 1)
        self.assertIn("Soup's nearly done", found[0])

    def test_detects_both_smoke_shot2_duplicates(self):
        found = find_duplicated_lines(SHOT2_MOTION, SHOT2_AUDIO)
        self.assertEqual(len(found), 2)
        self.assertIn("The light comes first", found[0])
        self.assertIn("Then eat fast", found[1])

    def test_sfx_only_shot_has_none(self):
        self.assertEqual(find_duplicated_lines(SHOT0_MOTION, SHOT0_AUDIO), [])

    def test_studio_convention_has_none(self):
        """The studio path quotes dialogue only in the audio half — nothing to
        strip. If this ever fails, the ported convention has regressed."""
        self.assertEqual(find_duplicated_lines(STUDIO_MOTION, STUDIO_AUDIO), [])

    def test_empty_inputs(self):
        self.assertEqual(find_duplicated_lines("", SHOT1_AUDIO), [])
        self.assertEqual(find_duplicated_lines(SHOT1_MOTION, ""), [])


class TestConservatism(unittest.TestCase):
    """Quoted text with no audio counterpart must survive untouched."""

    def test_sign_text_preserved(self):
        motion = 'Slow push-in on a neon sign reading "OPEN ALL NIGHT" above the door.'
        audio = "[Sound Effect] Buzzing neon, distant traffic."
        cleaned, removed = strip_duplicated_speech(motion, audio)
        self.assertEqual(removed, [])
        self.assertEqual(cleaned, motion)

    def test_title_card_preserved(self):
        motion = 'A lower-third title card wipes in reading "THE FORGIVENESS REACH."'
        audio = "[Music] Bright synth aerobics track."
        cleaned, removed = strip_duplicated_speech(motion, audio)
        self.assertEqual(removed, [])
        self.assertEqual(cleaned, motion)

    def test_short_fragments_not_stripped(self):
        """A 3-letter quote can coincide across unrelated text; removing it would
        be wrong."""
        motion = 'He points at the door marked "EXIT".'
        audio = "[Speaker] Guide (calm): Take the exit on your left."
        cleaned, removed = strip_duplicated_speech(motion, audio)
        self.assertEqual(removed, [])
        self.assertEqual(cleaned, motion)

    def test_partial_overlap_not_stripped(self):
        motion = 'She reads the label: "ORGANIC TOMATO PASSATA, 680G".'
        audio = "[Speaker] Cook (bored): Tomato."
        _, removed = strip_duplicated_speech(motion, audio)
        self.assertEqual(removed, [])


class TestStripping(unittest.TestCase):
    def test_shot1_keeps_the_speech_act_drops_the_words(self):
        cleaned, removed = strip_duplicated_speech(SHOT1_MOTION, SHOT1_AUDIO)
        self.assertEqual(len(removed), 1)
        # The visible mechanics survive — the model still animates the mouth.
        self.assertIn("speaks in a calm, caring tone", cleaned)
        self.assertIn("half-turns her head", cleaned)
        # The words are gone.
        self.assertNotIn("Soup's nearly done", cleaned)
        # No dangling attribution punctuation or stray quotes.
        self.assertNotIn(':"', cleaned)
        self.assertNotIn('"', cleaned)
        self.assertFalse(cleaned.rstrip().endswith(":"))
        self.assertTrue(cleaned.rstrip().endswith("."))

    def test_shot2_removes_both_lines(self):
        cleaned, removed = strip_duplicated_speech(SHOT2_MOTION, SHOT2_AUDIO)
        self.assertEqual(len(removed), 2)
        self.assertNotIn("The light comes first", cleaned)
        self.assertNotIn("Then eat fast", cleaned)
        # Surrounding action is intact.
        self.assertIn("sets a steaming bowl of soup down", cleaned)
        self.assertIn("He says", cleaned)
        self.assertIn("off-screen", cleaned)
        self.assertNotIn('"', cleaned)

    def test_mid_sentence_removal_keeps_the_clause_break(self):
        """Without a comma, `He says: "X" then a woman replies` collapses to
        "He says then a woman replies", which reads as a different clause."""
        cleaned, _ = strip_duplicated_speech(SHOT2_MOTION, SHOT2_AUDIO)
        self.assertIn("He says, then a woman replies off-screen.", cleaned)
        self.assertNotIn("saysthen", cleaned)
        self.assertNotIn("says then", cleaned)

    def test_end_of_sentence_removal_gets_no_stray_comma(self):
        cleaned, _ = strip_duplicated_speech(SHOT1_MOTION, SHOT1_AUDIO)
        self.assertTrue(cleaned.rstrip().endswith("caring tone."))
        self.assertNotIn(",.", cleaned)

    def test_no_double_periods_or_double_spaces(self):
        for motion, audio in ((SHOT1_MOTION, SHOT1_AUDIO), (SHOT2_MOTION, SHOT2_AUDIO)):
            cleaned, _ = strip_duplicated_speech(motion, audio)
            self.assertNotIn("..", cleaned)
            self.assertNotIn("  ", cleaned)
            self.assertNotIn(" .", cleaned)

    def test_clean_input_returned_unchanged(self):
        cleaned, removed = strip_duplicated_speech(STUDIO_MOTION, STUDIO_AUDIO)
        self.assertEqual(removed, [])
        self.assertEqual(cleaned, STUDIO_MOTION)

    def test_curly_quotes_handled(self):
        motion = "She turns and says: “The light comes first. It always has.”"
        audio = "[Speaker] Tomas: The light comes first. It always has."
        cleaned, removed = strip_duplicated_speech(motion, audio)
        self.assertEqual(len(removed), 1)
        self.assertNotIn("The light comes first", cleaned)


class TestBuildVideoPrompt(unittest.TestCase):
    def test_audio_desc_is_retained_verbatim(self):
        """audio_desc is the single authority for the words — it must never be
        stripped, only motion_desc is cleaned."""
        prompt, removed = build_video_prompt(SHOT1_MOTION, SHOT1_AUDIO)
        self.assertIn("Soup's nearly done. You should eat before the next bell.", prompt)
        # ...but exactly once, now.
        self.assertEqual(prompt.count("Soup's nearly done"), 1)
        self.assertEqual(len(removed), 1)

    def test_each_shot2_line_appears_once(self):
        prompt, _ = build_video_prompt(SHOT2_MOTION, SHOT2_AUDIO)
        self.assertEqual(prompt.count("The light comes first"), 1)
        self.assertEqual(prompt.count("Then eat fast"), 1)

    def test_sfx_only_prompt_is_the_plain_join(self):
        prompt, removed = build_video_prompt(SHOT0_MOTION, SHOT0_AUDIO)
        self.assertEqual(prompt, SHOT0_MOTION + "\n" + SHOT0_AUDIO)
        self.assertEqual(removed, [])

    def test_missing_audio_desc_does_not_crash(self):
        prompt, removed = build_video_prompt(SHOT0_MOTION, "")
        self.assertEqual(prompt, SHOT0_MOTION)
        self.assertEqual(removed, [])

    def test_missing_motion_desc_does_not_crash(self):
        prompt, removed = build_video_prompt("", SHOT0_AUDIO)
        self.assertEqual(prompt, SHOT0_AUDIO)
        self.assertEqual(removed, [])


class TestPromptConventionPorted(unittest.TestCase):
    """The field descriptions and agent prompts must carry the rule, since the
    guard is a backstop and the prompt is the primary fix."""

    def test_shot_description_fields_forbid_quoting_speech(self):
        with open(os.path.join(REPO_ROOT, "interfaces", "shot_description.py"),
                  encoding="utf-8") as f:
            source = f.read()
        self.assertIn("ONLY in audio_desc", source)
        self.assertIn("visible mechanics of speech", source)
        # The old instruction that caused the doubling must be gone.
        self.assertNotIn("please write down the content of the conversation", source)

    def test_storyboard_artist_prompts_forbid_quoting_speech(self):
        with open(os.path.join(REPO_ROOT, "agents", "storyboard_artist.py"),
                  encoding="utf-8") as f:
            source = f.read()
        self.assertIn("SPOKEN WORDS GO IN EXACTLY ONE PLACE", source)
        self.assertIn("NEVER write spoken words in the motion description", source)

    def test_pipeline_uses_the_guard_not_raw_concatenation(self):
        with open(os.path.join(REPO_ROOT, "pipelines", "script2video_pipeline.py"),
                  encoding="utf-8") as f:
            source = f.read()
        self.assertIn("build_video_prompt(", source)
        self.assertNotIn('motion_desc + "\\n" + shot_description.audio_desc', source)


if __name__ == "__main__":
    unittest.main()
