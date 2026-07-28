"""Strip duplicated spoken lines out of the video prompt.

The video prompt is ``motion_desc + "\\n" + audio_desc``. ``audio_desc`` owns the
spoken words; ``motion_desc`` should describe only the visible mechanics of
speech. When both carry the same line, the model receives it twice and speaks it
twice — the doubled-VO symptom.

The prompts in ``interfaces/shot_description.py`` and ``agents/storyboard_artist``
now say this explicitly, and the studio render path (which never doubled) proves
the convention works. But prompt compliance is probabilistic and this defect is
silent and audible-only, so the concatenation point also enforces it.

Deliberately conservative: a quoted span is removed ONLY when the same text also
appears in ``audio_desc``. Quoted text with no audio counterpart is left alone —
a sign reading "OPEN", a book title, a brand name on a package are all legitimate
things to quote in a visual description, and stripping those would silently
degrade the frame.
"""

import re
from typing import List, Tuple

# Matches a quoted span using straight or curly double quotes. Non-greedy so
# adjacent quoted lines in one sentence stay separate spans.
_QUOTED = re.compile(r"[\"“]([^\"“”]{4,}?)[\"”]")

# The attribution punctuation immediately before a quote ("... says: ") — removed
# with the quote so the sentence doesn't end on a dangling colon.
_ATTRIBUTION_TAIL = re.compile(r"[,:—-]\s*$")


def _normalize(text: str) -> str:
    """Casefold, drop quotes/punctuation, squeeze whitespace, for comparison only."""
    stripped = re.sub(r"[^\w\s]", " ", text or "")
    return re.sub(r"\s+", " ", stripped).strip().casefold()


def find_duplicated_lines(motion_desc: str, audio_desc: str) -> List[str]:
    """Quoted spans in ``motion_desc`` whose words also appear in ``audio_desc``."""
    if not motion_desc or not audio_desc:
        return []
    audio_norm = _normalize(audio_desc)
    if not audio_norm:
        return []
    duplicated = []
    for match in _QUOTED.finditer(motion_desc):
        quoted_norm = _normalize(match.group(1))
        # Require a substantive overlap: very short fragments ("yes", "no") can
        # coincide across unrelated text, and removing those would be wrong.
        if len(quoted_norm) >= 8 and quoted_norm in audio_norm:
            duplicated.append(match.group(1))
    return duplicated


def strip_duplicated_speech(
    motion_desc: str,
    audio_desc: str,
) -> Tuple[str, List[str]]:
    """Remove from ``motion_desc`` any quoted line already carried by ``audio_desc``.

    Returns ``(cleaned_motion_desc, removed_lines)``. The caller logs the removals
    — a silent strip would hide a prompt-compliance regression, which is the thing
    most likely to bring this defect back.

    The speech ACT is preserved: "and speaks in a calm tone: <quote>" becomes
    "and speaks in a calm tone." So the model still animates the mouth, it just
    isn't told the words a second time.
    """
    if not motion_desc:
        return motion_desc, []

    duplicated = find_duplicated_lines(motion_desc, audio_desc)
    if not duplicated:
        return motion_desc, []

    # Mark each removed span with a sentinel first, then tidy the surrounding
    # punctuation in one pass. Doing it per-match would mean re-deriving offsets
    # into a string that keeps shifting under the edits.
    cleaned = motion_desc
    for line in duplicated:
        pattern = re.compile(r"[\"“]" + re.escape(line) + r"[\"”]")
        cleaned = pattern.sub("\x00", cleaned)

    # Drop the attribution punctuation that preceded each removed quote, so
    # "He says: <quote>" doesn't leave "He says: ." behind. When the sentence
    # continues afterwards, leave a comma: without it, `He says: "X" then a woman
    # replies` collapses to "He says then a woman replies", which reads as a
    # different clause than it is.
    def _close_gap(match):
        # Inspect what follows in the string being scanned. A lowercase word
        # means the sentence continues, so the clause needs a comma.
        rest = match.string[match.end():].lstrip()
        return ", " if rest[:1].islower() else ""

    cleaned = re.sub(r"[,:—-]?\s*\x00\s*", _close_gap, cleaned)
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"([^.!?])\s*$", r"\1.", cleaned)
    # A removed line can leave a doubled terminator ("speaks..").
    cleaned = re.sub(r"\.{2,}", ".", cleaned)

    return cleaned, duplicated


def build_video_prompt(motion_desc: str, audio_desc: str) -> Tuple[str, List[str]]:
    """Assemble the video prompt with duplicated speech removed.

    The ONE place motion_desc and audio_desc are joined for the video model.
    """
    cleaned, removed = strip_duplicated_speech(motion_desc, audio_desc)
    parts = [p for p in (cleaned, audio_desc) if p]
    return "\n".join(parts), removed
