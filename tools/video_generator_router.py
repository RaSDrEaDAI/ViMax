"""Video generator router: local LTX 2.3 default with smart paid-API escalation.

Philosophy: free first, paid only when needed. Every shot starts at the local
backend (LTX 2.3, free, fast). The router escalates to the paid API only when
one of these gates fires:

1. LIPSYNC         - shot has on-camera dialogue that needs mouth movement
                     (LTX can't lip-sync; Veo/Kling can)
2. MULTIREF        - 2+ reference frames (first+last); LTX is single-ref only
3. LARGE_VARIATION - variation_type == "large" (significant camera move /
                     composition change that LTX struggles with)
4. HIGH_COMPLEXITY - agent metadata explicitly flags complexity="high"
5. EXPLICIT        - caller passes escalate=True (manual override)
6. FAILURE_RETRY   - local generation raised; retry on paid backend

Heuristic gates (1-3) use prompt text + ref count + variation_type.
Agent metadata gate (4) is driven by Director/StoryboardArtist output.
Explicit override (5) always wins, in either direction.
Failure retry (6) only fires when the local backend raises.

Backwards compatibility:
  - Old configs using ``dialogue_backend`` / ``broll_backend`` /
    ``default_route="broll"`` still work (silently aliased).
  - Old call sites using ``has_dialogue=True/False`` still work.
  - The new preferred names are ``local_backend`` + ``escalation_backend``
    with ``default_route="local"``.

Per-shot metadata the router understands (all optional kwargs):
  - ``variation_type``: "small" | "medium" | "large" (from StoryboardArtist)
  - ``complexity``:     "low" | "medium" | "high" (agent-flagged)
  - ``requires_lipsync``: bool (agent-flagged, overrides lipsync heuristic)
  - ``escalate``:       bool (manual force-escalate or force-local)
  - ``audio_desc``:     str (used for lipsync heuristic if present)
  - ``motion_desc``:    str (used for complex-motion heuristic if present)
"""

import logging
import re
from typing import Any, List, Optional

from interfaces.video_output import VideoOutput


# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

# Lipsync detection: signals that the shot has on-camera speech that needs
# mouth movement. Voice-over without face shots does NOT require escalation
# (can be added in post), so we look for camera-facing speech cues.
#
# IMPORTANT: V.O. (voice-over) and O.S. (off-screen) markers explicitly mean
# the speaker is NOT visible on camera. These shots do NOT need lipsync and
# should NOT escalate to Veo just because the word "says" appears in the
# audio_desc. The _VO_OVERRIDE_PATTERNS below are checked FIRST; if any
# matches, the shot is treated as non-lipsync regardless of other cues.
_VO_OVERRIDE_PATTERNS = [
    re.compile(r"\bV\.?O\.?\b", re.IGNORECASE),            # V.O., VO, v.o.
    re.compile(r"\bvoice[- ]?over\b", re.IGNORECASE),        # "voice-over"
    re.compile(r"\bO\.?S\.?\b"),                             # O.S. (off-screen) — case-sensitive to avoid false hits
    re.compile(r"\boff[- ]?screen\b", re.IGNORECASE),        # "off-screen"
    re.compile(r"\bnarration\b", re.IGNORECASE),             # narration
]

_LIPSYNC_PATTERNS = [
    # Screenplay-style speaker line: "MARELLA: I love this." (CAPS name + colon)
    re.compile(r"^[A-Z][A-Z0-9_ \-]{1,40}:\s*[\"\u201c\(]?", re.MULTILINE),
    # Attribution verbs near quotes
    re.compile(r"\b(says|said|asks|whispers|shouts|exclaims|replies)\b", re.IGNORECASE),
    # Explicit on-camera speech markers (speaks/addresses/talks ... camera)
    re.compile(r"\b(speak\w*|address\w*|talk\w*).*?\bcamera\b", re.IGNORECASE),
    re.compile(r"\b(on[- ]?camera|lipsync|lip[- ]?sync)\b", re.IGNORECASE),
]

# Complex motion detection: signals that the shot has camera moves or element
# motion that LTX 2.3 typically struggles with. Used as a fallback when
# variation_type is not supplied.
_COMPLEX_MOTION_PATTERNS = [
    re.compile(r"\b(drone|aerial|sweeping pan|dolly zoom|vertigo effect|crane shot)\b", re.IGNORECASE),
    re.compile(r"\b(smash cut|whip pan|fast tracking|rapid zoom)\b", re.IGNORECASE),
    re.compile(r"\b(multiple characters.*interact|fight scene|choreography)\b", re.IGNORECASE),
]


def _looks_like_lipsync(text: str) -> bool:
    """True if the text has cues suggesting on-camera dialogue.

    V.O. (voice-over), O.S. (off-screen), and narration markers explicitly
    indicate the speaker is NOT visible — those override any positive lipsync
    signals. A B-roll shot of a brush on canvas with V.O. should NOT escalate
    to Veo just because the audio_desc contains the word "says".
    """
    if not text:
        return False
    # V.O./O.S./narration override: if any of these are present, not lipsync.
    if any(pat.search(text) for pat in _VO_OVERRIDE_PATTERNS):
        return False
    return any(pat.search(text) for pat in _LIPSYNC_PATTERNS)


def _looks_like_complex_motion(text: str) -> bool:
    """True if the text describes motion LTX 2.3 typically handles poorly."""
    if not text:
        return False
    return any(pat.search(text) for pat in _COMPLEX_MOTION_PATTERNS)


# Backwards-compat alias for the old public name.
_looks_like_dialogue = _looks_like_lipsync


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class VideoGeneratorRouter:
    """Two-backend router with heuristic + metadata-driven escalation.

    Config shape (new preferred):

        video_generator:
          class_path: tools.VideoGeneratorRouter
          init_args:
            default_route: local              # "local" (default) or "escalate"
            local_backend:                    # LTX 2.3 (free)
              class_path: tools.VideoGeneratorComfyUI
              init_args:
                base_url: http://127.0.0.1:8189
                i2v_workflow: workflows/ltx23_i2v.json
            escalation_backend:               # Veo / Kling / etc (paid)
              class_path: tools.VideoGeneratorFalAI
              init_args: {...}
            # Optional gate toggles (all default to True)
            escalate_on_lipsync: true
            escalate_on_multiref: true
            escalate_on_large_variation: true
            escalate_on_high_complexity: true
            retry_escalation_on_local_failure: true

    Legacy config (still works, aliased):

        video_generator:
          class_path: tools.VideoGeneratorRouter
          init_args:
            default_route: broll              # treated as "local"
            broll_backend: {...}              # alias for local_backend
            dialogue_backend: {...}           # alias for escalation_backend
    """

    def __init__(
        self,
        # New preferred names:
        local_backend: Any = None,
        escalation_backend: Any = None,
        # Legacy aliases (backwards compat):
        broll_backend: Any = None,
        dialogue_backend: Any = None,
        # Routing:
        default_route: str = "local",
        # Gate toggles:
        escalate_on_lipsync: bool = True,
        escalate_on_multiref: bool = True,
        escalate_on_large_variation: bool = True,
        escalate_on_high_complexity: bool = True,
        retry_escalation_on_local_failure: bool = True,
    ):
        # Resolve aliases: new names win, fall back to legacy.
        self.local_backend = local_backend if local_backend is not None else broll_backend
        self.escalation_backend = (
            escalation_backend if escalation_backend is not None else dialogue_backend
        )

        if self.local_backend is None:
            raise ValueError(
                "VideoGeneratorRouter: local_backend (or broll_backend) is required."
            )
        if self.escalation_backend is None:
            raise ValueError(
                "VideoGeneratorRouter: escalation_backend (or dialogue_backend) is required."
            )

        # Normalize legacy default_route values.
        if default_route in ("broll", "local"):
            default_route = "local"
        elif default_route in ("dialogue", "escalate"):
            default_route = "escalate"
        else:
            raise ValueError(
                "default_route must be 'local', 'escalate', "
                "('broll' legacy) or ('dialogue' legacy)"
            )
        self.default_route = default_route

        self.escalate_on_lipsync = escalate_on_lipsync
        self.escalate_on_multiref = escalate_on_multiref
        self.escalate_on_large_variation = escalate_on_large_variation
        self.escalate_on_high_complexity = escalate_on_high_complexity
        self.retry_escalation_on_local_failure = retry_escalation_on_local_failure

    # ------------------------------------------------------------------
    # Routing decision
    # ------------------------------------------------------------------

    def _decide_route(
        self,
        prompt: str,
        reference_image_paths: List[str],
        **kwargs,
    ) -> str:
        """Returns "local" or "escalate".

        Order of precedence (highest wins):
          1. Explicit ``escalate=`` override from caller
          2. Agent metadata: ``requires_lipsync`` / ``complexity`` / ``variation_type``
          3. Heuristics on prompt text + ref count
          4. default_route
        """
        refs = reference_image_paths or []

        # (1) Explicit override always wins, in either direction.
        explicit = kwargs.get("escalate")
        if explicit is True:
            return "escalate"
        if explicit is False:
            return "local"

        # (2) Agent metadata (preferred over heuristics when present).

        # Agent-flagged lipsync (overrides heuristic).
        requires_lipsync = kwargs.get("requires_lipsync")
        if requires_lipsync is None:
            # Fall back to heuristic, combining prompt + optional audio_desc.
            audio_desc = kwargs.get("audio_desc") or ""
            combined_text = f"{prompt}\n{audio_desc}".strip()
            requires_lipsync = (
                self.escalate_on_lipsync and _looks_like_lipsync(combined_text)
            )
        elif requires_lipsync and not self.escalate_on_lipsync:
            # Agent said lipsync but gate disabled: respect the gate.
            requires_lipsync = False
        if requires_lipsync:
            return "escalate"

        # Agent-flagged complexity (overrides heuristic).
        complexity = kwargs.get("complexity")
        if complexity is None:
            # Heuristic fallback: look for complex-motion phrases in prompt +
            # optional motion_desc.
            motion_desc = kwargs.get("motion_desc") or ""
            combined_text = f"{prompt}\n{motion_desc}".strip()
            complexity = "high" if _looks_like_complex_motion(combined_text) else None
        if complexity == "high" and self.escalate_on_high_complexity:
            return "escalate"

        # Variation type from StoryboardArtist: "small" | "medium" | "large".
        variation_type = kwargs.get("variation_type")
        if (
            variation_type == "large"
            and self.escalate_on_large_variation
        ):
            return "escalate"

        # (3) Heuristic: multi-ref (first+last frame) interpolation.
        # LTX 2.3 is single-ref; 2+ refs means we need the paid backend for
        # reliable first->last interpolation.
        if len(refs) >= 2 and self.escalate_on_multiref:
            return "escalate"

        # (4) Fall through to default.
        return self.default_route

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str],
        **kwargs,
    ) -> VideoOutput:
        route = self._decide_route(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
            **kwargs,
        )

        # Strip router-only kwargs before delegating to the backend.
        backend_kwargs = dict(kwargs)
        for k in (
            "escalate",
            "requires_lipsync",
            "complexity",
            "variation_type",
            "audio_desc",
            "motion_desc",
            # Legacy:
            "has_dialogue",
        ):
            backend_kwargs.pop(k, None)

        # Legacy: has_dialogue=True used to force the dialogue backend.
        # Map it to the new explicit-override semantics.
        if kwargs.get("has_dialogue") is True and route == "local":
            route = "escalate"

        backend = (
            self.escalation_backend if route == "escalate" else self.local_backend
        )
        backend_name = type(backend).__name__
        logging.info(
            f"VideoGeneratorRouter: route={route} backend={backend_name} "
            f"refs={len(reference_image_paths or [])} "
            f"variation={kwargs.get('variation_type')} "
            f"complexity={kwargs.get('complexity')}"
        )

        if route == "local" and self.retry_escalation_on_local_failure:
            try:
                return await backend.generate_single_video(
                    prompt=prompt,
                    reference_image_paths=reference_image_paths,
                    **backend_kwargs,
                )
            except Exception as local_exc:
                logging.warning(
                    f"VideoGeneratorRouter: local backend failed "
                    f"({type(local_exc).__name__}: {local_exc}); "
                    f"retrying on escalation backend."
                )
                fallback = self.escalation_backend
                logging.info(
                    f"VideoGeneratorRouter: route=fallback backend={type(fallback).__name__}"
                )
                return await fallback.generate_single_video(
                    prompt=prompt,
                    reference_image_paths=reference_image_paths,
                    **backend_kwargs,
                )

        return await backend.generate_single_video(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
            **backend_kwargs,
        )
