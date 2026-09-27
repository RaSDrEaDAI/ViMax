"""Smoke test: escalation logic in VideoGeneratorRouter.

No network. Exercises both the heuristic patterns and the full routing
decision tree (metadata override > heuristics > default) so we know what
gets routed where.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.video_generator_router import (
    VideoGeneratorRouter,
    _looks_like_lipsync,
    _looks_like_complex_motion,
)


# ---------------------------------------------------------------------------
# Heuristic pattern tests
# ---------------------------------------------------------------------------

LIPSYNC_CASES = [
    ("MARELLA: I love this product.", True, "screenplay-style speaker line"),
    ("Marella says, 'It's amazing.'", True, "narrative dialogue"),
    ("A wide drone shot of the beach at sunset, no people.", False, "pure b-roll"),
    ("Slow push-in on the product on a marble counter.", False, "pure b-roll product shot"),
    ("Voice-over: This is the future.", False, "voice-over narration (no on-camera)"),
    ("She walks across the field as the camera tracks her.", False, "action b-roll"),
    ("BOB: \"Take care.\" The door closes.", True, "speaker + quoted speech"),
    ("Character speaks directly to camera.", True, "explicit on-camera speech"),
]

COMPLEX_MOTION_CASES = [
    ("Sweeping drone shot flying over the city.", True, "drone shot"),
    ("Dolly zoom into the product label.", True, "dolly zoom"),
    ("Static medium shot of the character.", False, "static shot"),
    ("Gentle pan across the bookshelf.", False, "simple pan"),
    ("Fight scene with multiple characters interacting.", True, "fight scene"),
]


def _run_pattern_tests():
    print("\n=== Lipsync pattern tests ===")
    failures = []
    for text, expected, label in LIPSYNC_CASES:
        got = _looks_like_lipsync(text)
        status = "OK " if got == expected else "FAIL"
        print(f"{status} ({label}): got={got} expected={expected}")
        if got != expected:
            failures.append(("lipsync", label))

    print("\n=== Complex motion pattern tests ===")
    for text, expected, label in COMPLEX_MOTION_CASES:
        got = _looks_like_complex_motion(text)
        status = "OK " if got == expected else "FAIL"
        print(f"{status} ({label}): got={got} expected={expected}")
        if got != expected:
            failures.append(("complex_motion", label))

    return failures


# ---------------------------------------------------------------------------
# Router decision tests
# ---------------------------------------------------------------------------

class _StubBackend:
    """Records the route it was called with; never actually generates."""

    def __init__(self, name: str):
        self.name = name
        self.calls = []

    async def generate_single_video(self, prompt, reference_image_paths, **kwargs):
        self.calls.append({
            "prompt": prompt,
            "reference_image_paths": reference_image_paths,
            "kwargs": kwargs,
        })
        # Return None; tests only inspect routing, not output.
        return None


ROUTER_CASES = [
    # (label, prompt, refs, kwargs, expected_route)
    (
        "simple b-roll, no metadata -> local",
        "Slow push-in on the product on a marble counter.",
        ["/tmp/ff.png"],
        {},
        "local",
    ),
    (
        "dialogue screenplay -> escalate (lipsync heuristic)",
        "MARELLA: I love this product.",
        ["/tmp/ff.png"],
        {},
        "escalate",
    ),
    (
        "two refs (first+last) -> escalate (multiref heuristic)",
        "Gentle pan across the room.",
        ["/tmp/ff.png", "/tmp/lf.png"],
        {},
        "escalate",
    ),
    (
        "variation_type=large -> escalate",
        "Wide shot of the city.",
        ["/tmp/ff.png"],
        {"variation_type": "large"},
        "escalate",
    ),
    (
        "variation_type=medium -> local (single ref, simple motion)",
        "Character turns to face camera.",
        ["/tmp/ff.png"],
        {"variation_type": "medium"},
        "local",
    ),
    (
        "variation_type=small -> local",
        "Character blinks.",
        ["/tmp/ff.png"],
        {"variation_type": "small"},
        "local",
    ),
    (
        "complexity=high (agent flag) -> escalate",
        "Two characters talk.",
        ["/tmp/ff.png"],
        {"complexity": "high"},
        "escalate",
    ),
    (
        "requires_lipsync=True (agent flag) -> escalate",
        "Character stands still.",
        ["/tmp/ff.png"],
        {"requires_lipsync": True},
        "escalate",
    ),
    (
        "explicit escalate=True overrides everything -> escalate",
        "Static landscape shot.",
        ["/tmp/ff.png"],
        {"escalate": True},
        "escalate",
    ),
    (
        "explicit escalate=False overrides dialogue -> local",
        "MARELLA: I love this product.",
        ["/tmp/ff.png"],
        {"escalate": False},
        "local",
    ),
    (
        "explicit escalate=False overrides large variation -> local",
        "Wide drone shot of city.",
        ["/tmp/ff.png", "/tmp/lf.png"],
        {"escalate": False, "variation_type": "large"},
        "local",
    ),
    (
        "complex motion heuristic (drone) -> escalate",
        "Sweeping drone shot flying over the city.",
        ["/tmp/ff.png"],
        {},
        "escalate",
    ),
    (
        "audio_desc with dialogue -> escalate (lipsync via metadata field)",
        "Wide shot of the room.",
        ["/tmp/ff.png"],
        {"audio_desc": "BOB: Welcome to the show."},
        "escalate",
    ),
]


def _run_router_tests():
    print("\n=== Router decision tests ===")
    failures = []
    for label, prompt, refs, kwargs, expected in ROUTER_CASES:
        local = _StubBackend("local")
        escalation = _StubBackend("escalation")
        router = VideoGeneratorRouter(
            local_backend=local,
            escalation_backend=escalation,
            default_route="local",
        )
        route = router._decide_route(
            prompt=prompt,
            reference_image_paths=refs,
            **kwargs,
        )
        status = "OK " if route == expected else "FAIL"
        print(f"{status} ({label}): got={route} expected={expected}")
        if route != expected:
            failures.append(label)
    return failures


def _run_legacy_compat_test():
    """Verify old broll_backend/dialogue_backend/default_route=broll still work."""
    print("\n=== Legacy compat test ===")
    local = _StubBackend("local")
    escalation = _StubBackend("escalation")
    try:
        router = VideoGeneratorRouter(
            broll_backend=local,           # legacy alias
            dialogue_backend=escalation,   # legacy alias
            default_route="broll",         # legacy value
        )
        ok = router.default_route == "local"
        print(f"{'OK ' if ok else 'FAIL'} legacy aliases resolve, default normalizes to 'local'")
        return [] if ok else ["legacy_compat"]
    except Exception as e:
        print(f"FAIL legacy compat raised: {e}")
        return ["legacy_compat"]


def _run_failure_retry_test():
    """Verify local failure falls back to escalation backend."""
    print("\n=== Failure retry test ===")

    class _FailingLocal:
        async def generate_single_video(self, prompt, reference_image_paths, **kwargs):
            raise RuntimeError("simulated local failure")

    escalation = _StubBackend("escalation")
    router = VideoGeneratorRouter(
        local_backend=_FailingLocal(),
        escalation_backend=escalation,
        default_route="local",
        retry_escalation_on_local_failure=True,
    )
    import asyncio
    asyncio.run(router.generate_single_video(
        prompt="any shot",
        reference_image_paths=["/tmp/ff.png"],
    ))
    ok = len(escalation.calls) == 1
    print(f"{'OK ' if ok else 'FAIL'} local failure fell back to escalation backend")
    return [] if ok else ["failure_retry"]


def main():
    all_failures = []
    all_failures += _run_pattern_tests()
    all_failures += _run_router_tests()
    all_failures += _run_legacy_compat_test()
    all_failures += _run_failure_retry_test()

    print("\n" + "=" * 50)
    if all_failures:
        print(f"FAILED suites: {all_failures}")
        raise SystemExit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
