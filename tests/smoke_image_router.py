"""Smoke test: image router + CompositionPlanner schema.

No network. Exercises:
  1. Routing decision tree (text → qwen, environment → zimage, multiref →
     qwen_edit, multi-char → ideogram, portrait → qwen, fallback → flux)
  2. CompositionPlanner output schema validation (bbox format, ranges, etc.)
  3. Legacy config compatibility (simple_backend/multiref_backend aliases)
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.image_generator_router import (
    ImageGeneratorRouter,
    _frame_needs_text,
    _is_environment_frame,
)
from agents.composition_planner import (
    CompositionPlanner,
    IdeogramJsonPrompt,
    IdeogramElement,
    StyleDescription,
    CompositionalDeconstruction,
)
from interfaces.character import CharacterInScene


# ---------------------------------------------------------------------------
# Heuristic pattern tests
# ---------------------------------------------------------------------------

def _run_heuristic_tests():
    print("\n=== Heuristic pattern tests ===")
    failures = []

    TEXT_CASES = [
        ("A sign reads 'Welcome to the show'", True, "explicit reads"),
        ("The headline says 'Breaking News'", True, "headline cue"),
        ("Bob standing near the window", False, "no text cue"),
        ("A wide landscape shot of the desert", False, "no text cue"),
    ]
    for prompt, expected, label in TEXT_CASES:
        got = _frame_needs_text(prompt)
        ok = got == expected
        print(f"{'OK ' if ok else 'FAIL'} (text: {label}): got={got} expected={expected}")
        if not ok:
            failures.append(("text", label))

    ENV_CASES = [
        ("Establishing shot of the city skyline at dusk", True, "establishing"),
        ("Wide aerial view of the coastline", True, "aerial"),
        ("Bob and Alice talking in the kitchen", False, "has characters"),
        ("Empty room with sunlight streaming in", True, "empty room"),
    ]
    for prompt, expected, label in ENV_CASES:
        got = _is_environment_frame(prompt, visible_character_count=0)
        ok = got == expected
        print(f"{'OK ' if ok else 'FAIL'} (env: {label}): got={got} expected={expected}")
        if not ok:
            failures.append(("env", label))

    return failures


# ---------------------------------------------------------------------------
# Router decision tests
# ---------------------------------------------------------------------------

class _StubImageBackend:
    def __init__(self, name: str):
        self.name = name
        self.calls = []

    async def generate_single_image(self, prompt, reference_image_paths=None, **kwargs):
        self.calls.append({
            "prompt": prompt,
            "reference_image_paths": reference_image_paths or [],
            "kwargs": kwargs,
        })
        return None  # tests only inspect routing


def _make_character(name: str) -> CharacterInScene:
    return CharacterInScene(
        idx=0,
        identifier_in_scene=name,
        is_visible=True,
        static_features=f"{name} has distinct features",
        dynamic_features="wearing casual clothes",
    )


ROUTER_CASES = [
    # (label, prompt, refs, kwargs, expected_backend)
    (
        "simple text-only prompt, no metadata -> qwen (default)",
        "A bowl of fruit on a table.",
        [],
        {},
        "qwen",
    ),
    (
        "multi-char (2 visible) -> ideogram",
        "Bob and Alice talking in the kitchen.",
        [],
        {"visible_characters": [_make_character("Bob"), _make_character("Alice")]},
        "ideogram",
    ),
    (
        "multi-ref (2 refs) -> qwen_edit",
        "Compose character in environment.",
        ["/tmp/char.png", "/tmp/env.png"],
        {},
        "qwen_edit",
    ),
    (
        "frame with text cue -> qwen",
        "A poster reads 'Welcome Home'",
        [],
        {},
        "qwen",
    ),
    (
        "environment plate (no chars) -> zimage",
        "Establishing shot of the city skyline",
        [],
        {"visible_characters": []},
        "zimage",
    ),
    (
        "single-char portrait (no LoRA) -> qwen",
        "Close-up of Marella's face.",
        [],
        {"visible_characters": [_make_character("Marella")]},
        "qwen",
    ),
    (
        "single-char portrait WITH LoRA -> qwen (not flux)",
        "Close-up of Marella's face.",
        [],
        {"visible_characters": [_make_character("Marella")], "lora_name": "marella.safetensors"},
        "qwen",
    ),
    (
        "explicit backend override -> chosen backend",
        "anything",
        [],
        {"backend": "zimage"},
        "zimage",
    ),
    (
        "pre-planned json_prompt -> ideogram",
        "anything",
        [],
        {"json_prompt": '{"high_level_description": "test"}'},
        "ideogram",
    ),
]


def _run_router_tests():
    print("\n=== Router decision tests ===")
    failures = []

    for label, prompt, refs, kwargs, expected in ROUTER_CASES:
        backends = {name: _StubImageBackend(name) for name in ["flux", "qwen", "zimage", "qwen_edit", "ideogram"]}
        # Mock composition_planner so we don't need a real LLM
        planner = MagicMock()

        router = ImageGeneratorRouter(
            flux_backend=backends["flux"],
            qwen_backend=backends["qwen"],
            zimage_backend=backends["zimage"],
            qwen_edit_backend=backends["qwen_edit"],
            ideogram_backend=backends["ideogram"],
            composition_planner=planner,
            default_backend="qwen",
        )
        backend_name = router._decide_backend(
            prompt=prompt,
            reference_image_paths=refs,
            **kwargs,
        )
        ok = backend_name == expected
        print(f"{'OK ' if ok else 'FAIL'} ({label}): got={backend_name} expected={expected}")
        if not ok:
            failures.append(label)

    return failures


# ---------------------------------------------------------------------------
# CompositionPlanner schema tests
# ---------------------------------------------------------------------------

def _run_schema_tests():
    print("\n=== CompositionPlanner schema tests ===")
    failures = []

    # Valid bbox
    try:
        elem = IdeogramElement(type="obj", desc="Bob, tall, blue shirt", bbox=[100, 50, 900, 450])
        print(f"OK  (valid bbox accepted)")
    except Exception as e:
        print(f"FAIL (valid bbox rejected): {e}")
        failures.append("valid_bbox")

    # Invalid bbox: y_min >= y_max
    try:
        IdeogramElement(type="obj", desc="x", bbox=[500, 50, 500, 450])
        print(f"FAIL (y_min == y_max should reject)")
        failures.append("y_min_eq_y_max")
    except ValueError:
        print(f"OK  (y_min == y_max rejected)")

    # Invalid bbox: out of range
    try:
        IdeogramElement(type="obj", desc="x", bbox=[100, 50, 900, 1500])
        print(f"FAIL (x_max > 1000 should reject)")
        failures.append("x_max_out_of_range")
    except ValueError:
        print(f"OK  (x_max > 1000 rejected)")

    # Invalid bbox: wrong length
    try:
        IdeogramElement(type="obj", desc="x", bbox=[100, 50, 900])
        print(f"FAIL (3-element bbox should reject)")
        failures.append("bbox_length_3")
    except ValueError:
        print(f"OK  (3-element bbox rejected)")

    # Full schema construction
    try:
        layout = IdeogramJsonPrompt(
            high_level_description="Two friends meeting in a cafe.",
            style_description=StyleDescription(aesthetics="cinematic photoreal, warm tones"),
            compositional_deconstruction=CompositionalDeconstruction(
                background="Cozy cafe interior, warm light, wooden tables",
                elements=[
                    IdeogramElement(type="obj", desc="Bob, tall, blue shirt, sitting", bbox=[100, 50, 900, 450]),
                    IdeogramElement(type="obj", desc="Alice, short hair, green dress, standing", bbox=[100, 550, 900, 950]),
                ],
            ),
        )
        dumped = layout.model_dump(exclude_none=True)
        assert "compositional_deconstruction" in dumped
        assert len(dumped["compositional_deconstruction"]["elements"]) == 2
        print(f"OK  (full schema construction)")
    except Exception as e:
        print(f"FAIL (full schema construction): {e}")
        failures.append("schema_construction")

    # JSON serialization via to_json_prompt_str
    try:
        planner = CompositionPlanner(chat_model=MagicMock())
        s = planner.to_json_prompt_str(layout)
        parsed = json.loads(s)
        assert parsed["high_level_description"] == "Two friends meeting in a cafe."
        assert len(parsed["compositional_deconstruction"]["elements"]) == 2
        print(f"OK  (JSON serialization)")
    except Exception as e:
        print(f"FAIL (JSON serialization): {e}")
        failures.append("json_serialization")

    return failures


# ---------------------------------------------------------------------------
# Legacy compat test
# ---------------------------------------------------------------------------

def _run_legacy_compat_test():
    print("\n=== Legacy compat test ===")
    simple = _StubImageBackend("simple")
    multiref = _StubImageBackend("multiref")
    try:
        router = ImageGeneratorRouter(
            simple_backend=simple,
            multiref_backend=multiref,
        )
        ok = router.flux_backend is simple and router.qwen_edit_backend is multiref
        print(f"{'OK ' if ok else 'FAIL'} legacy aliases resolve to new backend names")
        return [] if ok else ["legacy_compat"]
    except Exception as e:
        print(f"FAIL legacy compat raised: {e}")
        return ["legacy_compat"]


def main():
    all_failures = []
    all_failures += _run_heuristic_tests()
    all_failures += _run_router_tests()
    all_failures += _run_schema_tests()
    all_failures += _run_legacy_compat_test()

    print("\n" + "=" * 50)
    if all_failures:
        print(f"FAILED suites: {all_failures}")
        raise SystemExit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
