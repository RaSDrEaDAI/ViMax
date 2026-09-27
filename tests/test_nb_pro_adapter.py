"""Unit tests for the Nano Banana Pro fal adapter.

No network. The spend gate, the ref cap, and the kwarg rejection all raise
before any fal call, so they are testable without fal-client installed. The
refusal path stubs ``fal_client`` to raise a mock 422 body.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.image_generator_nb_pro_fal import (  # noqa: E402
    ImageGeneratorNanoBananaProFalAI,
    NbProRefusalError,
    SAFETY_MARKER,
    _coerce_bool,
    fal_error_detail,
    is_nb_safety_refusal,
)


def _gen(**kwargs):
    defaults = dict(api_key="test-key", allow_paid_image_spend=True)
    defaults.update(kwargs)
    return ImageGeneratorNanoBananaProFalAI(**defaults)


class _FakeApiError(Exception):
    """Mimics the fal client's ApiError shape (detail lives on ``body``)."""

    def __init__(self, body):
        super().__init__("422 Unprocessable Entity")
        self.body = body


class _FakeFalClient:
    """Stub fal_client: records dispatched arguments, or raises on demand."""

    def __init__(self, result=None, raises=None):
        self._result = result if result is not None else {
            "images": [{"url": "https://fal.media/out.png"}],
            "request_id": "req-123",
        }
        self._raises = raises
        self.calls = []
        self.uploads = []

    async def upload_file_async(self, path):
        self.uploads.append(path)
        return f"https://fal.media/uploaded/{os.path.basename(path)}"

    async def subscribe_async(self, model, arguments=None, with_logs=False, client_timeout=None):
        self.calls.append({"model": model, "arguments": arguments, "client_timeout": client_timeout})
        if self._raises is not None:
            raise self._raises
        return self._result


class TestSpendGate(unittest.TestCase):
    """The gate must raise before any network I/O, and default to closed."""

    def test_default_is_not_authorized(self):
        gen = ImageGeneratorNanoBananaProFalAI(api_key="k")
        self.assertFalse(gen.allow_paid_image_spend)

    def test_gate_raises_when_unauthorized(self):
        gen = ImageGeneratorNanoBananaProFalAI(api_key="k")
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(gen.generate_single_image(prompt="anything"))
        self.assertIn("allow_paid_image_spend is false", str(ctx.exception))

    def test_gate_raises_before_fal_import(self):
        """An unauthorized run must not even require fal-client."""
        gen = ImageGeneratorNanoBananaProFalAI(api_key="k")
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            side_effect=AssertionError("fal must not be touched when gated"),
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(gen.generate_single_image(prompt="anything"))

    def test_string_false_reads_as_unauthorized(self):
        """${VAR} substitution yields strings; 'false' is truthy in Python."""
        gen = ImageGeneratorNanoBananaProFalAI(
            api_key="k", allow_paid_image_spend="false",
        )
        self.assertFalse(gen.allow_paid_image_spend)

    def test_string_true_reads_as_authorized(self):
        gen = ImageGeneratorNanoBananaProFalAI(
            api_key="k", allow_paid_image_spend="true",
        )
        self.assertTrue(gen.allow_paid_image_spend)

    def test_coerce_bool_table(self):
        for affirmative in ("1", "true", "TRUE", " yes ", "on", True):
            self.assertTrue(_coerce_bool(affirmative), affirmative)
        for negative in ("0", "false", "FALSE", "no", "off", "", None, False):
            self.assertFalse(_coerce_bool(negative), negative)


class TestResolutionValidation(unittest.TestCase):
    def test_rejects_half_k(self):
        """'0.5K' exists on non-Pro nano-banana but not on Pro."""
        with self.assertRaises(ValueError) as ctx:
            ImageGeneratorNanoBananaProFalAI(api_key="k", resolution="0.5K")
        self.assertIn("0.5K", str(ctx.exception))

    def test_accepts_valid(self):
        for res in ("1K", "2K", "4K"):
            self.assertEqual(_gen(resolution=res).resolution, res)


class TestReferenceCap(unittest.TestCase):
    def test_over_cap_raises_with_count_and_paths(self):
        paths = [f"/tmp/ref{i}.png" for i in range(7)]
        gen = _gen(max_reference_images=6)
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(gen.generate_single_image(
                prompt="p", reference_image_paths=paths,
            ))
        msg = str(ctx.exception)
        self.assertIn("got 7", msg)
        self.assertIn("max is 6", msg)
        self.assertIn("ref6.png", msg)

    def test_at_cap_is_allowed(self):
        fake = _FakeFalClient()
        paths = [f"/tmp/ref{i}.png" for i in range(6)]
        gen = _gen(max_reference_images=6)
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ):
            asyncio.run(gen.generate_single_image(
                prompt="p", reference_image_paths=paths,
            ))
        self.assertEqual(len(fake.calls[0]["arguments"]["image_urls"]), 6)


class TestModeDerivation(unittest.TestCase):
    def test_zero_refs_uses_t2i_and_omits_image_urls(self):
        fake = _FakeFalClient()
        gen = _gen()
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ):
            asyncio.run(gen.generate_single_image(prompt="p"))
        call = fake.calls[0]
        self.assertEqual(call["model"], "fal-ai/nano-banana-pro")
        # Never send an empty image_urls to /edit — fal 422s.
        self.assertNotIn("image_urls", call["arguments"])

    def test_one_ref_uses_edit(self):
        fake = _FakeFalClient()
        gen = _gen()
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ):
            asyncio.run(gen.generate_single_image(
                prompt="p", reference_image_paths=["/tmp/a.png"],
            ))
        self.assertEqual(fake.calls[0]["model"], "fal-ai/nano-banana-pro/edit")
        self.assertEqual(len(fake.calls[0]["arguments"]["image_urls"]), 1)


class TestArguments(unittest.TestCase):
    def test_aspect_ratio_sent_on_both_paths(self):
        """The existing edit-path omission is the bug this fixes."""
        for refs in ([], ["/tmp/a.png"]):
            fake = _FakeFalClient()
            gen = _gen()
            with patch(
                "tools.image_generator_nb_pro_fal._require_fal_client",
                return_value=fake,
            ):
                asyncio.run(gen.generate_single_image(
                    prompt="p", reference_image_paths=refs, aspect_ratio="16:9",
                ))
            args = fake.calls[0]["arguments"]
            self.assertEqual(args["aspect_ratio"], "16:9", f"refs={refs}")
            self.assertEqual(args["resolution"], "2K", f"refs={refs}")

    def test_seed_forwarded_only_when_set(self):
        fake = _FakeFalClient()
        gen = _gen()
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ):
            asyncio.run(gen.generate_single_image(prompt="p"))
            self.assertNotIn("seed", fake.calls[0]["arguments"])
            asyncio.run(gen.generate_single_image(prompt="p", seed=42))
            self.assertEqual(fake.calls[1]["arguments"]["seed"], 42)

    def test_legacy_size_translated(self):
        fake = _FakeFalClient()
        gen = _gen()
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ):
            asyncio.run(gen.generate_single_image(
                prompt="p", aspect_ratio=None, size="1600x900",
            ))
        self.assertEqual(fake.calls[0]["arguments"]["aspect_ratio"], "16:9")

    def test_legacy_size_conflicting_with_aspect_ratio_raises(self):
        gen = _gen()
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(gen.generate_single_image(
                prompt="p", aspect_ratio="9:16", size="1600x900",
            ))
        self.assertIn("Conflicting geometry", str(ctx.exception))

    def test_unknown_size_raises(self):
        gen = _gen()
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(gen.generate_single_image(prompt="p", size="1234x567"))
        self.assertIn("Cannot translate legacy size", str(ctx.exception))

    def test_unknown_kwarg_raises_instead_of_swallowing(self):
        gen = _gen()
        with self.assertRaises(TypeError) as ctx:
            asyncio.run(gen.generate_single_image(prompt="p", nonsense=1))
        self.assertIn("nonsense", str(ctx.exception))

    def test_router_metadata_kwargs_are_accepted_and_ignored(self):
        fake = _FakeFalClient()
        gen = _gen()
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ):
            asyncio.run(gen.generate_single_image(
                prompt="p",
                visible_characters=[],
                frame_desc="a frame",
                shot_notes="shot 0",
                style="cinematic",
            ))
        args = fake.calls[0]["arguments"]
        # Routing metadata must not leak into the fal payload.
        for leaked in ("visible_characters", "frame_desc", "shot_notes", "style"):
            self.assertNotIn(leaked, args)


class TestSentInput(unittest.TestCase):
    def test_sent_input_matches_dispatched_arguments(self):
        fake = _FakeFalClient()
        gen = _gen()
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ):
            out = asyncio.run(gen.generate_single_image(
                prompt="p", reference_image_paths=["/tmp/a.png"],
            ))
        self.assertEqual(out.sent_input, fake.calls[0]["arguments"])
        self.assertEqual(out.data, "https://fal.media/out.png")

    def test_sent_input_is_a_copy(self):
        """Mutating the returned record must not rewrite history."""
        fake = _FakeFalClient()
        gen = _gen()
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ):
            out = asyncio.run(gen.generate_single_image(prompt="p"))
        out.sent_input["prompt"] = "tampered"
        self.assertEqual(fake.calls[0]["arguments"]["prompt"], "p")


class TestRefusalPath(unittest.TestCase):
    REFUSAL_BODY = {
        "detail": [{"msg": f"{SAFETY_MARKER}. The prompt was rejected."}],
    }

    def test_refusal_detected_from_body(self):
        err = _FakeApiError(self.REFUSAL_BODY)
        self.assertTrue(is_nb_safety_refusal(err))
        self.assertIn(SAFETY_MARKER, fal_error_detail(err))

    def test_transport_error_is_not_a_refusal(self):
        self.assertFalse(is_nb_safety_refusal(RuntimeError("connection reset")))

    def test_refusal_raises_nb_pro_refusal_error(self):
        fake = _FakeFalClient(raises=_FakeApiError(self.REFUSAL_BODY))
        gen = _gen()
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ):
            with self.assertRaises(NbProRefusalError) as ctx:
                asyncio.run(gen.generate_single_image(prompt="the bad prompt"))
        self.assertEqual(ctx.exception.assembled_prompt, "the bad prompt")
        self.assertIn(SAFETY_MARKER, ctx.exception.detail)

    def test_refusal_is_not_retried(self):
        fake = _FakeFalClient(raises=_FakeApiError(self.REFUSAL_BODY))
        gen = _gen()
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ):
            with self.assertRaises(NbProRefusalError):
                asyncio.run(gen.generate_single_image(prompt="p"))
        self.assertEqual(len(fake.calls), 1, "refusal must not be retried")

    def test_refusal_detail_truncated_to_2000_chars(self):
        long_body = SAFETY_MARKER + ("x" * 5000)
        fake = _FakeFalClient(raises=_FakeApiError(long_body))
        gen = _gen()
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ):
            with self.assertRaises(NbProRefusalError) as ctx:
                asyncio.run(gen.generate_single_image(prompt="p"))
        self.assertEqual(len(ctx.exception.detail), 2000)

    def test_transport_error_is_retried(self):
        fake = _FakeFalClient(raises=RuntimeError("connection reset"))
        gen = _gen()
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ), patch("asyncio.sleep", new=_noop_sleep):
            with self.assertRaises(RuntimeError):
                asyncio.run(gen.generate_single_image(prompt="p"))
        self.assertEqual(len(fake.calls), 3, "transport errors retry 3x")


async def _noop_sleep(_seconds):
    """Skip the 5/10/20s retry backoff in tests."""
    return None


class TestCostLedger(unittest.TestCase):
    def test_row_written_per_successful_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeFalClient()
            gen = _gen(working_dir=tmp)
            with patch(
                "tools.image_generator_nb_pro_fal._require_fal_client",
                return_value=fake,
            ):
                asyncio.run(gen.generate_single_image(
                    prompt="hello", reference_image_paths=["/tmp/a.png"], seed=7,
                ))
            rows = _read_ledger(tmp)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["endpoint"], "fal-ai/nano-banana-pro/edit")
        self.assertEqual(row["num_images"], 1)
        self.assertEqual(row["prompt_chars"], 5)
        self.assertEqual(row["ref_count"], 1)
        self.assertEqual(row["seed"], 7)
        self.assertEqual(row["request_id"], "req-123")
        self.assertEqual(row["result_url"], "https://fal.media/out.png")
        self.assertNotIn("refused", row)

    def test_refusal_recorded_with_refused_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeFalClient(
                raises=_FakeApiError(f"{SAFETY_MARKER} nope"),
            )
            gen = _gen(working_dir=tmp)
            with patch(
                "tools.image_generator_nb_pro_fal._require_fal_client",
                return_value=fake,
            ):
                with self.assertRaises(NbProRefusalError):
                    asyncio.run(gen.generate_single_image(prompt="p"))
            rows = _read_ledger(tmp)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["refused"])
        self.assertIsNone(rows[0]["result_url"])

    def test_row_count_equals_call_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeFalClient()
            gen = _gen(working_dir=tmp)
            with patch(
                "tools.image_generator_nb_pro_fal._require_fal_client",
                return_value=fake,
            ):
                for i in range(3):
                    asyncio.run(gen.generate_single_image(prompt=f"p{i}"))
            self.assertEqual(len(_read_ledger(tmp)), 3)

    def test_no_working_dir_is_not_fatal(self):
        fake = _FakeFalClient()
        gen = _gen(working_dir=None)
        with patch(
            "tools.image_generator_nb_pro_fal._require_fal_client",
            return_value=fake,
        ):
            out = asyncio.run(gen.generate_single_image(prompt="p"))
        self.assertEqual(out.data, "https://fal.media/out.png")


def _read_ledger(working_dir):
    path = os.path.join(working_dir, "cost_ledger.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestConfigTemplate(unittest.TestCase):
    """The committed template must stay loadable and correctly wired."""

    def setUp(self):
        import yaml
        path = os.path.join(
            REPO_ROOT, "configs", "script2video_nbpro.example.yaml",
        )
        with open(path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def test_keyframes_route_to_nb_pro(self):
        self.assertEqual(
            self.config["image_generator"]["class_path"],
            "tools.ImageGeneratorNanoBananaProFalAI",
        )

    def test_spend_gate_defaults_to_false(self):
        args = self.config["image_generator"]["init_args"]
        self.assertEqual(
            args["allow_paid_image_spend"],
            "${VIMAX_ALLOW_PAID_IMAGE_SPEND:-false}",
        )

    def test_working_dir_placeholder_present(self):
        args = self.config["image_generator"]["init_args"]
        self.assertEqual(args["working_dir"], "${working_dir}")

    def test_sheet_backend_is_local(self):
        self.assertEqual(
            self.config["sheet_image_generator"]["class_path"],
            "tools.ImageGeneratorRouter",
        )


if __name__ == "__main__":
    unittest.main()
