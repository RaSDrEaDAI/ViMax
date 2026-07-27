"""Unit tests for the RenderBackend seams added for the NB Pro keyframe path.

Covers the two new seams and asserts the pre-existing behaviour they must not
disturb: configs without a ``sheet_image_generator`` section keep working, and
``${working_dir}`` resolves AFTER ``working_dir_override`` so concurrent
orchestrator jobs get separate cost ledgers.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.render_backend import RenderBackend  # noqa: E402


class _Recorder:
    """Instantiable stub that records the init_args RenderBackend passed it."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def generate_single_image(self, prompt, reference_image_paths=None, **kw):
        raise AssertionError("test stub should never generate")

    async def generate_single_video(self, prompt, reference_image_paths=None, **kw):
        raise AssertionError("test stub should never generate")


def _config(**overrides):
    config = {
        "image_generator": {
            "class_path": "tests.test_render_backend_seams._Recorder",
            "init_args": {"tag": "keyframe"},
        },
        "video_generator": {
            "class_path": "tests.test_render_backend_seams._Recorder",
            "init_args": {"tag": "video"},
        },
        "working_dir": ".working_dir/from_config",
    }
    config.update(overrides)
    return config


class TestSheetImageGenerator(unittest.TestCase):
    def test_absent_section_falls_back_to_image_generator(self):
        """Pre-existing configs have no sheet section — one backend serves both."""
        backend = RenderBackend.from_config(_config())
        self.assertIs(backend.sheet_image_generator, backend.image_generator)

    def test_present_section_is_instantiated_separately(self):
        backend = RenderBackend.from_config(_config(sheet_image_generator={
            "class_path": "tests.test_render_backend_seams._Recorder",
            "init_args": {"tag": "sheet"},
        }))
        self.assertIsNot(backend.sheet_image_generator, backend.image_generator)
        self.assertEqual(backend.sheet_image_generator.kwargs["tag"], "sheet")
        self.assertEqual(backend.image_generator.kwargs["tag"], "keyframe")

    def test_section_without_class_path_falls_back(self):
        """A commented-out / partial section must not crash the run."""
        backend = RenderBackend.from_config(_config(
            sheet_image_generator={"max_requests_per_minute": 8},
        ))
        self.assertIs(backend.sheet_image_generator, backend.image_generator)


class TestWorkingDirPlaceholder(unittest.TestCase):
    def test_placeholder_resolves_from_config(self):
        backend = RenderBackend.from_config(_config(image_generator={
            "class_path": "tests.test_render_backend_seams._Recorder",
            "init_args": {"working_dir": "${working_dir}"},
        }))
        self.assertEqual(
            backend.image_generator.kwargs["working_dir"],
            ".working_dir/from_config",
        )

    def test_override_wins_for_per_job_isolation(self):
        """Two concurrent jobs must not share one cost_ledger.jsonl."""
        backend = RenderBackend.from_config(
            _config(image_generator={
                "class_path": "tests.test_render_backend_seams._Recorder",
                "init_args": {"working_dir": "${working_dir}"},
            }),
            working_dir_override=".working_dir/orchestrator/job-abc",
        )
        self.assertEqual(
            backend.image_generator.kwargs["working_dir"],
            ".working_dir/orchestrator/job-abc",
        )

    def test_placeholder_reaches_nested_backend_specs(self):
        backend = RenderBackend.from_config(_config(image_generator={
            "class_path": "tests.test_render_backend_seams._Recorder",
            "init_args": {
                "nested": {
                    "class_path": "tests.test_render_backend_seams._Recorder",
                    "init_args": {"working_dir": "${working_dir}"},
                },
            },
        }))
        nested = backend.image_generator.kwargs["nested"]
        self.assertEqual(nested.kwargs["working_dir"], ".working_dir/from_config")

    def test_unset_working_dir_resolves_to_none(self):
        """None, not the literal '${working_dir}' string — a truthy path-shaped
        placeholder would have the adapter write a ledger into a bogus dir."""
        config = _config()
        del config["working_dir"]
        config["image_generator"] = {
            "class_path": "tests.test_render_backend_seams._Recorder",
            "init_args": {"working_dir": "${working_dir}"},
        }
        backend = RenderBackend.from_config(config)
        self.assertIsNone(backend.image_generator.kwargs["working_dir"])


if __name__ == "__main__":
    unittest.main()
