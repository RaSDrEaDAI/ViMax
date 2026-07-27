"""Integration tests for MiniMax provider support.

These tests verify end-to-end flow through the pipeline config loading
and ``init_chat_model`` invocation.  They mock the LangChain factory so
no real API calls are made.

Heavy multimedia dependencies (moviepy, scenedetect, cv2, google-genai,
etc.) are stubbed at the module level so the pipeline modules can be
imported in a lightweight test environment.

IMPORTANT (isolation): a module is stubbed ONLY when it cannot really be
imported. These stubs go into ``sys.modules`` at import time and are never
removed — ``unittest discover`` imports every test module before running any
test, so an unconditional stub here replaces PIL for the whole process and any
test that genuinely uses PIL then fails on a MagicMock. (``PIL.ImageFilter``
resolving to an auto-created MagicMock attribute rather than the real module is
what that looks like from the far end.) On a full environment nothing is stubbed
and nothing leaks; on a lean one the stubs still do their job.
"""

import importlib
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

# ---- stub heavy deps (only those genuinely unavailable) ----
_STUB_MODULES = [
    "moviepy", "cv2", "scenedetect", "scenedetect.detectors",
    "PIL", "PIL.Image",
    "faiss",
    "google", "google.genai", "google.genai.types", "google.genai.errors",
    "langchain_community", "langchain_community.vectorstores",
    "langchain_community.vectorstores.FAISS",
]


def _is_really_importable(name: str) -> bool:
    if name in sys.modules:
        return not isinstance(sys.modules[name], MagicMock)
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


_stubbed = []
for _mod in _STUB_MODULES:
    if _is_really_importable(_mod):
        continue
    mock = MagicMock()
    # Give stub a __spec__ so importlib.util.find_spec() works
    mock.__spec__ = importlib.machinery.ModuleSpec(_mod, None)
    mock.__path__ = []
    sys.modules[_mod] = mock
    _stubbed.append(_mod)

from utils.provider_presets import resolve_chat_model_config


class TestPipelineConfigResolution(unittest.TestCase):
    """Integration: config dict -> resolve -> init_chat_model kwargs."""

    def _make_minimax_config(self, **overrides):
        base = {
            "model": "MiniMax-M2.7",
            "model_provider": "minimax",
            "api_key": "test-key",
        }
        base.update(overrides)
        return base

    def test_full_minimax_config_resolution(self):
        config = self._make_minimax_config()
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["model_provider"], "openai")
        self.assertEqual(resolved["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(resolved["model"], "MiniMax-M2.7")
        self.assertEqual(resolved["api_key"], "test-key")

    def test_minimax_highspeed_model(self):
        config = self._make_minimax_config(model="MiniMax-M2.7-highspeed")
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["model"], "MiniMax-M2.7-highspeed")
        self.assertEqual(resolved["model_provider"], "openai")

    def test_minimax_m25_model(self):
        config = self._make_minimax_config(model="MiniMax-M2.5")
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["model"], "MiniMax-M2.5")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "env-api-key"})
    def test_env_key_fallback_in_config(self):
        config = {
            "model": "MiniMax-M2.7",
            "model_provider": "minimax",
        }
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["api_key"], "env-api-key")

    def test_openrouter_config_unchanged(self):
        """Existing OpenRouter configs must not be affected."""
        config = {
            "model": "google/gemini-2.5-flash-lite-preview-09-2025",
            "model_provider": "openai",
            "api_key": "or-key",
            "base_url": "https://openrouter.ai/api/v1",
        }
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["model_provider"], "openai")
        self.assertEqual(resolved["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(resolved["model"], "google/gemini-2.5-flash-lite-preview-09-2025")

    def test_init_chat_model_receives_openai_provider(self):
        """Verify that resolved kwargs have model_provider='openai'."""
        config = self._make_minimax_config()
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["model_provider"], "openai")
        self.assertEqual(resolved["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(resolved["model"], "MiniMax-M2.7")

    def test_temperature_clamping_in_pipeline_flow(self):
        config = self._make_minimax_config(temperature=2.0)
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["temperature"], 1.0)

    def test_extra_kwargs_preserved(self):
        config = self._make_minimax_config(max_tokens=4096, top_p=0.9)
        resolved = resolve_chat_model_config(config)
        self.assertEqual(resolved["max_tokens"], 4096)
        self.assertEqual(resolved["top_p"], 0.9)


class TestPipelineInitFromConfig(unittest.TestCase):
    """Integration: full pipeline init_from_config with MiniMax config."""

    @patch("pipelines.idea2video_pipeline.init_chat_model")
    @patch("pipelines.idea2video_pipeline.RenderBackend.from_config")
    def test_idea2video_pipeline_minimax_config(self, mock_backend, mock_init):
        mock_model = MagicMock()
        mock_init.return_value = mock_model
        mock_backend.return_value = MagicMock(image_generator=MagicMock(), video_generator=MagicMock())

        from pipelines.idea2video_pipeline import Idea2VideoPipeline
        pipeline = Idea2VideoPipeline.init_from_config("configs/idea2video_minimax.yaml")

        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        self.assertEqual(call_kwargs["model_provider"], "openai")
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7")

    @patch("pipelines.script2video_pipeline.init_chat_model")
    @patch("pipelines.script2video_pipeline.RenderBackend.from_config")
    def test_script2video_pipeline_minimax_config(self, mock_backend, mock_init):
        mock_model = MagicMock()
        mock_init.return_value = mock_model
        mock_backend.return_value = MagicMock(image_generator=MagicMock(), video_generator=MagicMock())

        from pipelines.script2video_pipeline import Script2VideoPipeline
        pipeline = Script2VideoPipeline.init_from_config("configs/script2video_minimax.yaml")

        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        self.assertEqual(call_kwargs["model_provider"], "openai")
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7")

    @patch("pipelines.idea2video_pipeline.init_chat_model")
    @patch("pipelines.idea2video_pipeline.RenderBackend.from_config")
    def test_existing_openrouter_config_still_works(self, mock_backend, mock_init):
        mock_model = MagicMock()
        mock_init.return_value = mock_model
        mock_backend.return_value = MagicMock(image_generator=MagicMock(), video_generator=MagicMock())

        from pipelines.idea2video_pipeline import Idea2VideoPipeline
        pipeline = Idea2VideoPipeline.init_from_config("configs/idea2video.yaml")

        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        self.assertEqual(call_kwargs["model_provider"], "openai")
        self.assertEqual(call_kwargs["base_url"], "https://openrouter.ai/api/v1")


if __name__ == "__main__":
    unittest.main()
