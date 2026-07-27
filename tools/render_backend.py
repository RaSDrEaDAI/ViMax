"""RenderBackend: config-driven factory for image and video generators.

Reads the ``image_generator`` and ``video_generator`` sections from a
ViMax YAML config, instantiates the concrete classes via *class_path*,
and wires up rate limiters.

Usage::

    backend = RenderBackend.from_config(config)
    image = await backend.image_generator.generate_single_image(...)
    video = await backend.video_generator.generate_single_video(...)
"""

import importlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.rate_limiter import RateLimiter


# Pattern for ${ENV_VAR} or ${ENV_VAR:-default} substitution in config values.
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-(.*))?\}")


def _load_dotenv(env_path: str = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Does NOT override existing env vars (env wins over file). Simple parser:
    skips comments and blank lines, handles quotes, no shell expansion.
    """
    if env_path is None:
        env_path = str(Path(__file__).resolve().parent.parent / ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # Strip surrounding quotes if present.
            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val


def load_config(config_path: str, source_env: bool = True) -> Dict[str, Any]:
    """Load a ViMax YAML config, optionally sourcing .env first.

    Args:
        config_path: Path to YAML config (absolute or relative to repo root).
        source_env: If True, load D:/3.AIProjs/3.ViMax/.env into os.environ
            before parsing. Existing env vars are NOT overridden.

    Returns:
        Parsed config dict. ${ENV_VAR} substitution is NOT applied here;
        it happens in RenderBackend.from_config().
    """
    if source_env:
        _load_dotenv()
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${ENV_VAR} placeholders in strings.

    Supports:
      ${VAR}          — substitute from env, fail if unset
      ${VAR:-default} — substitute from env, use default if unset

    Non-string values pass through unchanged. Lists and dicts recurse.
    """
    if isinstance(value, str):
        def _replace(match):
            var_name = match.group(1)
            default = match.group(2)
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            raise KeyError(
                f"Config substitution failed: environment variable '{var_name}' is not set "
                f"and no default was provided. Set it in the environment or D:/3.AIProjs/3.ViMax/.env"
            )
        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(v) for v in value]
    return value


@dataclass
class RenderBackend:
    """Bundles a chat model with an image generator and a video generator."""

    image_generator: Any
    video_generator: Any
    chat_model: Any = None
    # Backend for sheet-class assets: character portraits, rotation grids, and
    # environment plates. These are NOT keyframes — they are the identity and
    # location authorities the keyframes reference, so they can render on a
    # different (typically local, free) backend than the keyframe path. Falls
    # back to ``image_generator`` when the config omits the section.
    sheet_image_generator: Any = None

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        chat_model: Any = None,
        working_dir_override: Optional[str] = None,
    ) -> "RenderBackend":
        """Build a RenderBackend from a YAML config path or a parsed dict.

        Args:
            config: Either a path to a YAML config file (str/PathLike) or an
                already-parsed config dict. When a path is given it is loaded
                with .env sourcing via ``load_config()``. (The planning bridge
                passes a path; the pipelines pass a pre-parsed dict.)
            chat_model: Optional LangChain chat model. If omitted and the config
                has a ``chat_model`` section, one is built from it and exposed as
                ``backend.chat_model``. Injected into any nested spec whose
                init_args has ``chat_model: "${chat_model}"``.
            working_dir_override: If set, overrides ``config["working_dir"]``.
                Used by the orchestrator bridge for per-job isolation.

        Rate limiters are created from ``max_requests_per_minute`` /
        ``max_requests_per_day`` if present in each generator section.

        Environment variable substitution (${VAR} / ${VAR:-default}) is
        applied to all string values in the config before instantiation.

        If config has a top-level ``vision_chat_model`` section, it is
        instantiated as a VisionModelClient and injected into nested specs
        via ``${vision_model}`` placeholder.
        """
        # Accept a path (or PathLike) as well as a pre-parsed dict — the planning
        # bridge calls from_config(config_path). load_config() also sources .env.
        if isinstance(config, (str, Path)):
            config = load_config(str(config))

        # Apply env-var substitution to the entire config tree.
        config = _substitute_env_vars(config)

        # Apply working_dir override (for per-job isolation).
        if working_dir_override:
            config["working_dir"] = working_dir_override

        # Build the chat model from config when the caller didn't pass one, so
        # callers can use backend.chat_model directly (the planning bridge does).
        if chat_model is None:
            chat_cfg = config.get("chat_model") or {}
            chat_args = chat_cfg.get("init_args")
            if chat_args:
                from langchain.chat_models import init_chat_model
                from utils.provider_presets import resolve_chat_model_config
                chat_model = init_chat_model(**resolve_chat_model_config(chat_args))
                logging.info("RenderBackend: chat_model=%s", chat_args.get("model"))

        # Instantiate vision model if configured (optional).
        vision_model = None
        vision_cfg = config.get("vision_chat_model")
        if vision_cfg and "class_path" in vision_cfg:
            vision_model = _instantiate(vision_cfg, _build_rate_limiter(vision_cfg))
            logging.info("RenderBackend: vision=%s", vision_cfg["class_path"])

        img_cfg = config["image_generator"]
        vid_cfg = config["video_generator"]
        working_dir = config.get("working_dir")

        image_gen = _instantiate(
            img_cfg, _build_rate_limiter(img_cfg),
            chat_model=chat_model, vision_model=vision_model,
            working_dir=working_dir,
        )
        video_gen = _instantiate(
            vid_cfg, _build_rate_limiter(vid_cfg),
            chat_model=chat_model, vision_model=vision_model,
            working_dir=working_dir,
        )

        # Optional: a separate backend for sheet-class assets (portraits,
        # rotation grids, environment plates). Absent -> keyframes and sheets
        # share one backend, which is the pre-existing behaviour.
        sheet_cfg = config.get("sheet_image_generator")
        if sheet_cfg and "class_path" in sheet_cfg:
            sheet_gen = _instantiate(
                sheet_cfg, _build_rate_limiter(sheet_cfg),
                chat_model=chat_model, vision_model=vision_model,
                working_dir=working_dir,
            )
            logging.info("RenderBackend: sheet_image=%s", sheet_cfg["class_path"])
        else:
            sheet_gen = image_gen

        logging.info("RenderBackend: image=%s, video=%s, working_dir=%s",
                     img_cfg["class_path"], vid_cfg["class_path"],
                     working_dir or "<unset>")

        return cls(
            image_generator=image_gen,
            video_generator=video_gen,
            chat_model=chat_model,
            sheet_image_generator=sheet_gen,
        )


def _build_rate_limiter(section: Dict[str, Any]) -> RateLimiter | None:
    rpm = section.get("max_requests_per_minute")
    rpd = section.get("max_requests_per_day")
    if rpm or rpd:
        return RateLimiter(max_requests_per_minute=rpm, max_requests_per_day=rpd)
    return None


def _instantiate(
    section: Dict[str, Any],
    rate_limiter: RateLimiter | None,
    chat_model: Any = None,
    vision_model: Any = None,
    working_dir: Any = None,
) -> Any:
    module_path, cls_name = section["class_path"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    init_args = dict(section.get("init_args", {}))

    # Substitute chat_model / vision_model / working_dir placeholders before
    # recursing. Always replace the placeholder with the resolved object — even
    # when it is None (e.g. vision_chat_model retired). Leaving the literal
    # "${vision_model}" string in place is a footgun: it is truthy, so consumers
    # like CompositionPlanner would treat it as a live model and call methods on
    # a str.
    #
    # ${working_dir} resolves AFTER working_dir_override is applied, so a
    # per-job orchestrator run gets its own cost ledger instead of concurrent
    # jobs all appending to one file.
    for key, value in list(init_args.items()):
        if value == "${chat_model}":
            init_args[key] = chat_model
        elif value == "${vision_model}":
            init_args[key] = vision_model
        elif value == "${working_dir}":
            init_args[key] = working_dir

    # Recursively instantiate any nested backend specs. A nested backend is
    # signalled by a dict that contains a "class_path" key. This lets routers
    # take other generators as init args without manual pre-wiring.
    for key, value in list(init_args.items()):
        if isinstance(value, dict) and "class_path" in value:
            init_args[key] = _instantiate(
                value, _build_rate_limiter(value),
                chat_model=chat_model, vision_model=vision_model,
                working_dir=working_dir,
            )

    if rate_limiter is not None:
        init_args["rate_limiter"] = rate_limiter
    try:
        return cls(**init_args)
    except TypeError as e:
        # Routers may not accept rate_limiter; retry without it for those.
        if rate_limiter is not None and "rate_limiter" in str(e):
            init_args.pop("rate_limiter", None)
            return cls(**init_args)
        raise
