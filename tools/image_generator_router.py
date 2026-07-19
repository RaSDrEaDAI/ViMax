"""Image generator router: multi-model smart routing.

Routes each frame to the best model based on frame metadata (visible
characters, reference images, text requirements, complexity).

Backends (all optional; router degrades gracefully to whatever is configured):

  - ``flux_backend``      : Flux 1-dev + LoRA (general t2i, character LoRA)
  - ``qwen_backend``      : Qwen 2512 (best portraits, character identity, text)
  - ``zimage_backend``    : Z-image Turbo (fast environments, concept frames)
  - ``qwen_edit_backend`` : Qwen Image Edit 2511 (multi-ref composition, 1-3 refs)
  - ``ideogram_backend``  : Ideogram V4 JSON (multi-char bbox layout, paid)

Routing matrix (first match wins):

  | Signal                                 | Backend          | Why                                    |
  |----------------------------------------|------------------|----------------------------------------|
  | explicit `backend=<name>` override     | caller's choice  | Manual control                         |
  | `json_prompt` kwarg present            | ideogram_backend | Caller already planned the layout      |
  | `len(reference_image_paths) >= 2`      | qwen_edit_backend| Multi-ref needs Qwen Edit 2511         |
  | `len(visible_characters) >= 2`         | ideogram_backend | Multi-char → bbox layout               |
  | `has_text == True` (frame needs text)  | qwen_backend     | Qwen 2512 best at text-in-image        |
  | `is_portrait == True` (1 char, no LoRA)| qwen_backend     | Qwen 2512 best identity fidelity       |
  | `is_environment == True` (no chars)    | zimage_backend   | Z-image fast for pure environments     |
  | LoRA configured + 1 char               | flux_backend     | LoRA still wins for locked identity    |
  | fallback                               | flux_backend     | Safe default                           |

The CompositionPlanner agent is invoked transparently when routing to
ideogram_backend for multi-char frames: caller passes visible_characters +
frame_desc + style, router plans the layout, then submits.

Backwards compatibility:
  Legacy ``simple_backend`` / ``multiref_backend`` / ``multiref_threshold``
  are aliased to ``flux_backend`` / ``qwen_edit_backend`` for old configs.
"""

import logging
from typing import Any, List, Optional

from interfaces.image_output import ImageOutput


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

# Cues that a frame needs rendered text (signage, titles, screen content).
_TEXT_CUE_PATTERNS = [
    # "reads", "sign says", "title reads", "headline", "label reads"
    r"\b(reads|says|displays)\s+[\"']",
    r"\b(sign|headline|title|label|poster|billboard|marquee)\b",
    r"\b(text|typography|lettering|caption|subtitle)\s+(reads|says|displaying)",
    r"[\u201c\"][^\u201d\"]{2,40}[\u201d\"]\s+(reads|above|below|on)",
]

import re

_TEXT_REGEXES = [re.compile(p, re.IGNORECASE) for p in _TEXT_CUE_PATTERNS]


def _frame_needs_text(prompt: str, **kwargs) -> bool:
    """True if the frame description cues literal rendered text."""
    if kwargs.get("has_text") is True:
        return True
    if kwargs.get("has_text") is False:
        return False
    if not prompt:
        return False
    return any(r.search(prompt) for r in _TEXT_REGEXES)


# Cues that a frame is an environment / establishing plate (no characters).
_ENV_CUE_PATTERNS = [
    r"\b(establishing shot|wide shot of|exterior|landscape|aerial|skyline|panorama)\b",
    r"\b(interior|empty room|vacant|no people|deserted|uninhabited)\b",
]
_ENV_REGEXES = [re.compile(p, re.IGNORECASE) for p in _ENV_CUE_PATTERNS]


def _is_environment_frame(prompt: str, visible_character_count: int) -> bool:
    """True if the frame is a pure environment / establishing plate."""
    if visible_character_count > 0:
        return False
    if not prompt:
        return False
    return any(r.search(prompt) for r in _ENV_REGEXES)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ImageGeneratorRouter:
    """Multi-model image router with metadata-driven backend selection.

    Config shape (all backends optional):

        image_generator:
          class_path: tools.ImageGeneratorRouter
          init_args:
            flux_backend:
              class_path: tools.ImageGeneratorComfyUI
              init_args:
                t2i_workflow: workflows/flux1_dev_t2i_lora.json
                i2i_workflow: workflows/flux1_kontext_i2i.json
            qwen_backend:
              class_path: tools.ImageGeneratorComfyUI
              init_args:
                t2i_workflow: workflows/qwen2512_t2i.json
            zimage_backend:
              class_path: tools.ImageGeneratorComfyUI
              init_args:
                t2i_workflow: workflows/zimage_turbo_t2i.json
            qwen_edit_backend:
              class_path: tools.ImageGeneratorComfyUI
              init_args:
                i2i_workflow: workflows/qwen_edit_2511_multiref.json
            ideogram_backend:
              class_path: tools.ImageGeneratorComfyUI
              init_args:
                t2i_workflow: workflows/ideogram_v4_json.json
            composition_planner:
              class_path: agents.CompositionPlanner
              init_args:
                chat_model: ${chat_model}
            default_backend: flux
            route_multichar_to_ideogram: true
            route_text_to_qwen: true
            route_environments_to_zimage: true
    """

    def __init__(
        self,
        # New backend names:
        flux_backend: Any = None,
        qwen_backend: Any = None,
        zimage_backend: Any = None,
        qwen_edit_backend: Any = None,
        ideogram_backend: Any = None,
        composition_planner: Any = None,
        # Legacy aliases:
        simple_backend: Any = None,
        multiref_backend: Any = None,
        multiref_threshold: int = 2,
        # Routing policy:
        default_backend: str = "flux",
        route_multichar_to_ideogram: bool = True,
        route_text_to_qwen: bool = True,
        route_environments_to_zimage: bool = True,
        route_multiref_to_qwen_edit: bool = True,
        route_portraits_to_qwen: bool = True,
        # Fallback when chosen backend is not configured:
        fallback_to_default: bool = True,
    ):
        # Resolve legacy aliases.
        self.flux_backend = flux_backend if flux_backend is not None else simple_backend
        self.qwen_backend = qwen_backend
        self.zimage_backend = zimage_backend
        self.qwen_edit_backend = qwen_edit_backend if qwen_edit_backend is not None else multiref_backend
        self.ideogram_backend = ideogram_backend
        self.composition_planner = composition_planner
        self.multiref_threshold = multiref_threshold

        self.default_backend = default_backend
        self.route_multichar_to_ideogram = route_multichar_to_ideogram
        self.route_text_to_qwen = route_text_to_qwen
        self.route_environments_to_zimage = route_environments_to_zimage
        self.route_multiref_to_qwen_edit = route_multiref_to_qwen_edit
        self.route_portraits_to_qwen = route_portraits_to_qwen
        self.fallback_to_default = fallback_to_default

        if self.flux_backend is None and self.qwen_backend is None and self.zimage_backend is None:
            raise ValueError(
                "ImageGeneratorRouter: at least one backend must be configured."
            )

    # ------------------------------------------------------------------
    # Routing decision
    # ------------------------------------------------------------------

    def _decide_backend(
        self,
        prompt: str,
        reference_image_paths: List[str],
        **kwargs,
    ) -> str:
        """Returns backend name: flux | qwen | zimage | qwen_edit | ideogram."""
        refs = reference_image_paths or []

        # (1) Explicit override always wins.
        explicit = kwargs.get("backend")
        if explicit:
            return str(explicit)

        # (2) Pre-planned JSON layout goes straight to ideogram.
        if kwargs.get("json_prompt"):
            return "ideogram"

        # (3) Multi-ref composition (character-in-location, etc.)
        if (
            len(refs) >= self.multiref_threshold
            and self.route_multiref_to_qwen_edit
            and self.qwen_edit_backend is not None
        ):
            return "qwen_edit"

        visible_characters = kwargs.get("visible_characters") or []
        has_lora = bool(kwargs.get("lora_name"))

        # (4) Multi-character frame → ideogram with bbox layout
        if (
            len(visible_characters) >= 2
            and self.route_multichar_to_ideogram
            and self.ideogram_backend is not None
            and self.composition_planner is not None
        ):
            return "ideogram"

        # (5) Frame requires rendered text → Qwen 2512 (best at text)
        if (
            self.route_text_to_qwen
            and _frame_needs_text(prompt, **kwargs)
            and self.qwen_backend is not None
        ):
            return "qwen"

        # (6) Pure environment plate → Z-image (fast)
        if (
            self.route_environments_to_zimage
            and _is_environment_frame(prompt, len(visible_characters))
            and self.zimage_backend is not None
        ):
            return "zimage"

        # (7) Single-character portrait (LoRA or not) → Qwen 2512.
        # Qwen 2512 + Qwen character LoRA beats Flux+LoRA on identity fidelity.
        # Flux is kept as a backend only for explicit `backend="flux"` overrides.
        if (
            self.route_portraits_to_qwen
            and len(visible_characters) == 1
            and self.qwen_backend is not None
        ):
            return "qwen"

        # (8) LoRA-locked shot without a clearer signal → Qwen 2512 (preferred)
        if has_lora and self.qwen_backend is not None:
            return "qwen"

        # (9) Fallback
        return self.default_backend

    def _get_backend(self, name: str) -> Any:
        backends = {
            "flux": self.flux_backend,
            "qwen": self.qwen_backend,
            "zimage": self.zimage_backend,
            "qwen_edit": self.qwen_edit_backend,
            "ideogram": self.ideogram_backend,
        }
        backend = backends.get(name)
        if backend is None:
            if self.fallback_to_default:
                default = backends.get(self.default_backend)
                if default is not None:
                    logging.warning(
                        f"ImageGeneratorRouter: backend '{name}' not configured, "
                        f"falling back to default '{self.default_backend}'."
                    )
                    return default
            raise ValueError(
                f"ImageGeneratorRouter: backend '{name}' not configured and no fallback available."
            )
        return backend

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = None,
        **kwargs,
    ) -> ImageOutput:
        reference_image_paths = reference_image_paths or []
        backend_name = self._decide_backend(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
            **kwargs,
        )

        backend_kwargs = dict(kwargs)

        # Strip router-only kwargs.
        router_kwargs = [
            "backend",
            "visible_characters",
            "has_text",
            "lora_name",
            "frame_desc",
            "style",
            "shot_notes",
        ]
        router_only = {}
        for k in router_kwargs:
            if k in backend_kwargs:
                router_only[k] = backend_kwargs.pop(k)

        # If routing to ideogram for multi-char, plan the composition first.
        if backend_name == "ideogram" and not backend_kwargs.get("json_prompt"):
            if self.composition_planner is None:
                logging.warning(
                    "ImageGeneratorRouter: ideogram backend selected but no "
                    "composition_planner configured; falling back to text prompt."
                )
            else:
                visible_characters = router_only.get("visible_characters") or []
                frame_desc = router_only.get("frame_desc") or prompt
                style = router_only.get("style") or "cinematic photoreal"
                shot_notes = router_only.get("shot_notes")

                if visible_characters:
                    layout = await self.composition_planner.plan_composition(
                        frame_desc=frame_desc,
                        visible_characters=visible_characters,
                        style=style,
                        shot_notes=shot_notes,
                    )
                    backend_kwargs["json_prompt"] = (
                        self.composition_planner.to_json_prompt_str(layout)
                    )
                    logging.info(
                        f"ImageGeneratorRouter: planned ideogram layout with "
                        f"{len(layout.compositional_deconstruction.elements)} elements."
                    )

        backend = self._get_backend(backend_name)
        logging.info(
            f"ImageGeneratorRouter: backend={backend_name} "
            f"({type(backend).__name__}) refs={len(reference_image_paths)} "
            f"visible_chars={len(router_only.get('visible_characters') or [])} "
            f"json={'yes' if backend_kwargs.get('json_prompt') else 'no'}"
        )

        return await backend.generate_single_image(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
            **backend_kwargs,
        )
