"""Nano Banana Pro image generator (fal.ai) — the keyframe authority.

Routes:
  - 0 reference images -> ``fal-ai/nano-banana-pro``        (text-to-image)
  - 1+ reference images -> ``fal-ai/nano-banana-pro/edit``  (multi-ref edit)

Deliberately a SEPARATE class from ``ImageGeneratorFalAI`` rather than a
subclass or a flag on it. The behavioural contract differs in four ways that
each invite a silent regression if overloaded onto the non-Pro adapter:

  1. Paid spend is GATED. Without ``allow_paid_image_spend`` the call raises
     before any network I/O. No local fallback, no downgrade — a run that
     wasn't authorized to spend must stop, not quietly render something worse.
  2. ``aspect_ratio`` is sent on BOTH paths. The non-Pro adapter omits it on
     ``/edit``, so edit-path keyframes silently inherited the ref's aspect.
  3. Unknown kwargs are REJECTED, not swallowed. ``ImageGeneratorFalAI`` eats
     ``size="1600x900"`` via ``**kwargs`` — the pipeline had been passing a
     resolution that never reached fal for months.
  4. A Gemini safety refusal is an ERROR (``NbProRefusalError``), never a
     routing decision. Ported from destack-app ``lib/fal/nb_edit.ts``: when
     the classifier trips erratically per frame and the call site silently
     drops ``image_urls`` to retry text-only, identity for that frame degrades
     from image-locked to prose-only. A narrative ends up a mix of ref-locked
     and prose-only frames — that mix IS the character drift operators report.

Reference images are uploaded to fal's CDN via ``fal_client.upload_file_async``
and the returned URLs are passed in ``image_urls``.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

# NOTE: ``fal_client`` is imported lazily (see ``_require_fal_client``) so the
# module stays importable and this class stays instantiable on the local rail
# where fal-client is intentionally not installed. Only an actual call needs it.
from interfaces.image_output import ImageOutput
from utils.rate_limiter import RateLimiter


# The generic body fal returns when Gemini's safety classifier rejects a call.
# Matching this literal is how a safety refusal (recoverable by an operator
# rewriting the prompt) is distinguished from a real transport error (retryable).
SAFETY_MARKER = "Could not generate images"

# NB Pro resolutions. '0.5K' exists on non-Pro nano-banana but NOT on Pro.
VALID_RESOLUTIONS = ("1K", "2K", "4K")

# Aspect ratios NB Pro accepts, with the pixel geometry each legacy ``size=``
# string maps onto. Used to translate the dead ``size=`` kwarg instead of
# swallowing it.
_SIZE_TO_ASPECT_RATIO = {
    "1600x900": "16:9",
    "1920x1080": "16:9",
    "1280x720": "16:9",
    "1024x1024": "1:1",
    "512x512": "1:1",
    "900x1600": "9:16",
    "1080x1920": "9:16",
    "768x1024": "3:4",
    "1024x768": "4:3",
}


def _require_fal_client():
    """Import fal_client on demand, with an actionable error if it's missing."""
    try:
        import fal_client  # noqa: PLC0415 (deliberately lazy)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Nano Banana Pro backend called but 'fal-client' is not installed. "
            "This is the paid keyframe backend. Either install it "
            "(`pip install fal-client` + set FAL_API_KEY), or select a local "
            "config (configs/script2video_local.yaml)."
        ) from exc
    return fal_client


class NbProRefusalError(Exception):
    """Raised when a Nano Banana Pro call is refused by the safety classifier.

    Callers must NOT retry this and must NOT downgrade to a text-only call.
    It surfaces to the shot level so the run reports which frames refused;
    an operator decides the rewrite, not automation.
    """

    def __init__(self, detail: str, assembled_prompt: str):
        super().__init__(
            f"Nano Banana Pro refused (content safety): {detail[:300]}"
        )
        # The fal error body, truncated — full bodies can be enormous.
        self.detail = detail[:2000]
        # The prompt that was refused, for logs and the operator's rewrite.
        self.assembled_prompt = assembled_prompt


def fal_error_detail(err: BaseException) -> str:
    """Extract a detail string from a thrown fal error.

    The fal client wraps HTTP errors in different shapes by version — ApiError
    ``body``, ``response.data``, or ``detail``. Checking all of them means every
    call site detects refusals identically. Ported from destack-app
    ``falErrorDetail``.
    """
    try:
        body = getattr(err, "body", None)
        if body:
            return body if isinstance(body, str) else json.dumps(body, default=str)
        response = getattr(err, "response", None)
        data = getattr(response, "data", None) if response is not None else None
        if data:
            return data if isinstance(data, str) else json.dumps(data, default=str)
        detail = getattr(err, "detail", None)
        if detail:
            return detail if isinstance(detail, str) else json.dumps(detail, default=str)
    except Exception:  # noqa: BLE001 — fall through to str()
        pass
    return str(err)


def is_nb_safety_refusal(err: BaseException | str) -> bool:
    """True when a thrown fal error (or an error string) is a safety refusal."""
    detail = err if isinstance(err, str) else fal_error_detail(err)
    return SAFETY_MARKER in detail


class ImageGeneratorNanoBananaProFalAI:
    """Nano Banana Pro via fal.ai. Gated, fail-loud, forensically logged."""

    def __init__(
        self,
        api_key: str,
        t2i_model: str = "fal-ai/nano-banana-pro",
        i2i_model: str = "fal-ai/nano-banana-pro/edit",
        resolution: str = "2K",
        allow_paid_image_spend: bool = False,
        max_reference_images: int = 6,
        working_dir: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        client_timeout: float = 600.0,
    ):
        os.environ["FAL_KEY"] = api_key
        self.client_timeout = client_timeout
        self.t2i_model = t2i_model
        self.i2i_model = i2i_model

        if resolution not in VALID_RESOLUTIONS:
            raise ValueError(
                f"Invalid NB Pro resolution {resolution!r}. "
                f"Valid: {', '.join(VALID_RESOLUTIONS)} "
                f"('0.5K' exists on non-Pro nano-banana but not on Pro)."
            )
        self.resolution = resolution

        # Config carries this as a ${VAR} substitution, so it can arrive as the
        # string "false"/"true" rather than a bool. Coerce explicitly: a bare
        # truthiness test would read the string "false" as authorized spend.
        self.allow_paid_image_spend = _coerce_bool(allow_paid_image_spend)
        self.max_reference_images = max_reference_images
        self.working_dir = working_dir
        self.rate_limiter = rate_limiter

    # -- spend gate ---------------------------------------------------------

    def _assert_spend_authorized(self) -> None:
        if not self.allow_paid_image_spend:
            raise RuntimeError(
                "NB Pro spend not authorized: allow_paid_image_spend is false "
                "for this run"
            )

    # -- cost ledger --------------------------------------------------------

    @property
    def cost_ledger_path(self) -> Optional[str]:
        if not self.working_dir:
            return None
        return os.path.join(self.working_dir, "cost_ledger.jsonl")

    def _append_cost_ledger(self, row: Dict[str, Any]) -> None:
        """Append one JSONL row per paid attempt, refusals included.

        Box-side price is unknown, so this is a reconciliation record rather
        than a spend total — the row set is what a real ledger gets priced
        against later. Never fatal: a ledger write failure must not lose an
        image that was already paid for.
        """
        path = self.cost_ledger_path
        if not path:
            logging.debug("NB Pro cost ledger skipped: no working_dir configured")
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            logging.warning("NB Pro cost ledger append failed (%s): %s", path, e)

    # -- generation ---------------------------------------------------------

    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: Optional[List[str]] = None,
        aspect_ratio: Optional[str] = "16:9",
        seed: Optional[int] = None,
        **kwargs,
    ) -> ImageOutput:
        # Spend gate FIRST — before validation, before any upload, before any
        # network I/O. An unauthorized run must not even touch fal's CDN.
        self._assert_spend_authorized()

        aspect_ratio = self._resolve_aspect_ratio(aspect_ratio, kwargs)
        self._reject_unknown_kwargs(kwargs)

        reference_image_paths = list(reference_image_paths or [])
        if len(reference_image_paths) > self.max_reference_images:
            raise ValueError(
                f"NB Pro reference cap exceeded: got {len(reference_image_paths)} "
                f"reference images, max is {self.max_reference_images}. "
                f"Slot-priority truncation belongs upstream in the pipeline "
                f"where reference roles are known — the adapter must not guess "
                f"which reference to drop. Offending paths: {reference_image_paths}"
            )

        fal_client = _require_fal_client()

        # Mode is derived from the ref count and never auto-switched mid-call.
        # Sending an empty image_urls to /edit makes fal 422.
        is_t2i = len(reference_image_paths) == 0
        model = self.t2i_model if is_t2i else self.i2i_model

        arguments: Dict[str, Any] = {
            "prompt": prompt,
            "num_images": 1,
            "output_format": "png",
            # Sent on BOTH paths. The non-Pro adapter omits this on /edit,
            # which is how edit-path keyframes silently drifted off 16:9.
            "aspect_ratio": aspect_ratio,
            "resolution": self.resolution,
        }
        if seed is not None:
            arguments["seed"] = seed
        if not is_t2i:
            image_urls = await asyncio.gather(
                *(fal_client.upload_file_async(p) for p in reference_image_paths)
            )
            arguments["image_urls"] = list(image_urls)

        logging.info(
            "Calling %s (t2i=%s, refs=%d, resolution=%s, aspect_ratio=%s)...",
            model, is_t2i, len(reference_image_paths), self.resolution, aspect_ratio,
        )

        if self.rate_limiter:
            await self.rate_limiter.acquire()

        result = await self._subscribe_with_retry(fal_client, model, arguments, prompt)

        images = result.get("images") or []
        if not images:
            self._append_cost_ledger(self._ledger_row(
                model, arguments, result_url=None, no_image=True,
            ))
            raise ValueError(f"No image returned by {model}: {result}")

        image_url = images[0]["url"]
        self._append_cost_ledger(self._ledger_row(
            model, arguments,
            result_url=image_url,
            request_id=result.get("request_id"),
        ))

        # sent_input is the EXACT dispatched object, so the JSON a call site
        # persists next to the frame cannot disagree with what fal received.
        return ImageOutput(
            fmt="url", ext="png", data=image_url, sent_input=dict(arguments),
        )

    async def _subscribe_with_retry(
        self,
        fal_client,
        model: str,
        arguments: Dict[str, Any],
        prompt: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Retry transport errors (5/10/20s). NEVER retry a safety refusal."""
        for attempt in range(max_retries):
            try:
                # fal's subscribe_async waits indefinitely without client_timeout.
                # A timeout is NOT retried: the retry would resubmit (and re-bill)
                # a paid NB Pro job that may still be running on fal's side.
                result = await fal_client.subscribe_async(
                    model, arguments=arguments, with_logs=False,
                    client_timeout=self.client_timeout,
                )
                return result if isinstance(result, dict) else dict(result)
            except TimeoutError:
                raise
            except Exception as e:
                if is_nb_safety_refusal(e):
                    detail = fal_error_detail(e)
                    # Refusals are recorded — an operator needs to see which
                    # frames the classifier rejected and how often.
                    self._append_cost_ledger(self._ledger_row(
                        model, arguments, result_url=None, refused=True,
                    ))
                    raise NbProRefusalError(detail, prompt) from e
                if attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)
                    logging.warning(
                        "NB Pro %s transport error: %s. Retrying in %ss "
                        "(attempt %d/%d)", model, e, wait, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise
        # Unreachable: the loop either returns or raises.
        raise RuntimeError("NB Pro retry loop exited without a result")

    # -- kwarg handling -----------------------------------------------------

    def _resolve_aspect_ratio(
        self, aspect_ratio: Optional[str], kwargs: Dict[str, Any],
    ) -> str:
        """Resolve the effective aspect ratio, translating the legacy ``size=``.

        ``size="1600x900"`` was silently swallowed by the non-Pro adapter's
        ``**kwargs`` for months. Here it is either translated or raises — a
        pixel size the backend cannot honour must not pass unnoticed.
        """
        size = kwargs.pop("size", None)
        if size is None:
            if not aspect_ratio:
                raise ValueError(
                    "NB Pro requires an aspect_ratio (got None). Keyframes are "
                    "16:9 to match the video contract."
                )
            return aspect_ratio

        translated = _SIZE_TO_ASPECT_RATIO.get(str(size))
        if translated is None:
            raise ValueError(
                f"Cannot translate legacy size={size!r} to an NB Pro aspect "
                f"ratio. NB Pro takes aspect_ratio + resolution, not pixel "
                f"dimensions. Known sizes: "
                f"{', '.join(sorted(_SIZE_TO_ASPECT_RATIO))}. "
                f"Pass aspect_ratio= instead."
            )
        if aspect_ratio and aspect_ratio != translated:
            raise ValueError(
                f"Conflicting geometry: size={size!r} means aspect_ratio="
                f"{translated!r} but aspect_ratio={aspect_ratio!r} was also "
                f"passed. Pass only one."
            )
        logging.debug("Translated legacy size=%s to aspect_ratio=%s", size, translated)
        return translated

    @staticmethod
    def _reject_unknown_kwargs(kwargs: Dict[str, Any]) -> None:
        """Raise on kwargs this backend cannot honour.

        The router passes routing metadata (``visible_characters``,
        ``frame_desc``, ``shot_notes``, ``style``) that local backends consume
        for workflow selection. NB Pro has no routing decision to make, so
        those are accepted-and-ignored by name. Anything else is a caller bug
        and fails loud rather than vanishing into ``**kwargs``.
        """
        ignorable = {
            "visible_characters", "frame_desc", "shot_notes", "style",
            "variation_type", "env_slugline",
        }
        unknown = sorted(set(kwargs) - ignorable)
        if unknown:
            raise TypeError(
                f"ImageGeneratorNanoBananaProFalAI.generate_single_image() got "
                f"unexpected keyword argument(s): {', '.join(unknown)}. "
                f"This backend rejects unknown kwargs rather than swallowing "
                f"them — a silently dropped parameter is how size='1600x900' "
                f"went unnoticed on the non-Pro adapter."
            )

    # -- ledger row ---------------------------------------------------------

    @staticmethod
    def _ledger_row(
        model: str,
        arguments: Dict[str, Any],
        *,
        result_url: Optional[str],
        request_id: Optional[str] = None,
        refused: bool = False,
        no_image: bool = False,
    ) -> Dict[str, Any]:
        # time is imported here rather than at module scope purely to keep the
        # import list honest about what the hot path needs.
        import time
        row: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "endpoint": model,
            "num_images": arguments.get("num_images", 1),
            "prompt_chars": len(arguments.get("prompt") or ""),
            "ref_count": len(arguments.get("image_urls") or []),
            "seed": arguments.get("seed"),
            "result_url": result_url,
        }
        if request_id:
            row["request_id"] = request_id
        if refused:
            row["refused"] = True
        if no_image:
            row["no_image"] = True
        return row


def _coerce_bool(value: Any) -> bool:
    """Coerce a config-supplied flag to a bool.

    ${VAR} substitution yields strings, so ``allow_paid_image_spend`` can
    arrive as "false" — which is truthy. Anything not explicitly affirmative
    reads as NOT authorized: the safe direction for a spend gate.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)
