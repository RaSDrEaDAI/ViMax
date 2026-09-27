import tenacity
import traceback
import logging
import os
import json
from datetime import datetime


def _dump_error(exc: BaseException) -> None:
    """Dump the full exception details (including API response bodies) to a
    debug file in the working directory. This preserves information that
    ``repr(exc)`` strips out — specifically the HTTP response body from z.ai /
    OpenAI BadRequestError, which is critical for diagnosing 400s.

    Writes to ``<cwd>/.vimax_errors.log`` (or working_dir if discoverable).
    Safe to call when no working_dir is set — just skips the dump.
    """
    try:
        # Try to find a working directory. Cwd is usually the ViMax repo root
        # when run from the bridge; the orchestrator passes working_dir via
        # the pipeline instance, but we don't have that here. Fall back to cwd.
        dump_dir = os.environ.get("VIMAX_WORKING_DIR") or os.getcwd()
        dump_path = os.path.join(dump_dir, ".vimax_errors.log")

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }

        # Capture HTTP response body from openai.BadRequestError and friends.
        # The openai SDK stores the raw response on exc.response and a parsed
        # body on exc.body. Both are gold for debugging 400s.
        if hasattr(exc, "response"):
            resp = exc.response
            entry["response_status"] = getattr(resp, "status_code", None)
            try:
                entry["response_body"] = resp.json()
            except Exception:
                try:
                    entry["response_text"] = resp.text[:4000]
                except Exception:
                    pass
        if hasattr(exc, "body") and exc.body:
            entry["error_body"] = exc.body if isinstance(exc.body, (dict, list)) else str(exc.body)[:4000]

        # Capture openai API request info if available (without leaking keys).
        if hasattr(exc, "request"):
            req = exc.request
            entry["request_url"] = getattr(req, "url", None)
            entry["request_method"] = getattr(req, "method", None)

        entry["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[:8000]

        with open(dump_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, indent=2, default=str) + "\n---\n")
    except Exception:
        # Never let the dumper itself mask the original error.
        pass


def after_func(retry_state: tenacity.RetryCallState) -> None:
    if retry_state.outcome.failed:
        exc = retry_state.outcome.exception()
        # Dump full error context (including z.ai response body) before logging
        # the short form. Without this, BadRequestError's response body is lost
        # when the ViMax subprocess exits.
        _dump_error(exc)
        logging.warning(f"Retrying {retry_state.fn.__name__} due to {repr(exc)} (Attempt {retry_state.attempt_number})")
        logging.debug(traceback.format_exception(type(exc), exc, exc.__traceback__))
