"""ComfyUI HTTP/WS client.

Talks to a local ComfyUI server (default 127.0.0.1:8189) over HTTP for
queueing prompts, uploading reference images, and downloading outputs.
Polls /history for completion. Output binaries are fetched via /view.

The workflow format is the API JSON exported from the ComfyUI editor
(File -> Export (API)). Workflows are stored in workflows/ and templated
at runtime by overriding fields on specific nodes (prompt text, seed,
loaded image filename, model paths, dimensions, etc.).
"""

import asyncio
import functools
import logging
import os
import uuid
import json
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


# Connection-level errors that are safe to retry (transient, network-level).
# Deliberately does NOT include HTTP 4xx/5xx status codes — those are
# legitimate responses from a healthy server (e.g. 400 validation errors
# from a malformed workflow) and must surface immediately so the caller can
# diagnose and fix the workflow, not silently retry the same bad request.
_RETRYABLE_EXC = (
    aiohttp.ClientConnectorError,      # host refused / unreachable (the run-4 bug)
    aiohttp.ServerDisconnectedError,   # server closed the connection mid-request
    aiohttp.ClientOSError,             # OS-level socket errors (reset, broken pipe)
    asyncio.TimeoutError,              # request-level timeout (not overall poll timeout)
)


def _retry_on_conn(
    max_attempts: int = 3,
    backoff_base: float = 3.0,
    label: str = "request",
):
    """Retry transient ComfyUI connection errors with exponential backoff.

    Catches aiohttp connection-level exceptions and asyncio timeouts, retries
    with backoff (3s, 6s, 12s by default). Total worst-case added latency is
    ~21s before giving up — short enough not to stall a render, long enough
    to ride out a ComfyUI restart or transient network blip.

    HTTP non-200 status codes are NOT retried — the caller sees those and can
    act on them (a 400 means the workflow is bad, not that the server is down).
    """

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except _RETRYABLE_EXC as e:
                    last_exc = e
                    if attempt >= max_attempts:
                        logging.error(
                            "ComfyUI %s gave up after %d attempts: %s: %s",
                            label, max_attempts, type(e).__name__, e,
                        )
                        raise
                    wait = backoff_base * (2 ** (attempt - 1))
                    logging.warning(
                        "ComfyUI %s transient error (attempt %d/%d): %s: %s. Retrying in %.1fs",
                        label, attempt, max_attempts, type(e).__name__, e, wait,
                    )
                    await asyncio.sleep(wait)
            # Unreachable — loop either returns or raises on final attempt.
            assert last_exc is not None
            raise last_exc

        return wrapper

    return deco


class ComfyUIClient:
    """Minimal async ComfyUI client.

    Args:
        base_url: ComfyUI server, e.g. http://127.0.0.1:8189.
        client_id: Stable client UUID; generated if not given.
        request_timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8189",
        client_id: Optional[str] = None,
        request_timeout: float = 1800.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())
        self.request_timeout = request_timeout

    async def ping(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/system_stats", timeout=5) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def upload_image(self, image_path: str, subfolder: str = "vimax") -> str:
        """Upload an image to ComfyUI's input directory.

        Returns the filename ComfyUI knows it as (e.g. "vimax/abc123.png").
        Retries on transient connection errors (3 attempts, 3s/6s/12s backoff).
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {image_path}")

        name = f"{uuid.uuid4().hex}{path.suffix or '.png'}"

        @_retry_on_conn(max_attempts=3, backoff_base=3.0, label="upload_image")
        async def _do_upload() -> Dict[str, Any]:
            async with aiohttp.ClientSession() as session:
                with open(path, "rb") as f:
                    form = aiohttp.FormData()
                    form.add_field("image", f, filename=name, content_type="image/png")
                    form.add_field("subfolder", subfolder)
                    form.add_field("type", "input")
                    form.add_field("overwrite", "false")
                    async with session.post(
                        f"{self.base_url}/upload/image", data=form, timeout=self.request_timeout
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(f"ComfyUI upload failed: {resp.status} {text}")
                        return await resp.json()

        data = await _do_upload()
        # ComfyUI returns {"name": "<filename>", "subfolder": "...", "type": "input"}
        sub = data.get("subfolder") or subfolder
        fname = data["name"]
        return f"{sub}/{fname}" if sub else fname

    async def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Queue a workflow and return the prompt_id.

        Retries on transient connection errors (3 attempts, 3s/6s/12s backoff).
        This is the call that crashed run 4: ComfyUI had been knocked over by
        a mid-render cancel and the immediate re-claim hit a connection-refused
        on this /prompt POST. A 3-second retry would have saved the run.
        """
        payload = {"prompt": workflow, "client_id": self.client_id}

        @_retry_on_conn(max_attempts=3, backoff_base=3.0, label="queue_prompt")
        async def _do_post() -> str:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/prompt", json=payload, timeout=self.request_timeout
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"ComfyUI /prompt failed: {resp.status} {text}")
                    data = await resp.json()
            return data["prompt_id"]

        return await _do_post()

    async def wait_for_completion(self, prompt_id: str, poll_interval: float = 2.0) -> Dict[str, Any]:
        """Poll /history/{prompt_id} until complete; return the history entry.

        Resilient to transient ComfyUI connection drops mid-render: a poll that
        raises a connection-level exception is treated like a non-200 (log,
        sleep, continue) rather than killing the wait. The overall
        `request_timeout` still caps the total wait, so a permanently-dead
        server will eventually time out rather than hang forever.
        """
        url = f"{self.base_url}/history/{prompt_id}"
        elapsed = 0.0
        consecutive_conn_errors = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=30) as resp:
                        if resp.status != 200:
                            await asyncio.sleep(poll_interval)
                            elapsed += poll_interval
                            continue
                        data = await resp.json()
                consecutive_conn_errors = 0  # reset on any successful HTTP exchange
            except _RETRYABLE_EXC as e:
                # Transient connection error — ComfyUI hiccupped or restarting.
                # Don't kill the wait; log periodically and keep polling until
                # the overall request_timeout fires.
                consecutive_conn_errors += 1
                if consecutive_conn_errors == 1 or consecutive_conn_errors % 10 == 0:
                    logging.warning(
                        "ComfyUI poll connection error #%d for prompt %s: %s: %s. "
                        "Continuing to poll (overall timeout still governs).",
                        consecutive_conn_errors, prompt_id, type(e).__name__, e,
                    )
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                if elapsed >= self.request_timeout:
                    raise TimeoutError(
                        f"ComfyUI prompt {prompt_id} did not complete in {self.request_timeout}s "
                        f"(last connection error: {type(e).__name__}: {e})"
                    )
                continue

            if prompt_id in data:
                entry = data[prompt_id]
                status = entry.get("status", {})
                if status.get("completed"):
                    return entry
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI prompt failed: {entry}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            if elapsed >= self.request_timeout:
                raise TimeoutError(f"ComfyUI prompt {prompt_id} did not complete in {self.request_timeout}s")

    async def fetch_output(self, filename: str, subfolder: str = "", out_type: str = "output") -> bytes:
        """Download an output file by ComfyUI's view endpoint.

        Retries on transient connection errors (3 attempts, 3s/6s/12s backoff).
        """
        params = {"filename": filename, "type": out_type}
        if subfolder:
            params["subfolder"] = subfolder

        @_retry_on_conn(max_attempts=3, backoff_base=3.0, label="fetch_output")
        async def _do_fetch() -> bytes:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/view", params=params, timeout=self.request_timeout
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"ComfyUI /view failed: {resp.status} {text}")
                    return await resp.read()

        return await _do_fetch()

    @staticmethod
    def collect_outputs(history_entry: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """Return [(filename, subfolder, type), ...] from a completed history entry.

        Walks every node's `outputs.images` and `outputs.gifs` (videos sometimes
        come back as gifs in animate workflows). Caller picks the one it wants.
        """
        results: List[Tuple[str, str, str]] = []
        outputs = history_entry.get("outputs", {})
        for _, node_out in outputs.items():
            for key in ("images", "gifs", "videos", "audio"):
                for item in node_out.get(key, []) or []:
                    fname = item.get("filename")
                    if fname:
                        results.append((fname, item.get("subfolder", ""), item.get("type", "output")))
        return results


def load_workflow(workflow_path: str) -> Dict[str, Any]:
    """Load a workflow JSON (API format) from disk."""
    with open(workflow_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_node_by_title(workflow: Dict[str, Any], title: str) -> Optional[str]:
    """Find a node ID by its `_meta.title`. Returns the first match or None."""
    for node_id, node in workflow.items():
        meta = node.get("_meta") or {}
        if meta.get("title") == title:
            return node_id
    return None


def set_node_input(workflow: Dict[str, Any], title: str, key: str, value: Any) -> None:
    """Set workflow[<node-by-title>].inputs[key] = value. Raises if not found."""
    node_id = find_node_by_title(workflow, title)
    if node_id is None:
        raise KeyError(f"No node with title {title!r} in workflow")
    workflow[node_id]["inputs"][key] = value
