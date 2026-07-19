"""Tests for comfyui_client.py connection-level retry behavior.

Validates:
1. queue_prompt retries on ClientConnectorError (the run-4 bug)
2. upload_image retries on ServerDisconnectedError
3. fetch_output retries on ClientOSError
4. wait_for_completion survives transient connection drops mid-render
5. HTTP 4xx/5xx responses are NOT retried (validation errors must surface)
6. Retry gives up after max_attempts and raises the last error

Uses aiohttp's actual exception types against a server that doesn't exist
(127.0.0.1:1 — guaranteed refused) and monkeypatched session factory for
status-code and mid-render scenarios.

Run: ./.venv/Scripts/python.exe tests/test_comfyui_client_retry.py
"""

import asyncio
import logging
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiohttp
from tools.comfyui_client import ComfyUIClient, _retry_on_conn, _RETRYABLE_EXC


# ─── Helpers ──────────────────────────────────────────────────────────────

def _make_client():
    return ComfyUIClient(base_url="http://127.0.0.1:1", client_id="test-client")


def _conn_error():
    """A real ClientConnectorError like ComfyUI refusing a connection."""
    return aiohttp.ClientConnectorError(
        connection_key=MagicMock(),
        os_error=ConnectionRefusedError(111, "Connection refused"),
    )


# ─── Tests: retry fires on transient errors ───────────────────────────────

async def test_queue_prompt_retries_on_connection_refused():
    """Run-4 scenario: ComfyUI down, /prompt POST refused."""
    client = _make_client()

    # Patch the inner POST to always raise ClientConnectorError
    call_count = {"n": 0}

    class FakeResp:
        status = 200
        async def json(self): return {"prompt_id": "p1"}
        async def text(self): return ""
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class FakeSession:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def post(self, *a, **kw):
            call_count["n"] += 1
            # Raise on first 2 attempts, succeed on 3rd
            if call_count["n"] < 3:
                raise _conn_error()
            return FakeResp()

    with patch("tools.comfyui_client.aiohttp.ClientSession", FakeSession), \
         patch("asyncio.sleep", new=AsyncMock()):  # skip real backoff waits
        prompt_id = await client.queue_prompt({"node": {}})

    assert prompt_id == "p1", f"expected prompt_id p1, got {prompt_id}"
    assert call_count["n"] == 3, f"expected 3 attempts, got {call_count['n']}"
    print("OK  queue_prompt retried twice on ClientConnectorError, succeeded on 3rd attempt")


async def test_upload_image_retries_on_disconnect():
    """upload_image should retry on ServerDisconnectedError."""
    client = _make_client()
    call_count = {"n": 0}

    class FakeResp:
        status = 200
        async def json(self): return {"name": "abc.png", "subfolder": "vimax"}
        async def text(self): return ""
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class FakeSession:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def post(self, *a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise aiohttp.ServerDisconnectedError("server closed")
            return FakeResp()

    # Create a real temp image to upload
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\\x89PNG\\r\\n\\x1a\\n" + b"\\x00" * 100)
        tmp_path = f.name

    try:
        with patch("tools.comfyui_client.aiohttp.ClientSession", FakeSession), \
             patch("asyncio.sleep", new=AsyncMock()):
            result = await client.upload_image(tmp_path)
        assert "abc.png" in result, f"expected abc.png in result, got {result}"
        assert call_count["n"] == 2, f"expected 2 attempts, got {call_count['n']}"
        print("OK  upload_image retried on ServerDisconnectedError, succeeded on 2nd attempt")
    finally:
        os.unlink(tmp_path)


async def test_fetch_output_retries_on_oserror():
    """fetch_output should retry on ClientOSError."""
    client = _make_client()
    call_count = {"n": 0}

    class FakeResp:
        status = 200
        async def read(self): return b"\\x00\\x01\\x02videodata"
        async def text(self): return ""
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class FakeSession:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def get(self, *a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise aiohttp.ClientOSError("socket reset")
            return FakeResp()

    with patch("tools.comfyui_client.aiohttp.ClientSession", FakeSession), \
         patch("asyncio.sleep", new=AsyncMock()):
        data = await client.fetch_output("video.mp4")

    assert data == b"\\x00\\x01\\x02videodata", f"unexpected data: {data!r}"
    assert call_count["n"] == 2, f"expected 2 attempts, got {call_count['n']}"
    print("OK  fetch_output retried on ClientOSError, succeeded on 2nd attempt")


# ─── Tests: HTTP errors are NOT retried ───────────────────────────────────

async def test_queue_prompt_does_not_retry_http_400():
    """The validation-error bug we already fixed must NOT be retried.

    A 400 from /prompt means the workflow is malformed (empty LoadImage slot,
    bad node wiring, etc.). Retrying would waste time and hide the real error.
    """
    client = _make_client()
    call_count = {"n": 0}

    class FakeResp:
        status = 400
        async def text(self):
            return '{"error": "validation failed: invalid image file reference_3.png"}'
        async def json(self): return {}
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class FakeSession:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def post(self, *a, **kw):
            call_count["n"] += 1
            return FakeResp()

    with patch("tools.comfyui_client.aiohttp.ClientSession", FakeSession):
        raised = False
        try:
            await client.queue_prompt({"node": {}})
        except RuntimeError as e:
            raised = True
            assert "400" in str(e), f"error should mention 400, got: {e}"
        assert raised, "expected RuntimeError on 400"
    assert call_count["n"] == 1, f"HTTP 400 must NOT retry, got {call_count['n']} attempts"
    print("OK  HTTP 400 not retried (validation errors surface immediately)")


async def test_queue_prompt_does_not_retry_http_500():
    """500 is a legitimate server error, not a connection drop. Don't retry."""
    client = _make_client()
    call_count = {"n": 0}

    class FakeResp:
        status = 500
        async def text(self): return "Internal Server Error"
        async def json(self): return {}
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class FakeSession:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def post(self, *a, **kw):
            call_count["n"] += 1
            return FakeResp()

    with patch("tools.comfyui_client.aiohttp.ClientSession", FakeSession):
        try:
            await client.queue_prompt({"node": {}})
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
    assert call_count["n"] == 1, f"HTTP 500 must NOT retry, got {call_count['n']} attempts"
    print("OK  HTTP 500 not retried")


# ─── Tests: retry gives up after max_attempts ─────────────────────────────

async def test_retry_gives_up_after_max_attempts():
    """When all attempts fail with connection errors, the last error is raised."""
    client = _make_client()
    call_count = {"n": 0}

    class FakeSession:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def post(self, *a, **kw):
            call_count["n"] += 1
            raise _conn_error()

    with patch("tools.comfyui_client.aiohttp.ClientSession", FakeSession), \
         patch("asyncio.sleep", new=AsyncMock()):
        raised = False
        try:
            await client.queue_prompt({"node": {}})
        except aiohttp.ClientConnectorError:
            raised = True
        assert raised, "expected ClientConnectorError after all retries fail"
    assert call_count["n"] == 3, f"expected 3 attempts before giving up, got {call_count['n']}"
    print("OK  retry gave up after 3 attempts and raised ClientConnectorError")


# ─── Tests: wait_for_completion survives transient drops ──────────────────

async def test_wait_for_completion_survives_transient_drop():
    """A brief ComfyUI hiccup mid-render should not kill the poll loop.

    Simulates: poll 1 succeeds (no entry yet), poll 2-3 connection-refused,
    poll 4+ succeeds and shows completed. Should return the entry, not raise.
    """
    client = _make_client()
    client.request_timeout = 30  # short for test
    get_count = {"n": 0}  # increments every get() call (incl. failures)
    json_count = {"n": 0}  # increments only on successful json() (success path)

    class FakeResp:
        status = 200
        async def json(self):
            json_count["n"] += 1
            # First successful json: empty. Second+: completed.
            if json_count["n"] < 2:
                return {}  # prompt_id not in history yet
            return {
                "p1": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {},
                }
            }
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class FakeSession:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def get(self, *a, **kw):
            get_count["n"] += 1
            # get #1: success (empty). get #2-3: connection refused. get #4+: success.
            if get_count["n"] in (2, 3):
                raise _conn_error()
            return FakeResp()

    with patch("tools.comfyui_client.aiohttp.ClientSession", FakeSession), \
         patch("asyncio.sleep", new=AsyncMock()):
        entry = await client.wait_for_completion("p1", poll_interval=0.01)

    assert entry["status"]["completed"] is True
    assert get_count["n"] >= 4, f"expected ≥4 polls (incl. 2 failures), got {get_count['n']}"
    print(f"OK  wait_for_completion survived 2 transient drops over {get_count['n']} polls")


# ─── Tests: decorator unit tests ──────────────────────────────────────────

async def test_retryable_exc_tuple_contents():
    """Confirm the retryable exception tuple covers the documented cases."""
    names = [e.__name__ for e in _RETRYABLE_EXC]
    assert "ClientConnectorError" in names
    assert "ServerDisconnectedError" in names
    assert "ClientOSError" in names
    assert "TimeoutError" in names  # asyncio.TimeoutError
    # And critically does NOT include generic Exception or RuntimeError
    assert Exception not in _RETRYABLE_EXC
    assert RuntimeError not in _RETRYABLE_EXC
    print(f"OK  _RETRYABLE_EXC = {names} (no over-broad Exception/RuntimeError)")


# ─── Runner ───────────────────────────────────────────────────────────────

async def main():
    print("=== ComfyUI client retry tests ===\\n")
    logging.basicConfig(level=logging.CRITICAL)  # silence warning logs in tests
    await test_queue_prompt_retries_on_connection_refused()
    await test_upload_image_retries_on_disconnect()
    await test_fetch_output_retries_on_oserror()
    await test_queue_prompt_does_not_retry_http_400()
    await test_queue_prompt_does_not_retry_http_500()
    await test_retry_gives_up_after_max_attempts()
    await test_wait_for_completion_survives_transient_drop()
    await test_retryable_exc_tuple_contents()
    print("\\n==================================================")
    print("ALL TESTS PASSED")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())
