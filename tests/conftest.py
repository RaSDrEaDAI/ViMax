"""pytest harness for the ViMax suite.

Two jobs:

1. No test touches the network. An autouse fixture refuses every socket
   connect to a non-loopback address, so a mock that silently fails to apply
   surfaces as a loud ConnectionRefusedError instead of a real (possibly paid)
   model or generator call. Loopback stays open: the ComfyUI retry tests
   deliberately dial 127.0.0.1:1 to get a genuine connection-refused.

2. Bare ``async def test_*`` functions run. Several files were written to be
   executed as scripts and have module-level coroutine tests; without an async
   plugin pytest skips-with-failure on every one. This runs them on a fresh
   event loop. (unittest.IsolatedAsyncioTestCase classes are unaffected.)
"""

import asyncio
import inspect
import ipaddress
import os
import socket
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _is_loopback(address) -> bool:
    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX / socketpair-style addresses are local by definition
    host = address[0]
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _guarded_connect(self, address):
    if not _is_loopback(address):
        raise ConnectionRefusedError(f"tests may not reach the network (attempted {address!r})")
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    if not _is_loopback(address):
        raise ConnectionRefusedError(f"tests may not reach the network (attempted {address!r})")
    return _real_connect_ex(self, address)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    yield


@pytest.fixture(autouse=True)
def _isolated_error_dump(monkeypatch, tmp_path):
    # utils.retry.after_func appends every retried exception to
    # $VIMAX_WORKING_DIR/.vimax_errors.log (falling back to cwd = the repo root).
    # Tests that exercise retries must not write into the real log.
    monkeypatch.setenv("VIMAX_WORKING_DIR", str(tmp_path))
    yield


@pytest.fixture
def fake_chat_model():
    """Factory: ``fake_chat_model([resp1, resp2, ...])``."""
    from tests.fakes import FakeChatModel
    return FakeChatModel


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None
    if pyfuncitem.config.pluginmanager.hasplugin("asyncio"):
        return None  # a real async plugin is installed; let it own these
    names = pyfuncitem._fixtureinfo.argnames
    kwargs = {name: pyfuncitem.funcargs[name] for name in names}
    asyncio.run(pyfuncitem.obj(**kwargs))
    return True
