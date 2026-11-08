"""One SDK client per provider target, shared by every call in the run.

Each OpenAI or Anthropic client holds its own pool of open connections. The
pipeline used to build a new client for every model call and never close it,
and tracing's wrapper on ``create`` ties each client into a reference cycle,
so its sockets stayed open until Python's cycle collector ran. In the 8 Nov
local Qwen run from the Mac, after about 7,000 calls in 2.5 hours, every new
connection to the model server failed with "Connection error" (match checks,
dispute checks and the whole grounding pass), while curl from another
process still reached the server. One shared client reuses its connections.
"""

import threading
from typing import Any, Callable, Dict, Hashable

_CLIENTS: Dict[Hashable, Any] = {}
_LOCK = threading.RLock()  # an Instructor client makes its raw one inside


def shared_client(key: Hashable, make: Callable[[], Any]) -> Any:
    """The client stored under ``key``, made with ``make()`` the first time."""
    with _LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = _CLIENTS[key] = make()
        return client


def clear_clients() -> None:
    """Forget every stored client (tests patch the SDK classes per test)."""
    with _LOCK:
        _CLIENTS.clear()
