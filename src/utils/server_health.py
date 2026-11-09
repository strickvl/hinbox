"""Stop a run when the model server can't be reached at all.

On 9 Nov a 150-article local run lost its route to the model server after
article 43 ("No route to host" on every call for 40 minutes, while a new
process on the same machine got through). The run kept going: each article's
calls failed, and it went on to the next. ``watch_connections()`` wraps an SDK
client's ``create`` and counts connection failures in a row across all
threads; any answer from the server resets the count. Once
``HINBOX_MAX_CONNECTION_FAILURES`` (default 20, ``0`` = never stop) have
failed in a row, ``unreachable()`` says why, and the article loop stops the
run after saving what is merged (``ModelServerUnreachable``). A timeout is not
counted: the server is there, just slow.
"""

import functools
import os
import threading
from typing import Any, Callable, Optional

_lock = threading.Lock()
_in_a_row = 0
_last_error = ""


class ModelServerUnreachable(RuntimeError):
    """Raised between articles once every recent model call failed to connect."""


def failure_limit() -> int:
    return int(os.getenv("HINBOX_MAX_CONNECTION_FAILURES", "20"))


def is_connection_error(exc: BaseException) -> bool:
    """A request that never reached the server (not a timeout or an HTTP error)."""
    for module in ("openai", "anthropic"):
        try:
            sdk = __import__(module)
        except ImportError:
            continue
        if isinstance(exc, sdk.APIConnectionError) and not isinstance(
            exc, sdk.APITimeoutError
        ):
            return True
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with both SDKs
        return False
    return isinstance(exc, httpx.ConnectError)


def record(error: Optional[BaseException]) -> None:
    """Count one finished request: ``None`` for an answer, else its error."""
    global _in_a_row, _last_error
    with _lock:
        if error is None or not is_connection_error(error):
            _in_a_row = 0  # the server answered, even if with an error
        else:
            _in_a_row += 1
            cause = error.__cause__ or error
            _last_error = f"{type(cause).__name__}: {cause}"


def unreachable() -> Optional[str]:
    """Why the run should stop, or None while the server is answering."""
    limit = failure_limit()
    with _lock:
        if limit <= 0 or _in_a_row < limit:
            return None
        return f"the last {_in_a_row} model calls could not connect ({_last_error})"


def reset() -> None:
    global _in_a_row, _last_error
    with _lock:
        _in_a_row, _last_error = 0, ""


def watch_connections(create: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an SDK ``create`` so every request is counted by ``record()``."""

    @functools.wraps(create)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            response = create(*args, **kwargs)
        except Exception as e:
            record(e)
            raise
        record(None)
        return response

    return wrapper
