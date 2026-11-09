"""Run the reader: ``uv run python -m src.frontend`` (or ``just frontend``).

It listens on 127.0.0.1 only, so the profiles and article texts of a private
corpus are not served to the local network. Set ``HINBOX_FRONTEND_HOST``
(e.g. ``0.0.0.0``) to share it on purpose, and ``HINBOX_FRONTEND_PORT`` to
change the port.
"""

import os
import sys

import uvicorn

from ..constants import DEFAULT_FRONTEND_PORT
from .app import app


def main() -> None:
    host = os.getenv("HINBOX_FRONTEND_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("HINBOX_FRONTEND_PORT", "") or DEFAULT_FRONTEND_PORT)
    print(
        f"Hinbox reader: http://{'localhost' if host == '127.0.0.1' else host}:{port}"
    )
    try:
        uvicorn.run(app, host=host, port=port, reload=False)
    except OSError as e:
        if "address already in use" in str(e).lower():
            print(f"\nPort {port} is already in use. Stop the other server, or set")
            print(f"HINBOX_FRONTEND_PORT={port + 1} and run again.")
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
