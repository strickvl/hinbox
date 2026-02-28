"""
Refactored aggregator for the Guantánamo Entities Browser.
All the old code has been split among data_access.py, utils.py, filters.py, app_config.py, and routes/*.
We simply import them here to ensure everything is registered with 'app, rt'.
"""

# Import and expose the app at module level for FastHTML serve() to find
import sys

import uvicorn

from ..constants import DEFAULT_FRONTEND_PORT
from .app_config import app

# Import the routes so they get registered
from .routes import eval, events, home, locations, organizations, people  # noqa: F401

# We only need to define the run logic if this file is run directly under -m syntax:
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=DEFAULT_FRONTEND_PORT, reload=False)
    except OSError as e:
        if "address already in use" in str(e).lower():
            print(f"\n❌ Error: Port {DEFAULT_FRONTEND_PORT} is already in use!")
            print("💡 Try one of these solutions:")
            print("   • Stop the existing server (Ctrl+C in the other terminal)")
            print(
                f"   • Kill existing process: pkill -f 'uvicorn.*{DEFAULT_FRONTEND_PORT}'"
            )
            print(
                f"   • Use a different port: uvicorn src.frontend:app --port {DEFAULT_FRONTEND_PORT + 1}"
            )
            print()
            sys.exit(1)
        else:
            raise
