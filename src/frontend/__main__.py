"""
Refactored aggregator for the Guantánamo Entities Browser.
All the old code has been split among data_access.py, utils.py, filters.py, app_config.py, and routes/*.
We simply import them here to ensure everything is registered with 'app, rt'.
"""

# Import and expose the app at module level for FastHTML serve() to find
from .app_config import app

# Import the routes so they get registered
from .routes import events, home, locations, organizations, people  # noqa: F401

# We only need to define the run logic if this file is run directly under -m syntax:
if __name__ == "__main__":
    import sys

    import uvicorn

    try:
        uvicorn.run(app, host="0.0.0.0", port=5001, reload=False)
    except OSError as e:
        if "address already in use" in str(e).lower():
            print("\n❌ Error: Port 5001 is already in use!")
            print("💡 Try one of these solutions:")
            print("   • Stop the existing server (Ctrl+C in the other terminal)")
            print("   • Kill existing process: pkill -f 'uvicorn.*5001'")
            print("   • Use a different port: uvicorn src.frontend:app --port 5002")
            print()
            sys.exit(1)
        else:
            raise
