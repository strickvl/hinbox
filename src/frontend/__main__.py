"""
Refactored aggregator for the Guantánamo Entities Browser.
All the old code has been split among data_access.py, utils.py, filters.py, app_config.py, and routes/*.
We simply import them here to ensure everything is registered with 'app, rt'.
"""

from .app_config import app

# Import the routes so they get registered

# We only need to define the run logic if this file is run directly under -m syntax:
if __name__ == "__main__":
    from fasthtml.common import serve

    serve(app)
