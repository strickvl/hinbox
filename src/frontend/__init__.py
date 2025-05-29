# Export app for ASGI loading
from .app_config import app

# Import routes to register them
from .routes import events, home, locations, organizations, people  # noqa: F401

__all__ = ["app"]
