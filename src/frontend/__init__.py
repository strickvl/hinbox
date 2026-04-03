# Export app for ASGI loading
from .app_config import app

# Import routes to register them
from .routes import (  # noqa: F401
    eval,
    eval_relevance,
    events,
    home,
    locations,
    organizations,
    people,
)

__all__ = ["app"]
