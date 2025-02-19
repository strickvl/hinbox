import logging
import os
from pathlib import Path
from typing import Optional

from sqlmodel import SQLModel, create_engine

# Set up logging
logger = logging.getLogger(__name__)


def get_database_url(database_path: Optional[str] = None) -> str:
    """Get the database URL, creating the directory if needed.

    Args:
        database_path: Optional path to the database file. If not provided,
            defaults to 'data/hinbox.db' in the project root.

    Returns:
        str: The database URL in SQLAlchemy format
    """
    if database_path is None:
        # Default to data/hinbox.db in project root
        database_path = os.path.join("data", "hinbox.db")

    # Ensure the directory exists
    db_dir = os.path.dirname(database_path)
    if db_dir:
        Path(db_dir).mkdir(parents=True, exist_ok=True)

    # Convert to SQLAlchemy URL format
    return f"sqlite:///{database_path}"


def get_engine(database_url: Optional[str] = None):
    """Get SQLAlchemy engine instance.

    Args:
        database_url: Optional database URL. If not provided, uses default location.

    Returns:
        Engine: SQLAlchemy engine instance
    """
    if database_url is None:
        database_url = get_database_url()

    # Create engine with some good defaults for SQLite
    connect_args = {"check_same_thread": False}
    engine = create_engine(database_url, echo=False, connect_args=connect_args)
    return engine


def init_db(database_path: Optional[str] = None) -> None:
    """Initialize the database with all models.

    This will create all tables if they don't exist. It's safe to call this
    multiple times as it won't recreate existing tables.

    Args:
        database_path: Optional path to the database file. If not provided,
            uses the default location.
    """
    try:
        engine = get_engine(get_database_url(database_path))
        logger.info("Creating database tables...")

        # Import all models here to avoid circular imports
        # This also ensures they're registered with SQLModel
        import src.models  # noqa

        # Create all tables
        SQLModel.metadata.create_all(engine)
        logger.info("Database initialization complete")

    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
