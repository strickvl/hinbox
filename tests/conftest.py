"""Test configuration and fixtures."""

from pathlib import Path
from typing import Generator

import pytest
from sqlmodel import Session, SQLModel

from src.database import get_engine


@pytest.fixture(name="test_db_path")
def fixture_test_db_path(tmp_path: Path) -> str:
    """Create a temporary database path for testing.

    Args:
        tmp_path: Pytest fixture providing a temporary directory

    Returns:
        str: Path to the test database
    """
    db_path = tmp_path / "test.db"
    return str(db_path)


@pytest.fixture(name="engine")
def fixture_engine(test_db_path: str):
    """Create a new database engine for testing.

    Args:
        test_db_path: Path to the test database

    Returns:
        Engine: SQLAlchemy engine instance
    """
    engine = get_engine(f"sqlite:///{test_db_path}")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture(name="session")
def fixture_session(engine) -> Generator[Session, None, None]:
    """Create a new database session for testing.

    Args:
        engine: SQLAlchemy engine instance

    Yields:
        Session: SQLAlchemy session
    """
    with Session(engine) as session:
        yield session
