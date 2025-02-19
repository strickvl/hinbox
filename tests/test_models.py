"""Tests for database models."""

from datetime import datetime
from uuid import uuid4

from sqlmodel import Session, select

from src.models import (
    Detainee,
    Event,
    EventType,
    Location,
    Organisation,
    OrganisationType,
)


def test_create_location(session: Session):
    """Test creating a location."""
    location = Location(
        id=uuid4(),
        facility="Camp Delta",
        latitude=19.9031,
        longitude=-75.0967,
    )
    session.add(location)
    session.commit()

    # Query to verify
    db_location = session.exec(select(Location)).first()
    assert db_location is not None
    assert db_location.facility == "Camp Delta"
    assert db_location.latitude == 19.9031
    assert db_location.longitude == -75.0967


def test_create_event_with_relationships(session: Session):
    """Test creating an event with all its relationships."""
    # Create a location
    location = Location(
        id=uuid4(),
        facility="Camp Delta",
    )

    # Create a detainee
    detainee = Detainee(
        id=uuid4(),
        name="John Doe",
        nationality="Unknown",
        birth_date=datetime(1980, 1, 1),
        detainee_number="ISN123",
        detention_date=datetime(2002, 1, 1),
    )

    # Create an organisation
    organisation = Organisation(
        id=uuid4(),
        name="Department of Defense",
        type=OrganisationType.MILITARY,
        country="USA",
    )

    # Create an event linking everything
    event = Event(
        id=uuid4(),
        title="Initial Transfer",
        description="Transfer to Guantanamo Bay",
        classification=EventType.DETAINEE_TRANSFER,
        start=datetime(2002, 1, 1),
        location=location,
        involved_detainees=[detainee],
        involved_organisations=[organisation],
        tags=["transfer", "initial_detention"],
    )

    # Add and commit
    session.add(event)
    session.commit()

    # Query to verify
    db_event = session.exec(
        select(Event).where(Event.title == "Initial Transfer")
    ).first()
    assert db_event is not None
    assert db_event.location.facility == "Camp Delta"
    assert len(db_event.involved_detainees) == 1
    assert db_event.involved_detainees[0].detainee_number == "ISN123"
    assert len(db_event.involved_organisations) == 1
    assert db_event.involved_organisations[0].name == "Department of Defense"
    assert len(db_event.tags) == 2
