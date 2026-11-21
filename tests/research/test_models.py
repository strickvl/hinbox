from src.research.models import AnchorCreate, AnchorKind, new_research_id


def test_stable_ids_are_opaque_and_distinct() -> None:
    first = new_research_id("anchor")
    second = new_research_id("anchor")

    assert first.startswith("anchor_")
    assert first != second


def test_same_display_name_does_not_define_identity() -> None:
    first = AnchorCreate(kind=AnchorKind.PERSON, display_name="Muhammad")
    second = AnchorCreate(kind=AnchorKind.PERSON, display_name="Muhammad")

    assert first.id != second.id
    assert first.display_name == second.display_name
