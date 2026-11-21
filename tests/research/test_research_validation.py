import pytest

from src.research.extraction import (
    KnowledgeBasis,
    KnowledgePositionAction,
    NormalizedClaimAction,
    PassageAction,
    ProposalBatch,
    RelationshipAction,
    SourceAssertionAction,
)
from src.research.locators import TextLocator
from src.research.validation import acceptance_eligible, validate_proposal_batch


def test_deterministic_quote_validation_precedes_acceptance_eligibility():
    passage = PassageAction(
        action_id="passage_1",
        locator=TextLocator(representation_id="representation_1", start=0, end=5),
        selected_quote="wrong",
    )
    assertion = SourceAssertionAction(
        action_id="assertion_1",
        passage_action_ids=[passage.action_id],
        assertion_text="The source says hello.",
        attribution="Witness memoir",
    )

    observations = validate_proposal_batch(
        ProposalBatch(actions=[passage, assertion]),
        lambda representation_id, start, end: "hello",
        representation_id="representation_1",
    )

    assert [item.code for item in observations] == ["quote_mismatch"]
    assert acceptance_eligible(observations) is False


def test_claim_and_assessment_must_resolve_through_batch_passages_and_assertions():
    claim = NormalizedClaimAction(
        passage_action_ids=["missing_passage"],
        assertion_action_ids=["missing_assertion"],
        normalized_text="A normalized proposition",
        rationale="Proposed only to compare accounts",
    )
    assessment = KnowledgePositionAction(
        assertion_action_id="missing_assertion",
        passage_action_ids=["missing_passage"],
        basis=KnowledgeBasis.HEARSAY,
        rationale="The speaker attributes the account to another person.",
        assessor="research-agent/test",
        uncertainty="The intermediate speaker is not identified.",
    )

    observations = validate_proposal_batch(
        ProposalBatch(actions=[claim, assessment]),
        lambda representation_id, start, end: "unused",
        representation_id="representation_1",
    )

    assert {item.code for item in observations} == {
        "missing_passage_support",
        "missing_assertion_support",
        "missing_assessed_assertion",
    }
    assert acceptance_eligible(observations) is False


def test_relationship_allegation_retains_attribution_and_provisional_semantics():
    passage = PassageAction(
        action_id="passage_1",
        locator=TextLocator(representation_id="representation_1", start=0, end=5),
        selected_quote="claim",
    )
    relationship = RelationshipAction(
        passage_action_ids=[passage.action_id],
        subject_candidate="A",
        predicate="allegedly harmed",
        object_candidate="B",
        attribution="The complaint alleges",
        allegation=True,
    )
    observations = validate_proposal_batch(
        ProposalBatch(actions=[passage, relationship]),
        lambda representation_id, start, end: "claim",
        representation_id="representation_1",
    )

    assert acceptance_eligible(observations) is True
    assert relationship.allegation is True
    assert relationship.attribution == "The complaint alleges"


def test_normalized_claim_cannot_launder_an_allegation():
    passage = PassageAction(
        action_id="passage_1",
        locator=TextLocator(representation_id="representation_1", start=0, end=5),
        selected_quote="claim",
    )
    assertion = SourceAssertionAction(
        action_id="assertion_1",
        passage_action_ids=[passage.action_id],
        assertion_text="A witness alleges misconduct.",
        attribution="The complaint's witness alleges",
        allegation=True,
    )
    laundering_claim = NormalizedClaimAction(
        assertion_action_ids=[assertion.action_id],
        passage_action_ids=[passage.action_id],
        normalized_text="Misconduct occurred.",
        rationale="Collapsed wording",
        allegation=False,
        attribution=None,
    )

    observations = validate_proposal_batch(
        ProposalBatch(actions=[passage, assertion, laundering_claim]),
        lambda representation_id, start, end: "claim",
        representation_id="representation_1",
    )

    assert {item.code for item in observations} >= {
        "allegation_status_laundered",
        "unattributed_normalized_allegation",
    }
    assert acceptance_eligible(observations) is False


@pytest.mark.parametrize("basis", list(KnowledgeBasis))
def test_knowledge_position_categories_remain_claim_relative(basis: KnowledgeBasis):
    assessment = KnowledgePositionAction(
        assertion_action_id="assertion_1",
        passage_action_ids=["passage_1"],
        basis=basis,
        rationale="This rationale applies only to the supported assertion.",
        assessor="historian-or-run-id",
        uncertainty="The basis remains open to revision.",
    )

    assert assessment.basis is basis
    assert not hasattr(assessment, "source_credibility_score")
