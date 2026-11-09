"""The entity audit counts lost facts, repeats and dispute claims without a model."""

from scripts.audit_entities import (
    audit_history,
    cited,
    repeats_and_disputes,
    version_texts,
)


def test_lost_and_orphaned_facts():
    texts = [
        "Nashiri was charged in 2011 over the Cole bombing in Yemen^[a1].",
        "Nashiri was charged in 2011 over the Cole bombing in Yemen^[a1]. "
        "His lawyer Richard Kammen resigned from the defense team^[a2].",
        "Nashiri still faces a capital trial at Guantanamo^[a2].",
    ]
    history = audit_history(texts)
    lost = {x["sentence"][:7]: x for x in history["lost"]}
    assert set(lost) == {"Nashiri", "His law"}
    assert lost["Nashiri"]["orphaned"] and not lost["His law"]["orphaned"]
    assert history["shrinks"] == [{"version": 3, "before": 127, "after": 55}]


def test_reworded_fact_is_kept():
    texts = [
        "The attack killed seventeen American sailors aboard the destroyer^[a].",
        "The attack aboard the destroyer killed seventeen American sailors^[a].",
    ]
    assert audit_history(texts)["lost"] == []


def test_repeats_and_disputes():
    text = (
        "Judge Pohl suppressed the FBI interrogation accounts of the suspects^[a]. "
        "Prosecutors appealed the ruling^[a]. "
        "Judge Pohl suppressed the FBI interrogation accounts of the suspects^[b]. "
        "Sources disagree on the date of the ruling^[c]."
    )
    repeats, disputes = repeats_and_disputes(text)
    assert len(repeats) == 1 and repeats[0].endswith("^[b].")
    assert disputes == ["Sources disagree on the date of the ruling^[c]."]
    assert cited("x^[a, b] y^[c]") == ["a", "b", "c"]


def test_a_sentence_a_historian_struck_is_not_lost():
    from scripts.audit_entities import corrected_versions

    old = "Qadir was held.^[a1] He was force-fed for several weeks.^[a3]"
    row = {
        "profile": {"text": "Qadir was held.^[a1]"},
        "profile_versions": {
            "versions": [
                {"version_number": 1, "profile_data": {"text": old}},
                {
                    "version_number": 2,
                    "profile_data": {"text": "Qadir was held.^[a1]"},
                    "trigger_article_id": "corrections:c-0001",
                },
            ]
        },
    }
    history = audit_history(version_texts(row), corrected_versions(row))
    assert history["lost"] == [] and len(history["struck"]) == 1
