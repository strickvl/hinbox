"""Writing an encyclopedia page from a profile's notes, and the reader's view
of it. Made-up pages only (the repository is public)."""

import json
import time

from src.engine.written_page import (
    WRITTEN_KEY,
    Facts,
    SentenceChecks,
    WPage,
    condense,
    missing_details,
    page_notes,
    store_page,
    stored_pages,
    write_page,
)
from src.utils.corrections import sentence_key
from src.utils.file_ops import write_entities_table
from tests.test_frontend import A1, A2, A3, _arts, _report, client  # noqa: F401

CITY = (
    "### Background\n\n"
    "Port Elm is a harbour city in the north of Elmland.^[a1]\n\n"
    "### Arrests\n\n"
    "Police arrested Tom Vale in Port Elm in March 2004.^[a2] "
    "Tom Vale was 19 years old when he was arrested.^[a2] "
    "Police arrested Ann Roe in Port Elm in May 2005.^[a3]\n"
)
TAGS = {
    sentence_key("Tom Vale was 19 years old when he was arrested."): {
        "verdict": "other"
    }
}


def _city(**profile):
    arts = [{"article_id": a, "article_published_date": d}
            for a, d in (("a1", "2003-01-02"), ("a2", 1078444800), ("a3", "2005-05-06"))]  # fmt: skip
    return {"name": "Port Elm", "type": "city", "articles": arts,
            "profile": {"text": CITY, "note_relevance_json": json.dumps(TAGS), **profile}}  # fmt: skip


def test_notes_leave_out_off_topic_ones_and_carry_dates():
    notes = page_notes(_city())
    assert [n.n for n in notes] == [1, 2, 3]
    assert "19 years old" not in " ".join(n.text for n in notes)
    assert notes[1].date == "2004-03-05" and notes[1].heading == "Arrests"


def test_the_code_check_catches_added_names_numbers_and_dates():
    notes = ["Police arrested Tom Vale in Port Elm in March 2004."]
    assert (
        missing_details(
            "Police arrested Tom Vale there in March 2004.", notes, "Port Elm"
        )
        == []
    )
    # Pakistani for Pakistan, a possessive, a common opening word
    assert (
        missing_details("However, Elmland's police acted.", ["Elmlandish police"], "X")
        == []
    )
    assert missing_details("Police arrested Tom Vale in 2005.", notes, "Port Elm") == [
        "2005"
    ]
    assert missing_details(
        "Police and Interpol arrested Tom Vale.", notes, "Port Elm"
    ) == ["Interpol"]
    assert (
        missing_details("He was arrested in June.", notes, "Port Elm") == []
    )  # months pass


def _ask(page, checks=None, facts=None):
    calls = []

    def ask(msgs, model, max_tokens):
        calls.append(model)
        if model is WPage:
            return WPage.model_validate(page)
        if model is SentenceChecks:
            return SentenceChecks.model_validate(checks or {"checks": []})
        return Facts.model_validate(facts)

    return ask, calls


PAGE = {
    "opening": [{"text": "Port Elm is a harbour city in the north of Elmland.", "notes": [1]}],
    "sections": [{"heading": "Arrests", "paragraphs": [{"sentences": [
        {"text": "Police arrested Tom Vale there in March 2004 and Ann Roe in May 2005.", "notes": [2, 3]},
        {"text": "Interpol helped with both arrests.", "notes": [2, 3]},  # invented
        {"text": "Both were later freed.", "notes": [3]},
        {"text": "A sentence citing nothing real.", "notes": [9]},
    ]}]}],
}  # fmt: skip


def test_a_written_page_cites_its_notes_articles_and_drops_unbacked_sentences():
    checks = {"checks": [
        {"sentence": 1, "starts_with": "Port Elm is", "problem": "none", "verdict": "supported"},
        {"sentence": 2, "starts_with": "Police arrested Tom", "problem": "none", "verdict": "supported"},
        {"sentence": 3, "starts_with": "Both were later", "problem": "no release", "verdict": "not_supported"},
    ]}  # fmt: skip
    ask, calls = _ask(PAGE, checks)
    page = write_page(_city(), "locations", "m", ask, [("Overview", "what it is")])
    assert calls == [WPage, SentenceChecks]
    assert page["text"] == (
        "Port Elm is a harbour city in the north of Elmland.^[a1]\n\n"
        "## Arrests\n\n"
        "Police arrested Tom Vale there in March 2004 and Ann Roe in May 2005.^[a2, a3]\n"
    )
    r = page["report"]
    assert (r["sentences"], r["kept"], r["dropped_code"], r["dropped_check"]) == (
        5,
        2,
        2,
        1,
    )
    assert r["notes_used"] == 3 and r["headings"] == ["Arrests"]
    dropped = {s["text"]: s["dropped"] for s in page["sentences"]}
    assert dropped["Interpol helped with both arrests."].endswith("Interpol")
    assert dropped["A sentence citing nothing real."] == "cites no note"
    # The prompt offered the kind's headings and the right length
    profile = {}
    store_page(profile, page)
    assert list(stored_pages(profile)) == ["m"]
    assert isinstance(profile[WRITTEN_KEY], str)


def test_long_notes_are_condensed_into_facts_that_keep_their_notes(monkeypatch):
    import src.engine.written_page as wp

    monkeypatch.setattr(wp, "WRITE_CHARS", 120)
    monkeypatch.setattr(wp, "CONDENSE_CHARS", 10_000)
    notes = page_notes(_city())
    facts = {"facts": [{"text": "Police arrested Tom Vale and Ann Roe.", "notes": [2, 3]},
                       {"text": "A fact naming no shown note.", "notes": [7]}]}  # fmt: skip
    ask, calls = _ask(PAGE, facts=facts)
    report = {}
    out = condense(notes, "Port Elm", "place", ask, report)
    assert len(out) == 1 and out[0].n == 1
    assert out[0].articles == ["a2", "a3"] and len(out[0].keys) == 2
    assert out[0].date == "2004-03-05" and report["condense_rounds"] == 1


def test_the_reader_shows_the_written_page_first_and_its_notes_in_the_panel(
    client,  # noqa: F811
    tmp_path,
):
    from tests.test_frontend import TEXT

    written = {
        "model": "ollama/made-up-qwen",
        "written_at": "2026-11-21T16:00:00+00:00",
        "text": f"Yusuf Qadir is a made-up detainee who studied engineering.^[{A2}, {A1}]\n",
        "sentences": [{"text": "Yusuf Qadir is a made-up detainee who studied engineering.",
                       "notes": [sentence_key("Yusuf Qadir is a made-up detainee held at the base."),
                                 sentence_key("Qadir studied engineering at a local college.")],
                       "dropped": "", "check": "supported", "problem": ""}],
        "report": {"notes": 5, "notes_used": 2, "dropped_code": 0, "dropped_check": 0},
    }  # fmt: skip
    person = {
        "name": "Yusuf Qadir", "type": "detainee", "articles": _arts(A1, A2, A3),
        "profile": {"text": TEXT, "sources": [A1, A2, A3],
                    WRITTEN_KEY: json.dumps({written["model"]: written})},
        "profile_grounding": _report(),
    }  # fmt: skip
    time.sleep(0.01)
    write_entities_table("people", [person], str(tmp_path / "entities"))
    html = client.get("/people/Yusuf%20Qadir").text
    assert "who studied engineering" in html and "force-fed" not in html
    assert "Written by local Qwen from" in html and "Notes and evidence" in html
    assert 'data-written="ollama/made-up-qwen"' in html
    assert "mainly about other subjects" not in html
    notes = client.get("/people/Yusuf%20Qadir?view=notes").text
    assert "force-fed" in notes and "who studied engineering" not in notes
    panel = client.get(
        f"/panel?t=people&k=Yusuf%20Qadir&a={A1}&c=0&w=ollama/made-up-qwen"
    ).text
    assert "Notes behind this sentence" in panel
    assert "Qadir studied engineering at a local college." in panel
    assert "made-up detainee held at the base" in panel
    # The panel without a written page is the usual source panel
    assert (
        "Notes behind"
        not in client.get(f"/panel?t=people&k=Yusuf%20Qadir&a={A1}&c=1").text
    )
