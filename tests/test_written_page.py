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


def test_note_references_written_into_sentences_are_taken_out():
    from src.engine.written_page import WPage, flatten

    page = WPage.model_validate({"opening": [{"text": (
        "Port Elm is a city {1}. Police arrested Tom Vale there in 2004 (N2, N3)."),
        "notes": []}], "sections": [{"heading": "Arrests", "paragraphs": [{"sentences": [
        {"text": "The U.S. Army took Ann Roe in 2005 [3].", "notes": [3]},
        {"text": "The port had 4,000 workers (2004).", "notes": [1]}]}]}]})  # fmt: skip
    got = [(w.text, w.notes) for w in flatten(page, max_n=3)]
    assert got == [
        ("Port Elm is a city.", [1]),
        ("Police arrested Tom Vale there in 2004.", [2, 3]),
        ("The U.S. Army took Ann Roe in 2005.", [3]),
        ("The port had 4,000 workers (2004).", [1]),  # a year, not a note
    ]
    # A sentence's ordinary first word is not taken for a name
    assert missing_details("Reports differ on the date.", ["a date"], "X") == []


def _ask(page, checks=None, facts=None):
    calls = []

    def ask(msgs, model, max_tokens):
        calls.append(model)
        if model is WPage:
            return WPage.model_validate(page)
        if model is SentenceChecks:
            return SentenceChecks.model_validate(checks or {"checks": []})
        if model.__name__ == "Revisions":
            return model.model_validate({"revisions": []})
        return Facts.model_validate(facts)

    return ask, calls


PAGE = {
    "opening": [{"text": "Port Elm is a harbour city in the north of Elmland.", "notes": [1]}],
    "sections": [{"heading": "Arrests", "paragraphs": [{"sentences": [
        {"text": "Police arrested Tom Vale there in March 2004 and Ann Roe in May 2005.", "notes": [2, 3]},
        {"text": "Both arrests were helped by Interpol.", "notes": [2, 3]},  # invented
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
    page = write_page(
        _city(), "locations", "m", ask, [("Overview", "what it is")], approach="single"
    )
    assert calls[:2] == [WPage, SentenceChecks] and page["model"] == "m"
    assert calls[2].__name__ == "Revisions" and len(calls) == 3
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
    assert dropped["Both arrests were helped by Interpol."].endswith("Interpol")
    assert dropped["A sentence citing nothing real."] == "cites no note"
    # The prompt offered the kind's headings and the right length
    profile = {}
    store_page(profile, page)
    assert list(stored_pages(profile)) == ["m"]
    assert isinstance(profile[WRITTEN_KEY], str)


def test_a_planned_page_is_written_part_by_part_with_conflicts_weighed():
    from src.engine.written_page import WBody, WParagraph
    from src.engine.written_plan import Conflicts, Plan

    plan = {
        "identity": "a harbour city in Elmland",
        "why_it_matters": "two arrests were made there",
        "parts": [
            {"heading": "Overview", "covers": "what it is", "notes": [1]},
            {"heading": "Arrests", "covers": "who was arrested", "notes": [2, 3, 3, 9]},
            {"heading": "## Empty", "covers": "nothing", "notes": [9]},
        ],
        "left_out": [{"note": 2, "why": "someone else"}],
        "opening_notes": [],
    }
    conflicts = {
        "conflicts": [
            {
                "about": "arrest year",
                "one_side": "2004",
                "one_notes": [2],
                "other_side": "2005",
                "other_notes": [3],
            },
            # an exact date and a vaguer one that fits it: no conflict
            {
                "about": "arrest",
                "one_side": "March 2004",
                "one_notes": [2],
                "other_side": "March 5, 2004",
                "other_notes": [3],
            },
            {
                "about": "a note of another part",
                "one_side": "2003",
                "one_notes": [1],
                "other_side": "2005",
                "other_notes": [3],
            },
        ]
    }
    body = {
        "paragraphs": [
            {
                "sentences": [
                    {
                        "text": "Police arrested Tom Vale there in March 2004 and Ann Roe in May 2005.",
                        "notes": [2, 3],
                    },
                    {
                        "text": "Police arrested Tom Vale and Ann Roe there.",
                        "notes": [2, 3],
                    },
                ]
            }
        ]
    }  # a repeat
    opening = {
        "sentences": [
            {
                "text": "Port Elm is a harbour city in the north of Elmland.",
                "notes": [1],
            }
        ]
    }
    seen = []

    def ask(msgs, model, max_tokens):
        seen.append((model, msgs[0]["content"], msgs[1]["content"]))
        reply = {Plan: plan, Conflicts: conflicts, WBody: body, WParagraph: opening}
        return model.model_validate(reply.get(model) or {"checks": []})

    page = write_page(_city(), "locations", "m", ask, check=False)
    assert page["model"] == "m#plan"
    assert [m for m, _, _ in seen] == [Plan, Conflicts, WBody, WParagraph]
    section_prompt, section_notes = seen[2][1], seen[2][2]
    assert "one event with two accounts" in section_prompt
    assert "arrest year: 2004 (notes 2) against 2005 (notes 3)" in section_prompt
    assert "March 5" not in section_prompt and "another part" not in section_prompt
    assert "N1 " not in section_notes and "N2 " in section_notes  # its notes only
    # The opening is written last, from the sections and the overview's note
    assert "Police arrested Tom Vale there in March 2004" in seen[3][2]
    assert "N1 " in seen[3][2]
    assert page["text"].startswith("Port Elm is a harbour city")
    assert "## Arrests" in page["text"] and "Empty" not in page["text"]
    assert "Overview" not in page["text"]
    plan_out = page["report"]["plan"]
    assert plan_out["parts"] == [{"heading": "Arrests", "notes": [2, 3]}]
    assert plan_out["left_out"] == []  # note 2 is used after all
    assert [s["dropped"] for s in page["sentences"]][
        -1
    ] == "repeats an earlier sentence"


def test_a_dropped_sentence_is_rewritten_once_and_checked_again():
    from src.engine.written_page import Written, page_notes
    from src.engine.written_revise import Revisions, revise

    notes = page_notes(_city())
    bad = Written("Police and Interpol arrested Tom Vale in 2004.", [2], "Arrests", 1)
    bad.dropped = "not in its notes: Interpol"
    seen = []

    def ask(msgs, model, max_tokens):
        seen.append(model)
        if model is Revisions:
            assert "Problem: not in its notes: Interpol" in msgs[1]["content"]
            return Revisions.model_validate({"revisions": [
                {"sentence": 1, "starts_with": "Police and Interpol",
                 "text": "Police arrested Tom Vale in Port Elm in 2004."}]})  # fmt: skip
        return SentenceChecks.model_validate({"checks": [
            {"sentence": 1, "starts_with": "Police arrested Tom", "problem": "none",
             "verdict": "supported"}]})  # fmt: skip

    report = {}
    out = revise([bad], notes, "Port Elm", ask, report)
    assert [w.text for w in out][1] == "Police arrested Tom Vale in Port Elm in 2004."
    assert out[1].revised == bad.text and not out[1].dropped
    assert out[1].check == "supported" and seen == [Revisions, SentenceChecks]
    assert (report["revised"], report["revised_kept"]) == (1, 1)


def test_an_opening_summary_is_checked_against_the_sections():
    from src.engine.written_page import Written, summary_check

    notes = page_notes(_city())
    written = [
        Written("Port Elm is a harbour city.", [1]),
        Written("Police made arrests there in 2004 and 2005.", [], summary=True),
        Written("The entry covers its arrests.", [2], summary=True),
        Written("Police arrested Tom Vale there in March 2004.", [2], summary=True),
        Written("Police arrested Tom Vale there in March 2004.", [2], "Arrests", 1),
        Written("Ann Roe was arrested in May 2005.", [3], "Arrests", 1),
    ]
    summary_check(written, notes, "Port Elm")
    assert written[1].check == "summary" and written[1].notes == [2, 3]
    assert written[2].dropped == "talks about the page"
    assert written[3].dropped == "copies a sentence of the page"


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


def test_counts_from_different_months_and_false_repeats_are_caught():
    from src.engine.written_page import page_notes
    from src.engine.written_plan import (
        Conflict,
        LeftOut,
        Plan,
        PlanPart,
        clean_conflicts,
        restore_repeats,
    )

    counts = Conflict(about="prisoners", one_side="40 prisoners", one_notes=[1],
                      other_side="41 captives", other_notes=[2])  # fmt: skip
    dates = {1: "2018-10-02", 2: "2018-02-10"}
    assert clean_conflicts([counts], {1, 2}, dates) == []
    assert (
        len(clean_conflicts([counts, counts], {1, 2}, {})) == 1
    )  # same month: kept once
    notes = {n.n: n for n in page_notes(_city())}
    plan = Plan(identity="", why_it_matters="", opening_notes=[],
                parts=[PlanPart(heading="Arrests", covers="", notes=[2])],
                left_out=[LeftOut(note=3, why="repeats 2")])  # fmt: skip
    restore_repeats(plan, notes, "Port Elm")  # note 3 names Ann Roe, note 2 does not
    assert plan.parts[0].notes == [2, 3] and plan.left_out == []


def test_a_fact_told_again_in_other_words_from_the_same_notes_is_dropped():
    from src.engine.written_page import WPage, Written, drop_repeats, flatten

    page = WPage.model_validate({"opening": [], "sections": [{"heading": "Arrests",
        "paragraphs": [{"sentences": [
            {"text": "Police arrested Tom Vale in Port Elm in March 2004.", "notes": [2]},
            {"text": "Ann Roe was arrested in May 2005.", "notes": [3]},
            {"text": "The article states that police arrested Tom Vale in March 2004 in Port Elm, Elmland.", "notes": [2]},
        ]}]}]})  # fmt: skip
    written = flatten(page)
    assert written[2].text.startswith("Police arrested Tom Vale in March 2004")
    drop_repeats(written)
    assert [w.dropped for w in written] == ["", "", "repeats an earlier sentence"]
    # mostly the same words, but from a note not yet cited: kept
    other = [Written("Police arrested Tom Vale.", [2], "A", 1),
             Written("Tom Vale was held by police in Port Elm.", [3], "A", 1)]  # fmt: skip
    drop_repeats(other)
    assert other[1].dropped == ""


def test_an_opening_summary_about_the_page_is_rewritten():
    from src.engine.written_page import Written, summary_check
    from src.engine.written_revise import Revisions, revise

    notes = page_notes(_city())
    written = [
        Written("Port Elm is a harbour city.", [1]),
        Written("The entry covers the arrests made there.", [], summary=True),
        Written("Police arrested Tom Vale there in March 2004.", [2], "Arrests", 1),
    ]
    summary_check(written, notes, "Port Elm")
    assert written[1].dropped == "talks about the page"

    def ask(msgs, model, max_tokens):
        assert model is Revisions and "talks about the entry" in msgs[1]["content"]
        return Revisions.model_validate({"revisions": [
            {"sentence": 1, "starts_with": "The entry",
             "text": "Police made arrests there in 2004."}]})  # fmt: skip

    out = revise(written, notes, "Port Elm", ask, {})
    summary_check(out, notes, "Port Elm")
    assert out[2].summary and out[2].check == "summary" and not out[2].dropped


def test_pairs_naming_one_person_with_two_years_are_pointed_out():
    from src.engine.written_page import Note
    from src.engine.written_plan import candidate_pairs

    notes = [Note(1, "a", "Tom Vale was captured in Port Elm in 2002.", []),
             Note(2, "b", "Police arrested Tom Vale on April 29, 2003.", []),
             Note(3, "c", "Ann Roe was arrested in 2005.", [])]  # fmt: skip
    assert candidate_pairs(notes, "Port Elm") == [
        "notes 1 and 2 (Tom Vale: 2002 against 2003)"
    ]


def test_a_note_about_the_place_left_out_as_someone_else_goes_back():
    from src.engine.written_plan import LeftOut, Plan, PlanPart, restore_on_topic

    notes = {n.n: n for n in page_notes(_city())}
    plan = Plan(identity="", why_it_matters="", opening_notes=[],
                parts=[PlanPart(heading="Background", covers="", notes=[1]),
                       PlanPart(heading="Arrests", covers="", notes=[2])],
                left_out=[LeftOut(note=3, why="someone else")])  # fmt: skip
    restore_on_topic(plan, notes, "Port Elm", "place (city)")
    assert plan.parts[1].notes == [2, 3] and plan.left_out == []


def test_dates_from_article_dates_are_named_to_the_check():
    from src.engine.written_page import Note, from_article_dates

    shown = [Note(1, "a", "He said he would retire.", [], "2017-05-10")]
    assert from_article_dates("In May 2017, he said he would retire.", shown) == [
        "May 2017"
    ]
    assert from_article_dates("In June 2016, he said so.", shown) == []


def test_a_sentence_with_a_new_date_is_not_a_repeat():
    from src.engine.written_page import Written, drop_repeats

    w = [Written("Police arrested Tom Vale in Port Elm in April 2003.", [2], "A", 1),
         Written("Police arrested Tom Vale in Port Elm in September 2002.", [2], "A", 1)]  # fmt: skip
    drop_repeats(w)
    assert [x.dropped for x in w] == ["", ""]


def test_run_labels_show_in_the_reader():
    from src.frontend.written_view import model_label

    assert model_label("openai/gpt-6-luna#plan@b") == "GPT-6 Luna, planned (b)"
