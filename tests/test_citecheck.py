"""Tests for the tuned-checker data (``src/citecheck/``) on made-up tables.

Every name, sentence and article here is invented: the repository is public
and must not hold run outputs or article text.
"""

import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from starlette.testclient import TestClient

from src.citecheck import (
    export,
    items,
    label_app,
    merge_data,
    planted,
    review,
    score,
    teach,
)
from src.citecheck.__main__ import main
from src.citecheck.label_app import make_app, marked_pieces
from src.engine.match_checker import parse_sectioned_reply
from src.utils.file_ops import load_existing_entities, write_entities_table
from src.utils.grounding import _GROUNDING_SYSTEM_PROMPT


def _ids(split: str, count: int, prefix: str = "art") -> list:
    """Made-up article ids that ``article_split()`` puts in ``split``."""
    out, i = [], 0
    while len(out) < count:
        aid = f"{prefix}-{i:04d}-0000-4000-8000-000000000000"
        if items.article_split(aid) == split:
            out.append(aid)
        i += 1
    return out


TEST = _ids("test", 4)
TRAIN = _ids("train", 3)
ARTICLE = (
    "Police said Yusuf Qadir was flown to the base in 2006 with 14 other men. "
    "Col. Ann Lee said the transfer was allegedly approved by Mara Voss. "
    "Neighbours said Qadir had studied engineering at a local college."
)


def _entities(article_ids: list) -> dict:
    text = " ".join(
        f"Qadir was moved to the base, report {n} says.^[{aid}] "
        f"Qadir studied engineering at a college, report {n} adds.^[{aid}]"
        for n, aid in enumerate(article_ids)
    )
    verifs = [
        {"article_id": aid, "claim": c, "support_level": lvl}
        for n, aid in enumerate(article_ids)
        for c, lvl in (
            (f"Qadir was moved to the base, report {n} says.", "partial"),
            (f"Qadir studied engineering at a college, report {n} adds.", "supported"),
        )
    ]
    return {
        "people": {
            "Yusuf Qadir": {
                "name": "Yusuf Qadir",
                "profile": {"text": text},
                "profile_grounding": {"verifications": verifs, "model": "ollama/q"},
            },
            "Hub": {"name": "Hub", "profile": {"text": text, "linked_only": True}},
        }
    }


def test_article_split_is_fixed_and_spread():
    assert items.article_split(TEST[0]) == items.article_split(TEST[0]) == "test"
    counts = {"train": 0, "tune": 0, "test": 0}
    for i in range(2000):
        counts[items.article_split(f"a{i}")] += 1
    assert 0.11 < counts["test"] / 2000 < 0.19
    assert 0.07 < counts["tune"] / 2000 < 0.13


def test_claim_rows_carry_stored_verdicts_and_skip_hubs():
    rows = items.claim_rows(_entities(TEST[:2]))
    assert len(rows) == 4
    assert {r["entity_name"] for r in rows} == {"Yusuf Qadir"}
    assert {r["stored_verdict"] for r in rows} == {"partial", "supported"}
    assert all(r["split"] == "test" for r in rows)


def test_sample_test_set_draws_test_articles_blind_and_spread():
    rows = items.claim_rows(_entities(TEST + TRAIN))
    picked = items.sample_test_set(rows, n=6, seed=1)
    assert picked and all(
        items.article_split(r["article_id"]) == "test" for r in picked
    )
    assert all("stored_verdict" not in r for r in picked)
    assert {r["stratum"] for r in picked} <= {"doubted", "random"}
    # one entity: at most PER_ENTITY claims however many articles
    assert len(picked) == items.PER_ENTITY
    missing = items.sample_test_set(rows, n=6, have_text=[TEST[0]])
    assert {r["article_id"] for r in missing} == {TEST[0]}


def test_labels_latest_wins(tmp_path):
    path = str(tmp_path / "labels.jsonl")
    items.append_jsonl(path, items.label_row("a", "yes"))
    items.append_jsonl(path, items.label_row("a", "no", "wrong year"))
    assert items.latest_labels(path)["a"]["label"] == "no"
    with pytest.raises(ValueError):
        items.label_row("a", "maybe")


@pytest.mark.parametrize(
    "fn, sentence, expect",
    [
        (planted.change_year, "Qadir was flown to the base in 2006.", "in 20"),
        (planted.change_number, "He arrived with 14 other men.", "other men"),
        (planted.change_negation, "The transfer was approved.", "was not approved"),
        (planted.change_negation, "The transfer was not approved.", "was approved"),
        (planted.drop_hedge, "It was allegedly approved.", "It was approved."),
    ],
)
def test_each_change_alters_one_detail(fn, sentence, expect):
    changed, why = fn(sentence, ARTICLE, random.Random(0))
    assert changed != sentence and expect in changed and why


def test_year_change_avoids_values_in_the_article():
    rng = random.Random(0)
    for _ in range(20):
        changed, _ = planted.change_year("In 2006 he left.", "2005 2007 2008", rng)
        assert not any(y in changed for y in ("2005", "2007", "2008"))


def test_name_change_never_touches_the_entity():
    s = "Col. Ann Lee said Yusuf Qadir left."
    changed, why = planted.change_name(s, ARTICLE, random.Random(0), "Yusuf Qadir")
    assert "Yusuf Qadir" in changed and "Ann Lee" not in changed
    assert "Mara Voss" in changed


def test_planted_claims_label_copies_and_changes():
    made = planted.planted_claims(
        [("Yusuf Qadir", TRAIN[0]), ("Yusuf Qadir", TEST[0])],
        {TRAIN[0]: ARTICLE, TEST[0]: ARTICLE},
        per_pair=3,
    )
    assert made and all(r["article_id"] == TRAIN[0] for r in made)
    copies = [r for r in made if r["kind"] == "copy"]
    changed = [r for r in made if r["kind"] != "copy"]
    assert copies and all(r["level"] == "supported" for r in copies)
    assert changed and all(r["claim"] not in ARTICLE for r in changed)
    for r in made:  # the quote in the reasoning is the article's own words
        quote = r["reasoning"].split('"')[1]
        assert quote in ARTICLE


def test_planted_claims_change_years_too_when_negation_would_fit():
    """Every sentence here allows a negation ("was"), a year and a number; a
    fixed order of kinds planted only negations and names."""
    ids = [
        a for a in (f"art-{k}" for k in range(400)) if items.article_split(a) == "train"
    ][:40]
    texts = {
        a: f"Yusuf Qadir was flown to the base in 2006 with {k + 4} other men."
        for k, a in enumerate(ids)
    }
    made = planted.planted_claims([("Yusuf Qadir", a) for a in ids], texts, per_pair=1)
    kinds = {r["kind"] for r in made}
    assert {"year", "number", "negation"} <= kinds


def test_planted_claims_only_use_the_part_of_the_article_the_judge_sees(
    monkeypatch,
):
    # the article's second sentence naming him starts at character 141
    monkeypatch.setattr(planted, "GROUNDING_MAX_ARTICLE_CHARS", 120)
    made = planted.planted_claims(
        [("Yusuf Qadir", TRAIN[0])], {TRAIN[0]: ARTICLE}, per_pair=5
    )
    copies = [ARTICLE.find(r["claim"]) for r in made if r["kind"] == "copy"]
    assert copies == [0]


def test_the_random_stratum_is_drawn_from_every_test_claim():
    rows = [
        {
            "item_id": f"i{k}",
            "entity_table": "people",
            "entity_name": f"Person {k}",
            "article_id": f"t{k}",
            "claim": f"Claim {k}.",
            "split": "test",
            "stored_verdict": "partial" if k < 100 else "supported",
        }
        for k in range(200)
    ]
    picked = items.sample_test_set(rows, n=100, seed=3)
    random_half = [r for r in picked if r["stratum"] == "random"]
    doubted_in_random = sum(int(r["item_id"][1:]) < 100 for r in random_half)
    # about half the population is doubted, so about half the random stratum
    assert len(random_half) == 50 and 15 <= doubted_in_random <= 35
    assert sum(r["stratum"] == "doubted" for r in picked) == 50


def _labels(aid: str, n: int = 3) -> list:
    return [
        {
            "item_id": f"i{aid}{k}",
            "article_id": aid,
            "entity_name": "Yusuf Qadir",
            "claim": f"Claim {k}.",
            "level": "partial" if k % 2 else "supported",
            "reasoning": 'The article says "x".',
            "source": "teachers",
        }
        for k in range(n)
    ]


def test_build_examples_uses_pipeline_prompt_and_reply():
    examples, counts = export.build_examples(
        _labels(TRAIN[0], 5), {TRAIN[0]: ARTICLE}, max_claims=20, seed=1
    )
    assert sum(counts["claims"].values()) == 5
    for ex in examples:
        system, user, reply = ex["messages"]
        assert system["content"] == _GROUNDING_SYSTEM_PROMPT
        assert ARTICLE in user["content"]
        numbered = user["content"].count("[Yusuf Qadir]")
        got = score.read_reply(reply["content"], numbered)
        assert None not in got


def test_build_examples_refuses_test_articles():
    with pytest.raises(ValueError):
        export.build_examples(_labels(TEST[0]), {TEST[0]: ARTICLE})


def test_wilson_and_score():
    lo, hi = score.wilson(2, 42)
    assert 0.01 < lo < 0.05 and 0.14 < hi < 0.17
    its = [
        {"item_id": "a", "stratum": "random"},
        {"item_id": "b", "stratum": "doubted"},
        {"item_id": "c", "stratum": "doubted"},
        {"item_id": "d", "stratum": "random"},
    ]
    labels = {
        "a": {"label": "yes"},
        "b": {"label": "no"},
        "c": {"label": "no"},
        "d": {"label": "unsure"},
    }
    report = score.score(its, labels, {"a": "supported", "b": "supported", "c": None})
    assert report["labelled"] == 3 and report["unsure"] == 1
    assert report["wrong_passed"]["count"] == 1 and report["wrong_passed"]["of"] == 2
    assert report["good_doubted"]["count"] == 0
    assert report["errors"] == 1
    assert "wrong claims passed: 1/2" in score.describe(report, "m")


def test_read_reply_handles_fences_order_and_junk():
    text = '```json\n{"items":[{"claim_number":2,"support_level":"partial"},' \
        '{"claim_number":1,"support_level":"supported"}]}\n```'  # fmt: skip
    assert score.read_reply(text, 2) == ["supported", "partial"]
    assert score.read_reply("no json here", 2) == [None, None]
    assert score.read_reply('{"items":[{"support_level":"maybe"}]}', 1) == [None]
    preds = score.reply_predictions([{"item_ids": ["x", "y"], "reply": text}])
    assert preds == {"x": "supported", "y": "partial"}


def test_teachers_must_agree_and_a_supported_label_needs_a_quote():
    rows = []
    for k, t in enumerate(teach.TEACHERS):
        rows += [
            {"item_id": "same", "teacher": t, "level": "supported", "passages": ["p"]},
            {"item_id": "split", "teacher": t, "level": f"level{k}", "passages": ["p"]},
            {"item_id": "noquote", "teacher": t, "level": "supported", "passages": None},
            {"item_id": "partial", "teacher": t, "level": "partial", "passages": None},
            {"item_id": "absent", "teacher": t, "level": "not_supported"},
            # only the second teacher quoted the article: its reasoning is kept
            {"item_id": "second", "teacher": t, "level": "supported",
             "passages": ["q"] if k else [], "reasoning": f"by {k}"},
        ]  # fmt: skip
    rows.append({"item_id": "one", "teacher": teach.TEACHERS[0], "level": "supported"})
    labels, dropped = teach.agreed_labels(rows)
    by_id = {r["item_id"]: r for r in labels}
    assert set(by_id) == {"same", "partial", "absent", "second"}
    assert by_id["second"]["reasoning"] == "by 1"
    assert dropped == {"disagree": 1, "no_quote": 1, "missing": 1}


def test_cap_planted_keeps_every_teacher_label_and_samples_the_planted():
    teacher = [{"item_id": f"t{i}", "source": "teachers"} for i in range(3)]
    planted = [{"item_id": f"p{i}", "source": "planted"} for i in range(20)]
    kept = export.cap_planted(teacher + planted, ratio=2)
    assert [r for r in kept if r["source"] == "teachers"] == teacher
    assert sum(r["source"] == "planted" for r in kept) == 6
    assert kept == export.cap_planted(teacher + planted, ratio=2)  # same pick each time
    assert export.cap_planted(teacher + planted[:4], ratio=2) == teacher + planted[:4]


def test_article_chunks_cap_claims_per_call():
    claims = [{"article_id": "a", "item_id": str(i)} for i in range(45)]
    chunks = teach.article_chunks(claims, size=20)
    assert [len(g) for _, g in chunks] == [20, 20, 5]


def test_marked_pieces_keep_every_character():
    pieces = marked_pieces(ARTICLE, "Qadir studied engineering at a local college")
    assert "".join(p for p, _ in pieces) == ARTICLE
    assert any("engineering" in p for p, m in pieces if m)


def test_label_page_records_clicks(tmp_path):
    its = [
        {"item_id": "a", "entity_name": "Yusuf Qadir", "article_id": "x",
         "claim": "Qadir studied engineering."},
        {"item_id": "b", "entity_name": "Yusuf Qadir", "article_id": "x",
         "claim": "Qadir left in 2006.", "level": "partial", "reasoning": "r"},
    ]  # fmt: skip
    path = str(tmp_path / "labels.jsonl")
    client = TestClient(make_app(its, path, {"x": ARTICLE}))
    assert "How many in this sprint?" in client.get("/").text
    page = client.get("/?i=0")
    assert "Qadir studied engineering." in page.text and "<mark" in page.text
    r = client.post(
        "/label",
        data={"item_id": "a", "label": "yes", "index": "0"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"] == "/?i=1"
    assert items.latest_labels(path)["a"]["label"] == "yes"
    assert "verdict right" in client.get("/?i=1").text
    bad = client.post(
        "/label",
        data={"item_id": "b", "label": "no", "index": "1"},
        headers={"origin": "http://evil.example"},
    )
    assert bad.status_code == 403
    assert (
        client.post("/label", data={"item_id": "zz", "label": "no"}).status_code == 400
    )


def test_a_sprint_counts_its_own_labels_and_ends_with_a_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("HINBOX_LABEL_CATS", "")
    its = [
        {"item_id": f"i{k}", "entity_name": "Yusuf Qadir", "article_id": f"x{k % 2}",
         "claim": f"Qadir studied engineering, report {k} says."}
        for k in range(5)
    ]  # fmt: skip
    path = str(tmp_path / "labels.jsonl")
    meta = {"x0": {"title": "Detainee moved", "date": "2016-03-12T00:00:00"}}
    client = TestClient(make_app(its, path, {"x0": ARTICLE, "x1": ARTICLE}, meta=meta))
    r = client.get("/start?n=2", follow_redirects=False)
    sprint = dict(p.split("=") for p in r.headers["location"].split("?")[1].split("&"))
    assert sprint["n"] == "2" and sprint["b"] == "0"
    page = client.get(r.headers["location"])
    for step in range(2):
        assert f"{step} of 2</b> in this sprint" in page.text
        index = page.text.split('name="index" value="')[1].split('"')[0]
        iid = page.text.split('name="item_id" value="')[1].split('"')[0]
        page = client.post(
            "/label", data={"item_id": iid, "label": "yes", "index": index, **sprint}
        )
    assert "Sprint done" in page.text and "2 labelled" in page.text
    assert "Another 2" in page.text and "2 of 5" in page.text
    # a misclicked last answer can still be reopened and changed
    fix = page.text.split('<a href="')[-1].split('">Change the last answer')[0]
    page = client.get(fix.replace("&amp;", "&"))
    assert f'name="item_id" value="{iid}"' in page.text
    page = client.post(
        "/label", data={"item_id": iid, "label": "no", "index": index, **sprint}
    )
    assert "Sprint done" in page.text and "2 labelled" in page.text
    assert items.read_jsonl(path)[-1]["label"] == "no"
    # sentences of one article come together, so the second answer kept the article
    done = [r["item_id"] for r in items.read_jsonl(path)][:2]
    assert (
        len({next(i["article_id"] for i in its if i["item_id"] == d) for d in done})
        == 1
    )


def test_the_page_shows_the_date_and_underlines_details_the_article_lacks(tmp_path):
    its = [{"item_id": "a", "entity_name": "Yusuf Qadir", "article_id": "x",
            "claim": "Qadir was flown to the base in 2009 by Mara Voss."}]  # fmt: skip
    meta = {"x": {"title": "Detainee moved", "date": "2016-03-12T00:00:00"}}
    client = TestClient(
        make_app(its, str(tmp_path / "l.jsonl"), {"x": ARTICLE}, meta=meta)
    )
    text = client.get("/?i=0").text
    assert "12 March 2016" in text and "Detainee moved" in text
    assert '<span class="flag"' in text and ">2009</span>" in text
    assert ">Mara</span>" not in text  # in the article, so not flagged


def test_claim_pieces_flag_only_missing_years_numbers_and_names():
    pieces = label_app.claim_pieces(
        "In 2007, al-Nashiri met 14 men from Guantánamo and Lahore.",
        "Al Nashiri was held at Guantanamo in 2006 with 14 other men.",
    )
    assert "".join(p for p, _ in pieces).startswith("In 2007, al-Nashiri")
    assert [p for p, f in pieces if f] == ["2007", "Lahore"]


def test_label_order_keeps_each_articles_sentences_together():
    its = [{"article_id": a} for a in ["a", "b", "a", "c", "b", "a"]]
    order = label_app.label_order(its)
    arts = [its[i]["article_id"] for i in order]
    assert sorted(order) == list(range(6))
    assert all(
        arts.index(a) + arts.count(a) - 1 == len(arts) - 1 - arts[::-1].index(a)
        for a in arts
    )


def test_cats_come_now_and_then_never_twice_running_and_at_halfway():
    pets = label_app.cats("Tom:ginger, Kit:tortie, bad:dog")
    assert pets == [("Tom", "ginger"), ("Kit", "tortie")]
    assert label_app.cats("")[0][0] == "The ginger cat"
    shown = [label_app.cat_for(1700000000, d, 40, pets) for d in range(41)]
    assert shown[20] and "Halfway" in shown[20]["line"]
    assert not shown[0] and not shown[40]
    assert 3 <= sum(1 for c in shown if c) <= 15
    assert not any(shown[d] and shown[d + 1] for d in range(40))


def test_shorter_name_and_split_positives(monkeypatch):
    rng = random.Random(0)
    assert merge_data.shorter_name("Ann Beth Lee", rng) in ("Ann Lee", "Lee")
    ents = {
        "people": {
            "Ann Beth Lee": {
                "name": "Ann Beth Lee",
                "profile": {
                    "text": (
                        f"Ann Beth Lee is a made-up prosecutor who argued many cases.^[{TRAIN[0]}] "
                        f"She once worked in a made-up city office for years.^[{TRAIN[0]}]\n\n"
                        f"Lee later led the made-up prosecution team in a long trial.^[{TRAIN[1]}] "
                        f"She told reporters the trial would take months to finish.^[{TRAIN[1]}]"
                    )
                },
            },
            "Avoid Me": {"name": "Avoid Me", "profile": {"text": "x"}},
        }
    }
    monkeypatch.setattr(merge_data, "MIN_EXISTING_CHARS", 50)
    pairs = merge_data.split_positives(ents, 5, avoid_names={"Avoid Me"})
    assert len(pairs) == 1 and pairs[0]["truth"] is True
    p = pairs[0]["inputs"]
    assert p["existing_name"] == "Ann Beth Lee"
    new_id = TRAIN[0] if TRAIN[0] in p["new_profile_text"] else TRAIN[1]
    assert new_id not in p["existing_profile_text"]


def test_match_reply_round_trips_and_scores():
    reply = merge_data.match_reply("Both are the made-up judge.", True, 0.9)
    parsed = parse_sectioned_reply(reply)
    assert parsed.is_match is True and parsed.confidence == pytest.approx(0.9)
    pairs = [{"item_id": "p1", "truth": True}, {"item_id": "p2", "truth": False}]
    m = merge_data.score_replies(
        pairs,
        [{"item_ids": ["p1"], "reply": reply}, {"item_ids": ["p2"], "reply": "??"}],
    )
    assert m["tp"] == 1 and m["errors"] == 1


def test_agreed_pairs_and_examples():
    inputs = {
        "new_name": "Lee",
        "existing_name": "Ann Lee",
        "new_profile_text": "Lee spoke.",
        "existing_profile_text": "Ann Lee is a made-up judge.",
        "entity_type": "people",
    }
    pairs = [
        {"item_id": "a", "inputs": inputs, "truth": True, "split": "train"},
        {"item_id": "b", "inputs": inputs, "truth": False, "split": "train"},
        {"item_id": "c", "inputs": inputs, "truth": None, "split": "train"},
    ]
    t1, t2 = "x/one", "x/two"
    rows = [
        {
            "item_id": i,
            "teacher": t,
            "is_match": True,
            "confidence": c,
            "reasoning": "r",
        }
        for i in ("a", "b", "c")
        for t, c in ((t1, 0.9), (t2, 0.7))
    ]
    kept, dropped = merge_data.agreed_pairs(pairs, rows, (t1, t2))
    assert {r["item_id"] for r in kept} == {"a", "c"} and dropped == {
        "against_truth": 1
    }
    assert kept[0]["confidence"] == 0.7
    ex = merge_data.merge_examples(kept)
    assert ex[0]["messages"][-1]["role"] == "assistant"
    with pytest.raises(ValueError):
        merge_data.merge_examples([{**kept[0], "split": "test"}])


def _write_run(tmp_path):
    out = tmp_path / "entities"
    out.mkdir()
    ents = _entities(TEST + TRAIN)["people"]["Yusuf Qadir"]
    write_entities_table("people", [ents], str(out))
    for t in ("events", "locations", "organizations"):
        write_entities_table(t, [], str(out))
    arts = tmp_path / "articles.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"id": a, "content": ARTICLE} for a in TEST + TRAIN]),
        arts,
    )
    return str(out), str(arts)


def test_cli_sample_score_plant_export(tmp_path, capsys):
    ents, arts = _write_run(tmp_path)
    assert load_existing_entities(ents)["people"]
    out = str(tmp_path / "cc")
    main(["sample", "--entities", ents, "--articles", arts, "--out", out, "--n", "4"])
    test_items = items.read_jsonl(f"{out}/test_items.jsonl")
    assert test_items
    with pytest.raises(SystemExit):  # frozen once drawn
        main(["sample", "--entities", ents, "--articles", arts, "--out", out])
    labels = f"{out}/labels.jsonl"
    for it in test_items:
        items.append_jsonl(labels, items.label_row(it["item_id"], "no"))
    main(["score", "--items", f"{out}/test_items.jsonl", "--labels", labels,
          "--stored", ents, "--save", f"{out}/s.json"])  # fmt: skip
    saved = json.load(open(f"{out}/s.json"))
    assert saved["report"]["labelled"] == len(test_items)
    main(["plant", "--entities", ents, "--articles", arts, "--out", out])
    main(["prompts", "--items", f"{out}/test_items.jsonl", "--articles", arts,
          "--file", f"{out}/prompts.jsonl"])  # fmt: skip
    prompts = items.read_jsonl(f"{out}/prompts.jsonl")
    assert sum(len(p["item_ids"]) for p in prompts) == len(test_items)
    main(["export", "--articles", arts, "--out", out, "--no-teachers"])
    summary = json.load(open(f"{out}/export_summary.json"))
    assert summary["train"]["examples"] >= 0
    assert "wrong claims passed" in capsys.readouterr().out


def test_the_article_date_reads_unix_seconds_as_the_articles_file_stores_them():
    assert label_app._date(1371742594) == "20 June 2013"
    assert label_app._date("1371742594") == "20 June 2013"
    assert label_app._date(1371742594000) == "20 June 2013"
    assert label_app._date("2016-03-12T00:00:00") == "12 March 2016"
    assert label_app._date(None) == "" and label_app._date("soon") == "soon"


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_review_shows_disagreements_with_both_answers_and_marks_the_quotes(tmp_path):
    its = [
        {"item_id": k, "entity_name": "Yusuf Qadir", "article_id": "x", "claim": c}
        for k, c in [("a", "Qadir was flown to the base in 2006."),
                     ("b", "Voss approved the transfer."),
                     ("c", "Qadir studied engineering.")]
    ]  # fmt: skip
    mine, sol = tmp_path / "mine.jsonl", tmp_path / "sol.jsonl"
    _write(mine, [{"item_id": "a", "label": "yes"}, {"item_id": "b", "label": "yes"}])
    _write(sol, [
        {"item_id": "a", "label": "yes"},
        {"item_id": "b", "label": "no", "labeller": "sol", "note": "hedge dropped",
         "evidence": "“the transfer was allegedly approved by Mara Voss” … no more"},
        {"item_id": "c", "label": "no"},
    ])  # fmt: skip
    picked = review.review_items(its, str(mine), str(sol))
    assert [i["item_id"] for i in picked] == [
        "b"
    ]  # same label, or one side unlabelled: left out
    assert picked[0]["earlier"][1] == {
        "who": "sol",
        "label": "no",
        "note": "hedge dropped",
    }
    assert picked[0]["evidence"] == ["the transfer was allegedly approved by Mara Voss"]
    out = str(tmp_path / "review.jsonl")
    page = TestClient(make_app(picked, out, {"x": ARTICLE})).get("/?i=0").text
    assert "<b>sol</b> said" in page and "hedge dropped" in page
    assert (
        '<mark class="m">the transfer was allegedly approved by Mara Voss</mark>'
        in page
    )
    pieces = review.quote_pieces(ARTICLE, ["THE TRANSFER was 'allegedly'"])
    assert "".join(t for t, _ in pieces) == ARTICLE and not any(m for _, m in pieces)
    joined = "Col. Lee said so. Neighbours said Qadir had studied engineering at a local college."
    marked = [t for t, m in review.quote_pieces(ARTICLE, [joined]) if m]
    assert marked == [
        "Neighbours said Qadir had studied engineering at a local college."
    ]


def test_combine_lets_the_later_file_win(tmp_path):
    a, b, c = (tmp_path / n for n in ("sol.jsonl", "mine.jsonl", "review.jsonl"))
    _write(a, [{"item_id": "1", "label": "no"}, {"item_id": "2", "label": "no"}])
    _write(b, [{"item_id": "1", "label": "yes"}, {"item_id": "2", "label": "yes"}])
    _write(c, [{"item_id": "2", "label": "no"}])
    out = str(tmp_path / "all.jsonl")
    counts = review.combine_labels(
        [str(a), str(b), str(c), str(tmp_path / "none")], out
    )
    assert counts == {"yes": 1, "no": 1}
    assert items.latest_labels(out)["2"]["label"] == "no"


def test_combine_from_the_command_line_writes_a_file_not_a_folder(tmp_path):
    a, b = tmp_path / "sol.jsonl", tmp_path / "mine.jsonl"
    _write(a, [{"item_id": "1", "label": "no"}])
    _write(b, [{"item_id": "1", "label": "yes"}])
    out = tmp_path / "new" / "all.jsonl"
    main(["combine", "--labels", str(a), str(b), "--out", str(out)])
    assert out.is_file() and items.latest_labels(str(out))["1"]["label"] == "yes"
    main(["combine", "--labels", str(b), str(a), "--out", str(out)])  # file exists
    assert items.latest_labels(str(out))["1"]["label"] == "no"


def test_strict_rules_add_to_the_judge_prompt_and_the_judge_name(monkeypatch):
    from src.utils import grounding

    assert grounding.grounding_system_prompt() == _GROUNDING_SYSTEM_PROMPT
    strict = grounding.grounding_system_prompt("strict")
    assert "dropped attribution" in strict and '"the profile"' in strict
    # the rules come before the reply instructions, which stay last
    assert strict.index("dropped hedge") < strict.index("first write the reasoning")
    assert strict.replace(grounding._STRICT_RULES, "") == _GROUNDING_SYSTEM_PROMPT
    monkeypatch.setattr(grounding, "GROUNDING_RULES", "strict")
    system = grounding.grounding_messages("a", ARTICLE, [("X", "c")])[0]["content"]
    assert system == strict
    monkeypatch.setattr(grounding, "grounding_model", lambda _t: "cloud/judge")
    assert grounding.grounding_judge("cloud") == "cloud/judge (rules=strict)"


def test_teach_refuses_a_file_asked_under_other_rules(tmp_path, monkeypatch):
    from src.utils import grounding

    out = tmp_path / "teacher_verdicts.jsonl"
    out.write_text('{"item_id": "a", "teacher": "t", "level": "supported"}\n')
    monkeypatch.setattr(grounding, "GROUNDING_RULES", "strict")
    with pytest.raises(SystemExit, match="another --out"):
        teach.run_teachers([], {}, str(out), teachers=("t",))


def test_export_refuses_teacher_verdicts_asked_under_other_rules(tmp_path, monkeypatch):
    from src.utils import grounding

    row = {"item_id": "a", "teacher": "t", "level": "supported", "rules": "strict"}
    _write(tmp_path / "teacher_verdicts.jsonl", [row])
    args = ["export", "--articles", "unused.parquet", "--out", str(tmp_path)]
    with pytest.raises(SystemExit, match="HINBOX_GROUNDING_RULES"):
        main(args)
    monkeypatch.setattr(grounding, "GROUNDING_RULES", "strict")
    teach.same_rules([row], "x", "fix")  # matching rules pass
    teach.same_rules([{"item_id": "p", "level": "partial"}], "x", "fix")  # planted


def test_read_reply_reads_an_untuned_models_plain_text_answer():
    text = (
        "1. Reasoning: the article says so.\nSupport_level: supported\n\n"
        "2. Reasoning: the year differs.\n**Support level:** Not supported"
    )
    assert score.read_reply(text, 2) == ["supported", "not_supported"]
    assert score.read_reply(text, 3) == [None, None, None]  # one verdict short
    assert score.read_reply("no verdicts here", 1) == [None]
