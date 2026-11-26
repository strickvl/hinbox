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

from src.citecheck import export, items, merge_data, planted, score, teach
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


def test_teachers_must_agree_and_quote():
    rows = []
    for k, t in enumerate(teach.TEACHERS):
        rows += [
            {"item_id": "same", "teacher": t, "level": "supported", "passages": ["p"]},
            {"item_id": "split", "teacher": t, "level": f"level{k}", "passages": ["p"]},
            {"item_id": "noquote", "teacher": t, "level": "partial", "passages": None},
            {"item_id": "absent", "teacher": t, "level": "not_supported"},
        ]
    rows.append({"item_id": "one", "teacher": teach.TEACHERS[0], "level": "supported"})
    labels, dropped = teach.agreed_labels(rows)
    assert {r["item_id"] for r in labels} == {"same", "absent"}
    assert dropped == {"disagree": 1, "no_quote": 1, "missing": 1}


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
    page = client.get("/")
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
