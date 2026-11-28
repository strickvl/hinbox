"""The second "rewording or changes meaning" step after the differences checker."""

import json

from src.citecheck import meaning as m
from src.citecheck.__main__ import main


def _reply(*items):
    return json.dumps({"items": list(items)})


def _item(n, level, diffs=()):
    return {
        "claim_number": n,
        "differences": list(diffs),
        "reasoning": "r",
        "support_level": level,
    }


def test_reply_details_json_and_text():
    text = _reply(_item(2, "partial", ["x"]), _item(1, "supported", ["none"]))
    assert m.reply_details(text, 2) == [("supported", []), ("partial", ["x"])]
    plain = (
        "1. Differences: - says 2009, article 2010\nReasoning: r\nSupport_level: supported\n"
        "2. Differences: none\nReasoning: r\nSupport_level: supported"
    )
    assert m.reply_details(plain, 2) == [
        ("supported", ["says 2009, article 2010"]),
        ("supported", []),
    ]
    assert m.reply_details("garbled", 2) == [None, None]


def _setup():
    items = [
        {
            "item_id": "a",
            "article_id": "art",
            "entity_name": "Ann",
            "claim": "Ann left.",
        },
        {
            "item_id": "b",
            "article_id": "art",
            "entity_name": "Ann",
            "claim": "Ann won.",
        },
        {
            "item_id": "c",
            "article_id": "art",
            "entity_name": "Ann",
            "claim": "Ann ran.",
        },
    ]
    replies = [
        {
            "item_ids": ["a", "b", "c"],
            "sample": s,
            "reply": _reply(
                _item(1, "supported", ["tense"]),
                _item(2, "supported", ["drops 'allegedly'", "adds a year"]),
                _item(3, "supported"),
            ),
        }
        for s in (0, 1)
    ]
    return items, replies


def test_prompts_only_for_passed_claims_with_differences():
    items, replies = _setup()
    rows = m.meaning_prompts(items, {"art": "TEXT"}, m.details_by_sample(replies))
    assert [r["item_ids"] for r in rows] == [["a#s0"], ["b#s0"], ["a#s1"], ["b#s1"]]
    user = rows[1]["messages"][1]["content"]
    assert "TEXT" in user and "[Ann] Ann won." in user
    assert "1. drops 'allegedly'\n2. adds a year" in user


def test_predictions_keep_supported_only_when_every_difference_is_rewording():
    items, replies = _setup()
    second = [
        {
            "item_ids": ["a#s0"],
            "reply": _reply({"difference_number": 1, "answer": "rewording"}),
        },
        {
            "item_ids": ["b#s0"],
            "reply": _reply(
                {"difference_number": 1, "answer": "rewording"},
                {"difference_number": 2, "answer": "changes_meaning"},
            ),
        },
        {"item_ids": ["a#s1"], "reply": "Answer: rewording"},
        {
            "item_ids": ["b#s1"],
            "reply": _reply({"difference_number": 1, "answer": "rewording"}),
        },
    ]
    out, counts = m.meaning_predictions(m.details_by_sample(replies), second)
    assert out["model"][0] == {"a": "supported", "b": "supported", "c": "supported"}
    assert out["code"][0] == {"a": "partial", "b": "partial", "c": "supported"}
    assert out["meaning"][0] == {"a": "supported", "b": "partial", "c": "supported"}
    # b#s1 answered one of two differences: doubt, as code alone would
    assert out["meaning"][1] == {"a": "supported", "b": "partial", "c": "supported"}
    assert counts == {
        "asked": 4,
        "answered": 4,
        "kept_supported": 2,
        "doubted": 1,
        "unread": 1,
    }


def test_commands(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    items, replies = _setup()
    paths = {
        k: tmp_path / f"{k}.jsonl" for k in ("items", "replies", "labels", "second")
    }
    for k, rows in (("items", items), ("replies", replies)):
        paths[k].write_text("".join(json.dumps(r) + "\n" for r in rows))
    labels = [
        {"item_id": i, "label": lab}
        for i, lab in (("a", "yes"), ("b", "no"), ("c", "yes"))
    ]
    paths["labels"].write_text("".join(json.dumps(r) + "\n" for r in labels))
    art = tmp_path / "a.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["art"],
                "content": ["TEXT"],
                "title": ["T"],
                "published_date": ["2010-01-02"],
            }
        ),
        art,
    )
    prompts = tmp_path / "second_prompts.jsonl"
    main(
        [
            "meaning-prompts",
            "--items",
            str(paths["items"]),
            "--articles",
            str(art),
            "--replies",
            str(paths["replies"]),
            "--file",
            str(prompts),
            "--with-head",
        ]
    )
    rows = [json.loads(line) for line in prompts.read_text().splitlines()]
    assert (
        len(rows) == 4
        and "Headline: T\nPublished: 2010-01-02" in rows[0]["messages"][1]["content"]
    )
    second = [
        {
            "item_ids": r["item_ids"],
            "reply": "Answer: rewording\n"
            * (2 if r["item_ids"][0].startswith("b") else 1),
        }
        for r in rows
    ]
    paths["second"].write_text("".join(json.dumps(r) + "\n" for r in second))
    prefix = tmp_path / "dev_"
    main(
        [
            "meaning-score",
            "--items",
            str(paths["items"]),
            "--labels",
            str(paths["labels"]),
            "--replies",
            str(paths["replies"]),
            "--second",
            str(paths["second"]),
            "--save-prefix",
            str(prefix),
        ]
    )
    saved = json.loads((tmp_path / "dev_meaning.json").read_text())["predicted"]
    assert saved == {"a": "supported", "b": "supported", "c": "supported"}
    assert (
        json.loads((tmp_path / "dev_code.json").read_text())["predicted"]["a"]
        == "partial"
    )


def test_answers_must_cover_every_difference_once():
    def two(*pairs):
        return _reply(*({"difference_number": n, "answer": a} for n, a in pairs))

    assert m.meaning_answers(two((2, "changes_meaning"), (1, "rewording")), 2) == [
        "rewording",
        "changes_meaning",
    ]
    assert m.meaning_answers(two((1, "rewording"), (1, "rewording")), 2) is None
    assert m.meaning_answers(two((1, "rewording")), 2) is None
    assert m.meaning_answers(two((1, "rewording"), (2, "maybe")), 2) is None
    assert (
        m.meaning_answers("Answer: rewording\nAnswer: rewording", 2)
        == ["rewording"] * 2
    )
    assert m.meaning_answers("Answer: rewording", 2) is None
