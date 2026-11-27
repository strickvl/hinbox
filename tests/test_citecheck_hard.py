"""Tests for the hard-claim examples (``src/citecheck/hard.py``) on made-up
prompts. Every name, sentence and article here is invented."""

import json

from src.citecheck import hard
from src.citecheck.__main__ import main
from src.citecheck.export import reply_json
from src.citecheck.items import read_jsonl, write_jsonl
from src.utils.grounding import SupportLevel, grounding_messages

TEXT = "Ann Lee was flown to the base in 2006. She met Col. Ray Turner there."
CLAIMS = [
    ("Ann Lee", "Ann Lee was flown to the base in 2006."),
    ("Ann Lee", "Ann Lee met Ray Turner\nat the base."),
    ("Ray Turner", "Turner allegedly ran the base."),
]


def _example(
    aid: str = "a1", claims=CLAIMS, levels=("supported", "supported", "partial")
):
    messages = grounding_messages(aid, TEXT, claims)
    reply = {"role": "assistant", "content": reply_json([(lv, "") for lv in levels])}
    return {"messages": messages + [reply]}


def _log(example, answers, kept=None, gold=None):
    prompt = example["messages"][:-1]
    gold = gold or {"1": "supported", "2": "supported", "3": "partial"}
    return {"key": hard.row_key(prompt), "gold": gold, "kept": kept, "answers": answers}


def test_parse_prompt_reads_back_grounding_messages():
    aid, head, claims = hard.parse_prompt(_example()["messages"][:-1])
    assert aid == "a1" and head.endswith(TEXT)
    assert claims == CLAIMS  # the claim over two lines stays one claim


def test_hard_claims_are_wrong_in_every_answer():
    ex = _example()
    answers = [
        {"1": "supported", "2": "partial", "3": "supported"},
        {"1": "supported", "2": "supported", "3": "supported"},
        {"1": "partial", "2": "supported", "3": "supported"},
    ]
    found = hard.hard_claims([ex], [_log(ex, answers)])
    # claim 3 (partial) was passed every time; 1 and 2 were right at least once
    assert [h["claim"] for h in found] == [CLAIMS[2][1]]
    assert found[0]["answers"] == ["supported"] * 3


def test_hard_claims_skip_kept_examples_and_too_few_answers():
    ex = _example()
    wrong = [{"3": "supported"}] * 3
    assert hard.hard_claims([ex], [_log(ex, wrong, kept=0)]) == []
    assert hard.hard_claims([ex], [_log(ex, wrong[:2])]) == []
    assert hard.hard_claims([ex], []) == []


def test_hard_prompts_ask_one_claim_with_the_same_article():
    ex = _example()
    found = hard.hard_claims([ex], [_log(ex, [{"3": "supported"}] * 4)])
    kinds = hard.label_kinds(
        [
            {
                "article_id": "a1",
                "claim": CLAIMS[2][1],
                "source": "planted",
                "kind": "hedge",
            }
        ]
    )
    rows = hard.hard_prompts(found, kinds)
    assert len(rows) == 1 and rows[0]["hard"]["kind"] == "planted:hedge"
    prompt = rows[0]["messages"][:-1]
    assert prompt == grounding_messages("a1", TEXT, [CLAIMS[2]])
    gold = json.loads(rows[0]["messages"][-1]["content"])["items"]
    assert [g["support_level"] for g in gold] == ["partial"]
    assert hard.hard_summary(rows) == {"planted:hedge partial": 1}


def test_mixed_repeats_extra_and_drops_planted_years():
    year = _example("a2", [("Ann Lee", "Ann Lee was flown in 2007.")], ("partial",))
    other = _example("a3", [("Ann Lee", "She met him.")], ("supported",))
    kinds = hard.label_kinds(
        [{"article_id": "a2", "claim": "Ann Lee was flown in 2007.", "kind": "year"}]
    )
    extra = [{"messages": [], "hard": {}}]
    rows, counts = hard.mixed(
        [year, other], extra, repeat=4, kinds=kinds, drop_kinds=("planted:year",)
    )
    assert counts == {"base": 2, "base_dropped": 1, "extra": 1, "rows": 5}
    assert sum(r is extra[0] for r in rows) == 4 and year not in rows


def test_teacher_row_keeps_only_agreeing_reasoned_verdicts():
    row = hard.hard_prompts(
        hard.hard_claims([_example()], [_log(_example(), [{"3": "supported"}] * 3)])
    )[0]
    good = [(SupportLevel.NOT_SUPPORTED, "The article never says he ran it.", None)]
    out = hard.teacher_row(row, good, "openai/model")  # loose: not supported = partial
    parts = out["messages"][-1]["content"]
    assert parts[0] == {
        "type": "thinking",
        "thinking": "Claim 1: The article never says he ran it.",
    }
    assert json.loads(parts[1]["text"])["items"][0]["support_level"] == "not_supported"
    assert hard.teacher_row(row, good, "m", loose=False) is None
    assert hard.teacher_row(row, [(SupportLevel.SUPPORTED, "Yes.", None)], "m") is None
    assert hard.teacher_row(row, [(SupportLevel.PARTIAL, "", None)], "m") is None
    assert hard.teacher_row(row, [None], "m") is None


def test_missed_and_estimate():
    rows = hard.hard_prompts(
        hard.hard_claims([_example()], [_log(_example(), [{"3": "supported"}] * 3)])
    )
    key = hard.row_key(rows[0]["messages"][:-1])
    assert hard.missed(rows, [{"key": key, "kept": None}]) == rows
    assert hard.missed(rows, [{"key": key, "kept": 2}]) == []
    assert hard.missed(rows, []) == []  # never tried: not asked of a teacher
    assert 0 < hard.estimate_dollars(rows, 1.0, 1.0) < 0.01


def test_hard_and_mix_commands(tmp_path):
    export_dir, v3, out = tmp_path / "export", tmp_path / "v3", tmp_path / "out"
    ex = _example()
    for split in ("train", "tune"):
        write_jsonl(str(export_dir / f"{split}.jsonl"), [ex])
        write_jsonl(
            str(v3 / f"{split}.jsonl.log.jsonl"), [_log(ex, [{"3": "supported"}] * 4)]
        )
        write_jsonl(str(v3 / f"{split}.jsonl"), [_example("a9")])
    for name in ("test_items", "test_labels", "test_prompts"):
        write_jsonl(str(v3 / f"{name}.jsonl"), [{"x": 1}])
    main(
        ["hard", "--export", str(export_dir), "--distilled", str(v3), "--out", str(out)]
    )
    assert len(read_jsonl(str(out / "hard_train.jsonl"))) == 1
    summary = json.loads((out / "hard_summary.json").read_text())
    assert summary["train"]["by_kind"] == {"teachers partial": 1}
    kept = out / "hard_train.distilled.jsonl"
    write_jsonl(str(kept), [{"messages": [{"role": "user", "content": "x"}]}])
    main(
        [
            "mix",
            "--base",
            str(v3),
            "--extra",
            str(kept),
            "--repeat",
            "3",
            "--out",
            str(out),
        ]
    )
    assert len(read_jsonl(str(out / "train.jsonl"))) == 4
    assert len(read_jsonl(str(out / "tune.jsonl"))) == 1
    assert read_jsonl(str(out / "test_items.jsonl")) == [{"x": 1}]
