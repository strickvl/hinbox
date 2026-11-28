"""``prompts --with-head`` / ``score --with-head``: the article's headline
and date above its text. Every name and sentence here is invented."""

from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.citecheck.__main__ import main
from src.citecheck.items import article_head, read_jsonl, write_jsonl
from src.citecheck.score import prompt_rows

ITEM = {
    "item_id": "i1",
    "article_id": "a1",
    "entity_name": "Ann Lee",
    "claim": "Ann Lee was flown in May.",
}


def test_article_head_lines():
    assert article_head({"title": " Base news ", "date": "2009-05-02T00:00:00"}) == (
        "Headline: Base news\nPublished: 2009-05-02\n\n"
    )
    assert article_head({"title": None, "date": None}) == ""


def test_prompt_rows_put_the_head_above_the_text():
    rows = prompt_rows([ITEM], {"a1": "Body text."}, {"a1": "Headline: H\n\n"})
    user = rows[0]["messages"][1]["content"]
    assert "Headline: H\n\nBody text." in user
    plain = prompt_rows([ITEM], {"a1": "Body text."})[0]["messages"][1]["content"]
    assert "Headline" not in plain


def test_commands_read_title_and_date(tmp_path):
    arts = tmp_path / "a.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["a1"],
                "content": ["Body text."],
                "title": ["Base news"],
                "published_date": ["2009-05-02"],
            }
        ),
        arts,
    )
    items, labels = tmp_path / "items.jsonl", tmp_path / "labels.jsonl"
    write_jsonl(str(items), [ITEM])
    write_jsonl(str(labels), [{"item_id": "i1", "label": "no"}])
    out = tmp_path / "p.jsonl"
    main(
        [
            "prompts",
            "--items",
            str(items),
            "--articles",
            str(arts),
            "--file",
            str(out),
            "--with-head",
        ]
    )
    assert "Published: 2009-05-02" in read_jsonl(str(out))[0]["messages"][1]["content"]
    with patch("src.citecheck.score.judge_claims", return_value=[None]) as judge:
        main(
            [
                "score",
                "--items",
                str(items),
                "--labels",
                str(labels),
                "--articles",
                str(arts),
                "--model",
                "openai/x",
                "--with-head",
            ]
        )
    assert judge.call_args.kwargs["article_text"].startswith("Headline: Base news\n")
