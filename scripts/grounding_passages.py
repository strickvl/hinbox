"""Give a finished run's grounding verdicts the article passages they rest
on, without calling a model.

Runs from before passages were saved (the overnight 9 Nov Qwen run among
them) have verdicts whose reasoning often quotes the article: 'The article
says "Fourteen men were flown to the base", not fifteen.' This reads each
stored verdict without ``passages``, looks up the quotes in its reasoning in
the article text the judge saw, and stores the ones that are really there
(``backfill_passages()`` in ``src/engine/grounding_pass.py``). Verdicts,
reasoning and hashes are not touched, so the next pipeline run reuses the
reports as before. A verdict whose reasoning quotes nothing that is in the
article gets no passage; the reader then shows the whole article.

It writes copies of the four entity tables to ``--out`` and leaves the run
folder alone, so it is safe beside a run still in progress (point it at a
copy of that run's tables, or run it once the run is done). ``--in-place``
rewrites the run's own tables instead.

    uv run python -m scripts.grounding_passages --entities <dir> \\
        --articles <articles.parquet> --out <dir>
    HINBOX_OUTPUT_DIR=<out> just frontend

Outputs hold article text: keep them off GitHub.
"""

import argparse
import json
import os

import pyarrow.parquet as pq

from src.utils.grounding_passages import passage_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entities", required=True, help="folder with people.parquet")
    parser.add_argument("--articles", required=True, help="articles Parquet file")
    parser.add_argument("--out", help="folder to write the updated tables to")
    parser.add_argument(
        "--in-place", action="store_true", help="rewrite the tables in --entities"
    )
    args = parser.parse_args()
    if bool(args.out) == args.in_place:
        parser.error("give either --out <folder> or --in-place")
    out = args.entities if args.in_place else args.out
    if not args.in_place and os.path.realpath(out) == os.path.realpath(args.entities):
        parser.error("--out is the run folder; use --in-place to mean that")

    from src.engine.article_processor import ArticleProcessor
    from src.engine.grounding_pass import article_texts_from_rows, backfill_passages
    from src.process_and_extract import load_existing_entities, write_entities_to_files

    entities = load_existing_entities(args.entities)
    rows = pq.read_table(args.articles).to_pylist()
    texts = article_texts_from_rows(
        rows,
        ArticleProcessor.__new__(ArticleProcessor),  # stateless here
    )
    passage_stats(reset=True)
    counts = backfill_passages(entities, texts)
    os.makedirs(out, exist_ok=True)
    write_entities_to_files(entities, out)
    print(
        json.dumps({"counts": dict(counts), "quotes": dict(passage_stats())}, indent=1)
    )


if __name__ == "__main__":
    main()
