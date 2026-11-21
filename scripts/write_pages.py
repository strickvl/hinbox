"""Write encyclopedia pages from the notes of chosen pages of a finished run.

    uv run python -m scripts.write_pages <entities dir> <out dir> \\
        --names "Karachi,Central Intelligence Agency" \\
        --models ollama/local-qwen36-32k openai/gpt-6-luna [--no-check] [--dry-run]

Copies the four tables to <out dir> (unless it is the same folder), writes a
page per model for each named entity (``src/engine/written_page.py``),
stores it under ``profile["written_json"]`` and saves the tables after each
page. The source folder is never written. The reader shows a written page
above the notes (``HINBOX_OUTPUT_DIR=<out dir> just frontend``).

The tables hold article text: keep <out dir> off GitHub.
"""

import argparse
import os
import shutil
import time
from typing import Any, Dict, List, Tuple

TABLES = ("people", "organizations", "locations", "events")


def find(entities: Dict[str, Dict], name: str) -> List[Tuple[str, Any]]:
    """Every (table, key) whose page is called ``name``, most articles first."""
    hits = [
        (t, k, len(e.get("articles") or []))
        for t, rows in entities.items()
        for k, e in rows.items()
        if str(e.get("name") or e.get("title") or "").strip().lower() == name.lower()
    ]
    return [(t, k) for t, k, _ in sorted(hits, key=lambda h: -h[2])]


def main() -> None:
    from src.config_loader import load_domain_config
    from src.engine.section_reorganise import _starter_entries
    from src.engine.written_page import make_ask, page_notes, store_page, write_page
    from src.utils.file_ops import load_existing_entities, write_entities_table

    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--names", required=True, help="comma-separated page names")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--domain", default="guantanamo")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--workers", type=int, default=2, help="condensing calls at once")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if os.path.abspath(a.src) != os.path.abspath(a.out):
        os.makedirs(a.out, exist_ok=True)
        for t in TABLES:
            f = os.path.join(a.src, f"{t}.parquet")
            if os.path.exists(f) and not os.path.exists(
                os.path.join(a.out, f"{t}.parquet")
            ):
                shutil.copy2(f, os.path.join(a.out, f"{t}.parquet"))
    entities = load_existing_entities(a.out)
    config = load_domain_config(a.domain)

    chosen = []
    for name in (n.strip() for n in a.names.split(",") if n.strip()):
        hits = find(entities, name)
        if not hits:
            print(f"not found: {name}")
            continue
        t, k = hits[0]
        notes = page_notes(entities[t][k])
        print(
            f"{t}/{name}: {len(notes)} notes, {sum(len(n.text) for n in notes)} chars"
            + (f" ({len(hits) - 1} more pages with this name)" if len(hits) > 1 else "")
        )
        chosen.append((t, k))
    if a.dry_run:
        return

    for model in a.models:
        ask = make_ask(model)
        for t, k in chosen:
            ent = entities[t][k]
            kind = ent.get("type") or ent.get("event_type")
            start = time.monotonic()
            page = write_page(
                ent,
                t,
                model,
                ask,
                _starter_entries(config, t, kind),
                check=not a.no_check,
                workers=a.workers,
            )
            secs = round(time.monotonic() - start)
            page["report"]["seconds"] = secs
            r = page["report"]
            print(
                f"{model} {t}/{ent.get('name')}: {page.get('error') or 'ok'} in {secs}s; "
                f"notes {r.get('notes')} shown {r.get('shown')} used {r.get('notes_used')}; "
                f"sentences {r.get('sentences')} kept {r.get('kept')} "
                f"(dropped code {r.get('dropped_code')}, check {r.get('dropped_check')}, "
                f"partial {r.get('partial')}); {r.get('words')} words; calls {r.get('calls')}"
            )
            if page.get("text"):
                store_page(ent["profile"], page)
                write_entities_table(t, list(entities[t].values()), a.out)


if __name__ == "__main__":
    main()
