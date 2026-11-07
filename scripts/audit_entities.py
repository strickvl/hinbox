"""Spot-check a folder of entity tables: lost facts, citations, duplicates, misfiles.

Usage: uv run python -m scripts.audit_entities <entities_dir> [--articles <parquet>]
       [--replay <eval>/profile_replay/<model>] [--out report.md] [--examples 5]

Checks, all without a model:

- Lost facts: for each entity with 2+ profile versions, every cited sentence of
  every earlier version is looked for in the current profile with the profile
  eval's rule (a current sentence holding ``KEPT_OVERLAP`` of its content
  words). Sentences found in no later version count as lost. A lost sentence
  whose article is cited nowhere in the current profile is "orphaned": the
  profile no longer says anything from that article.
- Shrinks: single updates that cut the profile by more than 20%.
- Citations: cited ids that are not one of the entity's linked articles
  (garbled or invented), and linked articles the profile never cites.
- Duplicates: pairs of names in one table that ``names_are_variants`` would send
  to the match check, and names that appear in two tables (one is mis-typed).
- Misfiles: locations ``non_place_reason`` rejects, organizations named like a
  person in the people table.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow.parquet as pq

from src.evals.profile_cases import _CITATION, _content_words, _sentences
from src.evals.profile_update import KEPT_OVERLAP
from src.utils.name_variants import names_are_variants, normalize_for_match
from src.utils.non_place_names import non_place_reason

TYPES = ("people", "organizations", "locations", "events")
SHRINK = 0.8
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def entity_name(row: Dict[str, Any]) -> str:
    return row.get("name") or row.get("title") or ""


def cited(text: str) -> List[str]:
    return [i.strip() for g in _CITATION.findall(text or "") for i in g.split(",")]


def cited_facts(text: str) -> List[Tuple[str, Set[str], List[str]]]:
    """(sentence, content words, citation ids) for each cited sentence."""
    out = []
    for s in _sentences(text or ""):
        ids, words = cited(s), _content_words(s)
        if ids and len(words) >= 3:
            out.append((s, words, ids))
    return out


def kept(words: Set[str], pool: List[Set[str]]) -> bool:
    return any(len(words & c) / len(words) >= KEPT_OVERLAP for c in pool)


def version_texts(row: Dict[str, Any]) -> List[str]:
    versions = (row.get("profile_versions") or {}).get("versions") or []
    texts = [((v.get("profile_data") or {}).get("text") or "") for v in versions]
    current = (row.get("profile") or {}).get("text") or ""
    if not texts or texts[-1] != current:
        texts.append(current)
    return texts


REPEAT_OVERLAP = 0.8
_DISPUTE = re.compile(
    r"\b(sources? (?:disagree|differ|conflict)|conflicting (?:accounts|reports))", re.I
)


def repeats_and_disputes(text: str) -> Tuple[List[str], List[str]]:
    """Cited sentences that restate an earlier one, and "sources disagree" sentences."""
    seen: List[Set[str]] = []
    repeats = []
    for sentence, words, _ in cited_facts(text):
        if any(len(words & w) / len(words) >= REPEAT_OVERLAP for w in seen):
            repeats.append(sentence.strip())
        seen.append(words)
    disputes = [s.strip() for s in _sentences(text or "") if _DISPUTE.search(s)]
    return repeats, disputes


def audit_history(texts: List[str]) -> Dict[str, Any]:
    """Lost cited sentences and big shrinks across a profile's versions."""
    final = texts[-1]
    final_pool = [w for _, w, _ in cited_facts(final)]
    final_ids = set(cited(final))
    later_pools = [[w for _, w, _ in cited_facts(t)] for t in texts]
    lost, seen = [], set()
    for i, text in enumerate(texts[:-1]):
        for sentence, words, ids in cited_facts(text):
            key = frozenset(words)
            if key in seen or kept(words, final_pool):
                continue
            seen.add(key)
            # when did it go? first later version that no longer holds it
            gone_at = next(
                (
                    j
                    for j in range(i + 1, len(texts))
                    if not kept(words, later_pools[j])
                ),
                len(texts) - 1,
            )
            lost.append(
                {
                    "sentence": sentence.strip(),
                    "from_version": i + 1,
                    "gone_at_version": gone_at + 1,
                    "orphaned": not (set(ids) & final_ids),
                }
            )
    shrinks = [
        {"version": i + 1, "before": len(a), "after": len(b)}
        for i, (a, b) in enumerate(zip(texts, texts[1:]), start=1)
        if a and len(b) < SHRINK * len(a)
    ]
    peak = max(len(t) for t in texts)
    repeats, disputes = repeats_and_disputes(final)
    return {
        "repeats": repeats,
        "disputes": disputes,
        "versions": len(texts),
        "lengths": [len(t) for t in texts],
        "peak_chars": peak,
        "final_chars": len(final),
        "lost": lost,
        "shrinks": shrinks,
    }


def audit_citations(
    row: Dict[str, Any], corpus_ids: Optional[Set[str]]
) -> Dict[str, Any]:
    profile = row.get("profile") or {}
    text = profile.get("text") or ""
    linked = {a.get("article_id") for a in row.get("articles") or []}
    ids = cited(text)
    bad = [i for i in ids if i not in linked]
    return {
        "citations": len(ids),
        "not_linked": bad,
        "malformed": [i for i in bad if not UUID.match(i)],
        "in_corpus": [i for i in bad if corpus_ids and i in corpus_ids],
        # A frozen hub's later articles are linked on purpose, not forgotten
        "never_cited": sorted(
            linked - set(ids) - set(profile.get("linked_only") or [])
        ),
    }


def find_duplicates(
    tables: Dict[str, List[Dict[str, Any]]], groups: Dict[str, List[List[str]]]
) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {}
    for etype, rows in tables.items():
        names = sorted({entity_name(r) for r in rows if entity_name(r)})
        pairs = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if names_are_variants(
                    a, b, entity_type=etype, equivalence_groups=groups.get(etype)
                ):
                    pairs.append((a, b))
        out[etype] = pairs
    return out


def cross_type(tables: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, List[str]]]:
    where: Dict[str, Set[str]] = defaultdict(set)
    shown: Dict[str, str] = {}
    for etype, rows in tables.items():
        for r in rows:
            key = normalize_for_match(entity_name(r))
            if key:
                where[key].add(etype)
                shown.setdefault(key, entity_name(r))
    return sorted((shown[k], sorted(v)) for k, v in where.items() if len(v) > 1)


def load_tables(entities_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    tables = {}
    for etype in TYPES:
        path = os.path.join(entities_dir, f"{etype}.parquet")
        if os.path.exists(path):
            cols = [
                c
                for c in pq.read_schema(path).names
                if "embedding" not in c and c != "reflection_history"
            ]
            tables[etype] = pq.read_table(path, columns=cols).to_pylist()
    return tables


def equivalence_groups() -> Dict[str, List[List[str]]]:
    try:
        from src.config_loader import DomainConfig

        cfg = DomainConfig("guantanamo")
        return {
            t: cfg.get_name_variants_config(t).get("equivalence_groups", [])
            for t in TYPES
        }
    except Exception:
        return {}


def build_report(
    tables: Dict[str, List[Dict[str, Any]]],
    corpus_ids: Optional[Set[str]],
    n_examples: int,
) -> Tuple[str, Dict[str, Any]]:
    lines: List[str] = []
    data: Dict[str, Any] = {}
    for etype, rows in tables.items():
        hist = [(entity_name(r), audit_history(version_texts(r)), r) for r in rows]
        multi = [h for h in hist if h[1]["versions"] > 1]
        cites = [(entity_name(r), audit_citations(r, corpus_ids)) for r in rows]
        lost = sum(len(h["lost"]) for _, h, _ in multi)
        orphaned = sum(sum(x["orphaned"] for x in h["lost"]) for _, h, _ in multi)
        shrinks = sum(len(h["shrinks"]) for _, h, _ in multi)
        repeats = sum(len(h["repeats"]) for _, h, _ in hist)
        disputes = sum(len(h["disputes"]) for _, h, _ in hist)
        with_repeats = sum(1 for _, h, _ in hist if h["repeats"])
        total_cites = sum(c["citations"] for _, c in cites)
        not_linked = sum(len(c["not_linked"]) for _, c in cites)
        malformed = sum(len(c["malformed"]) for _, c in cites)
        uncited = sum(1 for _, c in cites if c["never_cited"])
        n_articles = Counter(len(r.get("articles") or []) for r in rows)
        qc_fail = sum(
            1
            for r in rows
            if ((r.get("profile") or {}).get("qc") or {}).get("passed") is False
        )
        data[etype] = {
            "rows": len(rows),
            "with_history": len(multi),
            "lost_sentences": lost,
            "orphaned": orphaned,
            "shrinks": shrinks,
            "repeated_sentences": repeats,
            "rows_with_repeats": with_repeats,
            "dispute_sentences": disputes,
            "citations": total_cites,
            "not_linked": not_linked,
            "malformed": malformed,
            "rows_with_uncited_articles": uncited,
            "qc_failed": qc_fail,
        }
        lines += [
            f"## {etype}: {len(rows)} rows",
            f"- Articles per row: {dict(sorted(n_articles.items()))}",
            f"- Rows with 2+ profile versions: {len(multi)}",
            f"- Cited sentences lost from earlier versions: {lost} "
            f"({orphaned} whose article is no longer cited at all)",
            f"- Single updates that cut the profile by >20%: {shrinks}",
            f"- Cited sentences that restate an earlier sentence of the same profile "
            f"(>= {REPEAT_OVERLAP:.0%} of its words): {repeats} in {with_repeats} rows",
            f"- 'Sources disagree' sentences: {disputes}",
            f"- Citations: {total_cites}; not one of the row's articles: {not_linked} "
            f"(malformed ids: {malformed})",
            f"- Rows with a linked article the profile never cites: {uncited}",
            f"- Rows whose profile QC failed: {qc_fail}",
            "",
        ]
        worst = sorted(multi, key=lambda h: -len(h[1]["lost"]))[:n_examples]
        for name, h, r in worst:
            if not h["lost"]:
                continue
            lines.append(
                f"### {name}: {h['versions']} versions, "
                f"{len(r.get('articles') or [])} articles, "
                f"chars {h['lengths'][0]} -> peak {h['peak_chars']} -> {h['final_chars']}, "
                f"{len(h['lost'])} lost"
            )
            for x in h["lost"][:4]:
                tag = " [orphaned]" if x["orphaned"] else ""
                lines.append(
                    f"- v{x['from_version']}, gone by v{x['gone_at_version']}{tag}: "
                    f"{x['sentence'][:300]}"
                )
            lines.append("")
        rep = sorted(hist, key=lambda h: -len(h[1]["repeats"]))[:n_examples]
        if any(h["repeats"] for _, h, _ in rep):
            lines.append("### Most repeated profiles")
            for name, h, _ in rep:
                if h["repeats"]:
                    lines.append(
                        f"- {name}: {len(h['repeats'])} repeats, e.g. {h['repeats'][0][:200]}"
                    )
            lines.append("")
        dis = [(n, d) for n, h, _ in hist for d in h["disputes"]][: n_examples * 2]
        if dis:
            lines.append("### 'Sources disagree' sentences")
            lines += [f"- {n}: {d[:300]}" for n, d in dis]
            lines.append("")
        bad = [(n, c) for n, c in cites if c["not_linked"]][:n_examples]
        if bad:
            lines.append("### Citations that don't point at a linked article")
            for n, c in bad:
                lines.append(f"- {n}: {', '.join(c['not_linked'][:3])}")
            lines.append("")
    groups = equivalence_groups()
    dups = find_duplicates(tables, groups)
    data["duplicates"] = {k: len(v) for k, v in dups.items()}
    lines.append("## Possible duplicates (names the match check should have seen)")
    for etype, pairs in dups.items():
        lines.append(f"- {etype}: {len(pairs)} pairs")
        for a, b in pairs[: n_examples * 3]:
            lines.append(f"  - {a}  /  {b}")
    xt = cross_type(tables)
    data["cross_type"] = len(xt)
    lines += ["", f"## Same name in two tables: {len(xt)}"]
    lines += [f"- {n}: {', '.join(t)}" for n, t in xt[: n_examples * 4]]
    locs = [entity_name(r) for r in tables.get("locations", [])]
    non_place = [(n, non_place_reason(normalize_for_match(n))) for n in locs]
    non_place = [(n, why) for n, why in non_place if why]
    people = {normalize_for_match(entity_name(r)) for r in tables.get("people", [])}
    org_people = [
        entity_name(r)
        for r in tables.get("organizations", [])
        if normalize_for_match(entity_name(r)) in people
    ]
    data["non_place_locations"] = len(non_place)
    data["orgs_named_as_people"] = len(org_people)
    lines += ["", f"## Locations that aren't places: {len(non_place)}"]
    lines += [f"- {n} ({why})" for n, why in non_place[: n_examples * 3]]
    lines += ["", f"## Organizations named like a person: {len(org_people)}"]
    lines += [f"- {n}" for n in org_people[: n_examples * 3]]
    return "\n".join(lines) + "\n", data


def audit_replay(replay_dir: str, n_examples: int) -> str:
    """Lost facts along ``just eval-replay`` snapshots (one folder per entity)."""
    lines = [f"## Replay snapshots: {replay_dir}"]
    for slug in sorted(os.listdir(replay_dir)):
        entity_dir = os.path.join(replay_dir, slug)
        steps = sorted(
            f for f in os.listdir(entity_dir) if re.match(r"step_\d+\.json$", f)
        )
        if len(steps) < 2:
            continue
        snaps = []
        for f in steps:
            with open(os.path.join(entity_dir, f), encoding="utf-8") as fh:
                snaps.append(json.load(fh))
        texts = [(s.get("profile") or {}).get("text") or "" for s in snaps]
        h = audit_history(texts)
        orphaned = sum(x["orphaned"] for x in h["lost"])
        first_cites = len(set(cited(texts[0])))
        sources = {s["article"]["id"] for s in snaps}
        final_sources = set(cited(texts[-1])) & sources
        lines.append(
            f"### {slug}: {len(texts)} steps, chars {h['lengths'][0]} -> peak "
            f"{h['peak_chars']} -> {h['final_chars']}; {len(h['lost'])} cited sentences "
            f"lost ({orphaned} orphaned); {len(h['shrinks'])} shrinks >20%; final cites "
            f"{len(final_sources)} of {len(sources)} articles (step 1 cited {first_cites}); "
            f"{len(h['repeats'])} repeated sentences, {len(h['disputes'])} 'sources disagree'; "
            f"snapshots dated {snaps[0].get('timestamp', '?')[:10]} to {snaps[-1].get('timestamp', '?')[:10]}"
        )
        for x in h["lost"][:n_examples]:
            tag = " [orphaned]" if x["orphaned"] else ""
            lines.append(
                f"- step {x['from_version']}, gone by step {x['gone_at_version']}{tag}: "
                f"{x['sentence'][:300]}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("entities_dir", nargs="?")
    parser.add_argument("--replay", help="a profile_replay/<model> folder of snapshots")
    parser.add_argument(
        "--articles", help="articles Parquet, to tell garbled ids from real ones"
    )
    parser.add_argument("--out", help="write the Markdown report here")
    parser.add_argument("--json", help="write the counts here")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    corpus = None
    if args.articles:
        corpus = set(
            pq.read_table(args.articles, columns=["id"]).column("id").to_pylist()
        )
    report, data = "", {}
    if args.entities_dir:
        report, data = build_report(
            load_tables(args.entities_dir), corpus, args.examples
        )
        report = f"# Entity audit: {args.entities_dir}\n\n" + report
    if args.replay:
        report += "\n" + audit_replay(args.replay, args.examples)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    print(report)


if __name__ == "__main__":
    main()
