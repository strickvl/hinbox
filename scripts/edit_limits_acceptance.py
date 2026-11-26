"""Review applied replay claims against the saved article and starting profile.

Run with one or more results.jsonl files to print a JSON review template. Fill
each verdict from the cited source and starting profile, then pass that file
back with --reviews. These files contain article text. Keep them local.

Use new_supported for a substantive new fact, already_covered for a supported
restatement, off_topic for a supported fact unrelated to the profile, and
unsupported when the cited source fails to support any material part. Include
the mismatch in the reason. Scoring exits nonzero for failures or pending review.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.grounding import CITATION_RE, _claim_key, _extract_cited_claims

VERDICTS = {
    "new_supported",
    "already_covered",
    "off_topic",
    "unsupported",
}


def _claims(text: str) -> Counter[Tuple[str, str]]:
    claims = Counter(
        (row["article_id"], _claim_key(row["claim"]).rstrip(" ."))
        for row in _extract_cited_claims(text)
    )
    end = 0
    for citation in CITATION_RE.finditer(text):
        end = citation.end()
    uncited = _claim_key(text[end:]).strip(" .")
    if any(char.isalnum() for char in uncited):
        claims["", uncited] += 1
    return claims


def claim_changes(before: str, after: str, article_id: str) -> Dict[str, Any]:
    """List new claim/source pairs and count retained old cited claims."""
    old, new = _claims(before), _claims(after)
    cited = Counter({key: n for key, n in old.items() if key[0]})
    profile_hash = hashlib.sha256(before.encode()).hexdigest()
    additions = []
    for source, claim in new - old:
        key = json.dumps([profile_hash, article_id, source, claim])
        additions.append(
            {
                "id": hashlib.sha256(key.encode()).hexdigest()[:20],
                "article_id": source,
                "claim": claim,
            }
        )
    return {
        "article_id": article_id,
        "old_cited": sum(cited.values()),
        "old_kept": sum((cited & new).values()),
        "additions": additions,
    }


def acceptance(
    row: Dict[str, Any], reviews: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"status": "fail", "new_supported": [], "unsupported": []}
    if row.get("finish") != "stop" or not row.get("parsed"):
        return {**result, "reason": "reply did not finish and parse"}
    changes = row.get("claim_changes")
    if changes is None:
        return {
            **result,
            "status": "unverified",
            "reason": "full edit cannot be replayed",
        }

    pending = []
    for claim in changes["additions"]:
        review = reviews.get(claim["id"], {})
        verdict = review.get("verdict")
        if not claim["article_id"]:
            result["unsupported"].append({**claim, "reason": "missing citation"})
        elif verdict == "unsupported":
            result["unsupported"].append({**claim, "reason": review.get("reason", "")})
        elif (
            verdict == "new_supported" and claim["article_id"] == changes["article_id"]
        ):
            result["new_supported"].append(claim)
        elif verdict not in VERDICTS:
            pending.append(claim["id"])

    if result["unsupported"]:
        return {**result, "reason": "unsupported additions"}
    if changes["old_kept"] < changes["old_cited"]:
        return {**result, "reason": "old cited claims changed or lost"}
    if pending:
        return {**result, "status": "unverified", "reason": "unreviewed additions"}
    if not result["new_supported"]:
        return {**result, "reason": "no new supported fact from the incoming article"}
    return {**result, "status": "pass"}


def review_template(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    entries = {}
    for row in rows:
        for claim in (row.get("claim_changes") or {}).get("additions", []):
            entries.setdefault(
                claim["id"],
                {
                    "entity": row.get("entity"),
                    "incoming_article": row.get("article"),
                    "cited_article": claim["article_id"],
                    "claim": claim["claim"],
                    "verdict": "",
                    "reason": "",
                },
            )
    return entries


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for path in args.results
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.reviews is None:
        print(json.dumps(review_template(rows), ensure_ascii=False, indent=2))
        return 0
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    failed = False
    for row in rows:
        result = acceptance(row, reviews)
        failed |= result["status"] != "pass"
        print(json.dumps({
            "entity": row.get("entity"),
            "article": row.get("article"),
            "limits": row.get("limits"),
            "rep": row.get("rep"),
            **result,
        }, ensure_ascii=False))  # fmt: skip
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
