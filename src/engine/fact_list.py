"""Update a profile from a list of sourced facts, without showing the model the profile.

Prototype, used only by ``just eval profile-update --approach facts``.

The edit-list updater (``profile_edits.py``) keeps the model's *output* small,
but every update still sends the whole profile. A 40k-character profile plus
an article is about 25k tokens, past a 16k local model's window, and the 9/11
trial has 100+ articles. Here no model call ever sees the whole profile:

1. **Extract** (model): the article alone, plus the profile's section headings,
   becomes a list of claims about the entity, each with an exact quote from the
   article that code checks.
2. **Look up** (code): for each claim, the few existing facts (rows) that share
   the most words and numbers with it.
3. **Compare** (model): each claim beside its few rows is ``same``, ``new`` or
   ``conflict``. One call for the whole article.
4. **Apply** (code). ``same`` adds nothing to the text: the article is counted
   on the row as a repeat, so the 50th report of the same flight doesn't add a
   50th citation. Only a different kind of source (a court record where the row
   rested on a newspaper) is cited. ``conflict`` keeps both: code writes the
   new version right after the old sentence. ``new`` goes to step 5.
5. **Write** (model, one call per changed section): the section's text and its
   new facts; the model says where each goes. Code inserts them, so existing
   text can't be lost.

Rows come from the existing profile (``rows_from_profile()``): one per cited
sentence, with its section and sources. Unchanged sections are copied as they
were, word for word.
"""

import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.engine.profile_edits import ProfileEdit, _parse_list_string, apply_edits
from src.evals.profile_cases import _content_words, _sentences
from src.utils.llm import cloud_generation, local_generation

FACT_MAX_TOKENS = 4096
MAX_CANDIDATES = 4  # existing rows shown beside each new claim
MIN_SHARED = 2  # word stems a row must share with a claim to be shown
MIN_QUOTE_WORDS = 0.9  # share of a loose quote's words that must be in the article
_CITATION = re.compile(r"\^\[([^\]]+)\]")
_HEADING = re.compile(r"^#{1,6} .*$", re.MULTILINE)
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")


def _list_from_string(value: Any) -> Any:
    # Bolt sometimes sends a list as a JSON string (see profile_edits.py)
    if isinstance(value, str):
        parsed = _parse_list_string(value)
        return value if parsed is None else parsed
    return value


class ArticleClaim(BaseModel):
    claim: str = Field(description="One sentence stating the fact, no citation")
    section: str = Field(description="An existing '## heading' or a new one")
    quote: str = Field(description="The article's words it rests on, copied exactly")
    source: str = Field(default="the reporter", description="Who says it")


class ArticleClaims(BaseModel):
    claims: List[ArticleClaim] = Field(default_factory=list)

    @field_validator("claims", mode="before")
    @classmethod
    def _parse(cls, value: Any) -> Any:
        return _list_from_string(value)


class ClaimVerdict(BaseModel):
    claim: int = Field(description="The claim's number")
    verdict: Literal["same", "new", "conflict"]
    row: str = Field(default="", description="The known fact's id, for same/conflict")
    new_evidence: bool = False


class ClaimVerdicts(BaseModel):
    verdicts: List[ClaimVerdict] = Field(default_factory=list)

    @field_validator("verdicts", mode="before")
    @classmethod
    def _parse(cls, value: Any) -> Any:
        return _list_from_string(value)


class Placement(BaseModel):
    fact: int = Field(description="The fact's number")
    after: str = Field(description="An existing sentence copied exactly, or END")
    sentence: str = Field(description="The fact as a sentence, ending in its citation")


class SectionPlacements(BaseModel):
    placements: List[Placement] = Field(default_factory=list)

    @field_validator("placements", mode="before")
    @classmethod
    def _parse(cls, value: Any) -> Any:
        return _list_from_string(value)


# ---------------------------------------------------------------- profile → rows


def split_sections(text: str) -> List[str]:
    """Cut the text before each heading. ``"".join()`` gives the text back."""
    starts = [0] + [m.start() for m in _HEADING.finditer(text) if m.start() > 0]
    return [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)]) if b > a]


def _heading(chunk: str) -> str:
    first = chunk.split("\n", 1)[0].strip()
    return first if first.startswith("#") else ""


def _ids(text: str) -> List[str]:
    return [i.strip() for g in _CITATION.findall(text) for i in g.split(",")]


def rows_from_profile(text: str) -> List[Dict[str, Any]]:
    """One row per cited sentence of the profile, keeping its section."""
    rows: List[Dict[str, Any]] = []
    for chunk in split_sections(text):
        section = _heading(chunk)
        for sentence in _sentences(chunk):
            if sentence.startswith("#") or not _CITATION.search(sentence):
                continue
            rows.append(
                {
                    "id": f"f{len(rows) + 1}",
                    "section": section,
                    "claim": _CITATION.sub("", sentence).strip(),
                    "text": sentence,
                    "sources": [{"article_id": i} for i in _ids(sentence)],
                    "repeats": [],
                }
            )
    return rows


# ---------------------------------------------------------------- checks by code


def _norm(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[–—]", "-", text)
    return " ".join(text.lower().split())


def quote_found(quote: str, article: str) -> bool:
    """The quote is in the article: exactly, or nearly all its words in a row
    are (models drop a comma or a "the")."""
    quote_n, article_n = _norm(quote).strip(" .\"'"), _norm(article)
    if not quote_n:
        return False
    if quote_n in article_n:
        return True
    words = re.findall(r"[a-z0-9']+", quote_n)
    present = set(re.findall(r"[a-z0-9']+", article_n))
    return len(words) >= 4 and sum(w in present for w in words) / len(words) >= (
        MIN_QUOTE_WORDS
    )


def _stems(text: str) -> set:
    return {w[:5] for w in _content_words(text)}


def candidates(claim: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The rows most likely to be about the same thing as ``claim``.

    Scored by shared word stems, plus one for each shared number (a year, a
    count). This is the step that has to keep working at 300 rows: it is
    code, so the model only ever sees ``MAX_CANDIDATES`` rows per claim.
    """
    stems, numbers = _stems(claim), set(_NUMBER.findall(claim))
    scored = []
    for row in rows:
        shared = len(stems & _stems(row["claim"]))
        if shared < MIN_SHARED:
            continue
        bonus = len(numbers & set(_NUMBER.findall(row["claim"])))
        scored.append((shared + bonus, row["id"], row))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [row for _, _, row in scored[:MAX_CANDIDATES]]


def _with_citation(sentence: str, article_id: str) -> str:
    """Add ``article_id`` to the sentence's last citation (or a new one)."""
    ids = _ids(sentence)
    if article_id in ids:
        return sentence
    matches = list(_CITATION.finditer(sentence))
    if not matches:
        return f"{sentence.rstrip()}^[{article_id}]"
    last = matches[-1]
    return f"{sentence[: last.end() - 1]}, {article_id}]{sentence[last.end() :]}"


def _clean_sentence(sentence: str, article_id: str) -> str:
    """Cite only the article: drop other ids a model wrote, add it if missing."""
    body = _CITATION.sub("", sentence).strip()
    return f"{body}^[{article_id}]"


# ---------------------------------------------------------------- model calls


Generate = Callable[..., Any]


def _call(generate: Generate, model: str, system: str, user: str, schema: Any) -> Any:
    response = generate(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_model=schema,
        model=model,
        max_tokens=FACT_MAX_TOKENS,
    )
    if response is None:
        raise RuntimeError(f"{schema.__name__} call returned nothing")
    return response


def _prompt(domain: str, name: str, model: str, **values: str) -> str:
    from src.config_loader import DomainConfig

    with open(DomainConfig(domain).prompt_path(name, model), encoding="utf-8") as f:
        template = f.read()
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def _numbered_claims(claims: List[ArticleClaim]) -> str:
    return "\n".join(
        f"{n}. {c.claim} (source: {c.source})" for n, c in enumerate(claims, 1)
    )


# ---------------------------------------------------------------- the update


def update_from_facts(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict[str, Any],
    article_text: str,
    article_id: str,
    model: Optional[str] = None,
    domain: str = "guantanamo",
    generate: Optional[Generate] = None,
) -> Dict[str, Any]:
    """Run steps 1-5 on one article. Raises if a model call fails.

    Returns the profile with new ``text`` and ``sources``, its ``fact_rows``
    after the update, and a ``fact_report`` of what each step did.
    """
    local = (model or "").startswith("ollama/")
    model = model or (OLLAMA_MODEL if local else CLOUD_MODEL)
    generate = generate or (local_generation if local else cloud_generation)
    text = existing_profile.get("text", "")
    rows = rows_from_profile(text)
    chunks = split_sections(text)
    headings = [h for h in (_heading(c) for c in chunks) if h]
    names = {"entity_type": entity_type, "entity_name": entity_name}
    report: Dict[str, Any] = {"rows_before": len(rows), "prompt_chars": []}

    def ask(name: str, user: str, schema: Any, **values: str) -> Any:
        system = _prompt(domain, name, model, article_id=article_id, **names, **values)
        report["prompt_chars"].append(len(system) + len(user))
        return _call(generate, model, system, user, schema)

    # 1. Extract, keeping only claims whose quote is really in the article
    extracted = ask(
        "fact_extract",
        f"Article (ID: {article_id}):\n----------------\n{article_text}",
        ArticleClaims,
        headings="\n".join(headings) or "(none yet)",
    ).claims
    claims = [
        c for c in extracted if c.claim.strip() and quote_found(c.quote, article_text)
    ]
    report["claims"] = len(extracted)
    report["dropped_quotes"] = [c.claim for c in extracted if c not in claims]

    # 2-3. Look up the likely rows, then compare (claims with none are new)
    found = [candidates(c.claim, rows) for c in claims]
    verdicts: Dict[int, ClaimVerdict] = {}
    if any(found):
        blocks = []
        for n, (claim, rows_for) in enumerate(zip(claims, found), 1):
            if not rows_for:
                continue
            known = "\n".join(f"  [{r['id']}] {r['claim']}" for r in rows_for)
            blocks.append(f"Claim {n}: {claim.claim} (source: {claim.source})\n{known}")
        answer = ask("fact_compare", "\n\n".join(blocks), ClaimVerdicts)
        verdicts = {v.claim: v for v in answer.verdicts}

    by_id = {r["id"]: r for r in rows}
    section_edits: Dict[str, List[ProfileEdit]] = {}
    new_facts: Dict[str, List[Tuple[int, ArticleClaim]]] = {}
    tally = {"same": 0, "new_evidence": 0, "conflict": 0, "new": 0}
    for n, (claim, rows_for) in enumerate(zip(claims, found), 1):
        verdict = verdicts.get(n)
        # A verdict naming a row it wasn't shown is a slip: treat it as new
        row = by_id.get(verdict.row) if verdict else None
        if verdict and row in rows_for and verdict.verdict == "same":
            tally["same"] += 1
            if verdict.new_evidence:
                tally["new_evidence"] += 1
                row["sources"].append(
                    {
                        "article_id": article_id,
                        "quote": claim.quote,
                        "kind": claim.source,
                    }
                )
                cited = _with_citation(row["text"], article_id)
                section_edits.setdefault(row["section"], []).append(
                    ProfileEdit(op="replace", anchor=row["text"], text=cited)
                )
            else:
                row["repeats"].append(article_id)
            continue
        if verdict and row in rows_for and verdict.verdict == "conflict":
            # Written by code, right after the old sentence: both stay, each cited
            tally["conflict"] += 1
            body = _CITATION.sub("", claim.claim).strip()
            section_edits.setdefault(row["section"], []).append(
                ProfileEdit(
                    op="add",
                    anchor=row["text"],
                    text=f"Another report differs: {body}^[{article_id}]",
                )
            )
            rows.append(
                _new_row(rows, row["section"], claim, article_id, disputes=row["id"])
            )
            continue
        tally["new"] += 1
        section = (
            claim.section.strip()
            if claim.section.strip() in headings
            else (
                claim.section.strip()
                if claim.section.strip().startswith("#")
                else f"## {claim.section.strip()}"
            )
        )
        new_facts.setdefault(section, []).append((n, claim))
    report.update(tally)

    # 5. Write the new facts, one call per existing section that gets any
    for section, items in new_facts.items():
        if section not in headings:
            continue
        chunk = next(c for c in chunks if _heading(c) == section)
        listing = "\n".join(f"{n}. {c.claim} ^[{article_id}]" for n, c in items)
        try:
            placed = ask(
                "fact_write",
                f"Section:\n{chunk.strip()}\n\nNew facts:\n{listing}",
                SectionPlacements,
                section=section,
            ).placements
        except Exception as e:  # noqa: BLE001 — fall back to the end of the section
            report.setdefault("write_errors", []).append(f"{type(e).__name__}: {e}")
            placed = []
        given = {p.fact: p for p in placed}
        for n, claim in items:
            p = given.get(n)
            sentence = _clean_sentence(
                p.sentence if p and p.sentence.strip() else claim.claim, article_id
            )
            anchor = (
                p.after.strip()
                if p and p.after.strip() and p.after.strip() != "END"
                else section
            )
            section_edits.setdefault(section, []).append(
                ProfileEdit(op="add", anchor=anchor, text=sentence)
            )
            rows.append(_new_row(rows, section, claim, article_id, text=sentence))

    # Apply each section's edits to that section's text only
    out = []
    for chunk in chunks:
        heading = _heading(chunk)
        edits = section_edits.pop(heading, None)
        if not edits:
            out.append(chunk)
            continue
        body = chunk.rstrip()
        edited, applied = apply_edits(body, edits)
        report.setdefault("unplaced", 0)
        report["unplaced"] += applied["appended"] + len(applied["skipped"])
        out.append(
            edited.rstrip() + chunk[len(body) :] if chunk[len(body) :] else edited
        )
    new_text = "".join(out)
    # New sections go at the end, in the order the article gave them
    for section, items in new_facts.items():
        if section in headings:
            continue
        sentences = [_clean_sentence(c.claim, article_id) for _, c in items]
        for (_, claim), sentence in zip(items, sentences):
            rows.append(_new_row(rows, section, claim, article_id, text=sentence))
        new_text = f"{new_text.rstrip()}\n\n{section}\n" + " ".join(sentences) + "\n"

    sources = list(existing_profile.get("sources") or [])
    report["rows_after"] = len(rows)
    return {
        **existing_profile,
        "text": new_text,
        "sources": sources + ([article_id] if article_id not in sources else []),
        "fact_rows": rows,
        "fact_report": report,
    }


def _new_row(
    rows: List[Dict[str, Any]],
    section: str,
    claim: ArticleClaim,
    article_id: str,
    text: str = "",
    disputes: Optional[str] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": f"f{len(rows) + 1}",
        "section": section,
        "claim": claim.claim,
        "text": text,
        "sources": [
            {"article_id": article_id, "quote": claim.quote, "kind": claim.source}
        ],
        "repeats": [],
    }
    if disputes:
        row["disputes"] = disputes
    return row
