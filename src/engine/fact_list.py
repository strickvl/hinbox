"""Update a profile from a list of sourced facts, without showing the model the profile.

Used by the pipeline with ``HINBOX_PROFILE_UPDATE_APPROACH=facts`` and by
``just eval profile-update --approach facts``.

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
   rested on a newspaper) is cited. ``conflict`` (accepted only when code
   finds a year, month or number that differs, ``values_differ()``) keeps both: code writes the
   new version right after the old sentence. ``new`` goes to step 5. Each
   claim is marked ``news`` (what the article reports) or background (history
   it recaps); news is handled first. From an article not mainly about the
   entity, background is dropped and at most ``MAX_PASSING_CLAIMS`` new news
   claims are kept.
4b. **Date check** (code pairs, one model call; ``date_disputes.py``): a date
   in a sentence step 1 skipped ("the bombing on Oct. 12, 1999") that differs
   from a profile sentence on the same event is written after it, like a conflict.
5. **Write** (model, one call per changed section): the section's text and its
   new facts; the model says where each goes. Code inserts them, so existing
   text can't be lost.

Rows come from the existing profile (``rows_from_profile()``): one per cited
sentence, with its section and sources. Unchanged sections are copied as they
were, word for word.
"""

import json
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.engine.date_disputes import (
    DISPUTE_PREFIX,
    date_check_enabled,
    dates_differ,
    find_date_disputes,
)
from src.engine.profile_edits import ProfileEdit, _parse_list_string, apply_edits
from src.evals.profile_cases import _content_words, _sentences
from src.utils.llm import cloud_generation, local_generation
from src.utils.step_timing import timed_step

FACT_MAX_TOKENS = 4096
MAX_CANDIDATES = 4  # existing rows shown beside each new claim
MIN_SHARED = 2  # word stems a row must share with a claim to be shown
MAX_PASSING_CLAIMS = 3  # news claims kept from an article not mainly about the entity
MIN_QUOTE_WORDS = 0.9
# Share of the shorter sentence's word stems a conflicting row must share
SAME_SUBJECT = 0.4
_CITATION = re.compile(r"\^\[([^\]]+)\]")
_HEADING = re.compile(r"^#{1,6} .*$", re.MULTILINE)
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_YEAR = re.compile(r"^(?:1[89]|20)\d\d$")
_MONTH = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|March|April|May|June|July|Aug(?:ust)?"
    r"|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b\.?"
)


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
    # On the labelled test cases, most missed facts were the article's news
    # cut by the passing-mention cap while recapped history ("Camp 7 is a
    # secretive prison") was kept: code now puts news first, and drops
    # background from passing mentions
    news: bool = Field(
        default=True,
        description="True if the article reports this as new (done, said or "
        "decided now); false for history or description it recaps",
    )


class ArticleClaims(BaseModel):
    # Asked first, so a passing mention can be capped by code: on dev, Luna
    # listed 5-6 claims from a hearing that only mentioned the Cole bombing
    mainly_about: bool = Field(
        default=True, description="Is the article mainly about the entity?"
    )
    claims: List[ArticleClaim] = Field(default_factory=list)

    @field_validator("claims", mode="before")
    @classmethod
    def _parse(cls, value: Any) -> Any:
        return _list_from_string(value)


class ClaimVerdict(BaseModel):
    claim: int = Field(description="The claim's number")
    verdict: Literal["same", "new", "update", "conflict"]
    row: str = Field(
        default="", description="The known fact's id, for same/update/conflict"
    )
    new_evidence: bool = False


class ClaimVerdicts(BaseModel):
    verdicts: List[ClaimVerdict] = Field(default_factory=list)

    @field_validator("verdicts", mode="before")
    @classmethod
    def _parse(cls, value: Any) -> Any:
        return _list_from_string(value)


class Placement(BaseModel):
    facts: List[int] = Field(
        description="The numbers of the facts this sentence states"
    )
    after: str = Field(description="An existing sentence copied exactly, or END")
    sentence: str = Field(
        description="The fact(s) as a sentence, ending in the citation"
    )

    @field_validator("facts", mode="before")
    @classmethod
    def _one_number(cls, value: Any) -> Any:
        return [value] if isinstance(value, (int, str)) else value


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
                    "id": f"f{_next_id(rows)}",
                    "section": section,
                    "claim": _CITATION.sub("", sentence).strip(),
                    "text": sentence,
                    "sources": [{"article_id": i} for i in _ids(sentence)],
                    "repeats": [],
                }
            )
    return rows


FACT_ROWS_KEY = "fact_rows_json"  # a JSON string, so Parquet sees one column


def load_rows(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The profile's saved rows, plus rows for cited sentences they don't cover.

    Saved rows keep what the text can't: each source's kind and quote, the
    articles that only repeated a fact, which row a conflict disputes. A saved
    row whose sentence is no longer in the text (the profile was rewritten
    another way) is dropped, so rows and text can't drift apart.
    """
    text = profile.get("text", "")
    try:
        saved = json.loads(profile.get(FACT_ROWS_KEY) or "[]")
    except (TypeError, ValueError):
        saved = []
    rows = [r for r in saved if r.get("text") and r["text"] in text]
    covered = {r["text"] for r in rows}
    used = {int(r["id"][1:]) for r in rows if str(r.get("id", ""))[1:].isdigit()}
    next_id = max(used, default=0) + 1
    for row in rows_from_profile(text):
        if row["text"] not in covered:
            rows.append({**row, "id": f"f{next_id}"})
            next_id += 1
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
    if len(words) < 4:
        return False
    # Nearly all the quote's words, in order, within one stretch of the
    # article. Words found anywhere would pass "Nashiri was not captured in
    # 2002" against "Nashiri was captured in 2002. ... not charged until 2011."
    tokens = re.findall(r"[a-z0-9']+", article_n)
    span, needed = len(words) + 3, MIN_QUOTE_WORDS * len(words)
    first = set(words[:2])
    for start, token in enumerate(tokens):
        if token not in first:
            continue
        window = tokens[start : start + span]
        blocks = SequenceMatcher(None, words, window, autojunk=False)
        if sum(b.size for b in blocks.get_matching_blocks()) >= needed:
            return True
    return False


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


def _values(text: str) -> Tuple[set, set, set]:
    """(years, months, other numbers) in the text, citations left out."""
    text = _CITATION.sub("", text)
    numbers = set(_NUMBER.findall(text))
    years = {n for n in numbers if _YEAR.match(n)}
    months = {m.lower()[:3] for m in _MONTH.findall(text)}
    return years, months, numbers - years


def values_differ(claim: str, row_claim: str) -> bool:
    """Code's check on a ``conflict`` verdict: for years, months or other
    numbers, each sentence has a value the other doesn't ("October 2003"
    against "November 2002"). Extra detail ("Camp 6 is a 200-cell building"
    against "Camp 6 opened in 2006") doesn't count.

    On dev, Qwen called 12 ordinary facts conflicts ("Camp 6 is a 200-cell
    penitentiary style building" against a sentence dated 2006; a Cole death
    toll that agreed with the profile), and after this check six more that
    were counts from different dates ("12 captives were on hunger strike").
    A planted dispute changes one year, so it passes; those don't.
    """
    (years, months, counts), (r_years, r_months, r_counts) = (
        _values(claim),
        _values(row_claim),
    )
    if (years - r_years and r_years - years) or (
        months - r_months and r_months - months
    ):
        return True
    # A count changes over time ("12 captives were on hunger strike" against
    # "about 100 were"), so it only conflicts when both give the same year
    return bool(counts - r_counts and r_counts - counts and years & r_years)


# "one" is left out: "one of the 17 sailors" isn't a count of one
_NUMBER_WORDS = dict(
    zip(
        "two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
        map(str, range(2, 21)),
    )
)
# A run of number words ("twenty-five", "two hundred", "thirty one") is
# converted only when it is a single word in the table: converting its parts
# would give "20-5" or "2 hundred", which read as new counts
_SPELLED = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen"
    "|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty"
    "|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|dozen"
)
_NUMBER_RUN = re.compile(
    rf"(?<![\w-])(?:{_SPELLED})(?:(?:[\s-]+|\s+and\s+)(?:{_SPELLED}))*(?![\w-])",
    re.IGNORECASE,
)


def _digits(text: str) -> str:
    """Spelled-out numbers as digits, so "seventeen sailors" matches "17"."""
    return _NUMBER_RUN.sub(
        lambda m: _NUMBER_WORDS.get(m.group(0).lower(), m.group(0)), text
    )


def adds_values(claim: str, row_claim: str, profile_text: str = "") -> bool:
    """The claim has a year, month or number that neither the row nor the
    rest of the profile has.

    Checked against the whole profile, not just the row: in the 60-article
    Qwen run, 29 of 38 ``same`` claims with a value the row lacked were
    background the profile already stated in another sentence ("killed 17
    sailors", "Oct. 12, 2000" on the Cole, Yemen and al-Qaida profiles) or in
    words ("seventeen"). Written as new facts, each article would have added
    the Cole bombing again.
    """
    known = _values(_digits(f"{row_claim} {profile_text}"))
    return any(mine - theirs for mine, theirs in zip(_values(_digits(claim)), known))


def same_subject(claim: str, row_claim: str) -> bool:
    """Code's second check on a ``conflict``: the two sentences are mostly
    about the same thing, numbers aside. Two different events with different
    years ("still deciding" in 2009, "decided" in 2011) share less."""
    mine, theirs = _stems(claim), _stems(row_claim)
    smaller = min(len(mine), len(theirs)) or 1
    return len(mine & theirs) / smaller >= SAME_SUBJECT


def _match_heading(
    section: str, headings: List[str], article: str = "", near: str = ""
) -> str:
    """The heading a new fact goes under.

    An existing heading with the same words at any level wins: Luna wrote
    "### Role and Position" beside "## Role and Position" and got a second
    section. A new heading copied from the article's own subheadings
    ("## ASTROTURF FOR CAMP?") goes to ``near`` (the section of the most
    similar existing fact), else the last section, instead.
    """
    words = section.strip().lstrip("#").strip()
    for heading in headings:
        if heading.lstrip("#").strip().lower() == words.lower():
            return heading
    letters = [c for c in words if c.isalpha()]
    shouting = len(letters) > 3 and sum(c.isupper() for c in letters) > 0.8 * len(
        letters
    )
    copied = words and any(
        line.strip().strip("#*: ").lower() == words.lower()
        for line in article.splitlines()
    )
    if shouting or copied or not words:
        return near or (headings[-1] if headings else "## Background")
    return f"## {words}"


def _kinds(row: Dict[str, Any]) -> str:
    kinds = sorted({s["kind"] for s in row["sources"] if s.get("kind")})
    return ", ".join(kinds) or "not recorded"


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
    date_check: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run steps 1-5 on one article. Raises if a model call fails.

    ``date_check`` (default: off, ``HINBOX_FACT_DATE_CHECK=on`` turns it on)
    adds step 4b, ``date_disputes.py``: background dates the extract step
    skipped are checked against the profile's dated sentences.

    Returns the profile with new ``text`` and ``sources``, its ``fact_rows``
    after the update, and a ``fact_report`` of what each step did.
    """
    local = (model or "").startswith("ollama/")
    model = model or (OLLAMA_MODEL if local else CLOUD_MODEL)
    generate = generate or (local_generation if local else cloud_generation)
    text = existing_profile.get("text", "")
    rows = load_rows(existing_profile)
    chunks = split_sections(text)
    headings = [h for h in (_heading(c) for c in chunks) if h]
    names = {"entity_type": entity_type, "entity_name": entity_name}
    report: Dict[str, Any] = {"rows_before": len(rows), "prompt_chars": []}

    def ask(name: str, user: str, schema: Any, **values: str) -> Any:
        with timed_step(name):  # each stage is its own task in traces
            system = _prompt(
                domain, name, model, article_id=article_id, **names, **values
            )
            report["prompt_chars"].append(len(system) + len(user))
            return _call(generate, model, system, user, schema)

    # 1. Extract, keeping only claims whose quote is really in the article
    answer = ask(
        "fact_extract",
        f"Article (ID: {article_id}):\n----------------\n{article_text}",
        ArticleClaims,
        headings="\n".join(headings) or "(none yet)",
    )
    extracted = answer.claims
    claims = [
        c for c in extracted if c.claim.strip() and quote_found(c.quote, article_text)
    ]
    report["claims"] = len(extracted)
    report["dropped_quotes"] = [c.claim for c in extracted if c not in claims]
    report["mainly_about"] = answer.mainly_about
    # News first, so the cap and the write step see it before background
    claims = sorted(claims, key=lambda c: not c.news)
    report["claim_log"] = [
        {"claim": c.claim, "news": c.news, "verdict": "new"} for c in claims
    ]

    # 2-3. Look up the likely rows, then compare (claims with none are new)
    found = [candidates(c.claim, rows) for c in claims]
    verdicts: Dict[int, ClaimVerdict] = {}
    if any(found):
        blocks = []
        for n, (claim, rows_for) in enumerate(zip(claims, found), 1):
            if not rows_for:
                continue
            # The source kinds, so ``new_evidence`` isn't a guess; rows read
            # back from profile text have only article ids
            known = "\n".join(
                f"  [{r['id']}] {r['claim']} (source: {_kinds(r)})" for r in rows_for
            )
            blocks.append(f"Claim {n}: {claim.claim} (source: {claim.source})\n{known}")
        compared = ask("fact_compare", "\n\n".join(blocks), ClaimVerdicts)
        verdicts = {v.claim: v for v in compared.verdicts}

    by_id = {r["id"]: r for r in rows}
    section_edits: Dict[str, List[ProfileEdit]] = {}
    new_facts: Dict[str, List[Tuple[int, ArticleClaim]]] = {}
    tally = {
        "same": 0,
        "new_evidence": 0,
        "conflict": 0,
        "not_conflict": 0,
        "update": 0,
        "same_with_detail": 0,
        "new": 0,
    }
    for n, (claim, rows_for) in enumerate(zip(claims, found), 1):
        verdict = verdicts.get(n)
        if verdict:
            report["claim_log"][n - 1].update(verdict=verdict.verdict, row=verdict.row)
        # A verdict naming a row it wasn't shown is a slip: treat it as new
        row = by_id.get(verdict.row) if verdict else None
        if (
            verdict
            and row in rows_for
            and verdict.verdict == "same"
            and adds_values(claim.claim, row["claim"], text)
        ):
            # "Same" but with a number or date the profile lacks ("a 78-minute
            # secret session"): the detail is new, so write it, don't drop it
            tally["same_with_detail"] += 1
            verdict = None
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
                # Rows that share this sentence follow it, or they'd be lost
                for other in rows:
                    if other is not row and other.get("text") == row["text"]:
                        other["text"] = cited
                row["text"] = cited
            else:
                row["repeats"].append(article_id)
            continue
        if verdict and row in rows_for and verdict.verdict == "conflict":
            if not (
                values_differ(claim.claim, row["claim"])
                and same_subject(claim.claim, row["claim"])
            ):
                tally["not_conflict"] += 1  # written as a plain new fact below
                verdict = None
        if verdict and row in rows_for and verdict.verdict == "update":
            # A later state of the same thing (moved camps, decided after
            # deciding): a new fact on the timeline, linked to the old row
            tally["update"] += 1
            claim.section = row["section"]  # next to the state it follows
            verdict = None
        if verdict and row in rows_for and verdict.verdict == "conflict":
            # Written by code, right after the old sentence: both stay, each cited
            tally["conflict"] += 1
            body = _CITATION.sub("", claim.claim).strip()
            sentence = f"{DISPUTE_PREFIX}{body}^[{article_id}]"
            section_edits.setdefault(row["section"], []).append(
                ProfileEdit(op="add", anchor=row["text"], text=sentence)
            )
            rows.append(
                _new_row(
                    rows,
                    row["section"],
                    claim,
                    article_id,
                    text=sentence,
                    disputes=row["id"],
                )
            )
            continue
        tally["new"] += 1
        # Capped among the *new* claims only: articles open with background
        # the profile has ("the bombing killed 17 sailors"), and capping the
        # first two claims kept those and dropped the new charge further down
        # A passing mention's background ("bin al Shibh is a Yemeni") is
        # dropped; only what the article reports about the entity is kept
        if not answer.mainly_about and (
            not claim.news or tally["new"] > MAX_PASSING_CLAIMS
        ):
            tally["new"] -= 1
            report.setdefault("dropped_passing", []).append(claim.claim)
            continue
        near = rows_for[0]["section"] if rows_for else ""
        section = _match_heading(claim.section, headings, article_text, near)
        new_facts.setdefault(section, []).append((n, claim))
    report.update(tally)

    # 4b. Dates in sentences the extract step skipped (see date_disputes.py)
    if date_check_enabled(date_check):
        _add_date_disputes(
            ask,
            rows,
            new_facts,
            section_edits,
            article_text,
            article_id,
            report,
        )

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
        # One sentence may state several facts; any fact the model left out
        # is written as extracted, at the end of the section
        # Each fact's row keeps the sentence that states it (shared when one
        # sentence states several), so its quote and source kind are saved
        numbers = {n for n, _ in items}
        written: Dict[int, str] = {}
        for p in placed:
            facts = [n for n in p.facts if n in numbers and n not in written]
            if not facts or not p.sentence.strip():
                continue
            sentence = _clean_sentence(p.sentence, article_id)
            written.update({n: sentence for n in facts})
            after = p.after.strip()
            section_edits.setdefault(section, []).append(
                ProfileEdit(
                    op="add",
                    anchor=after if after and after != "END" else section,
                    text=sentence,
                )
            )
        for n, claim in items:
            sentence = written.get(n) or _clean_sentence(claim.claim, article_id)
            if n not in written:
                section_edits.setdefault(section, []).append(
                    ProfileEdit(op="add", anchor=section, text=sentence)
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
        # Rows whose sentence made it into the text; saved for the next update
        FACT_ROWS_KEY: json.dumps(
            [r for r in rows if r.get("text") and r["text"] in new_text],
            ensure_ascii=False,
        ),
        "fact_rows": rows,
        "fact_report": report,
    }


def _add_date_disputes(
    ask: Callable[..., Any],
    rows: List[Dict[str, Any]],
    new_facts: Dict[str, List[Tuple[int, ArticleClaim]]],
    section_edits: Dict[str, List[ProfileEdit]],
    article_text: str,
    article_id: str,
    report: Dict[str, Any],
) -> None:
    """Write each date dispute right after its old sentence, like a conflict.

    A failed call raises like any other step, so the pipeline keeps the old
    profile and retries the article next run instead of marking it processed
    unchecked. A new fact that quotes the disputed article sentence *and*
    states the differing date is dropped, so the other date isn't written
    twice; a second fact from the same sentence ("...dismissed the charges on
    Tuesday") stays.
    """
    # A row re-cited in step 4 is still checked: its ``text`` is already the
    # re-cited sentence, and edits apply in order, so the dispute goes after it
    disputes, found = find_date_disputes(
        ask, rows, article_text, article_id, skip=set()
    )
    report.update(found)
    for dispute in disputes:
        row = dispute["row"]
        section_edits.setdefault(row["section"], []).append(
            ProfileEdit(op="add", anchor=row["text"], text=dispute["text"])
        )
        claim = ArticleClaim(
            claim=dispute["text"].removeprefix(DISPUTE_PREFIX).split("^[")[0],
            section=row["section"],
            quote=dispute["quote"],
        )
        rows.append(
            _new_row(
                rows,
                row["section"],
                claim,
                article_id,
                text=dispute["text"],
                disputes=row["id"],
            )
        )
        for section in list(new_facts):
            kept = [
                (n, c)
                for n, c in new_facts[section]
                if not (
                    quote_found(c.quote, dispute["quote"])
                    and dates_differ(c.claim, row["claim"])
                )
            ]
            if len(kept) < len(new_facts[section]):
                report["date_dropped_new"] = report.get("date_dropped_new", 0) + (
                    len(new_facts[section]) - len(kept)
                )
            if kept:
                new_facts[section] = kept
            else:
                del new_facts[section]


def _next_id(rows: List[Dict[str, Any]]) -> int:
    ids = [int(r["id"][1:]) for r in rows if str(r.get("id", ""))[1:].isdigit()]
    return max(ids, default=0) + 1


def _new_row(
    rows: List[Dict[str, Any]],
    section: str,
    claim: ArticleClaim,
    article_id: str,
    text: str = "",
    disputes: Optional[str] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": f"f{_next_id(rows)}",
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
