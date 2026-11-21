"""Tag each cited sentence ("note") of a profile: about the entity or not.

Scenario: in the 12 Nov full run, a city's page is mostly other men's
arrests, ages and intelligence assessments, and its "sources disagree"
sentences argue over one detainee's age; a country's page holds a sentence
on another country's peacekeeping. In the 21 Nov hand count, sentences
about something else were the most common complaint. The edit prompt asks
for facts about the entity only, but nothing checks it afterwards.

Here, once per run after grounding, each note not tagged before is put to
the run's profile model, a section's notes at a time (``BATCH`` per call),
with the entity's name and kind. For each it answers:

- ``about``: it tells the reader something about the entity itself;
- ``context``: mainly about something else, but needed to follow a nearby
  sentence about the entity;
- ``other``: about someone or something else.

Tags are stored as a JSON string under ``profile["note_relevance_json"]``
(a dict keyed by sentence text would become one Parquet column per
sentence), by ``sentence_key()`` (the note's words without markers), with
what the model said the note is about, and the model under
``profile["note_relevance_model"]``; a note is
asked about once, and a new model asks again. No text changes. Two things
use the tags:

1. the reader hides ``other`` notes until the reader asks to see them
   (``src/frontend/note_fold.py``);
2. the edit guards drop a new "sources disagree" sentence whose disputed
   sentences are all ``other`` (``off_topic_keys()`` in
   ``src/utils/note_tags.py``, ``guard_edits()``).

Off unless ``HINBOX_NOTE_RELEVANCE=on``, until it is measured on local Qwen
and a cloud model (``just eval note-relevance``).
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.logging_config import log
from src.utils.claim_spans import parse_claims
from src.utils.corrections import sentence_key
from src.utils.note_tags import MODEL_KEY, TAGS_KEY, stored_tags

ENV = "HINBOX_NOTE_RELEVANCE"
BATCH = 10  # notes per call
VERDICTS = ("about", "context", "other")
_HEADING = re.compile(r"^[ \t]*(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)

INSTRUCTIONS = """You sort the sentences of a profile of the {entity_type} \
"{entity_name}"{kind}. The profile was built from news articles, and some \
sentences in it are really about something else.

For each numbered sentence:
1. Copy its first three words.
2. Say in a few words what or whom it is mainly about.
3. Give one verdict:
- about: it tells the reader something about {entity_name} itself: what \
it is, what it did or said, what was done to it, what happened there or to \
it, or its part in the story.
- context: it is mainly about something else, but a reader of this \
profile needs it to follow a nearby sentence about {entity_name}.
- other: it is about someone or something else, and {entity_name} is only \
mentioned in passing, as a setting or a source, or not at all.

Rules:
- Judge by what the sentence is mainly about, not by whether the name \
appears in it.
- For a place: something that happened there is about it (an arrest in the \
city, a protest at the camp). Facts about the people involved that are not \
about the place (their age, their later trial, an assessment of them) are \
other.
- For an organization or office: what it did, decided, said or is \
responsible for is about it. A person's words count only when that person \
speaks for it.
- For a person: what they did, said, decided or went through is about them. \
Facts about the cases, people or events they dealt with are other unless \
they say what this person did.
- For an event: the event itself, who took part, what led to it and what \
followed from it are about it.
- A sentence saying that sources disagree is judged by what the \
disagreement is about.
- When unsure between about and context, answer about. When unsure between \
context and other, answer context."""


def _starts_required(schema: Dict[str, Any]) -> None:
    schema.setdefault("required", []).append("starts_with")


class NoteVerdict(BaseModel):
    model_config = ConfigDict(json_schema_extra=_starts_required)

    sentence: int = Field(description="the sentence number")
    starts_with: str = Field("", description="the sentence's first three words")
    about_what: str = Field(description="what or whom it is mainly about")
    verdict: str = Field(description="about, context or other")


class NoteVerdicts(BaseModel):
    verdicts: List[NoteVerdict]


Ask = Callable[[List[Dict[str, str]]], NoteVerdicts]


def enabled() -> bool:
    return os.getenv(ENV, "off").strip().lower() in ("on", "1", "true")


def notes(text: str) -> List[Dict[str, str]]:
    """The profile's cited claims, in order, cut as grounding cuts them, each
    with its key and the heading it sits under ("" above the first)."""
    claims, _ = parse_claims(text or "")
    heads = [(m.start(), m.group(2).strip()) for m in _HEADING.finditer(text or "")]
    out: List[Dict[str, str]] = []
    for c in claims:
        key = sentence_key(c.text)
        if not key:
            continue
        heading = ""
        for at, title in heads:
            if at <= c.start:
                heading = title
        out.append({"key": key, "text": key, "heading": heading})
    return out


def messages(
    sentences: List[str],
    entity_type: str,
    entity_name: str,
    kind: Optional[str] = None,
    heading: str = "",
) -> List[Dict[str, str]]:
    kind_note = f" (a {kind.replace('_', ' ')})" if kind else ""
    system = INSTRUCTIONS.format(
        entity_type=entity_type.rstrip("s") if entity_type != "people" else "person",
        entity_name=entity_name,
        kind=kind_note,
    )
    numbered = "\n\n".join(f"[{n}] {s}" for n, s in enumerate(sentences, 1))
    where = f'From the section "{heading}".\n\n' if heading else ""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{where}Sentences:\n\n{numbered}"},
    ]


def _norm(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _same_start(copied: str, sentence: str) -> bool:
    words = _norm(copied)[:3]
    return not words or words == _norm(sentence)[: len(words)]


def _verdict(value: str) -> Optional[str]:
    word = (_norm(value) or [""])[0]
    return word if word in VERDICTS else None


def read_reply(reply: NoteVerdicts, sentences: List[str]) -> List[Optional[Dict]]:
    """Each sentence's ``{"verdict", "about"}``, or None: missing, an unknown
    verdict, or copied first words that aren't its own (the numbering
    slipped). A repeated number keeps its first answer."""
    out: List[Optional[Dict]] = [None] * len(sentences)
    seen = set()
    for v in reply.verdicts:
        n = v.sentence
        if not 1 <= n <= len(sentences) or n in seen:
            continue
        seen.add(n)
        verdict = _verdict(v.verdict)
        if verdict and _same_start(v.starts_with, sentences[n - 1]):
            out[n - 1] = {"verdict": verdict, "about": v.about_what.strip()[:120]}
    return out


def make_asker(model: str, local: bool) -> Ask:
    from src.utils.llm import cloud_generation, local_generation
    from src.utils.step_timing import timed_step

    generate = local_generation if local else cloud_generation

    def ask(msgs: List[Dict[str, str]]) -> NoteVerdicts:
        with timed_step("profile.note_relevance"):
            reply = generate(
                messages=msgs, response_model=NoteVerdicts, model=model, max_tokens=4096
            )
        if reply is None:
            raise RuntimeError("note relevance returned nothing")
        return reply

    return ask


def _batches(items: List[Dict[str, str]]) -> Iterable[List[Dict[str, str]]]:
    """Notes in runs of one heading, at most ``BATCH`` each."""
    run: List[Dict[str, str]] = []
    for item in items:
        if run and (len(run) == BATCH or item["heading"] != run[0]["heading"]):
            yield run
            run = []
        run.append(item)
    if run:
        yield run


def tag_notes(
    items: List[Dict[str, str]],
    entity_type: str,
    entity_name: str,
    ask: Ask,
    kind: Optional[str] = None,
) -> Tuple[Dict[str, Dict], int]:
    """Tags for ``items`` by key, and how many calls failed (their notes stay
    untagged, to be asked next run). Once the model server has stopped
    answering (``unreachable()``), the remaining batches fail without a call."""
    from src.utils.server_health import unreachable

    tags: Dict[str, Dict] = {}
    failed = 0
    for run in _batches(items):
        if unreachable():
            failed += 1
            continue
        texts = [i["text"] for i in run]
        try:
            reply = ask(
                messages(texts, entity_type, entity_name, kind, run[0]["heading"])
            )
        except Exception as e:  # a failed call leaves its notes for next run
            log(f"Note relevance failed for {entity_name}: {e}", level="warning")
            failed += 1
            continue
        for item, tag in zip(run, read_reply(reply, texts)):
            if tag:
                tags[item["key"]] = tag
    return tags, failed


def _skip(entity: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    """Hub pages frozen as link-only are never tagged: nothing is added to
    them, and folding their few sentences would leave them empty."""
    return bool(profile.get("linked_only"))


def _name(key: Any, entity: Dict[str, Any]) -> str:
    name = entity.get("name") or entity.get("title")
    return str(name or (key[0] if isinstance(key, tuple) else key))


def plan(
    entity: Dict[str, Any], model: str
) -> Tuple[List[Dict[str, str]], Dict[str, Dict]]:
    """The notes still to ask about, and the stored tags still in the text
    (all dropped when another model tagged them)."""
    profile = entity.get("profile") or {}
    items = notes(str(profile.get("text") or ""))
    stored = stored_tags(profile)
    if profile.get(MODEL_KEY) != model:
        stored = {}
    keys = {i["key"] for i in items}
    kept = {k: v for k, v in stored.items() if k in keys}
    return [i for i in items if i["key"] not in kept], kept


def run_note_relevance(
    entities: Dict[str, Dict[Any, Dict[str, Any]]],
    model: str,
    local: bool,
    workers: int = 1,
    ask: Optional[Ask] = None,
) -> Dict[str, int]:
    """Tag every untagged note of every page (hubs aside), up to ``workers``
    pages at once. Returns counts: pages and notes asked about, calls
    failed, and notes per verdict over every stored tag after the pass."""
    ask = ask or make_asker(model, local)
    jobs = []
    for table, rows in entities.items():
        for key, entity in rows.items():
            profile = entity.get("profile")
            if not isinstance(profile, dict) or _skip(entity, profile):
                continue
            todo, kept = plan(entity, model)
            if todo or len(kept) != len(stored_tags(profile)):
                jobs.append((table, key, entity, todo, kept))

    def work(job):
        table, key, entity, todo, kept = job
        kind = entity.get("type") or entity.get("event_type")
        tags, failed = tag_notes(todo, table, _name(key, entity), ask, kind)
        return job, {**kept, **tags}, failed, len(todo)

    counts = {"pages": 0, "asked": 0, "failed_calls": 0}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for (_, _, entity, _, _), tags, failed, asked in pool.map(work, jobs):
            entity["profile"] = {
                **entity["profile"],
                TAGS_KEY: json.dumps(tags, ensure_ascii=False, sort_keys=True),
                MODEL_KEY: model,
            }
            counts["pages"] += 1
            counts["asked"] += asked
            counts["failed_calls"] += failed
    for v in VERDICTS:
        counts[v] = 0
    for rows in entities.values():
        for entity in rows.values():
            for tag in stored_tags(entity.get("profile") or {}).values():
                v = (tag or {}).get("verdict")
                if v in counts:
                    counts[v] += 1
    return counts


def relevance_pass(
    entities: Dict[str, Dict[Any, Dict[str, Any]]], model_type: str, workers: int = 1
) -> Optional[Dict[str, int]]:
    """The run's pass, after grounding, when ``HINBOX_NOTE_RELEVANCE=on``:
    tags new notes with the run's profile model and logs the counts. Raises
    ``ModelServerUnreachable`` when the server stopped answering, so the run
    saves and exits 1 as it does during grounding."""
    if not enabled():
        return None
    from src.utils.server_health import ModelServerUnreachable, unreachable
    from src.utils.step_timing import timed_step
    from src.utils.task_models import is_local_model, task_model

    model = task_model("profile", model_type)
    log("\nTagging notes as about each page's entity or not...", level="processing")
    with timed_step("note_relevance_pass"):
        counts = run_note_relevance(entities, model, is_local_model(model), workers)
    log(
        f"Note relevance: {counts['asked']} notes asked about on "
        f"{counts['pages']} pages, {counts['failed_calls']} calls failed; "
        f"stored tags: {counts['about']} about, {counts['context']} context, "
        f"{counts['other']} other",
        level="warning" if counts["failed_calls"] else "success",
    )
    why = unreachable()
    if why:
        raise ModelServerUnreachable(f"during note tagging, {why}")
    return counts
