"""Ask a model which heading each of a few profile sentences belongs under.

Scenario: in the 8 Nov local run Qwen filed Vance Spath's 2018 trial
suspension under "Background", and Obama's closure-policy news under "Role"
(whose note says posts and appointments only). Placement happens in three
places today: the reorganise (whole profile, one call), the edit call's
anchor for each new fact, and the split's sub-headings. Each asks the model
to do something else at the same time (keep paragraphs together, write the
edit, invent sub-headings), and a 3-run hand check found Qwen right on only
33 to 35 of 90 sampled sentences either way.

This prompt asks for placement alone: for each numbered sentence the model
first says in a few words what it mainly tells a reader about the entity,
then names one heading from the list. Code reads the heading back with
``section_reorganise._starter_for()``, so a heading copied with its note
still counts. Nothing here moves text yet: ``src/evals/placement.py``
scores it against hand labels first (``just eval placement``).
"""

from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

INSTRUCTIONS = """You file sentences from a profile of the {entity_type} \
"{entity_name}", built from news articles, under the profile's headings.

Headings, each with what belongs under it:

{listed}

For each numbered sentence:
1. Say in a few words what it mainly tells a reader about {entity_name}.
2. Pick the one heading whose description fits that best.

Rules:
- Go by each heading's description, not by which words appear in it.
- What happens in or around a court case (a ruling, order, motion, hearing, \
lawyers' argument, testimony or evidence request) goes under the legal \
heading even when it is about detention, CIA custody, torture or health.
- A general heading such as Background is only for who {entity_name} is and \
their life or career before the events covered, never for news that fits \
nowhere else.
- A heading about posts or roles is only for which post {entity_name} held \
and when, not for what they did in it.
- Answer with the heading name only, spelled exactly as given, without the \
description after the colon."""


class Placement(BaseModel):
    sentence: int = Field(description="the sentence number")
    about: str = Field(description="what it mainly tells about the entity")
    heading: str = Field(description="one heading name from the list")


class Placements(BaseModel):
    placements: List[Placement]


AskPlacements = Callable[[List[Dict[str, str]]], Placements]


def messages(
    sentences: List[str],
    starter: List[str],
    notes: Dict[str, str],
    entity_type: str,
    entity_name: str,
) -> List[Dict[str, str]]:
    listed = "\n".join(
        f"- {h}" + (f": {notes[h]}" if notes.get(h) else "") for h in starter
    )
    system = INSTRUCTIONS.format(
        entity_type=entity_type, entity_name=entity_name, listed=listed
    )
    numbered = "\n\n".join(f"[{n}] {s}" for n, s in enumerate(sentences, 1))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Sentences:\n\n{numbered}"},
    ]


def read_reply(
    reply: Placements, count: int, starter: List[str]
) -> List[Optional[str]]:
    """The starter heading each sentence got (None: missing, or a heading not
    in the list); a repeated number keeps its first answer."""
    from src.engine.section_reorganise import _starter_for
    from src.engine.section_split import _clean_heading

    out: List[Optional[str]] = [None] * count
    seen = set()
    for p in reply.placements:
        if 1 <= p.sentence <= count and p.sentence not in seen:
            seen.add(p.sentence)
            out[p.sentence - 1] = _starter_for(_clean_heading(p.heading), starter)
    return out


def make_asker(model: str, local: bool, step: str = "profile.section_place"):
    from src.utils.llm import cloud_generation, local_generation
    from src.utils.step_timing import timed_step

    generate = local_generation if local else cloud_generation

    def ask(msgs: List[Dict[str, str]]) -> Placements:
        with timed_step(step):
            reply = generate(
                messages=msgs, response_model=Placements, model=model, max_tokens=4096
            )
        if reply is None:
            raise RuntimeError("section placement returned nothing")
        return reply

    return ask
