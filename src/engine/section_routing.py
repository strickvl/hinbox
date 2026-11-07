"""Tell the edit model which section a new fact belongs in, once a profile has
its starter headings.

Scenario: al Nashiri's profile was reorganised under the detainee starter
headings, but it had no paragraph about detention, so "Detention at
Guantánamo" was left out. Later articles about his prison conditions were
anchored by the edit model on the last sentence it had read, under
"Military commission or legal case", and that sub-section grew to 8,542
characters. The reorganise only runs once, so nothing moved them back.

For a profile marked ``reorganised: done``, the edit call now also gets:

1. ``routing_note()``: the starter headings with what each is for (from
   ``profiles.starter_headings`` in the domain config) and the rule "anchor
   each new fact on the heading, or a sentence, of the section it belongs
   in; a sub-heading when the section has them".
2. ``insert_missing_heading()``: an ``add`` anchored on a starter heading the
   profile doesn't have yet creates that heading, at its place in the list
   (before the next starter heading the profile has), instead of going to
   the end of the profile.

Code still applies every edit word for word (``profile_edits.apply_edits``);
this only changes where an ``add`` lands.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.engine.section_split import _clean_heading, _headings

ROUTING_ENV = "HINBOX_SECTION_ROUTING"  # "off": no note, adds placed as before

ROUTING_NOTE = """

## Where new text goes

This profile is organised under these headings, each for one kind of fact:

{listed}

Put each new fact in the section it belongs in, by what it mainly tells \
about {entity_name}. A court ruling, order, motion or hearing goes under the \
legal heading even when it is about detention, clothing, mail or health. \
Anchor the `add` on a sentence in that section, or on \
the section's heading copied exactly (the text then goes at the end of the \
section's own text, before any of its sub-headings). When a section has sub-headings (one more "#"), anchor on the \
sub-heading or a sentence under it that fits. A heading marked "not in the \
profile yet" can be an anchor too: code adds the heading in its place. Use \
it for a fact that belongs there rather than putting it under another \
heading, even when a sentence on a related point already sits under that \
other heading. Use END only for a fact that fits none of the headings.

This only decides where a new fact goes, not whether to add one: the rules \
above on what to add still hold, so an article with nothing new about \
{entity_name} still gets no edits."""


def routing_sections(
    profile: Dict[str, Any], entity_type: str, kind: Optional[str], domain: str
) -> Optional[Tuple[List[str], Dict[str, str]]]:
    """The starter headings and their notes for a reorganised profile, else None."""
    from src.config_loader import get_domain_config
    from src.engine.section_reorganise import (
        MARK_KEY,
        starter_headings,
        starter_notes,
    )

    if profile.get(MARK_KEY) != "done":
        return None
    if os.getenv(ROUTING_ENV, "on").strip().lower() in ("off", "0", "false"):
        return None
    config = get_domain_config(domain).load_config()
    starter = starter_headings(config, entity_type, kind)
    if not starter:
        return None
    return starter, starter_notes(config, entity_type, kind)


def routing_note(
    starter: List[str], notes: Dict[str, str], entity_name: str, text: str = ""
) -> str:
    """The block added to the edit model's instructions; a heading ``text``
    lacks is marked "not in the profile yet"."""
    present = {_clean_heading(t).lower() for _, _, t in _headings(text.splitlines())}
    listed = "\n".join(
        f"- {h}"
        + (f": {notes[h]}" if notes.get(h) else "")
        + ("" if h.lower() in present else " (not in the profile yet)")
        for h in starter
    )
    return ROUTING_NOTE.format(listed=listed, entity_name=entity_name)


def starter_anchor(anchor: str, starter: List[str]) -> Optional[str]:
    """The starter heading a heading anchor names ("### Health and treatment",
    or one copied with its note), else None."""
    from src.engine.section_reorganise import _starter_for

    return _starter_for(_clean_heading(anchor), starter)


def insert_missing_heading(
    text: str, heading: str, addition: str, starter: List[str]
) -> Optional[str]:
    """``text`` with ``heading`` (a starter heading it lacks) added before the
    next starter heading it has, in list order, holding ``addition``; None
    when ``heading`` isn't a starter heading or the text already has it."""
    if heading not in starter:
        return None
    lines = text.splitlines()
    heads = _headings(lines)
    present = {_clean_heading(t).lower(): (i, lv) for i, lv, t in heads}
    if heading.lower() in present:
        return None
    low = min((lv for _, lv, _ in heads), default=3)
    top = [(i, lv, t) for i, lv, t in heads if lv == low]
    marks = "#" * low
    block = [f"{marks} {heading}", "", addition, ""]
    later = starter[starter.index(heading) + 1 :]
    for name in later:
        hit = present.get(name.lower())
        if hit and hit[1] == len(marks):
            at = hit[0]
            new = lines[:at] + block + lines[at:]
            return re.sub(r"\n{3,}", "\n\n", "\n".join(new)).strip() + "\n"
    # No later starter heading: before the first heading that isn't a starter
    # heading (the reorganise puts its new ones, "Civil lawsuits", last),
    # else at the end
    names = {n.lower() for n in starter}
    extra = [i for i, _, t in top if _clean_heading(t).lower() not in names]
    at = extra[0] if extra else len(lines)
    new = lines[:at] + [""] + block + lines[at:]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(new)).strip() + "\n"
