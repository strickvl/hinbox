"""Update a profile by applying a short list of edits instead of rewriting it.

The pipeline's ``update_profile`` asks the model for the whole new profile.
On long profiles that rewrite quietly drops facts: in the replay, single
updates cut a 31k-character profile to 6k. Here the model only says what to
add or change, and code applies that to the old text, so a sentence the model
never mentions can't disappear.

Two kinds of edit (see ``profile_edit.md``):

- ``add``: new text after an anchor (an existing sentence, a ``## heading``,
  or ``END``).
- ``replace``: a new version of one existing sentence. Any citation the old
  sentence had and the new one lacks is put back, so a replace can't drop a
  source.

There is no delete. An edit whose anchor can't be found is recorded; an
unplaced ``add`` goes to the end, an unplaced ``replace`` is skipped.
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from src.constants import CLOUD_MODEL, OLLAMA_MODEL, PROFILE_MAX_TOKENS
from src.utils.llm import cloud_generation, local_generation

END = "END"
_CITATION = re.compile(r"\^\[([^\]]+)\]")


class ProfileEdit(BaseModel):
    op: Literal["add", "replace"]
    anchor: str = Field(
        description="An existing sentence copied exactly, a '## heading', or END"
    )
    text: str = Field(description="The new sentence(s), each ending in a citation")


class Disagreement(BaseModel):
    profile_sentence: str = Field(description="The profile sentence, copied exactly")
    article_says: str = Field(description="What the article says instead")


class ProfileEditList(BaseModel):
    # Filled in before the edits, so the model checks for conflicts first
    # (on the dev set it ignored conflicting years when only asked for edits).
    # Asked too broadly, it called different events or extra detail a conflict
    # and wrote "the sources disagree" into the profile, so the prompt narrows it
    disagreements: List[Disagreement] = Field(
        default_factory=list,
        description="Claims where the article gives a different value for the "
        "very same thing as the profile, so both can't be true. Usually empty",
    )
    new_facts: List[str] = Field(
        default_factory=list,
        description="Facts about the entity in the article that the profile "
        "doesn't have yet, in a few words each",
    )
    edits: List[ProfileEdit] = Field(default_factory=list)

    @field_validator("disagreements", "new_facts", "edits", mode="before")
    @classmethod
    def _parse_json_string(cls, value: Any) -> Any:
        # Bolt sometimes sends a list as a JSON string ('[{"op": "add", ...}]'),
        # which failed validation on every retry
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value


def _ids(text: str) -> List[str]:
    return [i.strip() for g in _CITATION.findall(text) for i in g.split(",")]


def _squash(text: str) -> str:
    return " ".join(text.split())


def _find(text: str, anchor: str) -> Optional[Tuple[int, int]]:
    """Locate ``anchor`` in ``text``: exactly, else ignoring whitespace."""
    start = text.find(anchor)
    if start >= 0:
        return start, start + len(anchor)
    # Models often re-flow spaces or newlines inside a copied sentence
    words = [re.escape(w) for w in anchor.split()]
    if not words:
        return None
    match = re.search(r"\s+".join(words), text)
    return (match.start(), match.end()) if match else None


def _section_end(text: str, heading: str) -> Optional[int]:
    """Index just before the heading after ``heading``, or the text's end."""
    match = re.search(rf"^{re.escape(heading.strip())}\s*$", text, re.MULTILINE)
    if not match:
        return None
    following = re.search(r"^#{1,6} ", text[match.end() :], re.MULTILINE)
    return match.end() + following.start() if following else len(text)


def apply_edits(text: str, edits: List[ProfileEdit]) -> Tuple[str, Dict[str, Any]]:
    """Apply ``edits`` to ``text`` in order. Returns (new_text, report)."""
    report: Dict[str, Any] = {"applied": 0, "appended": 0, "skipped": []}
    # Several adds after one sentence go in list order: each one goes after
    # the text the previous one inserted there, not straight after the anchor
    last_added: Dict[str, str] = {}
    for edit in edits:
        addition = edit.text.strip()
        if not addition:
            report["skipped"].append(
                {"op": edit.op, "anchor": edit.anchor, "why": "empty"}
            )
            continue
        anchor = edit.anchor.strip()
        if edit.op == "replace":
            span = _find(text, anchor) if anchor and anchor != END else None
            if span is None:
                report["skipped"].append(
                    {"op": "replace", "anchor": anchor, "why": "not found"}
                )
                continue
            old = text[span[0] : span[1]]
            missing = [i for i in _ids(old) if i not in _ids(addition)]
            if missing:
                addition += f"^[{', '.join(missing)}]"
            text = text[: span[0]] + addition + text[span[1] :]
            report["applied"] += 1
            continue

        if anchor.startswith("#"):
            at = _section_end(text, anchor)
            if at is not None:
                head, tail = text[:at].rstrip("\n"), text[at:].lstrip("\n")
                text = f"{head}\n{addition}\n" + (f"\n{tail}" if tail else "")
                report["applied"] += 1
                continue
        elif anchor and anchor != END:
            span = _find(text, last_added.get(anchor, anchor))
            if span is not None:
                # Same paragraph, right after the anchor sentence
                text = f"{text[: span[1]]} {addition}{text[span[1] :]}"
                last_added[anchor] = addition
                report["applied"] += 1
                continue
        # END, or an anchor that couldn't be placed: append rather than lose it
        text = f"{text.rstrip()}\n\n{addition}\n"
        report["applied" if anchor == END else "appended"] += 1
    return text, report


def _messages(
    entity_type: str,
    entity_name: str,
    profile_text: str,
    article_text: str,
    article_id: str,
    domain: str,
) -> List[Dict[str, str]]:
    from src.config_loader import DomainConfig

    template = DomainConfig(domain).load_profile_prompt("edit")
    instructions = template.format(
        entity_type=entity_type, entity_name=entity_name, article_id=article_id
    )
    user = (
        f"Existing profile:\n----------------\n{profile_text}\n\n"
        f"New article (ID: {article_id}):\n----------------\n{article_text}"
    )
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user},
    ]


def edit_profile(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict[str, Any],
    article_text: str,
    article_id: str,
    model: Optional[str] = None,
    domain: str = "guantanamo",
    local: Optional[bool] = None,
) -> Dict[str, Any]:
    """Ask the model for edits and apply them. Raises if the call fails.

    ``local`` defaults to whether ``model`` starts with ``ollama/``.
    """
    local = (model or "").startswith("ollama/") if local is None else local
    generate = local_generation if local else cloud_generation
    response = generate(
        messages=_messages(
            entity_type,
            entity_name,
            existing_profile.get("text", ""),
            article_text,
            article_id,
            domain,
        ),
        response_model=ProfileEditList,
        model=model or (OLLAMA_MODEL if local else CLOUD_MODEL),
        max_tokens=PROFILE_MAX_TOKENS,
    )
    if response is None:
        raise RuntimeError("edit call returned nothing")
    text, report = apply_edits(existing_profile.get("text", ""), response.edits)
    sources = list(existing_profile.get("sources") or [])
    return {
        **existing_profile,
        "text": text,
        "sources": sources + ([article_id] if article_id not in sources else []),
        "edits": [e.model_dump() for e in response.edits],
        "disagreements": [d.model_dump() for d in response.disagreements],
        "edit_report": report,
    }
