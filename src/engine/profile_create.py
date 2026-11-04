"""Create a first profile with one draft, code checks and at most one repair call.

The older path (``reflect``) drafts a profile, then asks the model up to three
times whether the draft is valid and regenerates it from scratch each time it
says no. That reviewer never sees the article, so it can only judge format,
and on a 60-article local run it rejected 583 of 743 drafts, 918 of 935
rejections citing "structure" (mostly a missing literal "### Background").
Reflect and regenerate took 70% of profile-creation time.

This path (``check``) drafts once, then:

1. Fixes in code what code can fix: a first profile has exactly one source, so
   any citation is rewritten to that article's ID, a paragraph with no
   citation gets one, and a missing second tag becomes the entity type.
2. Calls the model once more only for what code can't fix: no section headings
   (the model adds heading lines; code refuses the reply if any sentence
   changed) or an empty/too-short draft (one fresh draft).

Checking claims against the article is left to the grounding pass
(``src/engine/grounding_pass.py``), which does see the article.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.constants import PROFILE_QC_MIN_TEXT_LENGTH
from src.logging_config import get_logger
from src.utils.step_timing import timed_step

logger = get_logger("profile_create")

CREATE_APPROACHES = ("reflect", "check")
CITATION_RE = re.compile(r"\^\[([^\]]*)\]")
HEADING_RE = re.compile(r"^\s*#{2,4}\s+\S", re.MULTILINE)


def type_tag(entity_type: str) -> str:
    """'person' for people (the merger passes "peopl"), else the singular type."""
    t = entity_type.strip().lower()
    return "person" if t.startswith("peopl") else t.rstrip("s")


HEADINGS_PROMPT = """You add markdown section headings to an entity profile.

Return the profile text with heading lines (starting with "### ") inserted
before groups of paragraphs, for example "### Background" or "### Role".
Change nothing else: every sentence, word and citation marker like ^[id] must
stay exactly as written and in the same order. Reply with the profile text
only, no JSON and no commentary."""


def parse_draft(text: str) -> Optional[Dict[str, Any]]:
    """The draft's JSON as a dict, or None if it isn't a JSON object."""
    try:
        data = json.loads(text) if text and text.strip() else None
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _cite_paragraphs(text: str, article_id: str) -> Tuple[str, int]:
    """Point every citation at ``article_id`` and cite each uncited paragraph.

    A first profile is written from one article, so a citation to any other ID
    is a garbled copy of it, and an uncited paragraph came from it too.
    Headings and blank lines are left alone. Returns (text, changes).
    """
    changes = 0
    marker = f"^[{article_id}]"

    def fix(m: "re.Match[str]") -> str:
        nonlocal changes
        if m.group(1) == article_id:
            return m.group(0)
        changes += 1
        return marker

    text = CITATION_RE.sub(fix, text)
    paragraphs = re.split(r"(\n\s*\n)", text)
    out = []
    for part in paragraphs:
        body = part.strip()
        if body and not body.startswith("#") and marker not in part:
            lines = part.rstrip().split("\n")
            prose = [i for i, ln in enumerate(lines) if not ln.lstrip().startswith("#")]
            if prose and lines[prose[-1]].strip():
                lines[prose[-1]] = lines[prose[-1]].rstrip() + marker
                part = "\n".join(lines) + part[len(part.rstrip()) :]
                changes += 1
        out.append(part)
    return "".join(out), changes


def code_fixes(
    profile: Dict[str, Any], article_id: str, entity_type: str
) -> Tuple[Dict[str, Any], List[str]]:
    """Apply the fixes that need no model; return the profile and what changed."""
    fixed: List[str] = []
    text, n = _cite_paragraphs(str(profile.get("text") or ""), article_id)
    if n:
        fixed.append(f"citations_fixed:{n}")
    profile["text"] = text
    profile["sources"] = [article_id]
    tags = [t for t in profile.get("tags") or [] if isinstance(t, str) and t.strip()]
    tag = type_tag(entity_type)
    if len(tags) < 2 and tag not in tags:
        tags.append(tag)
        fixed.append("type_tag_added")
    profile["tags"] = tags
    conf = profile.get("confidence")
    if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        profile["confidence"] = (
            min(1.0, max(0.0, float(conf))) if isinstance(conf, (int, float)) else 0.5
        )
        fixed.append("confidence_fixed")
    return profile, fixed


def model_problems(profile: Optional[Dict[str, Any]]) -> List[str]:
    """Problems only a model call can fix: no usable text, or no headings."""
    if profile is None:
        return ["unparseable"]
    text = str(profile.get("text") or "")
    if len(text.strip()) < PROFILE_QC_MIN_TEXT_LENGTH:
        return ["too_short"]
    if not HEADING_RE.search(text):
        return ["no_headings"]
    return []


def _words(text: str) -> List[str]:
    """The text's words with heading lines removed, for an unchanged-text check."""
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    return body.split()


def accept_headings(draft_text: str, reply: str) -> Optional[str]:
    """The reply if it only added heading lines to the draft, else None."""
    reply = reply.strip()
    if reply.startswith("```"):
        reply = re.sub(r"^```\w*\n?|\n?```$", "", reply).strip()
    if not HEADING_RE.search(reply) or _words(reply) != _words(draft_text):
        return None
    return reply


def create_with_checks(
    *,
    draft: Callable[[List[Dict[str, Any]]], str],
    write_text: Callable[[List[Dict[str, Any]]], str],
    generation_messages: List[Dict[str, Any]],
    article_id: str,
    entity_type: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Draft once, fix in code, and make at most one repair call.

    ``draft(messages)`` returns the profile as JSON text (the pipeline's
    structured profile call); ``write_text(messages)`` returns a plain-text
    reply. Returns (profile dict or None, history), where history has one entry
    per check in the same shape as the reflection history.
    """
    profile = parse_draft(draft(generation_messages))
    history: List[Dict[str, Any]] = []
    problems = model_problems(profile)
    if problems and problems[0] in ("unparseable", "too_short"):
        retry = generation_messages + [
            {
                "role": "user",
                "content": "The previous attempt returned an empty or very short "
                "profile. Write the full profile from the article now.",
            }
        ]
        with timed_step("regenerate"):
            profile = parse_draft(draft(retry)) or profile
        history.append({"iteration": 1, "check": "code", "problems": problems})
        problems = model_problems(profile)
    if profile is None:
        return None, history
    profile, fixed = code_fixes(profile, article_id, entity_type)
    if problems == ["no_headings"]:
        messages = [
            {"role": "system", "content": HEADINGS_PROMPT},
            {"role": "user", "content": profile["text"]},
        ]
        try:
            with timed_step("repair.headings"):
                reply = write_text(messages)
            new_text = accept_headings(profile["text"], reply)
        except Exception as e:  # a failed repair keeps the draft
            logger.warning(f"Heading repair failed, keeping the draft: {e}")
            new_text = None
        if new_text is not None:
            profile["text"] = new_text
            fixed.append("headings_added")
        else:
            fixed.append("headings_repair_refused")
    remaining = model_problems(profile)
    history.append(
        {
            "iteration": len(history) + 1,
            "check": "code",
            "valid": not remaining,
            "problems": problems,
            "fixed": fixed,
            "issues": remaining,
        }
    )
    return profile, history
