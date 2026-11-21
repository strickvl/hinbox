"""The written encyclopedia page on an entity page, and its panel.

When a page has a written text (``profile["written_json"]``, made by
``src/engine/written_page.py``), the entity page shows it first and the notes
(the stored profile) one click away (``?view=notes``). Each written sentence
carries one citation; clicking it opens the panel with the notes that
sentence was written from, each with the checker's verdict on it and its
sources, and below them the clicked source with those notes' passages
marked. So a sentence leads to its notes, and a note to its article.
"""

from typing import Any, Dict, List, Optional

from src.engine.written_page import stored_pages
from src.frontend.data_access import Entity, Store
from src.frontend.profile_render import RenderedProfile, render_profile
from src.utils.corrections import sentence_key

_NAMES = (("qwen", "local Qwen"), ("luna", "GPT-6 Luna"), ("haiku", "Claude Haiku"))


def model_label(model: str) -> str:
    """ "GPT-6 Luna", "local Qwen, planned", "by hand" ..."""
    base, _, approach = (model or "").partition("#")
    if base == "by-hand":
        return "by hand"
    low = base.lower()
    label = next((lab for word, lab in _NAMES if word in low), base)
    return f"{label}, planned" if approach == "plan" else label


def pick(ent: Entity, view: Optional[str]) -> Optional[Dict[str, Any]]:
    """The written page to show, or None for the notes. ``view`` is a model
    id, "notes", or empty (the first written page)."""
    pages = stored_pages(ent.profile)
    if not pages or view == "notes":
        return None
    return pages.get(view or "") or next(iter(pages.values()))


def written_context(
    ent: Entity, page: Dict[str, Any], domain_query: str
) -> Dict[str, Any]:
    rp: RenderedProfile = render_profile(
        page["text"],
        None,
        link="/sources/{article}" + domain_query,
        show_verdicts=False,
    )
    report = page.get("report") or {}
    return {
        "rendered": rp,
        "model": page["model"],
        "label": model_label(page["model"]),
        "models": [
            {"id": m, "label": model_label(m)} for m in stored_pages(ent.profile)
        ],
        "notes": report.get("notes", 0),
        "used": report.get("notes_used", 0),
        "dropped": (report.get("dropped_code") or 0)
        + (report.get("dropped_check") or 0),
        "written_at": str(page.get("written_at") or "")[:10],
    }


def _note_claims(ent: Entity) -> Dict[str, Any]:
    rp = render_profile(
        ent.profile.get("text") or "",
        ent.row.get("profile_grounding"),
        show_verdicts=True,
    )
    return {sentence_key(c.text): c for c in rp.claims}


def written_panel(
    store: Store, ent: Entity, model: str, index: Optional[int], article_id: str
) -> Dict[str, Any]:
    """What the panel shows for written sentence ``index`` and one source."""
    from src.frontend.views import _highlight, _mark_passages, plain_claim

    page = stored_pages(ent.profile).get(model) or {}
    kept = [s for s in page.get("sentences") or [] if not s.get("dropped")]
    sent = kept[index] if index is not None and 0 <= index < len(kept) else {}
    claims = _note_claims(ent)
    notes: List[Dict[str, Any]] = []
    passages: List[str] = []
    needle = ""
    for key in sent.get("notes") or []:
        c = claims.get(key)
        if c is None:
            notes.append({"text": key, "verdict": "", "sources": [], "here": False})
            continue
        here = article_id in c.articles
        if here:
            passages += c.passages.get(article_id, [])
            needle = needle or plain_claim(c.text)
        notes.append({
            "text": plain_claim(c.text), "verdict": c.verdict, "here": here,
            "sources": [store.source(a) for a in c.articles],
        })  # fmt: skip
    src = store.source(article_id)
    if passages:
        paras = _mark_passages(src.paragraphs(), passages)
    else:
        paras = [_highlight(p, needle.rstrip(".,;: ")) for p in src.paragraphs()]
    return {
        "source": src,
        "sentence": sent.get("text", ""),
        "check": sent.get("check", ""),
        "problem": sent.get("problem", ""),
        "label": model_label(model),
        "notes": notes,
        "paragraphs": paras,
        "passage_marked": any('<mark class="passage">' in p for p in paras),
        "ent": ent,
    }
