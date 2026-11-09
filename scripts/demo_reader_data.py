"""Write a small made-up corpus for trying the web reader without a run.

    uv run python -m scripts.demo_reader_data data/demo
    HINBOX_OUTPUT_DIR=data/demo HINBOX_ARTICLES_PATH=data/demo/articles.parquet just frontend

(`just demo-reader` does both.) Every person, article and sentence here is
invented, so the pages and screenshots made from it can be shared and
committed; real run outputs must never be (the repository is public).

The tables mimic what the pipeline saves: a long sectioned profile with
grounding verdicts (some partly supported, one not supported, one not
checked) and a version history, a stub, a link-only news outlet, and a few
events that share articles with the people.
"""

import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.file_ops import write_entities_table
from src.utils.grounding import _extract_cited_claims

OUTLET = "https://example.org/herald/"
# key: (title, date, paragraphs)
ARTICLES: Dict[str, Any] = {
    "a": ("Suspect held after Karachi raid", "2003-04-02", [
        "Pakistani police said Tuesday that they had arrested a Yemeni man, Yusuf Qadir, in a dawn raid on a flat in Karachi and handed him to American agents.",
        "Neighbours said Qadir, who they believed was about 28, had studied engineering at a local college. Relatives in Taiz said he was born there in 1975 and had worked for a shipping company before 2001."]),
    "b": ("Fourteen prisoners moved to Guantánamo", "2006-09-07", [
        "The Pentagon said fourteen men held by the CIA had been flown to Guantánamo Bay, ending years in which their whereabouts were secret.",
        "Among them was Yusuf Qadir, a Yemeni detainee whom officials accuse of helping plan an attack on a ship. Officials described all fourteen as high-value detainees."]),
    "c": ("Arraignment ends in silence", "2008-06-05", [
        "Prosecutors charged Yusuf Qadir with conspiracy and murder in violation of the law of war, and the charges were read at a hearing on Thursday.",
        "Asked whether he understood his rights, Qadir refused to answer the judge. A family statement read by his lawyer, Hassan Darwish, said he was born in 1977, not 1975 as earlier reports had said."]),
    "d": ("Order to close the prison", "2009-01-22", [
        "The new administration ordered the prison closed within a year and halted military commission cases for 120 days.",
        "Lawyers for several detainees, including Hassan Darwish, said they welcomed the pause."]),
    "e": ("Second arraignment runs into the night", "2012-05-05", [
        "Yusuf Qadir and four other men were arraigned again on Saturday in a hearing that ran late into the night.",
        "The judge read the charges aloud as the defendants declined to respond."]),
    "f": ("Hunger strike spreads", "2013-04-14", [
        "More than 100 prisoners have joined a hunger strike that began in February, the military said. Medical staff are force-feeding some of them.",
        "The military did not name the prisoners being fed. Lawyers said the strike started over searches of Korans in Camp 6."]),
    "g": ("Defence asks to see CIA cables", "2016-03-30", [
        "Lawyers for Yusuf Qadir asked the court to order the government to hand over CIA cables describing his interrogation at secret sites.",
        "In a filing, the defence said Qadir was kept awake for days at a time while in CIA custody. The filing did not say how long he was held at the sites."]),
    "h": ("Judge sets trial date", "2019-08-30", [
        "Army Col. Miriam Hale, who took over the case this year, set a trial date of January 2021 in the case against Yusuf Qadir.",
        "Hale said she would not allow further delays over discovery."]),
    "i": ("Trial date slips again", "2021-01-11", [
        "The January trial in the Qadir case will not go ahead, the court said, after the pandemic closed the courtroom at Guantánamo for most of the past year."]),
    "j": ("Plea deal announced, then withdrawn", "2024-07-31", [
        "Prosecutors announced a plea agreement with Yusuf Qadir that would take the death penalty off the table.",
        "Two days later the Secretary of Defense withdrew the agreement and took the case back from the official who had approved it."]),
}  # fmt: skip
ID = {k: f"00000000-demo-4000-8000-{i:012d}" for i, k in enumerate(ARTICLES, 1)}


def c(*keys: str) -> str:
    return "^[" + ", ".join(ID[k] for k in keys) + "]"


QADIR = f"""Yusuf Qadir is a Yemeni detainee held at Guantánamo Bay since 2006 and charged before a military commission.{c("b", "c")}

### Background

Qadir was born in Taiz in about 1975 and studied engineering in Karachi.{c("a")} Sources disagree on his year of birth: one gives 1975 and a later family statement gives 1977.{c("a", "c")} He worked for a shipping firm before 2001.{c("a")}

### Capture and CIA custody

Pakistani police arrested Qadir in a raid in Karachi in April 2003 and handed him to American agents.{c("a")} He was held for three years at secret CIA sites, where he was kept awake for days at a time.{c("g")}

### Detention at Guantánamo

He was one of fourteen prisoners moved to Guantánamo in September 2006.{c("b")} He joined a camp-wide hunger strike in 2013 and was force-fed for several weeks.{c("f")}

### Military commission

#### Charges and arraignment

Prosecutors charged Qadir in 2008 with conspiracy and murder in violation of the law of war.{c("c")} At his first arraignment he refused to answer the judge.{c("c")} He was arraigned a second time in May 2012 in a hearing that ran late into the night.{c("e")}

#### Pretrial litigation

His lawyers asked the court to order the government to hand over CIA cables about his interrogation.{c("g")} Col. Miriam Hale set a trial date of January 2021,{c("h")} which slipped after the pandemic closed the courtroom.{c("i")}

#### Plea negotiations

In July 2024 prosecutors announced a plea agreement that would take the death penalty off the table, but the Secretary of Defense withdrew it two days later.{c("j")}
"""
VERDICTS = {  # words in a claim -> (verdict, the checker's reason)
    "three years": ("partial", "The article says he was kept awake for days at CIA sites, but says it does not know how long he was held there. \"Three years\" is not in this article."),
    "force-fed": ("not_supported", "The article says more than 100 prisoners joined the strike and some were force-fed, but it names no one. It does not say Qadir was force-fed."),
    "late into the night": (None, ""),  # left unchecked: the call for this article failed
}  # fmt: skip


def _date(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp())


def _links(keys: str) -> List[Dict[str, Any]]:
    return [{"article_id": ID[k], "article_title": ARTICLES[k][0], "article_url": OUTLET + k,
             "article_published_date": _date(ARTICLES[k][1])} for k in keys]  # fmt: skip


def _report(text: str) -> Dict[str, Any]:
    out = []
    for cl in _extract_cited_claims(text):
        level, reason = "supported", "The article states this."
        for words, (v, why) in VERDICTS.items():
            if words in cl["claim"]:
                level, reason = v, why
        if level:
            out.append({"article_id": cl["article_id"], "citation": cl["citation"],
                        "claim": cl["claim"], "support_level": level, "reasoning": reason})  # fmt: skip
    return {"verifications": out, "model": "anthropic/claude-haiku-5-5"}


def _profile(text: str, keys: str, stub: bool = False) -> Dict[str, Any]:
    return {"text": text, "sources": [ID[k] for k in keys], "stub": stub or None}


def _versions(steps: List[tuple]) -> Dict[str, Any]:
    return {
        "current_version": len(steps),
        "versions": [
            {"version_number": i, "profile_data": {"text": text},
             "created_at": f"{ARTICLES[k][1]}T12:00:00", "trigger_article_id": ID[k]}
            for i, (k, text) in enumerate(steps, 1)
        ],
    }  # fmt: skip


def _cut(text: str, upto: str) -> str:
    """The profile as it stood once the article ``upto`` was merged in:
    sentences citing later articles dropped, then empty headings."""
    later = {ID[k] for k in ARTICLES if k > upto}
    paras = []
    for para in text.split("\n\n"):
        if para.startswith("#"):
            paras.append(para)
            continue
        pieces = re.findall(r".*?\^\[[^\]]*\][.,]?\s*", para)
        kept = [x for x in pieces if not any(aid in x for aid in later)]
        if kept:
            paras.append("".join(kept).strip())
    out: List[str] = []
    for para in reversed(paras):  # drop a heading with nothing under it
        level = len(para) - len(para.lstrip("#"))
        if level:
            nxt = out[-1] if out else ""
            nxt_level = len(nxt) - len(nxt.lstrip("#"))
            if not nxt or (nxt_level and nxt_level <= level):
                continue
        out.append(para)
    return "\n\n".join(reversed(out)) + "\n"


def build(out: str) -> None:
    rows = [{"id": ID[k], "title": t, "url": OUTLET + k, "published_date": str(_date(d)),
             "content": "\n".join(p)} for k, (t, d, p) in ARTICLES.items()]  # fmt: skip
    pq.write_table(pa.Table.from_pylist(rows), f"{out}/articles.parquet")

    hale = f"Army Col. Miriam Hale, who took over the case this year, set a trial date of January 2021 in the case against Yusuf Qadir.{c('h')} Hale said she would not allow further delays over discovery.{c('h')}"
    darwish = f"Hassan Darwish is a defence lawyer who represents Yusuf Qadir.{c('c')} In 2009 he welcomed the order halting military commission cases.{c('d')} He read a family statement at Qadir's first arraignment saying Qadir was born in 1977.{c('c')}"
    people = [
        {"name": "Yusuf Qadir", "type": "detainee", "articles": _links("abcdefghij"),
         "alternative_names": [{"name": "Yousef Kadir", "type": "detainee"}], "aliases": ["Abu Salim"],
         "profile": {**_profile(QADIR, "abcdefghij"), "reorganised": "done"},
         "profile_grounding": _report(QADIR),
         "profile_versions": _versions([("a", _cut(QADIR, "a")), ("c", _cut(QADIR, "c")), ("g", _cut(QADIR, "g")), ("j", QADIR)])},
        {"name": "Miriam Hale", "type": "military", "articles": _links("hi"),
         "profile": _profile(hale, "h", stub=True), "profile_grounding": _report(hale)},
        {"name": "Hassan Darwish", "type": "legal", "articles": _links("cd"),
         "profile": _profile(darwish, "cd"), "profile_grounding": _report(darwish)},
    ]  # fmt: skip
    omc = f"The Office of Military Commissions runs the war-crimes tribunals at Guantánamo Bay.{c('c')} Its cases were halted for 120 days in 2009.{c('d')} In 2024 it announced, then lost, a plea agreement in the Qadir case.{c('j')}"
    herald = f"The Example Herald reported the arrest of Yusuf Qadir in 2003.{c('a')}"
    orgs = [
        {"name": "Office of Military Commissions", "type": "legal", "articles": _links("cdej"),
         "profile": _profile(omc, "cdj"), "profile_grounding": _report(omc)},
        {"name": "Example Herald", "type": "media", "articles": _links("abcdefghij"),
         "profile": _profile(herald, "a", stub=True), "profile_grounding": _report(herald)},
    ]  # fmt: skip
    karachi = f"Pakistani police said Tuesday that they had arrested a Yemeni man, Yusuf Qadir, in a dawn raid on a flat in Karachi and handed him to American agents.{c('a')}"
    camp6 = (
        f"Lawyers said the strike started over searches of Korans in Camp 6.{c('f')}"
    )
    places = [
        {"name": "Karachi", "type": "city", "articles": _links("a"),
         "profile": _profile(karachi, "a", stub=True), "profile_grounding": _report(karachi)},
        {"name": "Camp 6", "type": "detention_facility", "articles": _links("f"),
         "profile": _profile(camp6, "f", stub=True), "profile_grounding": _report(camp6)},
    ]  # fmt: skip
    events = []
    for key, title, etype, desc in [
        ("c", "First arraignment of Yusuf Qadir", "EventType.LEGAL", "Charges against Qadir were read; he refused to answer the judge."),
        ("f", "2013 Guantánamo hunger strike", "EventType.PROTEST", "More than 100 prisoners joined a hunger strike; some were force-fed."),
        ("j", "Qadir plea agreement withdrawn", "EventType.LEGAL", "A plea agreement was announced and withdrawn two days later."),
    ]:  # fmt: skip
        text = f"{desc}{c(key)}"
        events.append({"title": title, "event_type": etype, "start_date": f"{ARTICLES[key][1]} 00:00:00+00:00",
                       "description": desc, "articles": _links(key), "profile": _profile(text, key, stub=True),
                       "profile_grounding": _report(text)})  # fmt: skip
    for table, ents in [
        ("people", people),
        ("organizations", orgs),
        ("locations", places),
        ("events", events),
    ]:
        write_entities_table(table, ents, out)


if __name__ == "__main__":
    import os

    target = sys.argv[1] if len(sys.argv) > 1 else "data/demo"
    os.makedirs(target, exist_ok=True)
    build(target)
    print(f"Made-up demo tables written to {target}/")
