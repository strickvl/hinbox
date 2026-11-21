"""Small fixes to a written page's order and opening, all in code.

``keep_identity()``: the opening's first sentence says what the entity is
("Karachi is a major city in Pakistan."). On 21 Nov the model check judged
it against the wrong note and dropped it, and the Karachi page then opened
with arrests. When a note says it, word for word or nearly, it stays.

``date_sort()``: sentences the fill stage adds go at the end of their
section, so on 21 Nov Zaki's long section ended in a tail out of date order
(June 2017 after February 2018) in three draws of three. Each filled
sentence is moved before the first sentence of its section that is dated
later than it.
"""

from typing import List, Optional, Sequence, Tuple

from src.engine.written_page import Note, Written


def keep_identity(written: List[Written], notes: Sequence[Note], name: str) -> bool:
    """Restore a dropped first sentence that a note gives with all its
    details and most of its words; True when it did."""
    from src.engine.written_checks import _content, missing_details

    first = next((w for w in written if not w.section and not w.summary), None)
    if first is None or not first.dropped.startswith("check"):
        return False
    words = _content(first.text)
    for n in notes:
        if missing_details(first.text, [n.text], name):
            continue
        if words and len(words & _content(n.text)) >= 0.8 * len(words):
            first.notes, first.dropped, first.check = [n.n], "", "identity"
            return True
    return False


def _when(w: Written, by_n: dict) -> Optional[Tuple[int, int]]:
    """(year, month) the sentence names first, else None."""
    from src.engine.written_checks import _MONTH_N, _WHEN

    m = _WHEN.search(w.text)
    if not m:
        return None
    month = (m.group(1) or "").lower()
    return int(m.group(2)), _MONTH_N.get(month, 0)


def date_sort(written: List[Written], notes: Sequence[Note]) -> int:
    """Move each kept filled sentence that names a date before the first
    kept, dated sentence of its section dated later; the number moved. A
    sentence with only a year sorts before that year's months."""
    by_n = {n.n: n for n in notes}
    moved = 0
    for w in [w for w in written if w.filled and not w.dropped and w.section]:
        when = _when(w, by_n)
        if when is None:
            continue
        mine = [
            o
            for o in written
            if o.section == w.section and not o.dropped and o is not w
        ]
        later = next(
            (o for o in mine if not o.filled and (_when(o, by_n) or (0, 0)) > when),
            None,
        )
        if later is None:
            continue
        written.pop(next(i for i, o in enumerate(written) if o is w))
        written.insert(next(i for i, o in enumerate(written) if o is later), w)
        w.para = later.para
        moved += 1
    return moved
