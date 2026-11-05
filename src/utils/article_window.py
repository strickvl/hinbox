"""Submit work a few items at a time and hand back the results in order.

The pipeline extracts articles in a thread pool and merges them one at a time
on the main thread, in article order. Submitting every article up front lets
extraction run far ahead of merging: with 150 articles and a slow merge, all
150 extractions finish first, their results sit in memory, and on a local
server their calls take the model's few slots while the merge thread waits
behind them. ``ordered_window`` keeps at most ``window`` articles submitted
but not yet merged, and submits the next one each time the merge thread moves
on to an article.
"""

from collections import deque
from itertools import islice
from typing import Callable, Deque, Iterable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def ordered_window(
    items: Iterable[T], submit: Callable[[T], R], window: int
) -> Iterator[R]:
    """Yield ``submit(item)`` for each item, in item order, ``window`` at a time.

    The first ``window`` items are submitted at once. The caller waits on what
    ``submit`` returned (a future, say) and processes it; when it asks for the
    next one, that item counts as done and one more is submitted. So the item
    being processed plus the ones submitted after it never exceed ``window``.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    remaining = iter(items)
    pending: Deque[R] = deque(submit(item) for item in islice(remaining, window))
    while pending:
        yield pending.popleft()
        # The caller has finished with that one: its slot is free
        pending.extend(submit(item) for item in islice(remaining, 1))
