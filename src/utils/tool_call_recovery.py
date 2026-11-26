"""Reading a list answer from a reply that broke Instructor's one-tool-call rule.

Instructor fails when a provider answers with several tool calls (one per list
item) or with plain text and no tool call. For ``List[Model]`` response types
these helpers read the items straight from the completion instead.
"""

import json
import re
from typing import Any, List, Optional, get_args, get_origin

from pydantic import BaseModel

from src.logging_config import get_logger

logger = get_logger("utils.llm")


def _list_item_model(response_model: Any) -> Optional[type[BaseModel]]:
    """Return the item model for a ``List[Model]`` response type."""
    origin = get_origin(response_model)
    if origin is list:
        args = get_args(response_model)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0]
    return None


def _count_tool_calls(error: Exception) -> str:
    """Tool calls in the completion behind an Instructor error, if known."""
    try:
        message = error.last_completion.choices[0].message  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError):
        return "an unknown number of"
    return str(len(getattr(message, "tool_calls", None) or []))


_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _recover_list_from_text(
    content: Optional[str], item_model: type[BaseModel]
) -> Optional[List[BaseModel]]:
    """Parse a list answer the model wrote as text (often a fenced JSON array).

    Instructor's JSON mode can't read a bare array: it keeps only the text
    from the first ``{`` to the last ``}``, which drops the brackets.
    """
    if not content:
        return None
    try:
        payload = json.loads(_CODE_FENCE.sub("", content.strip()))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and len(payload) == 1:
        payload = next(iter(payload.values()))
    if not isinstance(payload, list):
        return None
    recovered = [item_model.model_validate(item, strict=False) for item in payload]
    logger.warning(
        "Recovered %d items from a text answer without tool calls", len(recovered)
    )
    return recovered


def _recover_multiple_tool_calls(
    *,
    error: Exception,
    response_model: Any,
) -> Optional[Any]:
    """Recover from Instructor's "multiple tool calls" assertion when possible.

    Instructor fails fast when a provider emits more than one tool call. For
    ``List[Model]`` response shapes, each tool call is often one item. We can
    parse those arguments directly from ``last_completion`` and validate them.
    """
    completion = getattr(error, "last_completion", None)
    if completion is None:
        return None

    choices = getattr(completion, "choices", None)
    if not choices:
        return None

    message = getattr(choices[0], "message", None)
    if message is None:
        return None

    tool_calls = getattr(message, "tool_calls", None) or []
    item_model = _list_item_model(response_model)
    if item_model is None:
        return None
    if not tool_calls:
        return _recover_list_from_text(getattr(message, "content", None), item_model)
    if len(tool_calls) == 1:
        return None

    recovered: List[BaseModel] = []
    for call in tool_calls:
        fn = getattr(call, "function", None)
        args_payload = getattr(fn, "arguments", None)
        if args_payload is None:
            return None

        if isinstance(args_payload, str):
            payload = json.loads(args_payload)
        elif isinstance(args_payload, dict):
            payload = args_payload
        else:
            return None

        items = payload.get("items") if isinstance(payload, dict) else None
        if isinstance(items, list) and "items" not in item_model.model_fields:
            # one call per chunk of the list, each a {"items": [...]} object
            recovered.extend(item_model.model_validate(i, strict=False) for i in items)
        else:
            recovered.append(item_model.model_validate(payload, strict=False))

    logger.warning(
        "Recovered %d items from multiple tool calls without retry", len(recovered)
    )
    return recovered
