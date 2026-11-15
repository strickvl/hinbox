"""A reply cut off at the token limit is asked again once, not twice."""

from typing import List
from unittest.mock import patch

import instructor
import pytest
from instructor.core.exceptions import InstructorRetryException
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from src.utils.llm import local_generation


class Place(BaseModel):
    name: str


def _ask(finish_reason: str, content: str) -> int:
    """Calls the model server got for one local request that always fails."""
    raw = OpenAI(api_key="test", base_url="http://127.0.0.1:1/v1")
    calls = []

    def create(**_):
        calls.append(1)
        return ChatCompletion.model_validate(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            }
        )

    raw.chat.completions.create = create
    with (
        patch.dict("os.environ", {"HINBOX_OLLAMA_NUM_CTX": "32768"}),
        patch(
            "src.utils.llm._create_instructor_client",
            lambda target, mode: instructor.from_openai(raw, mode=mode),
        ),
        pytest.raises(InstructorRetryException),
    ):
        local_generation(
            [{"role": "user", "content": "List the places."}],
            List[Place],
            model="ollama/test",
        )
    return len(calls)


def test_cut_off_reply_is_asked_twice_in_all():
    looping = '{"items": [' + '{"name": "U.S."}, ' * 50
    assert _ask("length", looping) == 2


def test_unreadable_reply_still_gets_three_attempts():
    assert _ask("stop", '{"items": [{"nombre": "Cuba"}]}') == 3
