"""One SDK client per target for the whole run (src/utils/client_cache.py).

Scenario: the 8 Nov local Qwen run built a new OpenAI client, with its own
open connections, for every model call. After about 7,000 calls every new
connection to the model server failed with "Connection error".
"""

from unittest.mock import patch

from src.utils import llm
from src.utils.provider_routing import resolve_chat_target


def test_every_call_to_one_target_reuses_one_client(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_URL", "http://beast:8080/v1")
    target = resolve_chat_target("ollama/local-qwen36-32k")
    with patch.object(llm, "OpenAI") as fake_openai:
        first = llm._create_raw_client(target)
        assert llm._create_raw_client(target) is first
        json_mode = llm.instructor.Mode.JSON_SCHEMA
        wrapped = llm._create_instructor_client(target, json_mode)
        assert llm._create_instructor_client(target, json_mode) is wrapped
    assert fake_openai.call_count == 1  # the Instructor client wraps the same one


def test_another_target_gets_its_own_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OLLAMA_API_URL", "http://beast:8080/v1")
    with patch.object(llm, "OpenAI") as fake_openai:
        llm._create_raw_client(resolve_chat_target("ollama/local-qwen36-32k"))
        llm._create_raw_client(resolve_chat_target("openai/gpt-6-luna"))
    assert fake_openai.call_count == 2
