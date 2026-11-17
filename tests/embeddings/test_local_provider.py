"""Local embedding model loading."""

import sys
import threading
import time
from types import SimpleNamespace

import pytest

from src.utils.embeddings.base import EmbeddingConfig
from src.utils.embeddings.local import LocalEmbeddingProvider


@pytest.mark.parametrize("cached", [True, False], ids=["cached", "missing"])
def test_loads_from_cache_before_trying_to_download(monkeypatch, cached):
    attempts = []

    class Model:
        device = "cpu"

        def __init__(self, name, *, local_files_only=False, **kwargs):
            attempts.append(local_files_only)
            if local_files_only and not cached:
                raise OSError("Model files are missing from the cache")
            if not local_files_only and cached:
                raise OSError("Network is unavailable")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    provider = LocalEmbeddingProvider(EmbeddingConfig(model_name="all-MiniLM-L6-v2"))

    assert isinstance(provider._get_model(), Model)
    assert attempts == ([True] if cached else [True, False])


def test_four_threads_load_the_model_once(monkeypatch):
    loads = []

    class SlowModel:
        def __init__(self, name, **_):
            loads.append(name)
            time.sleep(0.2)  # a download: long enough for every thread to ask
            self.device = "cpu"

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=SlowModel),
    )
    provider = LocalEmbeddingProvider(EmbeddingConfig(model_name="all-MiniLM-L6-v2"))
    got = []
    threads = [
        threading.Thread(target=lambda: got.append(provider._get_model()))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(loads) == 1
    assert len(got) == 4 and all(m is got[0] for m in got)
