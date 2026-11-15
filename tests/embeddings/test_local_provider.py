"""The local embedding model loads once, however many merge threads ask."""

import sys
import threading
import time
from types import SimpleNamespace

from src.utils.embeddings.base import EmbeddingConfig
from src.utils.embeddings.local import LocalEmbeddingProvider


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
