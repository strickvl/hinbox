"""Tests for trace schema and JSONL trace writer."""

import json
import threading
from typing import List

from src.utils.trace_schema import TraceEvent, TraceStage
from src.utils.trace_writer import TraceWriter


def _read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_trace_writer_writes_jsonl_and_autofills_run_id_timestamp(tmp_path):
    writer = TraceWriter(output_dir=str(tmp_path))
    writer.emit(
        TraceEvent(
            stage=TraceStage.RELEVANCE,
            article_id="art-1",
            decision="relevant",
        )
    )
    writer.close()

    rows = _read_jsonl(writer.filepath)
    assert len(rows) == 1
    event = rows[0]
    assert event["run_id"] == writer.run_id
    assert event["stage"] == "relevance"
    assert event["decision"] == "relevant"
    assert "timestamp" in event
    assert writer.event_count == 1


def test_trace_writer_disabled_is_noop(tmp_path):
    writer = TraceWriter(output_dir=str(tmp_path), enabled=False)
    writer.emit(TraceEvent(stage=TraceStage.EXTRACTION, entity_type="people"))
    writer.close()

    assert writer.event_count == 0
    assert not (tmp_path / "traces").exists()


def test_trace_writer_run_id_is_unique_for_back_to_back_writers(tmp_path):
    writer_a = TraceWriter(output_dir=str(tmp_path))
    writer_b = TraceWriter(output_dir=str(tmp_path))

    try:
        assert writer_a.run_id != writer_b.run_id
        assert writer_a.filepath != writer_b.filepath
    finally:
        writer_a.close()
        writer_b.close()


def test_trace_writer_thread_safety_no_line_interleaving(tmp_path):
    writer = TraceWriter(output_dir=str(tmp_path))
    worker_count = 8
    per_worker = 50

    def _emit_many(worker_idx: int) -> None:
        for i in range(per_worker):
            writer.emit(
                TraceEvent(
                    stage=TraceStage.EXTRACTION,
                    article_id=f"art-{worker_idx}",
                    entity_type="people",
                    entity_name=f"Person {worker_idx}-{i}",
                    decision="extracted",
                )
            )

    threads = [
        threading.Thread(target=_emit_many, args=(idx,)) for idx in range(worker_count)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    writer.close()

    rows = _read_jsonl(writer.filepath)
    assert len(rows) == worker_count * per_worker
    assert writer.event_count == worker_count * per_worker
    for row in rows:
        assert row["stage"] == "extraction"
        assert row["decision"] == "extracted"
        assert "run_id" in row
