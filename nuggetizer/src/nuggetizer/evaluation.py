"""Reusable, resumable Nuggetizer assignment and scoring helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from .core.metrics import calculate_global_metrics, calculate_nugget_scores
from .core.types import ScoredNugget


def read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    records = json.loads(text) if text.startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Expected records in {path}")
    return records


def _index_unique(records: Iterable[dict], key: str, source: Path) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        if key not in record:
            raise ValueError(f"{source}: record is missing {key!r}")
        value = str(record[key])
        if value in indexed:
            raise ValueError(f"{source}: duplicate {key} {value!r}")
        indexed[value] = record
    return indexed


def _write_jsonl_atomic(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def process_record(
    answer_record: dict,
    nugget_record: dict,
    run_id: str,
    nuggetizer,
    logger: logging.Logger | None = None,
) -> dict:
    answer = answer_record.get("answer")
    if not isinstance(answer, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("text"), str)
        for item in answer
    ):
        raise ValueError(f"Answer {answer_record.get('topic_id')} has invalid answer segments")
    answer_text = " ".join(item["text"] for item in answer)
    nugget_items = nugget_record.get("nuggets")
    if not isinstance(nugget_items, list) or not nugget_items:
        raise ValueError(f"Nugget record {nugget_record.get('qid')} has no nuggets")
    nuggets = [
        ScoredNugget(text=item["text"], importance=item.get("importance", "vital"))
        for item in nugget_items
    ]
    assigned = nuggetizer.assign(nugget_record.get("query", "N/A"), answer_text, nuggets)
    return {
        "query": nugget_record.get("query", "N/A"),
        "qid": str(nugget_record["qid"]),
        "answer_text": answer_text,
        "response_length": answer_record.get("response_length", len(answer_text.split())),
        "run_id": run_id,
        "nuggets": [
            {
                "text": item.text,
                "importance": item.importance,
                "assignment": item.assignment,
            }
            for item in assigned
        ],
    }


def assign_answer_file(
    nugget_file: str | Path,
    answer_file: str | Path,
    output_file: str | Path,
    nuggetizer,
    *,
    selected_qids: set[str] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[int, set[str]]:
    """Append missing assignments and return (new count, expected qids)."""
    nugget_file, answer_file, output_file = map(Path, (nugget_file, answer_file, output_file))
    logger = logger or logging.getLogger(__name__)
    nuggets = _index_unique(read_records(nugget_file), "qid", nugget_file)
    answers = _index_unique(read_records(answer_file), "topic_id", answer_file)
    expected = set(nuggets) & set(answers)
    if selected_qids is not None:
        expected &= {str(qid) for qid in selected_qids}
    if not expected:
        raise ValueError(f"No answer qids in {answer_file} match the requested nuggets")

    completed_records = read_records(output_file) if output_file.exists() else []
    completed = set()
    if output_file.exists():
        completed = set(_index_unique(completed_records, "qid", output_file))
        unexpected = completed - expected
        if selected_qids is None and unexpected:
            raise ValueError(f"{output_file}: assignments contain unexpected qids {sorted(unexpected)}")
    pending = [qid for qid in nuggets if qid in expected - completed]
    run_id = answer_file.stem
    for index, qid in enumerate(pending, 1):
        logger.info("Assigning %s (%d/%d)", qid, index, len(pending))
        completed_records.append(
            process_record(answers[qid], nuggets[qid], run_id, nuggetizer)
        )
        _write_jsonl_atomic(output_file, completed_records)
    return len(pending), expected


def score_assignment_file(input_file: str | Path, output_file: str | Path) -> dict:
    """Calculate per-query and aggregate metrics and atomically replace output."""
    input_file, output_file = Path(input_file), Path(output_file)
    records = read_records(input_file)
    if not records:
        raise ValueError(f"No assignments to score in {input_file}")
    _index_unique(records, "qid", input_file)
    per_query = []
    for record in records:
        metrics = calculate_nugget_scores(str(record["qid"]), record["nuggets"])
        per_query.append({
            "qid": metrics.qid,
            "strict_vital_score": metrics.strict_vital_score,
            "strict_all_score": metrics.strict_all_score,
            "vital_score": metrics.vital_score,
            "all_score": metrics.all_score,
        })
    aggregate = calculate_global_metrics(records)
    _write_jsonl_atomic(output_file, [*per_query, aggregate])
    return aggregate
