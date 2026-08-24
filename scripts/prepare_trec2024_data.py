#!/usr/bin/env python3
"""Validate official TREC 2024 inputs and derive Nuggetizer nugget records."""

import argparse
import json
from pathlib import Path


def read_topic_qids(path: Path) -> list[str]:
    qids = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2 or not parts[0]:
            raise ValueError(f"Invalid topic row {path}:{line_number}")
        qids.append(parts[0])
    if len(qids) != len(set(qids)):
        raise ValueError(f"Duplicate qids in {path}")
    return qids


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Expected four qrels columns at {path}:{line_number}")
        qid, iteration, docid, relevance_text = parts
        if iteration != "0":
            raise ValueError(f"Unexpected qrels iteration at {path}:{line_number}")
        try:
            relevance = int(relevance_text)
        except ValueError as exc:
            raise ValueError(f"Invalid relevance at {path}:{line_number}") from exc
        if docid in qrels.setdefault(qid, {}):
            raise ValueError(f"Duplicate qid/docid at {path}:{line_number}")
        qrels[qid][docid] = relevance
    if not qrels:
        raise ValueError(f"No qrels in {path}")
    return qrels


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(record)
    if not records:
        raise ValueError(f"No JSONL records in {path}")
    return records


def nugget_signature(record: dict) -> tuple[tuple[str, str], ...]:
    if not isinstance(record.get("query"), str) or not record["query"]:
        raise ValueError(f"QID {record.get('qid')}: missing query")
    if not isinstance(record.get("nuggets"), list) or not record["nuggets"]:
        raise ValueError(f"QID {record.get('qid')}: missing nuggets")
    signature = []
    for nugget in record["nuggets"]:
        text = nugget.get("text") if isinstance(nugget, dict) else None
        importance = nugget.get("importance") if isinstance(nugget, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"QID {record.get('qid')}: invalid nugget text")
        if importance not in {"vital", "okay"}:
            raise ValueError(f"QID {record.get('qid')}: invalid importance {importance!r}")
        signature.append((text, importance))
    if len(signature) != len(set(signature)):
        raise ValueError(f"QID {record.get('qid')}: duplicate nuggets")
    return tuple(signature)


def convert_nugget_assignments(records: list[dict], topic_qids: list[str]) -> list[dict]:
    canonical: dict[str, dict] = {}
    for record in records:
        if "qid" not in record:
            raise ValueError("Nugget assignment record has no qid")
        qid = str(record["qid"])
        signature = nugget_signature(record)
        candidate = {
            "qid": qid,
            "query": record["query"],
            "nuggets": [
                {"text": text, "importance": importance}
                for text, importance in signature
            ],
        }
        if qid in canonical and canonical[qid] != candidate:
            raise ValueError(f"QID {qid}: inconsistent query or nugget set across runs")
        canonical[qid] = candidate

    expected = set(topic_qids)
    actual = set(canonical)
    if actual != expected:
        raise ValueError(
            f"Nugget qids do not match topics: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    return [canonical[qid] for qid in topic_qids]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topics", type=Path,
        default=repo / "querygym" / "queries" / "topics.original.txt",
    )
    parser.add_argument(
        "--qrels", type=Path, default=repo / "data" / "2024-retrieval-qrels.txt"
    )
    parser.add_argument(
        "--assignments", type=Path,
        default=repo / "data" / "nugget_assignment.20241218.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=repo / "data" / "hr_scored_nist_nuggets_20241218_rag24.test_qrels_nist.jsonl",
    )
    args = parser.parse_args()

    topic_qids = read_topic_qids(args.topics)
    if len(topic_qids) != 56:
        raise ValueError(f"Expected 56 experiment topics, found {len(topic_qids)}")
    qrels = read_qrels(args.qrels)
    missing_qrels = set(topic_qids) - set(qrels)
    if missing_qrels:
        raise ValueError(f"Official qrels miss experiment qids: {sorted(missing_qrels)}")
    derived = convert_nugget_assignments(read_jsonl(args.assignments), topic_qids)
    write_jsonl(args.output, derived)
    print(
        f"Validated 56/56 experiment qids in qrels ({len(qrels)} official qids); "
        f"wrote {len(derived)} Nuggetizer records to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
