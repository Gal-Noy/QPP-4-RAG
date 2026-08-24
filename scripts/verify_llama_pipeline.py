#!/usr/bin/env python3
"""Verify the complete 56 x 31 BM25/Cohere -> Llama -> Nuggetizer artifacts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


METHODS = ("genqr", "genqr_ensemble", "mugi", "qa_expand", "query2doc", "query2e")
VARIANTS = ("original",) + tuple(
    f"{method}_trial{trial}" for method in METHODS for trial in range(1, 6)
)
RETRIEVERS = ("retrieval", "retrieval_cohere")
SCORE_FIELDS = ("all_score", "strict_vital_score")


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def topics(path: Path) -> dict[str, str]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, text = line.split("\t", 1)
            output[qid] = text
    return output


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_queries(queries_dir: Path):
    expected = {f"topics.{variant}.txt" for variant in VARIANTS}
    actual = {path.name for path in queries_dir.glob("topics.*.txt")}
    require(actual == expected, f"Query file set mismatch: missing={expected-actual}, extra={actual-expected}")
    loaded = {variant: topics(queries_dir / f"topics.{variant}.txt") for variant in VARIANTS}
    original_qids = set(loaded["original"])
    require(len(original_qids) == 56, f"Expected 56 original qids, got {len(original_qids)}")
    for variant, rows in loaded.items():
        require(len(rows) == 56, f"{variant}: expected 56 unique queries, got {len(rows)}")
        require(set(rows) == original_qids, f"{variant}: qids differ from original")
    print("queries: 31 unchanged input files x 56 qids")
    return loaded


def read_trec(path: Path):
    grouped = defaultdict(list)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        require(len(parts) >= 6, f"{path}:{line_number}: invalid TREC row")
        grouped[parts[0]].append((int(parts[3]), parts[2]))
    return grouped


def verify_retrieval(root: Path, query_data):
    runs = {}
    qids = set(query_data["original"])
    for retriever in RETRIEVERS:
        directory = root / retriever
        expected = {f"run.{variant}.txt" for variant in VARIANTS}
        actual = {path.name for path in directory.glob("run.*.txt")}
        require(actual == expected, f"{retriever}: run file set mismatch")
        runs[retriever] = {}
        for variant in VARIANTS:
            path = directory / f"run.{variant}.txt"
            grouped = read_trec(path)
            require(set(grouped) == qids, f"{path}: expected all 56 qids")
            for qid, rows in grouped.items():
                require(len(rows) == 100, f"{path} {qid}: expected 100 hits")
                require([rank for rank, _ in rows] == list(range(1, 101)),
                        f"{path} {qid}: ranks are not 1..100 in order")
            runs[retriever][variant] = grouped
        print(f"{retriever}: 31 runs x 56 qids x 100 hits")
    return runs


def verify_prepared(root: Path, query_data, runs):
    prepared = {}
    for retriever in RETRIEVERS:
        directory = root / retriever
        prepared[retriever] = {}
        for variant in VARIANTS:
            path = directory / f"ragnarok_format_run.{variant}.json"
            require(path.exists(), f"Missing {path}")
            records = load_records(path)
            require(len(records) == 56, f"{path}: expected 56 requests")
            by_qid = {record["query"]["qid"]: record for record in records}
            require(len(by_qid) == 56, f"{path}: duplicate/missing qids")
            provenance = set()
            for qid, record in by_qid.items():
                candidates = record.get("candidates", [])
                require(len(candidates) == 5, f"{path} {qid}: expected exactly 5 candidates")
                expected_ids = [docid for _, docid in runs[retriever][variant][qid][:5]]
                actual_ids = [candidate.get("docid") for candidate in candidates]
                require(actual_ids == expected_ids, f"{path} {qid}: Top-5 order/docids changed")
                text = record["query"]["text"]
                if text == query_data[variant][qid]:
                    provenance.add("matching")
                elif text == query_data["original"][qid]:
                    provenance.add("original")
                else:
                    provenance.add("other")
            require("other" not in provenance, f"{path}: query.text is neither matching nor original")
            prepared[retriever][variant] = by_qid
            print(f"{path.name}: query.text provenance={','.join(sorted(provenance))}")
        print(f"{retriever} prepared: 31 x 56 requests with exact Top-5")
    return prepared


def verify_results(root: Path, prepared, tag: str):
    results = {}
    for retriever in RETRIEVERS:
        results[retriever] = {}
        for variant in VARIANTS:
            path = root / retriever / f"rag_results_run.{variant}_{tag}.json"
            require(path.exists(), f"Missing {path}")
            records = load_records(path)
            require(len(records) == 56, f"{path}: expected 56 answers")
            by_qid = {str(record["topic_id"]): record for record in records}
            require(len(by_qid) == 56, f"{path}: duplicate/missing qids")
            for qid, request in prepared[retriever][variant].items():
                answer = by_qid[qid]
                require(answer.get("topic") == request["query"]["text"],
                        f"{path} {qid}: prepared query.text was changed")
                expected_refs = [candidate["docid"] for candidate in request["candidates"]]
                require(answer.get("references") == expected_refs,
                        f"{path} {qid}: references/order changed")
                require(isinstance(answer.get("answer"), list), f"{path} {qid}: invalid answer")
            results[retriever][variant] = by_qid
        print(f"{retriever} generated: 31 x 56 Llama answers")
    return results


def verify_scores(root: Path, tag: str):
    for retriever in RETRIEVERS:
        for variant in VARIANTS:
            path = root / retriever / "scores" / f"rag_results_run.{variant}_{tag}_scores.jsonl"
            require(path.exists(), f"Missing {path}")
            records = load_records(path)
            per_query = [record for record in records if str(record.get("qid")) != "all"]
            require(len(per_query) == 56, f"{path}: expected 56 per-query scores")
            require(len({str(record["qid"]) for record in per_query}) == 56,
                    f"{path}: duplicate score qids")
            for record in per_query:
                for field in SCORE_FIELDS:
                    require(field in record and isinstance(record[field], (int, float)),
                            f"{path} {record.get('qid')}: missing {field}")
        print(f"{retriever} scores: 31 x 56 all_score + strict_vital_score")


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--querygym", type=Path, default=repo / "querygym")
    parser.add_argument(
        "--stage", choices=("queries", "retrieval", "prepared", "generated", "scores", "all"),
        default="all",
    )
    parser.add_argument("--generator-tag", default="llama_3_2_3b_instruct_top5")
    args = parser.parse_args()

    query_data = verify_queries(args.querygym / "queries")
    if args.stage == "queries":
        return 0
    runs = verify_retrieval(args.querygym, query_data)
    if args.stage == "retrieval":
        return 0
    prepared = verify_prepared(args.querygym / "rag_prepared", query_data, runs)
    if args.stage == "prepared":
        return 0
    verify_results(args.querygym / "rag_results", prepared, args.generator_tag)
    if args.stage == "generated":
        return 0
    verify_scores(args.querygym / "rag_nuggetized_eval", args.generator_tag)
    print("FULL VERIFIED: 2 retrievers x 31 variants x 56 topics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
