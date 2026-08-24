#!/usr/bin/env python3
"""Convert TREC runs to Ragnarok requests with exactly the first five passages."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def setup_java() -> None:
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        return
    for relative in ("lib/server/libjvm.so", "lib/libjvm.so"):
        candidate = Path(java_home) / relative
        if candidate.exists():
            os.environ.setdefault("JVM_PATH", str(candidate))
            break


def read_queries(path: Path) -> dict[str, str]:
    queries = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid topics row {path}:{line_number}")
            queries[parts[0]] = parts[1]
    return queries


def read_run(path: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Invalid TREC row {path}:{line_number}")
            qid, _, docid, rank, score = parts[:5]
            grouped[qid].append({
                "docid": docid,
                "rank": int(rank),
                "score": float(score),
            })
    return dict(grouped)


def run_name(path: Path) -> str:
    name = path.name
    if name.startswith("run.") and name.endswith(".txt"):
        return name[4:-4]
    return path.stem


def resolve_topics(run_file: Path, source: str | None, explicit: Path | None,
                   queries_dir: Path) -> Path:
    if explicit:
        return explicit
    if source == "original":
        return queries_dir / "topics.original.txt"
    if source == "matching":
        return queries_dir / f"topics.{run_name(run_file)}.txt"
    raise ValueError(
        "Choose query provenance explicitly with --query-text-source matching|original "
        "or provide --queries"
    )


def raw_document(searcher, docid: str) -> dict:
    stored = searcher.doc(docid)
    if stored is None or not stored.raw():
        raise KeyError(f"Document not found in index: {docid}")
    data = json.loads(stored.raw())
    if not data.get("segment"):
        raise ValueError(f"Document has no passage segment: {docid}")
    return data


def convert_run(queries: dict[str, str], results: dict[str, list[dict]], searcher,
                k: int = 5, only_qid: str | None = None,
                max_queries: int | None = None, document_cache: dict | None = None):
    if k != 5:
        raise ValueError("This reproduction pipeline requires exactly --k 5")
    qids = list(results)
    if only_qid:
        qids = [qid for qid in qids if qid == only_qid]
        if not qids:
            raise ValueError(f"QID {only_qid!r} is absent from the run")
    if max_queries is not None:
        qids = qids[:max_queries]
    cache = document_cache if document_cache is not None else {}
    requests = []
    for qid in qids:
        if qid not in queries:
            raise KeyError(f"QID {qid} is absent from the selected topics file")
        ranked = sorted(results[qid], key=lambda item: item["rank"])
        if len(ranked) < 5:
            raise ValueError(f"QID {qid} has only {len(ranked)} results")
        top_five = ranked[:5]
        if [item["rank"] for item in top_five] != [1, 2, 3, 4, 5]:
            raise ValueError(f"QID {qid} does not have contiguous ranks 1..5")
        candidates = []
        for item in top_five:
            docid = item["docid"]
            if docid not in cache:
                cache[docid] = raw_document(searcher, docid)
            candidates.append({
                "doc": cache[docid],
                "docid": docid,
                "score": item["score"],
            })
        requests.append({
            "query": {"text": queries[qid], "qid": qid},
            "candidates": candidates,
        })
    return requests


def output_is_complete(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data, list) and len(data) == expected and all(
            len(record.get("candidates", [])) == 5 for record in data
        )
    except (OSError, json.JSONDecodeError):
        return False


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--retrieval-dir", type=Path)
    inputs.add_argument("--run-file", "--single-file", dest="run_file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path,
                        help="Explicit topics file; takes precedence over query-text-source")
    parser.add_argument("--queries-dir", type=Path, default=repo / "querygym" / "queries")
    parser.add_argument("--query-text-source", choices=("matching", "original"),
                        help="Required unless --queries is supplied; records the provenance choice")
    parser.add_argument("--index", default="msmarco-v2.1-doc-segmented")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--qid")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.queries and not args.query_text_source:
        parser.error("Pass --query-text-source matching|original or --queries FILE")
    if args.k != 5:
        parser.error("The paper configuration and this pipeline require --k 5")

    run_files = [args.run_file] if args.run_file else sorted(args.retrieval_dir.glob("run.*.txt"))
    if not run_files or any(not path.exists() for path in run_files):
        parser.error("No selected TREC run files exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_java()
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as exc:
        parser.error(f"Pyserini is required to materialize passage text: {exc}")
    searcher = LuceneSearcher.from_prebuilt_index(args.index)
    document_cache = {}

    for run_file in run_files:
        topics_file = resolve_topics(
            run_file, args.query_text_source, args.queries, args.queries_dir
        )
        if not topics_file.exists():
            raise FileNotFoundError(f"Topics file not found: {topics_file}")
        results = read_run(run_file)
        expected = len(results)
        if args.qid:
            expected = int(args.qid in results)
        if args.max_queries is not None:
            expected = min(expected, args.max_queries)
        destination = args.output_dir / f"ragnarok_format_run.{run_name(run_file)}.json"
        if not args.overwrite and output_is_complete(destination, expected):
            print(f"Skipping complete file: {destination}")
            continue
        converted = convert_run(
            read_queries(topics_file), results, searcher, args.k,
            args.qid, args.max_queries, document_cache,
        )
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(converted, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        print(
            f"Wrote {len(converted)} requests with query text from {topics_file}: "
            f"{destination}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
