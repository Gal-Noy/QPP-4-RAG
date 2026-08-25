#!/usr/bin/env python3
"""Retrieve the checked-in query variants with the paper's BM25 setup."""

import argparse
import os
import sys
from pathlib import Path

from tqdm.auto import tqdm


def read_topics_file(topics_file: Path):
    queries = []
    with topics_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid topics row {topics_file}:{line_number}")
            queries.append((parts[0], parts[1]))
    return queries


def run_name_for(topics_file: Path) -> str:
    name = topics_file.name
    return name[7:-4] if name.startswith("topics.") and name.endswith(".txt") else topics_file.stem


def normalize_checkpoint(output_file: Path, k: int) -> set[str]:
    """Discard incomplete qid blocks so a resumed run cannot duplicate ranks."""
    grouped: dict[str, list[str]] = {}
    order = []
    if output_file.exists():
        for line in output_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 6:
                if parts[0] not in grouped:
                    order.append(parts[0])
                grouped.setdefault(parts[0], []).append(line)
    complete = {
        qid for qid, lines in grouped.items()
        if len(lines) == k and [int(line.split()[3]) for line in lines] == list(range(1, k + 1))
    }
    kept = [line for qid in order if qid in complete for line in grouped[qid]]
    if output_file.exists():
        output_file.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    return complete


def process_query_file(searcher, topics_file: Path, output_dir: Path, k: int,
                       only_qid: str | None = None, max_queries: int | None = None,
                       resume: bool = True) -> tuple[int, int]:
    queries = read_topics_file(topics_file)
    if only_qid:
        queries = [row for row in queries if row[0] == only_qid]
        if not queries:
            raise ValueError(f"QID {only_qid!r} is not in {topics_file}")
    if max_queries is not None:
        queries = queries[:max_queries]

    run_name = run_name_for(topics_file)
    output_file = output_dir / f"run.{run_name}.txt"
    done = normalize_checkpoint(output_file, k) if resume else set()
    pending = [row for row in queries if row[0] not in done]
    already_complete = len(queries) - len(pending)
    mode = "a" if resume and output_file.exists() else "w"
    succeeded = failed = 0

    with output_file.open(mode, encoding="utf-8") as output:
        progress = tqdm(
            pending,
            total=len(queries),
            initial=already_complete,
            desc=f"BM25 {run_name}",
            unit="query",
            dynamic_ncols=True,
            disable=None,
        )
        for query_id, query_text in progress:
            try:
                hits = searcher.search(query_text, k=k)
                lines = [
                    f"{query_id} Q0 {hit.docid} {rank} {float(hit.score):.6f} {run_name}"
                    for rank, hit in enumerate(hits, 1)
                    if getattr(hit, "docid", None) is not None
                ]
                if len(lines) != k:
                    raise RuntimeError(f"Expected {k} hits, received {len(lines)}")
                output.write("\n".join(lines) + "\n")
                output.flush()
                succeeded += 1
            except Exception as exc:
                print(f"ERROR {topics_file.name} {query_id}: {exc}", file=sys.stderr)
                failed += 1
    return succeeded, failed


def setup_java_environment() -> None:
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        return
    for relative in ("lib/server/libjvm.so", "lib/libjvm.so"):
        candidate = Path(java_home) / relative
        if candidate.exists():
            os.environ.setdefault("JVM_PATH", str(candidate))
            break


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-dir", type=Path, default=here / "queries")
    parser.add_argument("--output-dir", type=Path, default=here / "retrieval")
    parser.add_argument("--index", default="msmarco-v2.1-doc-segmented")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--test-file", help="Process one topics filename")
    parser.add_argument("--qid", help="Process one qid from each selected file")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    setup_java_environment()
    # Pyserini imports its optional OpenAI encoder alongside Lucene and recent
    # OpenAI clients reject an empty key at import time. BM25 never uses that
    # encoder, so provide a temporary placeholder only for the import.
    added_openai_placeholder = not os.environ.get("OPENAI_API_KEY")
    if added_openai_placeholder:
        os.environ["OPENAI_API_KEY"] = "unused-by-bm25"
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as exc:
        parser.error(f"Pyserini is required: {exc}")
    finally:
        if added_openai_placeholder:
            os.environ.pop("OPENAI_API_KEY", None)

    files = sorted(args.queries_dir.glob("topics.*.txt"))
    if args.test_file:
        files = [args.queries_dir / args.test_file]
    missing = [path for path in files if not path.exists()]
    if missing or not files:
        parser.error(f"Topics files not found: {missing or args.queries_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    searcher = LuceneSearcher.from_prebuilt_index(args.index)
    searcher.set_bm25(k1=0.9, b=0.4)

    total_ok = total_failed = 0
    for topics_file in files:
        ok, failed = process_query_file(
            searcher, topics_file, args.output_dir, args.k, args.qid,
            args.max_queries, not args.no_resume,
        )
        total_ok += ok
        total_failed += failed
        print(f"{topics_file.name}: {ok} retrieved, {failed} failed")
    print(f"BM25 complete: {total_ok} queries retrieved, {total_failed} failed")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
