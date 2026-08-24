#!/usr/bin/env python3
"""Dense retrieval with the paper's Cohere DiskVectorIndex, with batched embeds."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def load_local_env() -> None:
    """Load repository-root .env without overriding exported variables."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


def get_api_key() -> str:
    load_local_env()
    key = os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
    if not key:
        raise RuntimeError(
            "Set COHERE_API_KEY (or CO_API_KEY) in the environment or repository .env"
        )
    os.environ["COHERE_API_KEY"] = key.strip()
    return key.strip()


def read_topics_file(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid topics row {path}:{line_number}")
            rows.append((parts[0], parts[1]))
    return rows


def run_name_for(path: Path) -> str:
    return path.name[7:-4] if path.name.startswith("topics.") else path.stem


def normalize_checkpoint(path: Path, k: int) -> set[str]:
    grouped = {}
    order = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
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
    if path.exists():
        path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    return complete


def embed_batch(index, texts: list[str], retries: int = 5) -> np.ndarray:
    embedding_type = index.config.get("embedding_type", "float")
    for attempt in range(retries):
        try:
            response = index.co.embed(
                texts=texts,
                model=index.config["model"],
                input_type="search_query",
                embedding_types=[embedding_type],
            )
            values = getattr(response.embeddings, embedding_type)
            return np.asarray(values)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise AssertionError("unreachable")


def fetch_document(index, doc_idx: int) -> dict:
    """Use DiskVectorIndex's corpus layout without issuing another API call."""
    from indexed_zstd import IndexedZstdFile

    file_id = doc_idx // index.config["corpus_num_lines"]
    file_id_text = str(file_id).zfill(index.config["corpus_file_len"])
    folder = file_id_text[-index.config["corpus_folder_len"]:]
    corpus_relative = f"corpus/{folder}/{file_id_text}.jsonl.zst"
    offsets_relative = f"corpus/{folder}/{file_id_text}.jsonl.offsets"
    index.download_from_remote(corpus_relative)
    index.download_from_remote(offsets_relative)
    offsets = np.load(Path(index.local_dir) / offsets_relative, mmap_mode="r")
    with IndexedZstdFile(Path(index.local_dir) / corpus_relative) as corpus:
        corpus.seek(offsets[doc_idx % index.config["corpus_num_lines"]])
        return json.loads(corpus.readline())


def search_batch(index, texts: list[str], k: int, fetcher=fetch_document):
    embeddings = embed_batch(index, texts)
    scores, indices = index.index.search(embeddings, k)
    output = []
    for row_scores, row_indices in zip(scores, indices):
        docs = []
        for score, doc_idx in zip(row_scores.tolist(), row_indices.tolist()):
            if doc_idx < 0:
                continue
            docs.append({"doc": fetcher(index, doc_idx), "score": float(score)})
        output.append(docs)
    return output


def docid_of(result: dict) -> str:
    doc = result.get("doc", {})
    docid = doc.get("docid") or doc.get("id")
    if not docid:
        raise ValueError(f"Cohere result has no docid: {result}")
    return str(docid)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-dir", type=Path, default=here / "queries")
    parser.add_argument("--output-dir", type=Path, default=here / "retrieval_cohere")
    parser.add_argument("--index-name", default="Cohere/trec-rag-2024-index")
    parser.add_argument("--cache-dir", type=Path, default=Path("index_cache"))
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Queries per Cohere embed call (maximum supported is 96)")
    parser.add_argument("--test-file", help="Process one topics filename")
    parser.add_argument("--qid", help="Process one qid from each selected file")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 96:
        parser.error("--batch-size must be in [1, 96]")

    get_api_key()
    try:
        from DiskVectorIndex import DiskVectorIndex
    except ImportError as exc:
        parser.error(f"DiskVectorIndex is required: {exc}")

    files = sorted(args.queries_dir.glob("topics.*.txt"))
    if args.test_file:
        files = [args.queries_dir / args.test_file]
    if not files or any(not path.exists() for path in files):
        parser.error("Selected topics file(s) do not exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    handles = {}
    try:
        for topics_file in files:
            run_name = run_name_for(topics_file)
            output_file = args.output_dir / f"run.{run_name}.txt"
            resume = not args.no_resume
            done = normalize_checkpoint(output_file, args.k) if resume else set()
            rows = read_topics_file(topics_file)
            if args.qid:
                rows = [row for row in rows if row[0] == args.qid]
            if args.max_queries is not None:
                rows = rows[:args.max_queries]
            tasks.extend((output_file, run_name, qid, text) for qid, text in rows if qid not in done)
            mode = "a" if resume and output_file.exists() else "w"
            handles[output_file] = output_file.open(mode, encoding="utf-8")

        index = DiskVectorIndex(str(args.index_name), cache_dir=str(args.cache_dir))
        completed = 0
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start:start + args.batch_size]
            results = search_batch(index, [task[3] for task in batch], args.k)
            for (output_file, run_name, qid, _), docs in zip(batch, results):
                if len(docs) != args.k:
                    raise RuntimeError(f"{qid}: expected {args.k} results, got {len(docs)}")
                lines = [
                    f"{qid} Q0 {docid_of(result)} {rank} {result['score']:.6f} {run_name}"
                    for rank, result in enumerate(docs, 1)
                ]
                handles[output_file].write("\n".join(lines) + "\n")
                handles[output_file].flush()
                completed += 1
            print(f"Cohere dense retrieval: {completed}/{len(tasks)} queries")
    finally:
        for handle in handles.values():
            handle.close()
    print(f"Cohere dense retrieval complete: {len(tasks)} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
