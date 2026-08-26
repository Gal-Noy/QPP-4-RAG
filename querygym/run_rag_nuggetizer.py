#!/usr/bin/env python3
"""Assign fixed NIST nuggets with one reusable local Qwen3 judge."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from querygym.local_hf_chat import LocalChatModel
except ModuleNotFoundError:  # Direct execution from querygym/.
    from local_hf_chat import LocalChatModel


QWEN_JUDGE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def selected_directories(base: Path, output: Path) -> list[tuple[Path, Path]]:
    children = [name for name in ("retrieval", "retrieval_cohere") if (base / name).is_dir()]
    return [(base / name, output / name) for name in children] if children else [(base, output)]


def answer_files(directory: Path, max_files: int | None) -> list[Path]:
    files = sorted(set(directory.glob("*.json")) | set(directory.glob("*.jsonl")))
    return files[:max_files] if max_files is not None else files


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_manifest(output_dir: Path, manifest: dict) -> None:
    path = output_dir / "evaluation_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                f"Judge provenance mismatch in {path}; choose a fresh --output-dir "
                "instead of mixing evaluator configurations"
            )
        return
    existing_files = [candidate for candidate in output_dir.rglob("*") if candidate.is_file()]
    if existing_files:
        raise ValueError(
            f"Evaluator artifacts already exist in {output_dir} without a provenance "
            "manifest; choose a fresh --output-dir instead of mixing judge outputs"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parser_for(repo: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-results-dir", type=Path, default=repo / "querygym" / "rag_results")
    parser.add_argument("--nugget-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=repo / "querygym" / "rag_nuggetized_eval")
    parser.add_argument(
        "--model",
        default=QWEN_JUDGE_MODEL,
        help=f"Local evaluator model (experiment default: {QWEN_JUDGE_MODEL})",
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--qid", action="append", help="Only evaluate this qid; repeatable")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--context-size", type=int, default=16384)
    parser.add_argument(
        "--judge-batch-size", type=int, default=1,
        help="Local constrained-classification batch size",
    )
    parser.add_argument("--log-level", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    load_dotenv(repo / ".env", override=False)
    parser = parser_for(repo)
    args = parser.parse_args()
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive")
    if args.context_size < 1 or args.judge_batch_size < 1:
        parser.error("context size and judge batch size must be positive")
    for required in (args.rag_results_dir, args.nugget_file):
        if not required.exists():
            parser.error(f"Required path does not exist: {required}")

    sys.path.insert(0, str(repo / "nuggetizer" / "src"))
    from nuggetizer.evaluation import assign_answer_file, read_records, score_assignment_file
    from nuggetizer.models.constrained_nuggetizer import ConstrainedNuggetizer

    nugget_qids = {str(record["qid"]) for record in read_records(args.nugget_file)}
    selected_qids = set(args.qid) if args.qid else None
    if selected_qids and selected_qids - nugget_qids:
        parser.error(f"QID(s) absent from nugget file: {sorted(selected_qids - nugget_qids)}")
    work = [
        (answer_file, method_output)
        for answers_dir, method_output in selected_directories(args.rag_results_dir, args.output_dir)
        for answer_file in answer_files(answers_dir, args.max_files)
    ]
    if not work:
        parser.error("No answer JSON/JSONL files found")
    matched = 0
    for answer_file, _ in work:
        answer_qids = {str(record["topic_id"]) for record in read_records(answer_file)}
        matched += len(answer_qids & nugget_qids & (selected_qids or nugget_qids))
    if not matched:
        parser.error("No generated-answer qids match the requested nuggets")
    if args.validate_only:
        print(f"Validated {len(work)} answer files and {matched} answer/nugget pairs")
        return 0
    if args.model != QWEN_JUDGE_MODEL:
        print(
            f"WARNING: {args.model} is a code-path smoke model; evaluator runs "
            f"must use {QWEN_JUDGE_MODEL}",
            file=sys.stderr,
        )

    manifest = {
        "schema_version": 1,
        "judge_backend": "local-hf",
        "judge_model": args.model,
        "fixed_nugget_sha256": file_sha256(args.nugget_file),
        "assignment_mode": "support_grade_3",
        "local_generation": {
            "assignment_strategy": "constrained_next_token_abc_v1",
            "assignment_unit": "per_nugget",
            "context_size": args.context_size,
            "dtype": args.dtype,
            "load_in_4bit": args.load_in_4bit,
            "judge_batch_size": args.judge_batch_size,
        },
    }
    ensure_manifest(args.output_dir, manifest)

    logger = logging.getLogger(__name__)
    level = logging.DEBUG if args.log_level >= 2 else logging.INFO if args.log_level else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    runtime = LocalChatModel(
        args.model, args.device, args.dtype, args.local_files_only, args.load_in_4bit,
        context_size=args.context_size,
    )
    local_nuggetizer = ConstrainedNuggetizer(
        runtime,
        batch_size=args.judge_batch_size,
    )

    succeeded = failed = 0
    for number, (answer_file, method_output) in enumerate(work, 1):
        assignment_file = method_output / "assignments" / f"{answer_file.stem}_assignments.jsonl"
        score_file = method_output / "scores" / f"{answer_file.stem}_scores.jsonl"
        try:
            _added, expected = assign_answer_file(
                args.nugget_file, answer_file, assignment_file, local_nuggetizer,
                selected_qids=selected_qids, logger=logger,
            )
            assigned = {str(record["qid"]) for record in read_records(assignment_file)}
            if expected - assigned:
                raise RuntimeError(f"Missing assigned qids: {sorted(expected - assigned)}")
            score_assignment_file(assignment_file, score_file)
            print(f"[{number}/{len(work)}] evaluated {answer_file.name}: {len(expected)} answers")
            succeeded += 1
        except Exception as exc:
            print(f"ERROR evaluating {answer_file}: {exc}", file=sys.stderr)
            failed += 1
    print(f"Nuggetizer complete: {succeeded} files evaluated, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
