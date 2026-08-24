#!/usr/bin/env python3
"""Run the paper's GPT-4o Nuggetizer assigner and calculate nugget scores."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def selected_directories(base: Path, output: Path):
    children = [
        name for name in ("retrieval", "retrieval_cohere")
        if (base / name).is_dir()
    ]
    if children:
        return [(base / name, output / name) for name in children]
    return [(base, output)]


def expected_qids(answer_file: Path, nugget_qids: set[str]) -> set[str]:
    return {
        str(record["topic_id"])
        for record in read_records(answer_file)
        if str(record.get("topic_id")) in nugget_qids
    }


def run_command(command: list[str], timeout: int) -> None:
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr[-2000:]}"
        )


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rag-results-dir", type=Path, default=repo / "querygym" / "rag_results"
    )
    parser.add_argument("--nugget-file", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=repo / "querygym" / "rag_nuggetized_eval"
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="Nuggetizer assigner model; keep gpt-4o for paper consistency",
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--use-azure-openai", action="store_true")
    args = parser.parse_args()

    assign_script = repo / "nuggetizer" / "scripts" / "assign_nuggets.py"
    metrics_script = repo / "nuggetizer" / "scripts" / "calculate_metrics.py"
    for required in (args.rag_results_dir, args.nugget_file, assign_script, metrics_script):
        if not required.exists():
            parser.error(f"Required path does not exist: {required}")
    if "llama" in args.model.lower():
        parser.error("Do not replace the paper's Nuggetizer evaluator with Llama")

    nugget_qids = {
        str(record["qid"]) for record in read_records(args.nugget_file)
    }
    succeeded = failed = 0
    for answers_dir, method_output in selected_directories(
        args.rag_results_dir, args.output_dir
    ):
        assignments_dir = method_output / "assignments"
        scores_dir = method_output / "scores"
        assignments_dir.mkdir(parents=True, exist_ok=True)
        scores_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(set(answers_dir.glob("*.json")) | set(answers_dir.glob("*.jsonl")))
        if args.max_files is not None:
            files = files[:args.max_files]
        for number, answer_file in enumerate(files, 1):
            assignment_file = assignments_dir / f"{answer_file.stem}_assignments.jsonl"
            score_file = scores_dir / f"{answer_file.stem}_scores.jsonl"
            expected = expected_qids(answer_file, nugget_qids)
            if not expected:
                print(f"Skipping {answer_file.name}: no answer qids match the nugget file")
                continue
            try:
                assigned_before = {
                    str(record["qid"])
                    for record in read_records(assignment_file)
                } if assignment_file.exists() else set()
                if expected - assigned_before:
                    command = [
                        sys.executable, str(assign_script),
                        "--nugget_file", str(args.nugget_file),
                        "--answer_file", str(answer_file),
                        "--output_file", str(assignment_file),
                        "--model", args.model,
                    ]
                    if args.use_azure_openai:
                        command.append("--use_azure_openai")
                    run_command(command, timeout=1800)
                assigned = {
                    str(record["qid"]) for record in read_records(assignment_file)
                }
                missing = expected - assigned
                if missing:
                    raise RuntimeError(
                        f"Nuggetizer did not assign {len(missing)} qids: {sorted(missing)}"
                    )
                run_command([
                    sys.executable, str(metrics_script),
                    "--input_file", str(assignment_file),
                    "--output_file", str(score_file),
                ], timeout=120)
                print(
                    f"[{number}/{len(files)}] evaluated {answer_file.name}: "
                    f"{len(expected)} answers"
                )
                succeeded += 1
            except Exception as exc:
                print(f"ERROR evaluating {answer_file}: {exc}", file=sys.stderr)
                failed += 1
    print(f"Nuggetizer complete: {succeeded} files evaluated, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
