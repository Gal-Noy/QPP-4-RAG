#!/usr/bin/env python3
"""Assign fixed nuggets to generated answers with the hosted judge backend."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nuggetizer.evaluation import (  # noqa: E402,F401
    assign_answer_file,
    process_record,
    read_records,
)
from nuggetizer.models.nuggetizer import Nuggetizer  # noqa: E402


def setup_logging(log_level: int) -> None:
    level = logging.DEBUG if log_level >= 2 else logging.INFO if log_level else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nugget_file", type=Path, required=True)
    parser.add_argument("--answer_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--use_azure_openai", action="store_true")
    parser.add_argument("--qid", action="append", help="Only assign this qid; repeatable")
    parser.add_argument("--log_level", type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()
    setup_logging(args.log_level)
    nuggetizer = Nuggetizer(
        assigner_model=args.model,
        log_level=args.log_level,
        use_azure_openai=args.use_azure_openai,
    )
    try:
        added, expected = assign_answer_file(
            args.nugget_file,
            args.answer_file,
            args.output_file,
            nuggetizer,
            selected_qids=set(args.qid) if args.qid else None,
            logger=logging.getLogger(__name__),
        )
    except Exception:
        logging.getLogger(__name__).exception("Nugget assignment failed")
        return 1
    logging.getLogger(__name__).info(
        "Assignment complete: %d new, %d selected", added, len(expected)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
