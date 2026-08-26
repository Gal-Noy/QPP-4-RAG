#!/usr/bin/env python3
"""Calculate per-query and aggregate metrics from nugget assignments."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nuggetizer.evaluation import score_assignment_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()
    aggregate = score_assignment_file(args.input_file, args.output_file)
    print(aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
