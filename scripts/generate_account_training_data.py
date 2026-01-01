#!/usr/bin/env python3
"""
Generate account-specific training data from multi-account JSONL.

Usage:
    python3 scripts/generate_account_training_data.py \
        --input reports/pearltrees_targets_full_multi_account.jsonl \
        --account s243a \
        --output reports/pearltrees_targets_s243a.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Filter JSONL training data by account"
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Input multi-account JSONL file")
    parser.add_argument("--account", type=str, required=True,
                        help="Account to filter for")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output account-specific JSONL file")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    count = 0
    with open(args.input) as f_in, open(args.output, "w") as f_out:
        for line in f_in:
            record = json.loads(line)
            if record.get("account") == args.account:
                f_out.write(line)
                count += 1

    print(f"Wrote {count} records for account '{args.account}' to {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
