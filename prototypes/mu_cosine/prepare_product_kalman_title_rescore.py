#!/usr/bin/env python3
"""Extract only title-corrected campaign rows for audited judge rescoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


SCHEMA_VERSION = 1
SCORE_COLUMNS = 8


class TitleRescoreError(ValueError):
    pass


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json_atomic(path, data):
    atomic_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_audited_pairs(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "pair_id", "corpus", "descendant_title", "ancestor_title", "hop",
        "descendant_title_audit_action", "ancestor_title_audit_action",
    }
    missing = sorted(required - fields)
    if missing:
        raise TitleRescoreError(f"audited pair table is missing: {', '.join(missing)}")
    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise TitleRescoreError("audited pair_id values must be unique")
    corpora = {row["corpus"] for row in rows}
    if len(corpora) != 1:
        raise TitleRescoreError(f"audited pair table must contain one corpus, found {sorted(corpora)}")
    for row in rows:
        for side in ("descendant", "ancestor"):
            action = row[f"{side}_title_audit_action"]
            if action not in {"raw", "corrected"}:
                raise TitleRescoreError(
                    f"{row['pair_id']}: invalid {side} title audit action {action!r}"
                )
    return rows, next(iter(corpora))


def load_score_rows(path):
    header = None
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")
            if line.startswith("#"):
                if header is not None or rows:
                    raise TitleRescoreError("score input must have one leading comment header")
                header = line
                continue
            if not line:
                raise TitleRescoreError(f"score input contains a blank row at line {line_number}")
            columns = line.split("\t")
            if len(columns) != SCORE_COLUMNS:
                raise TitleRescoreError(
                    f"score input line {line_number} has {len(columns)} columns; expected {SCORE_COLUMNS}"
                )
            rows.append((line, columns))
    if header is None:
        raise TitleRescoreError("score input is missing its comment header")
    return header, rows


def validate_alignment(pair_rows, score_rows, title_view="audited"):
    if len(pair_rows) != len(score_rows):
        raise TitleRescoreError(
            f"row-count mismatch: {len(pair_rows)} audited pairs vs {len(score_rows)} score rows"
        )
    for index, (pair, (_line, score)) in enumerate(zip(pair_rows, score_rows)):
        suffix = "_raw_title" if title_view == "raw" else "_title"
        expected_fields = (f"descendant{suffix}", f"ancestor{suffix}")
        if any(field not in pair for field in expected_fields):
            raise TitleRescoreError(
                f"title view {title_view!r} requires fields {expected_fields!r}"
            )
        expected = tuple(pair[field] for field in expected_fields)
        observed = (score[0], score[1])
        if observed != expected:
            raise TitleRescoreError(
                f"row {index} ({pair['pair_id']}) title mismatch: expected {expected!r}, "
                f"found {observed!r}"
            )
        if not score[4].endswith(f"_h{pair['hop']}"):
            raise TitleRescoreError(
                f"row {index} ({pair['pair_id']}) neighborhood {score[4]!r} "
                f"does not encode hop {pair['hop']}"
            )


def build_subset(pair_rows, score_rows):
    subset_lines = []
    mapping = []
    for source_index, (pair, (line, _columns)) in enumerate(zip(pair_rows, score_rows)):
        actions = {
            "descendant": pair["descendant_title_audit_action"],
            "ancestor": pair["ancestor_title_audit_action"],
        }
        if "corrected" not in actions.values():
            continue
        subset_index = len(subset_lines)
        subset_lines.append(line)
        mapping.append({
            "subset_index": subset_index,
            "source_index": source_index,
            "pair_id": pair["pair_id"],
            "hop": int(pair["hop"]),
            "descendant_title_audit_action": actions["descendant"],
            "ancestor_title_audit_action": actions["ancestor"],
        })
    return subset_lines, mapping


def run(args):
    output_paths = {Path(args.out).resolve(), Path(args.mapping).resolve()}
    input_paths = {Path(args.audited_pairs).resolve(), Path(args.score_in).resolve()}
    if output_paths & input_paths:
        raise TitleRescoreError("outputs must not overwrite inputs")
    if len(output_paths) != 2:
        raise TitleRescoreError("subset and mapping outputs must be distinct")

    pair_rows, corpus = load_audited_pairs(args.audited_pairs)
    header, score_rows = load_score_rows(args.score_in)
    validate_alignment(pair_rows, score_rows, args.title_view)
    subset_lines, rows = build_subset(pair_rows, score_rows)
    output_text = header + "\n"
    if subset_lines:
        output_text += "\n".join(subset_lines) + "\n"
    atomic_text(args.out, output_text)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus": corpus,
        "view": "audited_title_rescore_subset",
        "title_view": args.title_view,
        "audited_pairs": str(args.audited_pairs),
        "audited_pairs_sha256": sha256_path(args.audited_pairs),
        "source_score_in": str(args.score_in),
        "source_score_in_sha256": sha256_path(args.score_in),
        "source_row_count": len(pair_rows),
        "subset_score_in": str(args.out),
        "subset_score_in_sha256": sha256_path(args.out),
        "subset_row_count": len(rows),
        "rows": rows,
        "guardrail": "only rows with at least one pre-scoring title correction are selected",
    }
    write_json_atomic(args.mapping, manifest)
    print(f"prepared {corpus} audited-title rescore subset: {len(rows)}/{len(pair_rows)} rows")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audited-pairs", required=True, type=Path)
    ap.add_argument("--score-in", required=True, type=Path)
    ap.add_argument("--title-view", choices=("audited", "raw"), default="audited")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mapping", required=True, type=Path)
    return ap


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
