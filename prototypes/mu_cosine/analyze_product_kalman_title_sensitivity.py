#!/usr/bin/env python3
"""Join audited subset responses and quantify frozen title-policy sensitivity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from emit_direction_blend import parse_responses


SCHEMA_VERSION = 1
DIR = ("subcategory", "subtopic", "element_of", "super_category")
SYM = ("see_also", "assoc")


class TitleSensitivityError(ValueError):
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


def load_pairs(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    required = {
        "pair_id", "corpus", "hop", "descendant_title_audit_action",
        "ancestor_title_audit_action",
    }
    missing = sorted(required - (set(rows[0]) if rows else set()))
    if missing:
        raise TitleSensitivityError(f"audited pair table is missing: {', '.join(missing)}")
    return rows


def load_mapping(path, pairs_path, pair_count):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise TitleSensitivityError(f"unsupported mapping schema {data.get('schema_version')!r}")
    if data.get("audited_pairs_sha256") != sha256_path(pairs_path):
        raise TitleSensitivityError("mapping does not match the audited pair table hash")
    if data.get("source_row_count") != pair_count:
        raise TitleSensitivityError("mapping source row count does not match audited pairs")
    rows = data.get("rows")
    if not isinstance(rows, list) or data.get("subset_row_count") != len(rows):
        raise TitleSensitivityError("mapping subset row count is inconsistent")
    expected_subset = list(range(len(rows)))
    if [row.get("subset_index") for row in rows] != expected_subset:
        raise TitleSensitivityError("mapping subset indices must be contiguous and ordered")
    source_indices = [row.get("source_index") for row in rows]
    if any(not isinstance(index, int) or index < 0 or index >= pair_count for index in source_indices):
        raise TitleSensitivityError("mapping contains an invalid source index")
    if len(source_indices) != len(set(source_indices)):
        raise TitleSensitivityError("mapping source indices must be unique")
    return data, rows


def require_ids(by_id, expected, label):
    observed = set(by_id)
    expected = set(expected)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise TitleSensitivityError(f"{label} response ids mismatch; missing={missing}, extra={extra}")


def response_hash(obj):
    payload = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def feature_value(obj, relation, field):
    value = (obj.get(relation, {}) or {}).get(field, 0.0)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TitleSensitivityError(f"non-numeric {relation}.{field}: {value!r}") from exc
    if not math.isfinite(result):
        raise TitleSensitivityError(f"non-finite {relation}.{field}: {value!r}")
    return result


def ds_features(obj):
    directional = [(feature_value(obj, rel, "mu_fwd"), rel) for rel in DIR]
    symmetric = [(feature_value(obj, rel, "mu"), rel) for rel in SYM]
    d_value, d_relation = max(directional)
    s_value, s_relation = max(symmetric)
    return d_value, s_value, d_relation, s_relation


def numeric_fields(obj):
    fields = {}
    for relation, values in obj.items():
        if relation == "id" or not isinstance(values, dict):
            continue
        for field, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                number = float(value)
                if not math.isfinite(number):
                    raise TitleSensitivityError(f"non-finite {relation}.{field}: {value!r}")
                fields[f"{relation}.{field}"] = number
    return fields


def metric_summary(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "changed_count": sum(abs(value) > 1e-12 for value in values),
        "mean_delta": sum(values) / len(values),
        "mean_absolute_delta": sum(abs(value) for value in values) / len(values),
        "root_mean_square_delta": math.sqrt(sum(value * value for value in values) / len(values)),
        "max_absolute_delta": max(abs(value) for value in values),
    }


def write_tsv(path, rows):
    fields = [
        "pair_id", "source_index", "subset_index", "hop",
        "descendant_title_audit_action", "ancestor_title_audit_action",
        "raw_D", "audited_D", "delta_D", "raw_S", "audited_S", "delta_S",
        "raw_D_relation", "audited_D_relation", "raw_S_relation", "audited_S_relation",
        "response_changed", "raw_response_sha256", "audited_response_sha256",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row[field]) for field in fields))
    atomic_text(path, "\n".join(lines) + "\n")


def run(args):
    outputs = {Path(args.merged_responses).resolve(), Path(args.out_tsv).resolve(), Path(args.report).resolve()}
    inputs = {
        Path(args.audited_pairs).resolve(), Path(args.raw_responses).resolve(),
        Path(args.audited_responses).resolve(), Path(args.mapping).resolve(),
    }
    if outputs & inputs or len(outputs) != 3:
        raise TitleSensitivityError("all inputs and outputs must use distinct paths")

    pairs = load_pairs(args.audited_pairs)
    mapping, mapping_rows = load_mapping(args.mapping, args.audited_pairs, len(pairs))
    raw = parse_responses(args.raw_responses)
    audited = parse_responses(args.audited_responses)
    require_ids(raw, range(len(pairs)), "raw")
    require_ids(audited, range(len(mapping_rows)), "audited subset")

    merged = {index: dict(obj) for index, obj in raw.items()}
    matched = []
    ds_deltas = {"D": [], "S": []}
    by_hop = defaultdict(lambda: {"D": [], "S": []})
    field_deltas = defaultdict(list)
    for map_row in mapping_rows:
        subset_index = map_row["subset_index"]
        source_index = map_row["source_index"]
        pair = pairs[source_index]
        if map_row["pair_id"] != pair["pair_id"]:
            raise TitleSensitivityError(f"mapping pair_id mismatch at source index {source_index}")
        raw_obj = raw[source_index]
        audited_obj = dict(audited[subset_index])
        audited_obj["id"] = source_index
        merged[source_index] = audited_obj

        raw_d, raw_s, raw_d_rel, raw_s_rel = ds_features(raw_obj)
        aud_d, aud_s, aud_d_rel, aud_s_rel = ds_features(audited_obj)
        delta_d, delta_s = aud_d - raw_d, aud_s - raw_s
        ds_deltas["D"].append(delta_d)
        ds_deltas["S"].append(delta_s)
        hop = int(pair["hop"])
        by_hop[hop]["D"].append(delta_d)
        by_hop[hop]["S"].append(delta_s)
        raw_fields, audited_fields = numeric_fields(raw_obj), numeric_fields(audited_obj)
        for field in sorted(set(raw_fields) | set(audited_fields)):
            field_deltas[field].append(audited_fields.get(field, 0.0) - raw_fields.get(field, 0.0))
        matched.append({
            "pair_id": pair["pair_id"],
            "source_index": source_index,
            "subset_index": subset_index,
            "hop": hop,
            "descendant_title_audit_action": pair["descendant_title_audit_action"],
            "ancestor_title_audit_action": pair["ancestor_title_audit_action"],
            "raw_D": f"{raw_d:.12g}",
            "audited_D": f"{aud_d:.12g}",
            "delta_D": f"{delta_d:.12g}",
            "raw_S": f"{raw_s:.12g}",
            "audited_S": f"{aud_s:.12g}",
            "delta_S": f"{delta_s:.12g}",
            "raw_D_relation": raw_d_rel,
            "audited_D_relation": aud_d_rel,
            "raw_S_relation": raw_s_rel,
            "audited_S_relation": aud_s_rel,
            "response_changed": str(raw_obj != audited_obj).lower(),
            "raw_response_sha256": response_hash(raw_obj),
            "audited_response_sha256": response_hash(audited_obj),
        })

    merged_list = [merged[index] for index in range(len(pairs))]
    atomic_text(
        args.merged_responses,
        json.dumps(merged_list, separators=(",", ":"), sort_keys=True) + "\n",
    )
    write_tsv(args.out_tsv, matched)
    report = {
        "schema_version": SCHEMA_VERSION,
        "corpus": mapping["corpus"],
        "view": "paired_judge_response_comparison",
        "comparison_label": args.comparison_label,
        "source_pair_count": len(pairs),
        "affected_pair_count": len(matched),
        "exact_response_changed_count": sum(row["response_changed"] == "true" for row in matched),
        "D": metric_summary(ds_deltas["D"]),
        "S": metric_summary(ds_deltas["S"]),
        "by_hop": {
            str(hop): {"D": metric_summary(values["D"]), "S": metric_summary(values["S"])}
            for hop, values in sorted(by_hop.items())
        },
        "numeric_fields": {
            field: metric_summary(values) for field, values in sorted(field_deltas.items())
        },
        "artifacts": {
            "audited_pairs_sha256": sha256_path(args.audited_pairs),
            "raw_responses_sha256": sha256_path(args.raw_responses),
            "audited_subset_responses_sha256": sha256_path(args.audited_responses),
            "mapping_sha256": sha256_path(args.mapping),
            "merged_responses_sha256": sha256_path(args.merged_responses),
            "matched_deltas_sha256": sha256_path(args.out_tsv),
        },
        "guardrail": "descriptive paired-response comparison only; no causal attribution or model selection",
    }
    atomic_text(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"analyzed {mapping['corpus']} {args.comparison_label}: {len(matched)} affected pairs, "
        f"{report['exact_response_changed_count']} exact response changes"
    )
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audited-pairs", required=True, type=Path)
    ap.add_argument("--raw-responses", required=True, type=Path)
    ap.add_argument("--audited-responses", required=True, type=Path)
    ap.add_argument("--mapping", required=True, type=Path)
    ap.add_argument("--comparison-label", default="corrected_titles")
    ap.add_argument("--merged-responses", required=True, type=Path)
    ap.add_argument("--out-tsv", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    return ap


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
