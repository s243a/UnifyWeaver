#!/usr/bin/env python3
"""Contrast corrected-title response shifts with an unchanged-title repeat control."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


SCHEMA_VERSION = 1
SUMMARY_METRICS = ("mean_absolute_delta", "root_mean_square_delta", "max_absolute_delta")


class TitleContrastError(ValueError):
    pass


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_report(path, expected_label):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise TitleContrastError(f"{path}: unsupported schema {data.get('schema_version')!r}")
    if data.get("comparison_label") != expected_label:
        raise TitleContrastError(
            f"{path}: expected comparison label {expected_label!r}, "
            f"found {data.get('comparison_label')!r}"
        )
    return data


def metric_contrast(intervention, repeat):
    if intervention.get("count") != repeat.get("count"):
        raise TitleContrastError("comparison metric counts do not match")
    out = {
        "count": intervention.get("count", 0),
        "corrected_title_changed_count": intervention.get("changed_count", 0),
        "raw_repeat_changed_count": repeat.get("changed_count", 0),
        "corrected_title_mean_delta": intervention.get("mean_delta"),
        "raw_repeat_mean_delta": repeat.get("mean_delta"),
    }
    for metric in SUMMARY_METRICS:
        observed = intervention.get(metric)
        noise = repeat.get(metric)
        if observed is None or noise is None:
            continue
        out[f"corrected_title_{metric}"] = observed
        out[f"raw_repeat_{metric}"] = noise
        out[f"excess_{metric}"] = observed - noise
        out[f"ratio_{metric}"] = observed / noise if noise > 0 else None
    return out


def run(args):
    corrected = load_report(args.corrected, "corrected_titles")
    repeat = load_report(args.repeat, "raw_title_repeat")
    for field in ("corpus", "source_pair_count", "affected_pair_count"):
        if corrected.get(field) != repeat.get(field):
            raise TitleContrastError(f"comparison reports differ on {field}")
    corrected_hops, repeat_hops = corrected.get("by_hop", {}), repeat.get("by_hop", {})
    if set(corrected_hops) != set(repeat_hops):
        raise TitleContrastError("comparison reports have different hop levels")
    corrected_fields = corrected.get("numeric_fields", {})
    repeat_fields = repeat.get("numeric_fields", {})
    if set(corrected_fields) != set(repeat_fields):
        raise TitleContrastError("comparison reports have different numeric fields")
    output = {
        "schema_version": SCHEMA_VERSION,
        "corpus": corrected["corpus"],
        "view": "corrected_title_vs_raw_repeat_contrast",
        "source_pair_count": corrected["source_pair_count"],
        "affected_pair_count": corrected["affected_pair_count"],
        "exact_response_changed_count": {
            "corrected_titles": corrected["exact_response_changed_count"],
            "raw_title_repeat": repeat["exact_response_changed_count"],
        },
        "D": metric_contrast(corrected["D"], repeat["D"]),
        "S": metric_contrast(corrected["S"], repeat["S"]),
        "by_hop": {
            hop: {
                "D": metric_contrast(corrected_hops[hop]["D"], repeat_hops[hop]["D"]),
                "S": metric_contrast(corrected_hops[hop]["S"], repeat_hops[hop]["S"]),
            }
            for hop in sorted(corrected_hops, key=int)
        },
        "numeric_fields": {
            field: metric_contrast(corrected_fields[field], repeat_fields[field])
            for field in sorted(corrected_fields)
        },
        "inputs": {
            "corrected_title_report": str(args.corrected),
            "raw_repeat_report": str(args.repeat),
        },
        "guardrail": (
            "single repeat control is descriptive; excess/ratio metrics do not identify a causal title effect"
        ),
    }
    atomic_json(args.out, output)
    print(
        f"contrasted {output['corpus']} corrected-title shifts with raw-title repeat noise "
        f"over {output['affected_pair_count']} pairs"
    )
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corrected", required=True, type=Path)
    ap.add_argument("--repeat", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    return ap


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
