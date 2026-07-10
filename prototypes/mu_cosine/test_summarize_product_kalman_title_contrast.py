#!/usr/bin/env python3
"""Tests for corrected-title versus repeat-noise contrast summaries."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from summarize_product_kalman_title_contrast import TitleContrastError, run


def metric(mae, rmse, maximum, mean=0.0, changed=2, count=3):
    return {
        "count": count,
        "changed_count": changed,
        "mean_delta": mean,
        "mean_absolute_delta": mae,
        "root_mean_square_delta": rmse,
        "max_absolute_delta": maximum,
    }


def report(label, scale=1.0):
    d = metric(0.2 * scale, 0.3 * scale, 0.4 * scale, 0.01 * scale)
    s = metric(0.1 * scale, 0.2 * scale, 0.3 * scale, -0.01 * scale)
    return {
        "schema_version": 1,
        "comparison_label": label,
        "corpus": "pearltrees",
        "source_pair_count": 250,
        "affected_pair_count": 3,
        "exact_response_changed_count": 3,
        "D": d,
        "S": s,
        "by_hop": {"1": {"D": d, "S": s}},
        "numeric_fields": {"assoc.mu": s},
    }


def write_reports(root):
    corrected = root / "corrected.json"
    repeat = root / "repeat.json"
    corrected.write_text(json.dumps(report("corrected_titles")), encoding="utf-8")
    repeat.write_text(json.dumps(report("raw_title_repeat", 0.5)), encoding="utf-8")
    return corrected, repeat


def test_builds_descriptive_excess_and_ratio_summary():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        corrected, repeat = write_reports(root)
        args = SimpleNamespace(corrected=corrected, repeat=repeat, out=root / "contrast.json")
        assert run(args) == 0
        data = json.loads(args.out.read_text(encoding="utf-8"))
        assert data["D"]["excess_mean_absolute_delta"] == 0.1
        assert data["D"]["ratio_mean_absolute_delta"] == 2.0
        assert data["by_hop"]["1"]["S"]["ratio_root_mean_square_delta"] == 2.0
        assert "do not identify a causal title effect" in data["guardrail"]


def test_rejects_mismatched_contracts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        corrected, repeat = write_reports(root)
        data = json.loads(repeat.read_text(encoding="utf-8"))
        data["affected_pair_count"] = 4
        repeat.write_text(json.dumps(data), encoding="utf-8")
        args = SimpleNamespace(corrected=corrected, repeat=repeat, out=root / "contrast.json")
        try:
            run(args)
        except TitleContrastError as exc:
            assert "affected_pair_count" in str(exc)
        else:
            raise AssertionError("expected mismatched comparison contract rejection")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} title contrast tests passed")
