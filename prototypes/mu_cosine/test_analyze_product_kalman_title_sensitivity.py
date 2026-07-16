#!/usr/bin/env python3
"""Tests for audited-title response joining and sensitivity summaries."""

import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from analyze_product_kalman_title_sensitivity import TitleSensitivityError, run, sha256_path


def response(index, d, s, applies=0.8):
    return {
        "id": index,
        "subcategory": {"mu_fwd": d, "mu_rev": 0.1, "applies": applies},
        "subtopic": {"mu_fwd": d - 0.1, "mu_rev": 0.1, "applies": applies - 0.1},
        "element_of": {"mu_fwd": 0.2, "mu_rev": 0.1, "applies": 0.2},
        "super_category": {"mu_fwd": 0.1, "mu_rev": d, "applies": 0.1},
        "see_also": {"mu": s, "applies": s},
        "assoc": {"mu": s - 0.1, "applies": s - 0.1},
    }


def write_fixture(root):
    pairs = root / "pairs.tsv"
    fields = [
        "pair_id", "corpus", "hop", "descendant_title_audit_action",
        "ancestor_title_audit_action",
    ]
    rows = [
        {"pair_id": "p0", "corpus": "pearltrees", "hop": "1", "descendant_title_audit_action": "raw", "ancestor_title_audit_action": "corrected"},
        {"pair_id": "p1", "corpus": "pearltrees", "hop": "2", "descendant_title_audit_action": "raw", "ancestor_title_audit_action": "raw"},
        {"pair_id": "p2", "corpus": "pearltrees", "hop": "1", "descendant_title_audit_action": "corrected", "ancestor_title_audit_action": "raw"},
    ]
    with open(pairs, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    raw = root / "raw.txt"
    raw.write_text(json.dumps([response(0, 0.7, 0.3), response(1, 0.6, 0.2), response(2, 0.5, 0.4)]) + "\n", encoding="utf-8")
    audited = root / "audited.txt"
    audited.write_text(json.dumps([response(0, 0.8, 0.2), response(1, 0.4, 0.6)]) + "\n", encoding="utf-8")
    mapping = root / "mapping.json"
    mapping.write_text(json.dumps({
        "schema_version": 1,
        "corpus": "pearltrees",
        "audited_pairs_sha256": sha256_path(pairs),
        "source_row_count": 3,
        "subset_row_count": 2,
        "rows": [
            {"subset_index": 0, "source_index": 0, "pair_id": "p0"},
            {"subset_index": 1, "source_index": 2, "pair_id": "p2"},
        ],
    }), encoding="utf-8")
    return pairs, raw, audited, mapping


def args_for(root, inputs):
    pairs, raw, audited, mapping = inputs
    return SimpleNamespace(
        audited_pairs=pairs,
        raw_responses=raw,
        audited_responses=audited,
        mapping=mapping,
        comparison_label="corrected_titles",
        merged_responses=root / "merged.txt",
        out_tsv=root / "deltas.tsv",
        report=root / "report.json",
    )


def expect_error(args, match):
    try:
        run(args)
    except TitleSensitivityError as exc:
        assert match in str(exc), str(exc)
    else:
        raise AssertionError(f"expected TitleSensitivityError containing {match!r}")


def test_joins_subset_and_reports_ds_and_numeric_deltas():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        args = args_for(root, write_fixture(root))
        assert run(args) == 0
        merged = json.loads(args.merged_responses.read_text(encoding="utf-8"))
        assert [item["id"] for item in merged] == [0, 1, 2]
        assert merged[0]["subcategory"]["mu_fwd"] == 0.8
        assert merged[1]["subcategory"]["mu_fwd"] == 0.6
        assert merged[2]["subcategory"]["mu_fwd"] == 0.4
        report = json.loads(args.report.read_text(encoding="utf-8"))
        assert report["affected_pair_count"] == 2
        assert report["comparison_label"] == "corrected_titles"
        assert report["exact_response_changed_count"] == 2
        assert abs(report["D"]["mean_delta"] - 0.0) < 1e-12
        assert abs(report["D"]["mean_absolute_delta"] - 0.1) < 1e-12
        assert abs(report["S"]["mean_delta"] - 0.05) < 1e-12
        assert report["numeric_fields"]["subcategory.mu_fwd"]["changed_count"] == 2
        delta_rows = list(csv.DictReader(args.out_tsv.open(encoding="utf-8"), delimiter="\t"))
        assert [row["pair_id"] for row in delta_rows] == ["p0", "p2"]
        first = (args.merged_responses.read_bytes(), args.out_tsv.read_bytes(), args.report.read_bytes())
        assert run(args) == 0
        assert (args.merged_responses.read_bytes(), args.out_tsv.read_bytes(), args.report.read_bytes()) == first


def test_rejects_response_and_mapping_drift():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs = write_fixture(root)
        args = args_for(root, inputs)
        raw = inputs[1]
        raw.write_text(json.dumps([response(0, 0.7, 0.3), response(1, 0.6, 0.2)]) + "\n", encoding="utf-8")
        expect_error(args, "raw response ids mismatch")

        inputs = write_fixture(root)
        args = args_for(root, inputs)
        mapping = json.loads(inputs[3].read_text(encoding="utf-8"))
        mapping["rows"][1]["pair_id"] = "wrong"
        inputs[3].write_text(json.dumps(mapping), encoding="utf-8")
        expect_error(args, "mapping pair_id mismatch")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} title sensitivity analysis tests passed")
