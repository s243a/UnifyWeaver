#!/usr/bin/env python3
"""Tests for deterministic audited-title rescore subset extraction."""

import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from prepare_product_kalman_title_rescore import TitleRescoreError, run


FIELDS = [
    "pair_id", "corpus", "descendant_title", "ancestor_title", "hop",
    "descendant_raw_title", "ancestor_raw_title",
    "descendant_title_audit_action", "ancestor_title_audit_action",
]


def write_fixture(root, actions=(("raw", "corrected"), ("raw", "raw"), ("corrected", "raw"))):
    pairs = root / "pairs.tsv"
    score = root / "score.tsv"
    pair_rows = []
    score_lines = ["# node_title\troot_title\tcur_relation\tconf\tneighborhood\tnode_type\troot_type\traw"]
    for index, (desc_action, anc_action) in enumerate(actions):
        pair_rows.append({
            "pair_id": f"p{index}",
            "corpus": "pearltrees",
            "descendant_title": f"Child {index}",
            "ancestor_title": f"Parent {index}",
            "descendant_raw_title": f"Raw Child {index}",
            "ancestor_raw_title": f"Raw Parent {index}",
            "hop": str(index + 1),
            "descendant_title_audit_action": desc_action,
            "ancestor_title_audit_action": anc_action,
        })
        score_lines.append(
            f"Child {index}\tParent {index}\tsubtopic\t1.0\tprincipal_h{index + 1}\t"
            "pearltrees_collection\tpearltrees_collection\t"
        )
    with open(pairs, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(pair_rows)
    score.write_text("\n".join(score_lines) + "\n", encoding="utf-8")
    return pairs, score


def args_for(root, pairs, score):
    return SimpleNamespace(
        audited_pairs=pairs,
        score_in=score,
        title_view="audited",
        out=root / "subset.tsv",
        mapping=root / "mapping.json",
    )


def expect_error(args, match):
    try:
        run(args)
    except TitleRescoreError as exc:
        assert match in str(exc), str(exc)
    else:
        raise AssertionError(f"expected TitleRescoreError containing {match!r}")


def test_extracts_corrected_rows_with_stable_mapping():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs, score = write_fixture(root)
        args = args_for(root, pairs, score)
        assert run(args) == 0
        lines = args.out.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("# node_title")
        assert [line.split("\t")[:2] for line in lines[1:]] == [
            ["Child 0", "Parent 0"], ["Child 2", "Parent 2"]
        ]
        manifest = json.loads(args.mapping.read_text(encoding="utf-8"))
        assert manifest["source_row_count"] == 3
        assert manifest["subset_row_count"] == 2
        assert [(row["subset_index"], row["source_index"], row["pair_id"]) for row in manifest["rows"]] == [
            (0, 0, "p0"), (1, 2, "p2")
        ]
        assert manifest["rows"][0]["ancestor_title_audit_action"] == "corrected"
        first = (args.out.read_bytes(), args.mapping.read_bytes())
        assert run(args) == 0
        assert (args.out.read_bytes(), args.mapping.read_bytes()) == first


def test_all_raw_rows_produce_header_only_subset():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs, score = write_fixture(root, actions=(("raw", "raw"),))
        args = args_for(root, pairs, score)
        assert run(args) == 0
        assert len(args.out.read_text(encoding="utf-8").splitlines()) == 1
        assert json.loads(args.mapping.read_text(encoding="utf-8"))["rows"] == []


def test_selects_same_affected_rows_from_raw_title_view():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs, score = write_fixture(root)
        text = score.read_text(encoding="utf-8")
        for index in range(3):
            text = text.replace(
                f"Child {index}\tParent {index}",
                f"Raw Child {index}\tRaw Parent {index}",
            )
        score.write_text(text, encoding="utf-8")
        args = args_for(root, pairs, score)
        args.title_view = "raw"
        assert run(args) == 0
        lines = args.out.read_text(encoding="utf-8").splitlines()[1:]
        assert [line.split("\t")[:2] for line in lines] == [
            ["Raw Child 0", "Raw Parent 0"], ["Raw Child 2", "Raw Parent 2"]
        ]
        assert json.loads(args.mapping.read_text(encoding="utf-8"))["title_view"] == "raw"


def test_rejects_title_hop_and_row_count_drift():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs, score = write_fixture(root)
        args = args_for(root, pairs, score)
        score.write_text(score.read_text(encoding="utf-8").replace("Child 0", "Wrong", 1), encoding="utf-8")
        expect_error(args, "title mismatch")

        pairs, score = write_fixture(root)
        score.write_text(score.read_text(encoding="utf-8").replace("principal_h1", "principal_h5", 1), encoding="utf-8")
        expect_error(args, "does not encode hop")

        pairs, score = write_fixture(root)
        score.write_text("\n".join(score.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8")
        expect_error(args, "row-count mismatch")


def test_rejects_invalid_actions_and_output_aliases():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs, score = write_fixture(root, actions=(("review", "raw"),))
        args = args_for(root, pairs, score)
        expect_error(args, "invalid descendant title audit action")
        pairs, score = write_fixture(root)
        args = args_for(root, pairs, score)
        args.out = pairs
        expect_error(args, "outputs must not overwrite inputs")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} title rescore tests passed")
