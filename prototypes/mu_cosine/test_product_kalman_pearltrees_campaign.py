#!/usr/bin/env python3
"""Tests for the Pearltrees principal-parent campaign sampler."""

import csv
import json
import tempfile
from pathlib import Path

from sample_product_kalman_pearltrees_campaign import (
    candidate_pools,
    load_principal_paths,
    main,
    pearltrees_quality_flags,
)


def write_fixture(root, conflict=False):
    titles = {
        **{str(node): f"A title {node}" for node in range(1, 7)},
        **{str(node): f"B title {node}" for node in range(10, 16)},
        "20": "Private root",
        "21": "Hidden leaf",
    }
    titles["1"] = "Component A"
    titles["10"] = "Component B"
    titles_path = root / "titles.tsv"
    titles_path.write_text(
        "".join(f"{node_id}\t{title}\n" for node_id, title in sorted(titles.items(), key=lambda item: int(item[0]))),
        encoding="utf-8",
    )

    records = [
        {
            "tree_id": "6",
            "title": "A title 6",
            "account": "s243a",
            "path_ids": ["account:s243a", "1", "2", "3", "4", "5", "6"],
            "target_text": "- s243a\n  - Component A\n    - A title 6",
        },
        {
            "tree_id": "15",
            "title": "B renamed 15",
            "account": "s243a",
            "path_ids": ["account:s243a", "10", "11", "12", "13", "14", "15"],
            "target_text": "- s243a\n  - Component B\n    - B title 15",
        },
        {
            "tree_id": "21",
            "title": "Private alias label",
            "account": "s243a",
            "path_ids": ["account:s243a", "20", "21"],
            "target_text": "- s243a\n  - *private*\n    - *private*",
        },
    ]
    if conflict:
        records.append({
            "tree_id": "1",
            "title": "Component A",
            "account": "s243a",
            "path_ids": ["account:s243a", "2", "1"],
            "target_text": "- s243a\n  - A title 2\n    - Component A",
        })
    paths_path = root / "paths.jsonl"
    paths_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return paths_path, titles_path


def test_quality_flags_preserve_encoded_raw_title():
    assert pearltrees_quality_flags("Art &amp; Science") == ["encoded_display_text", "html_entity"]


def test_private_paths_are_removed_and_cross_record_cycles_are_excluded():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths, titles = write_fixture(root)
        forest = load_principal_paths(paths, titles)
        assert forest["stats"]["path_records_total"] == 3
        assert forest["stats"]["path_records_private"] == 1
        assert forest["stats"]["path_records_retained"] == 2
        assert forest["stats"]["principal_path_nodes"] == 12
        assert forest["stats"]["principal_path_unique_edges"] == 10
        assert ("s243a", "20") not in forest["nodes"]
        assert ("s243a", "21") not in forest["nodes"]
        assert "21" not in forest["title_aliases"]

        conflict_root = root / "conflict"
        conflict_root.mkdir()
        conflict_paths, conflict_titles = write_fixture(conflict_root, conflict=True)
        conflict_forest = load_principal_paths(conflict_paths, conflict_titles)
        _pools, rejected, conflicts = candidate_pools(conflict_forest, hmax=1, seed=0)
        assert rejected["direction_conflict_pairs"] == 1
        assert rejected["direction_conflict_observations"] >= 2
        assert conflicts[0]["kind"] == "direction"


def test_cli_samples_balanced_components_and_writes_stable_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths, titles = write_fixture(root)
        pairs_path = root / "pairs.tsv"
        score_path = root / "score.tsv"
        manifest_path = root / "manifest.json"
        args = [
            "--paths-jsonl", str(paths),
            "--titles-tsv", str(titles),
            "--pairs", "10",
            "--hmax", "5",
            "--seed", "7",
            "--pairs-tsv", str(pairs_path),
            "--score-in", str(score_path),
            "--manifest", str(manifest_path),
            "--allow-small-sample",
        ]
        assert main(args) == 0

        with open(pairs_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert len(rows) == 10
        assert {row["corpus"] for row in rows} == {"pearltrees"}
        assert {row["graph_view"] for row in rows} == {"principal_path_lineage"}
        assert {row["account"] for row in rows} == {"s243a"}
        assert all(row["source_tree_ids"] for row in rows)
        assert all(int(row["source_record_count"]) >= 1 for row in rows)
        alias_rows = [row for row in rows if row["descendant_id"] == "15"]
        assert alias_rows and "B renamed 15" in alias_rows[0]["descendant_title_aliases"]
        for hop in range(1, 6):
            hop_rows = [row for row in rows if int(row["hop"]) == hop]
            assert len(hop_rows) == 2
            assert {row["branch_title"] for row in hop_rows} == {"Component A", "Component B"}
        assert len({tuple(sorted((row["descendant_id"], row["ancestor_id"]))) for row in rows}) == 10

        score_rows = [
            line.rstrip("\n").split("\t")
            for line in score_path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")
        ]
        assert len(score_rows) == 10
        assert all(row[2] == "subtopic" for row in score_rows)
        assert all(row[5:7] == ["pearltrees_collection", "pearltrees_collection"] for row in score_rows)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["hop_counts"] == {str(hop): 2 for hop in range(1, 6)}
        assert manifest["forest_stats"]["path_records_private"] == 1
        assert manifest["forest_stats"]["merged_parent_conflict_nodes"] == 0
        assert manifest["forest_stats"]["record_title_alias_nodes"] == 1
        assert manifest["secondary_edge_policy"].startswith("assembled DAG aliases")
        assert manifest["title_audit"]["semantic_corrections_applied"] is False

        first = (pairs_path.read_bytes(), score_path.read_bytes(), manifest_path.read_bytes())
        assert main(args) == 0
        assert (pairs_path.read_bytes(), score_path.read_bytes(), manifest_path.read_bytes()) == first


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} Pearltrees campaign sampler tests passed")
