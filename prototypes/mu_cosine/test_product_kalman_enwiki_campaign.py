#!/usr/bin/env python3
"""Tests for the branch-stratified numeric enwiki campaign sampler."""

import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

from lmdb_id import enc_id
from sample_product_kalman_enwiki_campaign import (
    endpoint_record,
    main,
    normalize_title,
    shortest_upward_distance,
    title_quality_flags,
)


class ParentFixture:
    def __init__(self, parents):
        self._parents = parents

    def parents(self, node_id):
        return self._parents.get(node_id, ())


class TitleFixture:
    def title(self, _node_id, missing_ok=False):
        return "Some_Title"


def build_campaign_lmdb(path):
    try:
        import lmdb
    except ImportError:
        return False

    root = 1
    branches = (10, 20)
    titles = {root: "Main_topic_classifications", 10: "Branch_A", 20: "Branch_B"}
    edges = [(branch, root) for branch in branches]
    next_id = 100
    for branch in branches:
        for chain in range(4):
            parent = branch
            for depth in range(1, 9):
                child = next_id
                next_id += 1
                edges.append((child, parent))
                titles[child] = f"branch_{branch}_chain_{chain}_depth_{depth}"
                parent = child

    env = lmdb.open(str(path), max_dbs=16, map_size=64 * 1024 * 1024, subdir=True)
    category_parent = env.open_db(b"category_parent", dupsort=True, create=True)
    category_child = env.open_db(b"category_child", dupsort=True, create=True)
    title_i2s = env.open_db(b"title_i2s", create=True)
    title_s2i = env.open_db(b"title_s2i", create=True)
    meta = env.open_db(b"meta", create=True)
    with env.begin(write=True) as txn:
        for child, parent in edges:
            txn.put(enc_id(child), enc_id(parent), db=category_parent, dupdata=True)
            txn.put(enc_id(parent), enc_id(child), db=category_child, dupdata=True)
        for node_id, title in titles.items():
            encoded = title.encode("utf-8")
            txn.put(enc_id(node_id), encoded, db=title_i2s)
            txn.put(encoded, enc_id(node_id), db=title_s2i)
        txn.put(b"title_layer_kind", b"test_mediawiki_titles", db=meta)
        txn.put(b"title_layer_count", str(len(titles)).encode("ascii"), db=meta)
    env.sync()
    env.close()
    return True


def test_shortest_upward_distance_uses_dag_shortcut():
    graph = ParentFixture({3: (2, 1), 2: (1,)})
    assert shortest_upward_distance(graph, 3, 1, max_hop=2) == 1
    assert shortest_upward_distance(graph, 3, 2, max_hop=2) == 1
    assert shortest_upward_distance(graph, 2, 3, max_hop=2) is None


def test_title_normalization_and_quality_flags_are_descriptive():
    assert normalize_title("  Main_topic__Name  ") == "main topic name"
    assert title_quality_flags(" conflicted copy... ") == [
        "outer_whitespace",
        "conflicted_copy",
        "repeated_separator",
    ]


def test_endpoint_exclusion_uses_normalized_title_identity():
    stats = Counter()
    assert endpoint_record(TitleFixture(), 5, {"some title"}, stats) is None
    assert stats["excluded_title"] == 1


def test_cli_samples_balanced_numeric_pairs_and_joins_titles_at_output():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lmdb_dir = root / "campaign.lmdb"
        if not build_campaign_lmdb(lmdb_dir):
            print("  skip test_cli_samples_balanced_numeric_pairs_and_joins_titles_at_output (python-lmdb unavailable)")
            return
        pairs_tsv = root / "pairs.tsv"
        score_in = root / "score_in.tsv"
        manifest_path = root / "manifest.json"

        rc = main([
            "--lmdb-dir", str(lmdb_dir),
            "--pairs", "10",
            "--hmax", "5",
            "--max-prefix-depth", "3",
            "--max-attempts-per-hop", "1000",
            "--pairs-tsv", str(pairs_tsv),
            "--score-in", str(score_in),
            "--manifest", str(manifest_path),
            "--allow-small-sample",
        ])
        assert rc == 0

        with open(pairs_tsv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert len(rows) == 10
        assert all(row["descendant_id"].isdigit() and row["ancestor_id"].isdigit() for row in rows)
        assert all(not row["descendant_title"].isdigit() and not row["ancestor_title"].isdigit() for row in rows)
        assert len({tuple(sorted((row["descendant_id"], row["ancestor_id"]))) for row in rows}) == 10
        for hop in range(1, 6):
            hop_rows = [row for row in rows if int(row["hop"]) == hop]
            assert len(hop_rows) == 2
            assert {row["branch_title"] for row in hop_rows} == {"Branch_A", "Branch_B"}

        score_rows = [line.rstrip("\n").split("\t") for line in score_in.read_text().splitlines() if not line.startswith("#")]
        assert len(score_rows) == 10
        assert all(row[4] == f"transitive_h{pairs_row['hop']}" for row, pairs_row in zip(score_rows, rows))

        manifest = json.loads(manifest_path.read_text())
        assert manifest["scope_root_id"] == 1
        assert manifest["scope_root_title"] == "Main_topic_classifications"
        assert manifest["eligible_branch_count"] == 2
        assert manifest["hop_counts"] == {str(hop): 2 for hop in range(1, 6)}
        assert manifest["title_i2s_db"] == "title_i2s"
        assert manifest["title_s2i_db"] == "title_s2i"
        assert manifest["title_layer_count"] == 67
        assert manifest["title_lookup_phase"].endswith("endpoint boundary only")
        assert manifest["title_audit"]["semantic_corrections_applied"] is False
        assert manifest["title_audit"]["quality_flag_counts"] == {}

        first_pairs = pairs_tsv.read_text()
        first_score_in = score_in.read_text()
        first_manifest = manifest_path.read_text()
        assert main([
            "--lmdb-dir", str(lmdb_dir),
            "--pairs", "10",
            "--hmax", "5",
            "--max-prefix-depth", "3",
            "--max-attempts-per-hop", "1000",
            "--pairs-tsv", str(pairs_tsv),
            "--score-in", str(score_in),
            "--manifest", str(manifest_path),
            "--allow-small-sample",
        ]) == 0
        assert pairs_tsv.read_text() == first_pairs
        assert score_in.read_text() == first_score_in
        assert manifest_path.read_text() == first_manifest


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} enwiki campaign sampler tests passed")
