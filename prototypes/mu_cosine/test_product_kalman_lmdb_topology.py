#!/usr/bin/env python3
"""Tests for the streaming numeric LMDB topology audit."""

import json
import tempfile
from pathlib import Path

from audit_product_kalman_lmdb_topology import audit_lmdb_topology, main
from lmdb_id import enc_id


def build_topology_lmdb(path):
    try:
        import lmdb
    except ImportError:
        return False

    edges = [(2, 1), (3, 1), (3, 2), (4, 2)]
    titles = {
        1: "Main_topic_classifications",
        2: "Branch_A",
        3: "Shared_child",
        4: "Leaf",
    }
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
    env.sync()
    env.close()
    return True


def test_exact_all_node_degree_statistics_and_root_join():
    with tempfile.TemporaryDirectory() as tmp:
        lmdb_dir = Path(tmp) / "graph.lmdb"
        if not build_topology_lmdb(lmdb_dir):
            print("  skip test_exact_all_node_degree_statistics_and_root_join (python-lmdb unavailable)")
            return

        report = audit_lmdb_topology(
            lmdb_dir,
            scope_root="Main_topic_classifications",
            corpus="enwiki",
            graph_view="category_dag",
        )
        assert report["corpus"] == "enwiki"
        assert report["graph_view"] == "category_dag"
        assert report["nodes"]["graph_node_count"] == 4
        assert report["nodes"]["title_i2s_entries"] == 4
        assert report["nodes"]["title_entries_per_graph_node"] == 1.0
        assert report["root"] == {
            "requested": "Main_topic_classifications",
            "id": 1,
            "title": "Main_topic_classifications",
            "direct_child_count": 2,
        }
        assert report["integrity"]["category_parent_entries"] == 4
        assert report["integrity"]["category_child_entries"] == 4
        assert report["integrity"]["reciprocal_entry_counts_match"] is True

        parent = report["parent_degree"]
        assert parent["histogram"] == {"0": 1, "1": 2, "2": 1}
        assert parent["zero_degree_nodes"] == 1
        assert parent["degree_gt_one_nodes"] == 1
        assert parent["degree_gt_one_fraction"] == 0.25
        assert parent["mean"] == 1.0
        assert parent["median"] == 1
        assert parent["p95"] == 2
        assert parent["max"] == 2

        child = report["child_degree"]
        assert child["histogram"] == {"0": 2, "2": 2}
        assert child["zero_degree_nodes"] == 2
        assert child["mean"] == 1.0
        assert child["median"] == 0
        assert child["p95"] == 2
        assert report["interpretation"]["confidence_use"].startswith("forbidden:")


def test_numeric_root_does_not_require_title_tables():
    with tempfile.TemporaryDirectory() as tmp:
        lmdb_dir = Path(tmp) / "graph.lmdb"
        if not build_topology_lmdb(lmdb_dir):
            print("  skip test_numeric_root_does_not_require_title_tables (python-lmdb unavailable)")
            return

        report = audit_lmdb_topology(
            lmdb_dir,
            scope_root="1",
            title_i2s_db="absent_i2s",
            title_s2i_db="absent_s2i",
        )
        assert report["root"]["id"] == 1
        assert report["root"]["title"] is None
        assert report["root"]["direct_child_count"] == 2
        assert report["nodes"]["title_i2s_entries"] is None
        assert report["source"]["title_lookup_boundary"] == "optional scope-root provenance only"


def test_cli_writes_stable_json_report():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lmdb_dir = root / "graph.lmdb"
        if not build_topology_lmdb(lmdb_dir):
            print("  skip test_cli_writes_stable_json_report (python-lmdb unavailable)")
            return
        output = root / "audit.json"
        args = [
            "--lmdb-dir", str(lmdb_dir),
            "--scope-root", "Main_topic_classifications",
            "--corpus", "enwiki",
            "--graph-view", "category_dag",
            "--output", str(output),
        ]
        assert main(args) == 0
        first = output.read_bytes()
        assert main(args) == 0
        assert output.read_bytes() == first
        report = json.loads(first)
        assert report["audit_kind"] == "numeric_lmdb_topology"
        assert report["source"]["scan_boundary"] == "numeric adjacency keys and duplicate counts"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} LMDB topology audit tests passed")
