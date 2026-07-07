#!/usr/bin/env python3
"""Synthetic checks for the fresh-corpus Sigma(hop) sampler."""

import json
import os
import tempfile
from unittest import mock

from lmdb_id import enc_id
from sample_sigma_hop_fresh_corpus import (
    FreshCorpusError,
    ancestors_by_hop,
    build_maps,
    filter_candidate_edges,
    load_edges,
    main,
    node_block,
    sample_balanced_pairs,
)


def _run_cli(argv):
    with mock.patch("sys.argv", argv):
        return main()


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _chain_graph(root="Fresh_root", chains=8, depth=5):
    rows = ["child\tparent\n"]
    for i in range(chains):
        parent = root
        for d in range(depth, 0, -1):
            child = f"fresh_{i}_{d}"
            rows.append(f"{child}\t{parent}\n")
            parent = child
    return "".join(rows)


def test_filter_candidate_edges_removes_exploratory_nodes():
    exploratory = {("old_child", "old_parent")}
    candidate = [("fresh_child", "fresh_parent"), ("fresh_child", "old_parent")]
    filtered, removed, blocked = filter_candidate_edges(candidate, node_block(exploratory))
    assert filtered == [("fresh_child", "fresh_parent")]
    assert removed["overlap_edges"] == 1
    assert blocked == {"fresh_child"}


def test_sample_balanced_pairs_excludes_duplicate_unordered_pairs():
    pool = {
        1: [("a", "b"), ("b", "a"), ("c", "d")],
        2: [("e", "f"), ("g", "h")],
    }
    rows, counts = sample_balanced_pairs(pool, total_pairs=4, hmax=2, seed=0)
    unordered = {tuple(sorted((desc, anc))) for desc, anc, _ in rows}
    assert len(unordered) == len(rows)
    assert counts[1] == 2
    assert counts[2] == 2


def test_sample_balanced_pairs_assigns_remainder_to_lower_hops():
    pool = {
        1: [(f"d1_{i}", f"a1_{i}") for i in range(5)],
        2: [(f"d2_{i}", f"a2_{i}") for i in range(5)],
        3: [(f"d3_{i}", f"a3_{i}") for i in range(5)],
    }
    _rows, counts = sample_balanced_pairs(pool, total_pairs=7, hmax=3, seed=0)
    assert counts == {1: 3, 2: 2, 3: 2}


def test_ancestors_by_hop_uses_shortest_path_in_dag():
    parents, _ = build_maps([("child", "mid"), ("mid", "root"), ("child", "root")])
    by_hop = ancestors_by_hop(parents, "child", hmax=2)
    assert by_hop[1] == ["mid", "root"]
    assert 2 not in by_hop or "root" not in by_hop[2]


def test_cli_rejects_toy_sample_without_explicit_opt_in():
    try:
        _run_cli([
            "sample_sigma_hop_fresh_corpus.py",
            "--candidate-graph", "candidate.tsv",
            "--exploratory-graph", "exploratory.tsv",
            "--pairs", "10",
            "--out", "pairs.tsv",
            "--manifest", "manifest.json",
        ])
    except FreshCorpusError as exc:
        assert "at least 250 pairs" in str(exc)
    else:
        raise AssertionError("expected toy sample to require --allow-small-sample")


def test_cli_rejects_nonpositive_pairs_even_for_small_sample():
    try:
        _run_cli([
            "sample_sigma_hop_fresh_corpus.py",
            "--candidate-graph", "candidate.tsv",
            "--exploratory-graph", "exploratory.tsv",
            "--pairs", "0",
            "--out", "pairs.tsv",
            "--manifest", "manifest.json",
            "--allow-small-sample",
        ])
    except FreshCorpusError as exc:
        assert "--pairs must be positive" in str(exc)
    else:
        raise AssertionError("expected nonpositive pair count to abort")


def test_cli_rejects_slice_depth_less_than_hmax():
    try:
        _run_cli([
            "sample_sigma_hop_fresh_corpus.py",
            "--candidate-graph", "candidate.tsv",
            "--exploratory-graph", "exploratory.tsv",
            "--pairs", "250",
            "--slice-depth", "4",
            "--out", "pairs.tsv",
            "--manifest", "manifest.json",
        ])
    except FreshCorpusError as exc:
        assert "slice-depth 4 < --hmax 5" in str(exc)
    else:
        raise AssertionError("expected impossible slice-depth/hmax combination to abort")


def test_cli_rejects_same_candidate_and_exploratory_graph():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write("child\tparent\nx\ty\n")
        path = f.name
    try:
        try:
            _run_cli([
                "sample_sigma_hop_fresh_corpus.py",
                "--candidate-graph", path,
                "--exploratory-graph", path,
                "--out", "pairs.tsv",
                "--manifest", "manifest.json",
            ])
        except FreshCorpusError as exc:
            assert "must be different files" in str(exc)
        else:
            raise AssertionError("expected identical graph paths to abort")
    finally:
        os.unlink(path)


def test_cli_rejects_scope_root_combined_with_root():
    try:
        _run_cli([
            "sample_sigma_hop_fresh_corpus.py",
            "--candidate-graph", "candidate.tsv",
            "--exploratory-graph", "exploratory.tsv",
            "--root", "Fresh_root",
            "--scope-root", "Broad_root",
            "--out", "pairs.tsv",
            "--manifest", "manifest.json",
            "--allow-small-sample",
        ])
    except FreshCorpusError as exc:
        assert "--root and --scope-root are mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected root/scope-root combination to abort")


def test_cli_rejects_all_roots_excluded():
    with tempfile.TemporaryDirectory() as td:
        candidate = os.path.join(td, "candidate.tsv")
        exploratory = os.path.join(td, "exploratory.tsv")
        _write(candidate, _chain_graph(chains=8, depth=5))
        _write(exploratory, "child\tparent\nold_child\told_parent\n")
        try:
            _run_cli([
                "sample_sigma_hop_fresh_corpus.py",
                "--candidate-graph", candidate,
                "--exploratory-graph", exploratory,
                "--root", "Fresh_root",
                "--exclude-root", "Fresh_root",
                "--out", os.path.join(td, "pairs.tsv"),
                "--manifest", os.path.join(td, "manifest.json"),
                "--allow-small-sample",
            ])
        except FreshCorpusError as exc:
            assert "all supplied --root values were excluded" in str(exc)
        else:
            raise AssertionError("expected all-roots-excluded config to abort")


def test_cli_writes_balanced_score_in_and_manifest():
    with tempfile.TemporaryDirectory() as td:
        candidate = os.path.join(td, "candidate.tsv")
        exploratory = os.path.join(td, "exploratory.tsv")
        out = os.path.join(td, "pairs.tsv")
        manifest = os.path.join(td, "manifest.json")
        _write(candidate, _chain_graph(chains=8, depth=5))
        _write(exploratory, "child\tparent\nold_child\told_parent\n")

        _run_cli([
            "sample_sigma_hop_fresh_corpus.py",
            "--candidate-graph", candidate,
            "--exploratory-graph", exploratory,
            "--root", "Fresh_root",
            "--pairs", "10",
            "--hmax", "5",
            "--min-descendants", "10",
            "--out", out,
            "--manifest", manifest,
            "--allow-small-sample",
        ])

        rows = [ln.rstrip("\n").split("\t") for ln in open(out, encoding="utf-8") if not ln.startswith("#")]
        assert len(rows) == 10
        assert all(row[4].startswith("transitive_h") for row in rows)
        hops = sorted(row[4] for row in rows)
        for h in range(1, 6):
            assert hops.count(f"transitive_h{h}") == 2
        with open(manifest, encoding="utf-8") as f:
            m = json.load(f)
        assert m["selected_roots"] == ["Fresh_root"]
        assert m["node_overlap_with_exploratory"] == 0
        assert m["hop_counts"] == {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2}
        assert m["target_hop_counts"] == {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2}
        assert m["selection_rule"] == "user-supplied root validation"
        assert m["hop_semantics"] == "shortest upward graph distance within retained slice"
        assert "node_filter_semantics" in m
        assert "traversal_order" in m


def test_load_edges_skips_header():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write("child\tparent\nx\ty\n")
        path = f.name
    try:
        assert load_edges(path) == [("x", "y")]
    finally:
        os.unlink(path)


def _build_lmdb(path, edges, titles, meta=None):
    try:
        import lmdb
    except ImportError:
        return False
    env = lmdb.open(path, max_dbs=16, map_size=64 * 1024 * 1024, subdir=True)
    cp = env.open_db(b"category_parent", dupsort=True, create=True)
    cc = env.open_db(b"category_child", dupsort=True, create=True)
    title_i2s = env.open_db(b"title_i2s", create=True)
    title_s2i = env.open_db(b"title_s2i", create=True)
    meta_db = env.open_db(b"meta", create=True)
    with env.begin(write=True) as txn:
        for child, parent in edges:
            txn.put(enc_id(child), enc_id(parent), db=cp, dupdata=True)
            txn.put(enc_id(parent), enc_id(child), db=cc, dupdata=True)
        for node_id, title in titles.items():
            raw_title = title.encode("utf-8")
            txn.put(enc_id(node_id), raw_title, db=title_i2s)
            txn.put(raw_title, enc_id(node_id), db=title_s2i)
        txn.put(b"title_layer_kind", b"test_real_titles", db=meta_db)
        for key, value in (meta or {}).items():
            txn.put(key.encode("utf-8"), str(value).encode("utf-8"), db=meta_db)
    env.sync()
    env.close()
    return True


def _lmdb_chain_fixture(path, identity_titles=False, scoped=False):
    root = 1
    edges = []
    titles = {root: "Fresh_root"}
    meta = {}
    if scoped:
        scope = 100
        edges.append((root, scope))
        titles[scope] = "Scope_root"
        meta["scoped_root"] = scope
    next_id = 2
    for i in range(8):
        parent = root
        for d in range(5, 0, -1):
            child = next_id
            next_id += 1
            edges.append((child, parent))
            titles[child] = f"fresh_{i}_{d}"
            parent = child
    if identity_titles:
        titles = {node_id: str(node_id) for node_id in titles}
    return _build_lmdb(path, edges, titles, meta=meta)


def test_cli_writes_from_lmdb_real_title_layer():
    with tempfile.TemporaryDirectory() as td:
        lmdb_dir = os.path.join(td, "candidate.lmdb")
        if not _lmdb_chain_fixture(lmdb_dir):
            print("  skip test_cli_writes_from_lmdb_real_title_layer (python-lmdb unavailable)")
            return
        exploratory = os.path.join(td, "exploratory.tsv")
        out = os.path.join(td, "pairs.tsv")
        manifest = os.path.join(td, "manifest.json")
        _write(exploratory, "child\tparent\nold_child\told_parent\n")

        _run_cli([
            "sample_sigma_hop_fresh_corpus.py",
            "--candidate-lmdb", lmdb_dir,
            "--exploratory-graph", exploratory,
            "--root", "Fresh_root",
            "--pairs", "10",
            "--hmax", "5",
            "--min-descendants", "10",
            "--out", out,
            "--manifest", manifest,
            "--allow-small-sample",
        ])

        rows = [ln.rstrip("\n").split("\t") for ln in open(out, encoding="utf-8") if not ln.startswith("#")]
        assert len(rows) == 10
        assert all(not row[0].isdigit() and not row[1].isdigit() for row in rows)
        with open(manifest, encoding="utf-8") as f:
            m = json.load(f)
        assert m["candidate_source_kind"] == "lmdb"
        assert m["selected_roots"] == ["Fresh_root"]
        assert m["title_i2s_db"] == "title_i2s"
        assert m["title_s2i_db"] == "title_s2i"
        assert m["hop_counts"] == {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2}


def test_cli_uses_lmdb_meta_scoped_root_when_root_omitted():
    with tempfile.TemporaryDirectory() as td:
        lmdb_dir = os.path.join(td, "candidate.lmdb")
        if not _lmdb_chain_fixture(lmdb_dir, scoped=True):
            print("  skip test_cli_uses_lmdb_meta_scoped_root_when_root_omitted (python-lmdb unavailable)")
            return
        exploratory = os.path.join(td, "exploratory.tsv")
        out = os.path.join(td, "pairs.tsv")
        manifest = os.path.join(td, "manifest.json")
        _write(exploratory, "child\tparent\nold_child\told_parent\n")

        _run_cli([
            "sample_sigma_hop_fresh_corpus.py",
            "--candidate-lmdb", lmdb_dir,
            "--exploratory-graph", exploratory,
            "--pairs", "10",
            "--hmax", "5",
            "--min-descendants", "10",
            "--out", out,
            "--manifest", manifest,
            "--allow-small-sample",
        ])

        with open(manifest, encoding="utf-8") as f:
            m = json.load(f)
        assert m["scope_root"] == "Scope_root"
        assert m["scope_root_id"] == 100
        assert m["selected_roots"] == ["Fresh_root"]


def test_cli_rejects_lmdb_identity_numeric_title_layer():
    with tempfile.TemporaryDirectory() as td:
        lmdb_dir = os.path.join(td, "candidate.lmdb")
        if not _lmdb_chain_fixture(lmdb_dir, identity_titles=True):
            print("  skip test_cli_rejects_lmdb_identity_numeric_title_layer (python-lmdb unavailable)")
            return
        exploratory = os.path.join(td, "exploratory.tsv")
        _write(exploratory, "child\tparent\nold_child\told_parent\n")

        try:
            _run_cli([
                "sample_sigma_hop_fresh_corpus.py",
                "--candidate-lmdb", lmdb_dir,
                "--exploratory-graph", exploratory,
                "--root", "1",
                "--pairs", "10",
                "--hmax", "5",
                "--min-descendants", "10",
                "--out", os.path.join(td, "pairs.tsv"),
                "--manifest", os.path.join(td, "manifest.json"),
                "--allow-small-sample",
            ])
        except FreshCorpusError as exc:
            assert "identity numeric" in str(exc)
        else:
            raise AssertionError("expected identity numeric LMDB title layer to abort")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} fresh-corpus sampler tests passed")
