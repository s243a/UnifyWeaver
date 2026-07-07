#!/usr/bin/env python3
"""Synthetic checks for the fresh-corpus Sigma(hop) sampler."""

import json
import os
import tempfile

from sample_sigma_hop_fresh_corpus import (
    FreshCorpusError,
    filter_candidate_edges,
    load_edges,
    main,
    node_block,
    sample_balanced_pairs,
)


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
    filtered, removed = filter_candidate_edges(candidate, node_block(exploratory))
    assert filtered == [("fresh_child", "fresh_parent")]
    assert removed["overlap_edges"] == 1


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


def test_cli_rejects_toy_sample_without_explicit_opt_in():
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "sample_sigma_hop_fresh_corpus.py",
            "--candidate-graph", "candidate.tsv",
            "--exploratory-graph", "exploratory.tsv",
            "--pairs", "10",
            "--out", "pairs.tsv",
            "--manifest", "manifest.json",
        ]
        try:
            main()
        except FreshCorpusError as exc:
            assert "at least 250 pairs" in str(exc)
        else:
            raise AssertionError("expected toy sample to require --allow-small-sample")
    finally:
        sys.argv = old_argv


def test_cli_writes_balanced_score_in_and_manifest():
    with tempfile.TemporaryDirectory() as td:
        candidate = os.path.join(td, "candidate.tsv")
        exploratory = os.path.join(td, "exploratory.tsv")
        out = os.path.join(td, "pairs.tsv")
        manifest = os.path.join(td, "manifest.json")
        _write(candidate, _chain_graph(chains=8, depth=5))
        _write(exploratory, "child\tparent\nold_child\told_parent\n")

        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
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
            ]
            main()
        finally:
            sys.argv = old_argv

        rows = [ln.rstrip("\n").split("\t") for ln in open(out, encoding="utf-8") if not ln.startswith("#")]
        assert len(rows) == 10
        assert all(row[4].startswith("transitive_h") for row in rows)
        hops = sorted(row[4] for row in rows)
        assert hops.count("transitive_h1") == 2
        assert hops.count("transitive_h5") == 2
        m = json.load(open(manifest, encoding="utf-8"))
        assert m["selected_roots"] == ["Fresh_root"]
        assert m["node_overlap_with_exploratory"] == 0
        assert m["hop_counts"] == {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2}


def test_load_edges_skips_header():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write("child\tparent\nx\ty\n")
        path = f.name
    try:
        assert load_edges(path) == [("x", "y")]
    finally:
        os.unlink(path)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} fresh-corpus sampler tests passed")
