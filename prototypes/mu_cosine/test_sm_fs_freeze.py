"""Fixture tests for sm_fs_freeze.py: privacy fallback, split isolation, duplicates, reproduction."""
import json
import os

import sm_fs_freeze


def make_tree(tmp_path):
    tmp_path = tmp_path / "tree"          # corpus root separate from test output dirs
    for p, maps in {
        "Subjects/sci/phy/mech": ["Newton", "Lagrange"],
        "Subjects/sci/phy/quantum": ["Qubits"],
        "Subjects/art/music": ["Jazz", "Blues"],
        "Subjects/art/paint": ["Oil"],
        "Private/journal": ["Secrets"],
        "Subjects/sci/private_notes": ["Hidden"],
        "Other/misc": ["Jazz"],                       # duplicate leaf title in another path
    }.items():
        d = tmp_path / p
        d.mkdir(parents=True, exist_ok=True)
        for m in maps:
            (d / f"{m}.smmx").write_text("<x/>")
    return tmp_path


def run(tmp_path, out):
    sm_fs_freeze.main(["--fs-root", str(tmp_path), "--out-dir", str(out),
                       "--holdout-frac", "0.4", "--split-seed", "0"])
    return json.load(open(out / "ledger.json"))


def test_privacy_fallback_and_split_isolation(tmp_path):
    led = run(make_tree(tmp_path), tmp_path / "out")
    paths = [r["map_path"] for r in led["rows"]]
    assert not any("Private/" in p or "private_notes" in p for p in paths)   # fallback filter
    assert led["privacy_source"] == "fallback_substring"
    # split isolation: no explore destination inside any reserved subtree block
    res = {r["dest"] for r in led["rows"] if r["split"] == "reserved"}
    exp = {r["dest"] for r in led["rows"] if r["split"] == "explore"}
    for e in exp:
        assert not any(e == d or e.startswith(d + "/") for d in res)
    # all maps of one destination share a split side
    side = {}
    for r in led["rows"]:
        assert side.setdefault(r["dest"], r["split"]) == r["split"]


def test_duplicate_titles_stay_distinct_and_targets_explore_only(tmp_path):
    out = tmp_path / "out"
    led = run(make_tree(tmp_path), out)
    jazz = [r for r in led["rows"] if r["title"] == "Jazz"]
    assert len(jazz) == 2 and len({r["map_path"] for r in jazz}) == 2       # exact-path identity
    hdr = open(out / "lineage_fs_targets.tsv").read().splitlines()
    assert "lineage(fs,decay=0.85)" in hdr[0]                                # process expression
    explore_titles = {r["title"] for r in led["rows"] if r["split"] == "explore"}
    for ln in hdr:
        if ln.startswith("#"):
            continue
        assert ln.split("\t")[0] in explore_titles                           # explore-only targets


def test_reproducible_membership(tmp_path):
    t = make_tree(tmp_path)
    l1 = run(t, tmp_path / "o1")
    l2 = run(t, tmp_path / "o2")
    m1 = {(r["map_path"], r["split"]) for r in l1["rows"]}
    assert m1 == {(r["map_path"], r["split"]) for r in l2["rows"]}           # identical membership
    assert l1["tree_snapshot"] == l2["tree_snapshot"]
    assert l1["e5_revision"] == l2["e5_revision"]
