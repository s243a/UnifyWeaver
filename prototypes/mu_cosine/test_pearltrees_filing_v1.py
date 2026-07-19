"""Focused tests for the Pearltrees Filing v1 pipeline (torch-free where possible)."""
import os

import numpy as np
import pytest

from sample_pearltrees_lateral import ancestors, build_graph, sample_lateral


def tiny_forest():
    """Two small account trees exercising sib/cous/rand pools."""
    records = []
    paths = [
        ["r", "a", "a1", "a1x"],
        ["r", "a", "a2"],
        ["r", "b", "b1"],
        ["r", "b", "b2"],
        ["r2", "c", "c1"],
        ["r2", "c", "c2"],
        ["r2", "d", "d1"],
    ]
    for p in paths:
        records.append({"account": "acct", "path_ids": p, "tree_id": p[-1]})
    titles = {nid: f"T_{nid}" for p in paths for nid in p}
    forest = {"records": records, "titles": titles,
              "nodes": {("acct", nid) for p in paths for nid in p}}
    return forest


def test_lateral_sampler_invariants_and_determinism():
    forest = tiny_forest()
    parents, children = build_graph(forest)
    out1 = sample_lateral(forest, parents, children, n_sib=4, n_cous=4, n_rand=4, seed=0, hmax=6)
    out2 = sample_lateral(forest, parents, children, n_sib=4, n_cous=4, n_rand=4, seed=0, hmax=6)
    assert out1 == out2  # deterministic, no RNG state
    for a, b in out1["pt_sib"]:
        assert parents[a] & parents[b], "siblings must share a parent"
        assert b not in ancestors(parents, a, 6) and a not in ancestors(parents, b, 6)
    for a, b in out1["pt_cous"]:
        assert not (parents.get(a, set()) & parents.get(b, set())), "cousins have disjoint parents"
    for a, b in out1["pt_rand"]:
        anc_a, anc_b = ancestors(parents, a, 6), ancestors(parents, b, 6)
        assert b not in anc_a and a not in anc_b
        assert not (parents.get(a, set()) & parents.get(b, set()))
    seen = set()
    for pairs_list in out1.values():
        for p in pairs_list:
            key = tuple(sorted(p))
            assert key not in seen, "strata must not overlap"
            seen.add(key)


def test_lineage_decay_targets():
    from fine_tune_pearltrees_filing import LINEAGE_DECAY

    targets = [LINEAGE_DECAY ** (h - 1) for h in range(1, 6)]
    assert targets[0] == 1.0
    assert all(a > b for a, b in zip(targets, targets[1:]))  # parent > grandparent > far


def test_strict_scored_loader_fails_closed(tmp_path):
    from run_pearltrees_fusion import load_scored_strict

    header = ("#\tnode\troot\tcur_rel\tneighborhood\tjudge\t"
              + "\t".join(f"P[{r}]" for r in ["element_of", "subcategory", "subtopic",
                                              "super_category", "see_also", "assoc",
                                              "unknown", "none"])
              + "\t" + "\t".join(f"mu[{r}]" for r in ["element_of", "subcategory", "subtopic",
                                                      "super_category", "see_also", "assoc"]))
    # the real ingest header starts with '# node' — build a minimal compatible one
    cols = header.lstrip("#\t").split("\t")

    def row(node, root, judge="gpt-5.6-luna"):
        vals = {c: "0.5" for c in cols}
        vals["node"], vals["root"], vals["judge"] = node, root, judge
        vals["cur_rel"], vals["neighborhood"] = "assoc", "pt_rand"
        return "\t".join(vals[c] for c in cols)

    ok = tmp_path / "ok.tsv"
    ok.write_text("# " + "\t".join(cols) + "\n" + row("a", "b") + "\n" + row("c", "d") + "\n")
    out = load_scored_strict(str(ok), "gpt-5.6-luna")
    assert set(out) == {("a", "b"), ("c", "d")}

    dup = tmp_path / "dup.tsv"
    dup.write_text("# " + "\t".join(cols) + "\n" + row("a", "b") + "\n" + row("a", "b") + "\n")
    with pytest.raises(SystemExit, match="duplicate"):
        load_scored_strict(str(dup), "gpt-5.6-luna")

    wrong = tmp_path / "wrong.tsv"
    wrong.write_text("# " + "\t".join(cols) + "\n" + row("a", "b", judge="gpt-5.5-low") + "\n")
    with pytest.raises(SystemExit, match="judge"):
        load_scored_strict(str(wrong), "gpt-5.6-luna")

    short = tmp_path / "short.tsv"
    short.write_text("# " + "\t".join(cols) + "\n" + "a\tb\ttruncated\n")
    with pytest.raises(SystemExit, match="short row"):
        load_scored_strict(str(short), "gpt-5.6-luna")


def test_alias_equivalence_ranking():
    import torch

    from eval_pearltrees_filing import ranks_from

    M = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    # true folder is column 2, but column 0 shares its title: best alias rank wins
    assert ranks_from(M, [[0, 2]]) == [1]
    assert ranks_from(M, [[2]]) == [3]


def test_escalation_curve_reports_kept_counts():
    import torch

    from eval_pearltrees_filing import escalation_curve

    M = torch.tensor([[0.9, 0.1], [0.6, 0.55], [0.8, 0.2]])
    rows = escalation_curve(M, [[0], [0], [1]], thresholds=(0.0, 0.1))
    t0, routed0, kept_n0, r1_0 = rows[0]
    assert kept_n0 == 3 and routed0 == 0.0
    t1, routed1, kept_n1, r1_1 = rows[1]
    assert kept_n1 == 2  # the 0.05-margin row routes at t=0.1
    assert abs(r1_1 - 0.5) < 1e-6  # of the kept: one hit (row0), one miss (row2 true col1)


def test_filing_ranker_boundary_accounting():
    """Every cut edge is accounted for: titled exterior neighbors → semantic shunts, untitled →
    recorded topological-c0 fallback; nothing silently dropped (audit rounds 1+2)."""
    from filing_ranker import load_graph_universe

    universe, titles, neighbors, _, _, _, cut_ext, cut_ext_untitled = load_graph_universe(hops=2)
    uset = set(universe)
    # rebuild the raw adjacency to compare totals
    from collections import defaultdict
    import os
    adj = defaultdict(set)
    dag = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", ".local", "data", "pearltrees_api", "assembled_dag.tsv")
    for ln in open(dag, encoding="utf-8"):
        p, c = ln.split()
        adj[p].add(c)
        adj[c].add(p)
    for n in universe:
        ext = adj[n] - uset
        titled = set(cut_ext.get(n, []))
        n_untitled = cut_ext_untitled.get(n, 0)
        assert titled <= ext
        assert len(ext) == len(titled) + n_untitled, f"cut edges unaccounted at node {n}"
        assert all(x in titles for x in titled)
    total_untitled = sum(cut_ext_untitled.values())
    assert total_untitled > 0  # the 97 the audit found — must be recorded, not dropped
