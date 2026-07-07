#!/usr/bin/env python3
"""Unit checks for the preregistered Σ(hop) confirmatory runner.

These tests stay synthetic: they validate the frozen mechanics without needing scored LLM data, e5 caches, or a
torch model checkpoint.
"""

import os
import tempfile
from argparse import Namespace

import numpy as np

from sigma_hop_confirmatory import (
    ConfirmatoryData,
    assert_no_node_overlap,
    descendant_disjoint_split,
    permutation_test,
    validate_preregistered_cli,
    valid_splits,
)


def _synthetic_data(n_desc=24, seed=0):
    rng = np.random.default_rng(seed)
    pairs, hop, D, S = [], [], [], []
    for d in range(n_desc):
        desc = f"desc_{d}"
        for h in range(1, 6):
            pairs.append((desc, f"ancestor_{d}_{h}"))
            hop.append(h)
            scale = 0.05 + 0.025 * h
            D.append(0.5 + rng.normal(0, scale))
            S.append(0.4 + rng.normal(0, scale * (1.2 if h >= 4 else 0.8)))
    hop = np.array(hop)
    D = np.array(D)
    S = np.array(S)
    X = np.ones((len(pairs), 1))
    return ConfirmatoryData(pairs=pairs, hop=hop, D=D, S=S, X=X)


def test_descendant_disjoint_split_keeps_all_hops_together():
    data = _synthetic_data(n_desc=8)
    train, held = descendant_disjoint_split(data.pairs, seed=3)
    train_desc = {data.pairs[i][0] for i in train}
    held_desc = {data.pairs[i][0] for i in held}
    assert train_desc.isdisjoint(held_desc)
    assert len(train) + len(held) == len(data.pairs)


def test_assert_no_node_overlap_blocks_exploratory_nodes():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write("old_child\told_parent\n")
        path = f.name
    try:
        try:
            assert_no_node_overlap([("fresh_child", "old_parent")], path)
        except SystemExit as exc:
            assert "overlap exploratory graph nodes" in str(exc)
        else:
            raise AssertionError("expected overlap check to abort")
    finally:
        os.unlink(path)


def test_permutation_result_reports_preregistered_decision_fields():
    data = _synthetic_data()
    splits, skipped = valid_splits(data, range(4), min_train=30, min_held=12)
    assert len(splits) == 4
    assert skipped == []
    result = permutation_test(data, splits, k=5, seed=2)
    for key in [
        "mean_gain",
        "constant_nll",
        "sigma_hop_nll",
        "null_mean",
        "null_p95",
        "permutation_k",
        "permutation_p",
        "confirmed",
    ]:
        assert key in result
    assert result["permutation_k"] == 5
    assert 0 < result["permutation_p"] <= 1


def test_cli_guard_rejects_non_preregistered_split_protocol():
    args = Namespace(splits=20, held_frac=0.30, min_train=30, min_held=12, permutations=1000)
    try:
        validate_preregistered_cli(args)
    except SystemExit as exc:
        assert "non-preregistered split protocol" in str(exc)
    else:
        raise AssertionError("expected non-preregistered protocol to abort")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} sigma-hop confirmatory tests passed")
