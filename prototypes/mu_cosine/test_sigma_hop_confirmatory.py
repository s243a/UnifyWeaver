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
    ConfirmatoryInputError,
    OverlapError,
    assert_no_node_overlap,
    descendant_disjoint_split,
    load_scored_pairs,
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
    n = len(pairs)
    X = np.column_stack([
        rng.uniform(0.1, 0.9, n),
        rng.uniform(0.1, 0.9, n),
        rng.uniform(0.0, 1.0, n),
        np.ones(n),
    ])
    return ConfirmatoryData(pairs=tuple(pairs), hop=hop, D=D, S=S, X=X)


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
        except OverlapError as exc:
            assert "overlap exploratory graph nodes" in str(exc)
        else:
            raise AssertionError("expected overlap check to abort")
    finally:
        os.unlink(path)


def test_assert_no_node_overlap_allows_clean_pairs():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write("old_child\told_parent\n")
        path = f.name
    try:
        assert assert_no_node_overlap((("fresh_child", "fresh_parent"),), path) == []
    finally:
        os.unlink(path)


def test_permutation_result_reports_preregistered_decision_fields():
    data = _synthetic_data()
    splits, skipped = valid_splits(data, range(4), min_train=30, min_held=12)
    assert len(splits) == 4
    assert skipped == []
    result = permutation_test(data, splits, k=5, seed=2, allow_small_k=True)
    for key in [
        "mean_gain",
        "constant_nll",
        "sigma_hop_nll",
        "null_mean",
        "null_p95",
        "permutation_k",
        "permutation_p",
        "decision_inputs",
        "confirmed",
        "decision",
    ]:
        assert key in result
    assert result["permutation_k"] == 5
    assert 0 < result["permutation_p"] <= 1


def test_permutation_test_rejects_small_k_by_default():
    data = _synthetic_data()
    splits, _ = valid_splits(data, range(4), min_train=30, min_held=12)
    try:
        permutation_test(data, splits, k=999, seed=2)
    except ValueError as exc:
        assert "k >= 1000" in str(exc)
    else:
        raise AssertionError("expected confirmatory permutation K guard to abort")


def test_load_scored_pairs_reports_bad_hop_with_context():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as score, tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False
    ) as resp:
        score.write("child\tparent\tx\ty\ttransitive_hx\n")
        resp.write(
            '[{"id": 0, "subcategory": {"mu_fwd": 0.7}, "assoc": {"mu": 0.2}}]'
        )
        score_path, resp_path = score.name, resp.name
    try:
        try:
            load_scored_pairs(score_path, resp_path)
        except ConfirmatoryInputError as exc:
            assert "cannot parse hop count" in str(exc)
            assert "row 0" in str(exc)
        else:
            raise AssertionError("expected malformed hop to abort")
    finally:
        os.unlink(score_path)
        os.unlink(resp_path)


def test_cli_guard_rejects_non_preregistered_split_protocol():
    args = Namespace(splits=20, held_frac=0.30, min_train=30, min_held=12, permutations=1000)
    try:
        validate_preregistered_cli(args)
    except SystemExit as exc:
        assert "non-preregistered split protocol" in str(exc)
    else:
        raise AssertionError("expected non-preregistered protocol to abort")


def test_cli_guard_rejects_too_few_permutations():
    args = Namespace(splits=40, held_frac=0.30, min_train=30, min_held=12, permutations=999)
    try:
        validate_preregistered_cli(args)
    except SystemExit as exc:
        assert "permutations must be >= 1000" in str(exc)
    else:
        raise AssertionError("expected too-few permutations to abort")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} sigma-hop confirmatory tests passed")
