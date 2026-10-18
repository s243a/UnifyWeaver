#!/usr/bin/env python3
"""Focused synthetic tests for node-disjoint validation and node-block CIs."""

import numpy as np

from node_disjoint_eval import (
    format_split_diagnostics,
    node_disjoint_pair_split,
    paired_node_bootstrap_ci,
)


def _dense_pairs(n=10):
    pairs, strata = [], []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((f"n{i}", f"n{j}"))
            strata.append("even" if (i + j) % 2 == 0 else "odd")
    return pairs, strata


def test_node_split_is_strictly_disjoint_and_drops_cross_pairs():
    pairs, strata = _dense_pairs()
    split = node_disjoint_pair_split(
        pairs,
        7,
        held_node_fraction=0.30,
        strata=strata,
        candidates=32,
    )
    assert split.train_nodes.isdisjoint(split.held_nodes)
    assert len(split.train) + len(split.held) + len(split.cross) == len(pairs)
    for i in split.train:
        assert set(pairs[i]) <= split.train_nodes
    for i in split.held:
        assert set(pairs[i]) <= split.held_nodes
    for i in split.cross:
        assert len(set(pairs[i]) & split.held_nodes) == 1


def test_node_split_is_seeded_and_input_order_invariant():
    pairs, strata = _dense_pairs()
    a = node_disjoint_pair_split(pairs, 19, strata=strata, candidates=24)
    b = node_disjoint_pair_split(pairs, 19, strata=strata, candidates=24)
    assert a.held_nodes == b.held_nodes
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.held, b.held)
    assert np.array_equal(a.cross, b.cross)

    order = np.random.default_rng(3).permutation(len(pairs))
    shuffled_pairs = [pairs[i] for i in order]
    shuffled_strata = [strata[i] for i in order]
    c = node_disjoint_pair_split(shuffled_pairs, 19, strata=shuffled_strata, candidates=24)
    assert a.held_nodes == c.held_nodes


def test_default_fraction_targets_thirty_percent_of_retained_pairs():
    pairs, strata = _dense_pairs(50)
    split = node_disjoint_pair_split(pairs, 0, strata=strata, candidates=1)
    retained_held_share = len(split.held) / (len(split.train) + len(split.held))
    assert np.isclose(len(split.held_nodes) / 50, 0.40)
    assert 0.29 < retained_held_share < 0.32, retained_held_share


def test_runner_and_helper_defaults_are_aligned():
    from run_sym_channel_fusion import build_arg_parser

    args = build_arg_parser().parse_args([])
    assert np.isclose(args.held_node_frac, 0.40)
    pairs, strata = _dense_pairs(10)
    split = node_disjoint_pair_split(pairs, 0, strata=strata, candidates=1)
    assert np.isclose(len(split.held_nodes) / 10, args.held_node_frac)


def test_candidate_search_never_worsens_stratum_coverage_deficit():
    pairs, strata = _dense_pairs(12)
    first = node_disjoint_pair_split(pairs, 4, strata=strata, candidates=1, minimum_per_stratum=3)
    searched = node_disjoint_pair_split(pairs, 4, strata=strata, candidates=100, minimum_per_stratum=3)

    def deficit(split):
        return sum(max(0, 3 - c.train) + max(0, 3 - c.held) for c in split.strata.values())

    assert deficit(searched) <= deficit(first)
    for counts in searched.strata.values():
        assert counts.train + counts.held + counts.cross == counts.total


def test_split_diagnostics_expose_partition_and_stratum_coverage():
    pairs, strata = _dense_pairs()
    split = node_disjoint_pair_split(pairs, 2, strata=strata, candidates=8)
    rendered = format_split_diagnostics(split)
    assert "pairs train/held/cross=" in rendered
    assert "retained" in rendered
    assert "even" in rendered and "odd" in rendered


def test_node_bootstrap_constant_gain_has_degenerate_interval():
    pairs = [("hub", f"leaf{i}") for i in range(8)]
    result = paired_node_bootstrap_ci(pairs, np.full(len(pairs), 0.25), n_resamples=300, seed=5)
    assert np.isclose(result.estimate, 0.25)
    assert np.isclose(result.low, 0.25)
    assert np.isclose(result.high, 0.25)
    assert result.n_resamples == 300
    assert result.n_attempts >= result.n_resamples


def test_node_bootstrap_is_deterministic_and_row_order_invariant():
    pairs = [("hub", f"leaf{i}") for i in range(8)] + [("leaf0", "leaf1"), ("leaf2", "leaf3")]
    gains = np.linspace(-0.2, 0.5, len(pairs))
    a = paired_node_bootstrap_ci(pairs, gains, n_resamples=400, seed=23, confidence=0.90)
    b = paired_node_bootstrap_ci(pairs, gains, n_resamples=400, seed=23, confidence=0.90)
    assert a == b

    order = np.random.default_rng(11).permutation(len(pairs))
    c = paired_node_bootstrap_ci(
        [pairs[i] for i in order],
        gains[order],
        n_resamples=400,
        seed=23,
        confidence=0.90,
    )
    assert np.isclose(a.estimate, c.estimate)
    assert np.isclose(a.low, c.low)
    assert np.isclose(a.high, c.high)


def test_invalid_split_and_bootstrap_inputs_fail_loudly():
    for fn in (
        lambda: node_disjoint_pair_split([], 0),
        lambda: node_disjoint_pair_split([("a", "b")], 0, held_node_fraction=1.0),
        lambda: node_disjoint_pair_split([("a", "b")], 0, strata=[]),
        lambda: paired_node_bootstrap_ci([("a", "b")], [], n_resamples=10),
        lambda: paired_node_bootstrap_ci([("a", "b")], [0.1], confidence=0.0),
    ):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid input to raise ValueError")


def _run_all():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
