#!/usr/bin/env python3
"""Focused tests for the post-#3648 JointPosterior decision-space comparison."""
import numpy as np

from run_cheap_judge_joint_posterior import (
    CLASSES,
    aggregate_decision_probabilities,
    component_bootstrap_aurc,
    endpoint_components,
    gaussian_bridge_proba,
    macro_decision,
    pool_relation_values,
)
from node_disjoint_eval import node_disjoint_pair_split


def test_macro_decision_aggregation_is_normalised_and_ordered():
    names = [
        "P[element_of]", "P[subcategory]", "P[subtopic]", "P[super_category]",
        "P[see_also]", "P[assoc]", "P[unknown]", "P[none]",
    ]
    col = {name: i for i, name in enumerate(names)}
    row = ["0.10", "0.20", "0.05", "0.05", "0.20", "0.10", "0.20", "0.10"]
    p = aggregate_decision_probabilities(row, col)
    assert CLASSES == ("directional", "symmetric", "other")
    assert np.allclose(p, [0.40, 0.30, 0.30])
    assert np.isclose(p.sum(), 1.0)


def test_macro_decision_persists_margin_and_resolves_ties_in_class_order():
    decision, margin, tied = macro_decision([0.4, 0.4, 0.2])
    assert decision == "directional"
    assert margin == 0.0
    assert tied
    decision, margin, tied = macro_decision([0.1, 0.7, 0.2])
    assert decision == "symmetric"
    assert np.isclose(margin, 0.5)
    assert not tied


def test_within_judge_pooling_is_bounded_and_not_expert_selection():
    mu = np.array([0.1, 0.5, 0.9])
    probability = np.array([0.7, 0.2, 0.1])
    hard, prob_weighted, soft = pool_relation_values(mu, probability, temperature=0.1)
    assert hard == 0.9
    assert np.isclose(prob_weighted, 0.26)
    assert mu.min() <= soft <= mu.max()
    assert soft > prob_weighted  # different within-judge reductions of the same three relation values


def test_audited_node_split_has_no_endpoint_leakage_and_is_deterministic():
    pairs = [(f"n{i}", f"n{(i + 1) % 20}") for i in range(20)] + [
        (f"n{i}", f"n{(i + 7) % 20}") for i in range(20)
    ]
    strata = ["even" if i % 2 == 0 else "odd" for i in range(len(pairs))]
    split1 = node_disjoint_pair_split(
        pairs, 7, held_node_fraction=0.40, strata=strata, candidates=16,
    )
    split2 = node_disjoint_pair_split(
        pairs, 7, held_node_fraction=0.40, strata=strata, candidates=16,
    )
    assert np.array_equal(split1.train, split2.train)
    assert np.array_equal(split1.held, split2.held)
    assert split1.selected_candidate == split2.selected_candidate
    train_nodes = {n for i in split1.train for n in pairs[i]}
    held_nodes = {n for i in split1.held for n in pairs[i]}
    assert train_nodes.isdisjoint(held_nodes)
    assert len(split1.cross) > 0


def test_endpoint_components_keep_shared_node_rows_together():
    pairs = [("a", "b"), ("b", "c"), ("x", "y"), ("q", "r"), ("r", "s")]
    components = sorted((sorted(x.tolist()) for x in endpoint_components(pairs)), key=lambda x: x[0])
    assert components == [[0, 1], [2], [3, 4]]


class _Bridge:
    def proba(self, X):
        X = np.asarray(X, float)
        logits = np.column_stack([X[:, 0], X[:, 1], np.zeros(len(X))])
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)


def test_gaussian_bridge_zero_covariance_equals_point_bridge():
    means = np.array([[0.2, 0.8], [0.7, 0.1]])
    cov = np.zeros((2, 2, 2))
    got = gaussian_bridge_proba(_Bridge(), means, cov, order=5)
    assert np.allclose(got, _Bridge().proba(means), atol=1e-12)


def test_gaussian_bridge_nonzero_covariance_is_finite_and_normalised():
    means = np.array([[0.2, 0.8], [0.7, 0.1]])
    cov = np.array([[[0.2, 0.05], [0.05, 0.1]], [[0.1, -0.02], [-0.02, 0.3]]])
    got = gaussian_bridge_proba(_Bridge(), means, cov, order=7)
    assert np.isfinite(got).all()
    assert np.all(got >= 0)
    assert np.allclose(got.sum(axis=1), 1.0)


def test_component_bootstrap_reports_blocks_not_rows():
    pairs = [("a", "b"), ("b", "c"), ("x", "y"), ("q", "r")]
    proba = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1],
                      [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    labels = np.array([0, 0, 1, 2])
    point, lo, hi, n_blocks, largest = component_bootstrap_aurc(
        proba, labels, pairs, B=50, seed=4,
    )
    assert n_blocks == 3 and largest == 2
    assert 0.0 <= lo <= point <= hi <= 1.0


def _run_all():
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} cheap-judge JointPosterior tests passed")


if __name__ == "__main__":
    _run_all()
