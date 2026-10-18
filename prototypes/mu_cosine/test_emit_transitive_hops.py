#!/usr/bin/env python3
"""Determinism checks for graph-native hit probability."""

import pytest

from emit_transitive_hops import hit_prob


def test_hit_prob_is_invariant_to_parent_container_order_with_cycle_guard():
    # The a<->b cycle exercises the documented in-progress memo guard. Without canonical parent
    # traversal, merely reversing x's parent list changes which branch encounters the guard first.
    forward = {"x": ["a", "b"], "a": ["b", "target"], "b": ["a"], "target": []}
    reversed_lists = {node: list(reversed(parents)) for node, parents in forward.items()}
    set_backed = {node: set(parents) for node, parents in forward.items()}

    expected = hit_prob(forward, "x", "target")
    assert expected == pytest.approx(0.25)
    assert hit_prob(reversed_lists, "x", "target") == pytest.approx(expected)
    assert hit_prob(set_backed, "x", "target") == pytest.approx(expected)
