#!/usr/bin/env python3
"""Tests for outcome-blind geometry inventory summaries."""
import numpy as np
import pytest

from run_graph_geometry_inventory import summarize_geometry_inventory


def test_inventory_uses_only_within_descendant_off_diagonal_pairs():
    pairs = (("x", "a"), ("x", "b"), ("x", "c"), ("y", "a"), ("y", "c"))
    base = np.eye(len(pairs))
    graph = base.copy()
    graph[0, 1] = graph[1, 0] = 0.8
    graph[0, 2] = graph[2, 0] = 0.2
    graph[1, 2] = graph[2, 1] = 0.4
    graph[3, 4] = graph[4, 3] = 0.2
    semantic = base.copy()
    semantic[0, 1] = semantic[1, 0] = 0.1
    semantic[0, 2] = semantic[2, 0] = 0.9
    semantic[1, 2] = semantic[2, 1] = 0.5
    semantic[3, 4] = semantic[4, 3] = 0.9
    kernels = {
        "graph_closed": graph,
        "graph_walk_same_hop": graph,
        "graph_walk_cumulative": graph,
        "minilm": semantic,
        "nomic": semantic,
        "shared_e5": semantic,
    }
    neighbors = {"a": {"b"}, "b": {"a"}, "c": set()}
    summary = summarize_geometry_inventory(pairs, kernels, neighbors)
    assert summary["within_descendant_row_pairs"] == 4
    assert summary["directly_adjacent_root_pairs"] == 1
    assert summary["distance_pearson"]["graph_walk_cumulative"]["graph_closed"] == pytest.approx(1.0)
    assert summary["distance_spearman"]["minilm"]["nomic"] == pytest.approx(1.0)
    assert summary["mean_distance_by_graph_adjacency"]["graph_walk_cumulative"][
        "adjacent_mean_distance"
    ] == pytest.approx(0.2)


def test_inventory_rejects_misaligned_or_no_within_descendant_kernels():
    with pytest.raises(ValueError, match="align"):
        summarize_geometry_inventory(
            (("x", "a"), ("x", "b")),
            {name: np.eye(3) for name in (
                "graph_closed", "graph_walk_same_hop", "graph_walk_cumulative",
                "minilm", "nomic", "shared_e5"
            )},
            {},
        )
    with pytest.raises(ValueError, match="within-descendant"):
        summarize_geometry_inventory(
            (("x", "a"), ("y", "b")),
            {name: np.eye(2) for name in (
                "graph_closed", "graph_walk_same_hop", "graph_walk_cumulative",
                "minilm", "nomic", "shared_e5"
            )},
            {},
        )
