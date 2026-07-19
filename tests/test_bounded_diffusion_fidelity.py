#!/usr/bin/env python3
"""Tests for outcome-blind bounded-domain fidelity selectors and closure."""

from pathlib import Path
import math
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from unifyweaver.graph.bounded_diffusion_fidelity import (  # noqa: E402
    ExperimentalBoundaryClosure,
    ExperimentalBoundaryClosureConfig,
    ExteriorTraversalLimitError,
    ProtectedSetCoverageError,
    build_experimental_boundary_closure,
    discover_exterior_components,
    ensure_matched_budget,
    evaluate_bounded_domain_fidelity,
    select_hop_budget_domain,
    select_semantic_resistance_domain,
    select_topology_skeleton_domain,
    select_union_hop_reference,
)
from unifyweaver.graph.leaky_diffusion import (  # noqa: E402
    build_grounded_semantic_diffusion,
)
from unifyweaver.graph.local_diffusion import (  # noqa: E402
    build_local_grounded_semantic_diffusion,
)
from prototypes.mu_cosine.benchmark_bounded_diffusion_fidelity import (  # noqa: E402
    _exact_two_port_exterior_dtn,
)


def _neighbors(edges, extra=()):
    output = {node: set() for node in extra}
    for left, right in edges:
        output.setdefault(left, set()).add(right)
        output.setdefault(right, set()).add(left)
    return {
        node: tuple(sorted(adjacent, key=repr))
        for node, adjacent in output.items()
    }


def _path(last):
    return _neighbors((node, node + 1) for node in range(last))


def test_hop_budget_selection_is_a_deterministic_prefix_with_stable_provenance():
    graph = _neighbors(
        (("s", "a"), ("s", "b"), ("a", "c"), ("b", "d"), ("c", "e"))
    )
    small = select_hop_budget_domain(("s",), graph, maximum_nodes=4)
    repeated = select_hop_budget_domain(("s",), graph, maximum_nodes=4)
    large = select_hop_budget_domain(("s",), graph, maximum_nodes=5)

    assert small.domain.nodes == repeated.domain.nodes
    assert small.selection_fingerprint == repeated.selection_fingerprint
    assert set(small.domain.nodes).issubset(large.domain.nodes)
    assert small.closure_policy == "none_full_dirichlet_beta"
    assert small.provider_calls == len(small.domain.nodes)


def test_diamond_revisit_is_convergence_not_an_artificial_cut():
    graph = _neighbors(
        (("s", "a"), ("s", "b"), ("a", "t"), ("b", "t"))
    )
    selection = select_hop_budget_domain(("s",), graph, maximum_nodes=4)
    model = build_local_grounded_semantic_diffusion(
        selection.domain, intrinsic_leakage_conductance=0.2
    )

    assert set(selection.domain.nodes) == {"s", "a", "b", "t"}
    assert np.array_equal(model.cut_conductance, np.zeros(4))


def test_exterior_diamond_collision_stays_one_two_port_component():
    graph = _neighbors(
        (("p", "a"), ("p", "b"), ("a", "t"), ("b", "t"), ("t", "q"))
    )
    domain = select_hop_budget_domain(
        ("p", "q"), graph, maximum_nodes=2
    ).domain
    first = discover_exterior_components(domain, graph)
    repeated = discover_exterior_components(domain, graph)

    assert first.component_count == 1
    assert first.cut_edge_count == 3
    assert first.discovery_fingerprint == repeated.discovery_fingerprint
    component = first.components[0]
    assert component.nodes == ("a", "b", "t")
    assert component.ports == ("p", "q")
    assert component.internal_edge_count == 2
    assert component.cut_edges == (("p", "a"), ("p", "b"), ("q", "t"))


def test_exterior_discovery_identifies_simple_two_port_component():
    graph = _neighbors((("p", "x"), ("x", "q")))
    domain = select_hop_budget_domain(
        ("p", "q"), graph, maximum_nodes=2
    ).domain
    discovery = discover_exterior_components(domain, graph)

    assert discovery.component_count == 1
    assert discovery.components[0].nodes == ("x",)
    assert discovery.components[0].ports == ("p", "q")
    assert discovery.components[0].cut_edges == (("p", "x"), ("q", "x"))


def test_exterior_discovery_keeps_multi_port_component_joint():
    graph = _neighbors((("p", "x"), ("q", "x"), ("r", "x")))
    domain = select_hop_budget_domain(
        ("p", "q", "r"), graph, maximum_nodes=3
    ).domain
    discovery = discover_exterior_components(domain, graph)

    assert discovery.component_count == 1
    assert discovery.components[0].ports == ("p", "q", "r")
    assert discovery.components[0].cut_edges == (
        ("p", "x"),
        ("q", "x"),
        ("r", "x"),
    )


def test_exterior_discovery_cap_fails_closed_without_frontier_grounding():
    graph = _neighbors((("p", "x"), ("x", "y"), ("y", "q")))
    domain = select_hop_budget_domain(
        ("p", "q"), graph, maximum_nodes=2
    ).domain
    with pytest.raises(ExteriorTraversalLimitError) as error:
        discover_exterior_components(
            domain, graph, maximum_component_nodes=1
        )

    assert error.value.maximum_component_nodes == 1
    assert error.value.visited_nodes == ("x",)
    assert error.value.blocked_neighbor == "y"
    assert "remain grounded" in str(error.value)


def test_topology_skeleton_retains_common_ancestry_before_deterministic_fill():
    graph = _neighbors(
        (
            ("a", "p"),
            ("b", "p"),
            ("p", "g"),
            ("a", "x"),
            ("b", "y"),
            ("g", "z"),
        )
    )
    parents = {"a": ("p",), "b": ("p",), "p": ("g",)}
    small = select_topology_skeleton_domain(
        ("a", "b"), graph, parents, maximum_nodes=5, ancestor_depth=2
    )
    large = select_topology_skeleton_domain(
        ("a", "b"), graph, parents, maximum_nodes=6, ancestor_depth=2
    )

    assert {"a", "b", "p", "g"}.issubset(small.domain.nodes)
    assert small.structural_nodes == 4
    assert small.shared_parent_nodes == 1
    assert set(small.domain.nodes).issubset(large.domain.nodes)
    assert small.domain.anchors == ("a", "b")


def test_topology_skeleton_fails_closed_on_budget_or_non_topological_parent():
    graph = _neighbors((("a", "p"), ("p", "g")))
    parents = {"a": ("p",), "p": ("g",)}
    with pytest.raises(ValueError, match="skeleton exceeds"):
        select_topology_skeleton_domain(
            ("a",), graph, parents, maximum_nodes=2, ancestor_depth=2
        )
    with pytest.raises(ValueError, match="absent from incident adjacency"):
        select_topology_skeleton_domain(
            ("a",), graph, {"a": ("missing",)}, maximum_nodes=2
        )


def test_semantic_resistance_selects_only_existing_low_resistance_paths():
    graph = _neighbors((("s", "a"), ("s", "b"), ("b", "c"), ("a", "d")))
    embeddings = {
        "s": np.array([0.0]),
        "a": np.array([10.0]),
        "b": np.array([0.1]),
        "c": np.array([0.2]),
        "d": np.array([10.1]),
    }
    small = select_semantic_resistance_domain(
        ("s",),
        graph,
        embeddings,
        maximum_nodes=3,
        length_scale=1.0,
        conductance_floor=0.1,
    )
    large = select_semantic_resistance_domain(
        ("s",),
        graph,
        embeddings,
        maximum_nodes=4,
        length_scale=1.0,
        conductance_floor=0.1,
    )

    assert set(small.domain.nodes) == {"s", "b", "c"}
    assert set(small.domain.nodes).issubset(large.domain.nodes)
    assert small.domain.selection_metric == "semantic_resistance"
    assert "a" not in small.domain.nodes
    with pytest.raises(ValueError, match="absent from embeddings"):
        select_semantic_resistance_domain(
            ("s",),
            graph,
            {"s": np.array([0.0])},
            maximum_nodes=2,
            length_scale=1.0,
            conductance_floor=0.1,
        )


def test_matched_budget_validation_does_not_require_cross_family_nesting():
    graph = _path(6)
    embeddings = {node: np.array([float(node)]) for node in graph}
    hop = select_hop_budget_domain((0,), graph, maximum_nodes=4)
    semantic = select_semantic_resistance_domain(
        (0,),
        graph,
        embeddings,
        maximum_nodes=4,
        length_scale=2.0,
        conductance_floor=0.2,
    )
    assert ensure_matched_budget((hop, semantic)) == 4
    with pytest.raises(ValueError, match="same K"):
        ensure_matched_budget(
            (hop, select_hop_budget_domain((0,), graph, maximum_nodes=5))
        )



def test_harness_two_port_strength_is_topology_only_not_embedding_weighted():
    graph = _neighbors((("p", "x"), ("x", "q")))
    selection = select_hop_budget_domain(
        ("p", "q"), graph, maximum_nodes=2
    )
    near_embeddings = {
        "p": np.array([0.0]),
        "q": np.array([0.0]),
        "x": np.array([0.0]),
    }
    far_embeddings = {
        "p": np.array([-100.0]),
        "q": np.array([100.0]),
        "x": np.array([37.0]),
    }
    arguments = {
        "intrinsic_leakage": 0.2,
        "length_scale": 1.0,
        "topology_conductance": 1.0,
        "maximum_pairs": 1,
        "maximum_component_nodes": 8,
    }
    near = _exact_two_port_exterior_dtn(
        selection, graph, near_embeddings, **arguments
    )
    far = _exact_two_port_exterior_dtn(
        selection, graph, far_embeddings, **arguments
    )

    assert near[0] == far[0]
    assert near[1] == far[1]
    assert near[0][0][2] == pytest.approx(1.0 / 2.2)
    assert near[2]["strength_source"] == "topology_only_c0"
    assert near[2]["semantic_role"] == "rank_graph_connected_pairs_only"
    assert near[2]["discovery"]["components"]


def test_exact_component_schur_can_exceed_c0_for_parallel_paths():
    graph = _neighbors(
        (
            ("p", "x1"),
            ("x1", "q"),
            ("p", "x2"),
            ("x2", "q"),
            ("p", "x3"),
            ("x3", "q"),
        )
    )
    selection = select_hop_budget_domain(
        ("p", "q"), graph, maximum_nodes=2
    )
    embeddings = {node: np.array([0.0]) for node in graph}
    pairs, self_return, provenance = _exact_two_port_exterior_dtn(
        selection,
        graph,
        embeddings,
        intrinsic_leakage=0.0,
        length_scale=1.0,
        topology_conductance=1.0,
        maximum_pairs=1,
        maximum_component_nodes=8,
    )
    config = ExperimentalBoundaryClosureConfig(
        maximum_edges=1,
        closure_mass_fraction=1.0,
        ordinary_branch_conductance=1.0,
        bridge_conductance_cap=0.25,
        pair_conductance_source="exact_component_schur",
        ledger_mode="explicit_self_return",
    )
    closure = build_experimental_boundary_closure(
        selection.domain,
        intrinsic_leakage_conductance=0.2,
        pair_conductances=pairs,
        self_return_conductance=self_return,
        config=config,
    )
    full = build_grounded_semantic_diffusion(
        ("p", "q", "x1", "x2", "x3"),
        graph,
        leakage_conductance={
            "p": 0.2,
            "q": 0.2,
            "x1": 0.0,
            "x2": 0.0,
            "x3": 0.0,
        },
    )
    schur = _schur_precision(full, retained=(0, 1), exterior=(2, 3, 4))

    assert pairs == (("p", "q", pytest.approx(1.5)),)
    assert closure.edges[0][2] > config.ordinary_branch_conductance
    assert provenance["approximate_caps_applied"] is False
    assert np.allclose(closure.model.precision, schur)


def test_exact_component_schur_adds_to_existing_retained_edge():
    graph = _neighbors((("p", "q"), ("p", "x"), ("x", "q")))
    selection = select_hop_budget_domain(
        ("p", "q"), graph, maximum_nodes=2
    )
    embeddings = {node: np.array([0.0]) for node in graph}
    pairs, self_return, provenance = _exact_two_port_exterior_dtn(
        selection,
        graph,
        embeddings,
        intrinsic_leakage=0.0,
        length_scale=1.0,
        topology_conductance=1.0,
        maximum_pairs=1,
        maximum_component_nodes=8,
    )
    config = ExperimentalBoundaryClosureConfig(
        maximum_edges=1,
        closure_mass_fraction=1.0,
        ordinary_branch_conductance=1.0,
        bridge_conductance_cap=0.25,
        pair_conductance_source="exact_component_schur",
        ledger_mode="explicit_self_return",
    )
    closure = build_experimental_boundary_closure(
        selection.domain,
        intrinsic_leakage_conductance=0.2,
        pair_conductances=pairs,
        self_return_conductance=self_return,
        config=config,
    )
    full = build_grounded_semantic_diffusion(
        ("p", "q", "x"),
        graph,
        leakage_conductance={"p": 0.2, "q": 0.2, "x": 0.0},
    )
    schur = _schur_precision(full, retained=(0, 1), exterior=(2,))

    assert pairs == (("p", "q", pytest.approx(0.5)),)
    assert provenance["parallel_retained_edge_pairs"] == 1
    assert closure.model.model.conductance[0, 1] == pytest.approx(1.5)
    assert np.allclose(closure.model.precision, schur)
def _schur_precision(full, retained, exterior):
    retained = np.asarray(retained, dtype=int)
    exterior = np.asarray(exterior, dtype=int)
    j_rr = full.precision[np.ix_(retained, retained)]
    j_rx = full.precision[np.ix_(retained, exterior)]
    j_xx = full.precision[np.ix_(exterior, exterior)]
    return j_rr - j_rx @ np.linalg.solve(j_xx, j_rx.T)


def _equal_series_closure_fixture():
    graph = _neighbors((("a", "x"), ("x", "b")))
    domain = select_hop_budget_domain(
        ("a", "b"), graph, maximum_nodes=2
    ).domain
    config = ExperimentalBoundaryClosureConfig(
        maximum_edges=1,
        closure_mass_fraction=1.0,
        ordinary_branch_conductance=1.0,
        bridge_conductance_cap=0.75,
        pair_conductance_source="two_terminal_series",
        ledger_mode="equal_return_transfer",
    )
    closure = build_experimental_boundary_closure(
        domain,
        intrinsic_leakage_conductance=0.2,
        pair_conductances=(("a", "b", 0.5),),
        config=config,
    )
    full = build_grounded_semantic_diffusion(
        ("a", "x", "b"),
        graph,
        leakage_conductance={"a": 0.2, "x": 0.0, "b": 0.2},
    )
    return graph, domain, config, closure, full


def test_factor_two_closure_is_exact_for_equal_zero_leakage_series_path():
    _, _, _, closure, full = _equal_series_closure_fixture()
    schur = _schur_precision(full, retained=(0, 2), exterior=(1,))

    assert closure.edges == (("a", "b", 0.5),)
    assert closure.pair_source == "two_terminal_series"
    assert closure.supplied_pair_count == 1
    assert closure.filtered_pair_count == 0
    assert closure.realized_pair_count == 1
    assert closure.total_transfer_mass == pytest.approx(1.0)
    assert closure.total_self_return_mass == pytest.approx(1.0)
    assert np.allclose(closure.transfer_degree, [0.5, 0.5])
    assert np.allclose(closure.self_return_conductance, [0.5, 0.5])
    assert np.allclose(closure.residual_ground_conductance, [0.0, 0.0])
    assert np.allclose(closure.model.precision, schur)
    assert np.allclose(
        closure.original_cut_conductance,
        closure.residual_ground_conductance
        + closure.self_return_conductance
        + closure.transfer_degree,
    )


def test_explicit_ledger_is_exact_for_unequal_leaky_series_path():
    graph = _neighbors((("a", "x"), ("x", "b")))
    semantic_distance = math.sqrt(2.0 * math.log(2.0))
    embeddings = {
        "a": np.array([0.0]),
        "x": np.array([0.0]),
        "b": np.array([semantic_distance]),
    }
    domain = select_hop_budget_domain(
        ("a", "b"), graph, maximum_nodes=2
    ).domain
    exterior_leakage = 0.3
    left_cut = 1.0
    right_cut = 0.5
    denominator = left_cut + right_cut + exterior_leakage
    transfer = left_cut * right_cut / denominator
    self_return = {
        "a": left_cut * left_cut / denominator,
        "b": right_cut * right_cut / denominator,
    }
    config = ExperimentalBoundaryClosureConfig(
        maximum_edges=1,
        closure_mass_fraction=1.0,
        ordinary_branch_conductance=1.0,
        bridge_conductance_cap=0.75,
        pair_conductance_source="two_terminal_series",
        ledger_mode="explicit_self_return",
    )
    closure = build_experimental_boundary_closure(
        domain,
        intrinsic_leakage_conductance=0.2,
        pair_conductances=(("a", "b", transfer),),
        self_return_conductance=self_return,
        node_embeddings=embeddings,
        semantic_length_scale=1.0,
        config=config,
    )
    full = build_grounded_semantic_diffusion(
        ("a", "x", "b"),
        graph,
        node_embeddings=embeddings,
        leakage_conductance={
            "a": 0.2,
            "x": exterior_leakage,
            "b": 0.2,
        },
        length_scale=1.0,
    )
    schur = _schur_precision(full, retained=(0, 2), exterior=(1,))

    assert np.allclose(closure.model.precision, schur)
    assert np.allclose(
        closure.residual_ground_conductance,
        [
            left_cut * exterior_leakage / denominator,
            right_cut * exterior_leakage / denominator,
        ],
    )
    assert not np.allclose(
        closure.self_return_conductance, closure.transfer_degree
    )


def test_two_terminal_source_rejects_naive_multi_pair_composition():
    graph = _neighbors((("a", "x"), ("b", "x"), ("c", "x")))
    domain = select_hop_budget_domain(
        ("a", "b", "c"), graph, maximum_nodes=3
    ).domain
    config = ExperimentalBoundaryClosureConfig(
        maximum_edges=3,
        closure_mass_fraction=1.0,
        ordinary_branch_conductance=1.0,
        bridge_conductance_cap=0.75,
        pair_conductance_source="two_terminal_series",
        ledger_mode="explicit_self_return",
    )
    with pytest.raises(ValueError, match="exactly one pair|sparse_dtn"):
        build_experimental_boundary_closure(
            domain,
            intrinsic_leakage_conductance=0.2,
            pair_conductances=(
                ("a", "b", 0.25),
                ("a", "c", 0.25),
                ("b", "c", 0.25),
            ),
            self_return_conductance={"a": 0.5, "b": 0.5, "c": 0.5},
            config=config,
        )


def test_empty_sparse_dtn_is_a_provenanced_noop_and_invents_no_pairs():
    graph = _neighbors((("a", "x"), ("x", "b")))
    domain = select_hop_budget_domain(
        ("a", "b"), graph, maximum_nodes=2
    ).domain
    config = ExperimentalBoundaryClosureConfig(
        maximum_edges=1,
        closure_mass_fraction=1.0,
        ordinary_branch_conductance=1.0,
        bridge_conductance_cap=0.75,
        pair_conductance_source="sparse_dtn",
        ledger_mode="equal_return_transfer",
    )
    closure = build_experimental_boundary_closure(
        domain,
        intrinsic_leakage_conductance=0.2,
        pair_conductances=(),
        config=config,
    )
    baseline = build_local_grounded_semantic_diffusion(
        domain, intrinsic_leakage_conductance=0.2
    )

    assert closure.edges == ()
    assert closure.supplied_pair_count == 0
    assert closure.filtered_pair_count == 0
    assert closure.realized_pair_count == 0
    assert np.array_equal(closure.model.precision, baseline.precision)
    assert len(closure.ledger_fingerprint) == 64


def test_closure_cap_and_ledger_reject_oversubscription_or_double_counting():
    _, domain, config, closure, _ = _equal_series_closure_fixture()
    with pytest.raises(ValueError, match="strictly below"):
        ExperimentalBoundaryClosureConfig(
            maximum_edges=1,
            closure_mass_fraction=1.0,
            ordinary_branch_conductance=1.0,
            bridge_conductance_cap=1.0,
            pair_conductance_source="two_terminal_series",
            ledger_mode="equal_return_transfer",
        )
    baseline = build_local_grounded_semantic_diffusion(
        domain, intrinsic_leakage_conductance=0.2
    )
    with pytest.raises(ValueError, match="model cut conductance"):
        ExperimentalBoundaryClosure(
            model=baseline,
            edges=closure.edges,
            original_cut_conductance=np.array([1.0, 1.0]),
            transfer_degree=np.array([0.5, 0.5]),
            self_return_conductance=np.array([0.5, 0.5]),
            residual_ground_conductance=np.array([0.0, 0.0]),
            supplied_pair_count=1,
            filtered_pair_count=0,
            config=config,
            ledger_fingerprint="0" * 64,
        )
    with pytest.raises(ValueError, match="split beta|factor-two"):
        ExperimentalBoundaryClosure(
            model=closure.model,
            edges=(("a", "b", 0.6),),
            original_cut_conductance=np.array([1.0, 1.0]),
            transfer_degree=np.array([0.6, 0.6]),
            self_return_conductance=np.array([0.6, 0.6]),
            residual_ground_conductance=np.array([0.0, 0.0]),
            supplied_pair_count=1,
            filtered_pair_count=0,
            config=config,
            ledger_fingerprint="0" * 64,
        )

def test_fidelity_uses_shared_alpha_explicit_protected_nodes_and_selected_resistance():
    graph = _path(8)
    candidate = select_hop_budget_domain((0,), graph, maximum_nodes=5)
    reference = select_hop_budget_domain((0,), graph, maximum_nodes=9)
    result = evaluate_bounded_domain_fidelity(
        candidate,
        reference,
        protected_nodes=(0, 1, 2, 3),
        intrinsic_leakage_conductance=0.2,
        rank_top_k=2,
        include_effective_resistance=True,
    )

    assert result.protected_nodes_count == 4
    assert result.protected_candidate_fraction == pytest.approx(4 / 5)
    assert result.effective_resistance_evaluated
    assert result.maximum_effective_resistance_absolute_error >= 0.0
    assert result.maximum_effective_resistance_relative_error >= 0.0
    assert result.closure_policy == "none_full_dirichlet_beta"
    assert result.closure_pair_source is None
    assert result.closure_supplied_pairs == 0
    assert result.closure_realized_pairs == 0
    assert result.rank_excludes_source is True
    assert result.maximum_raw_absolute_error > 0.0
    assert -1.0 <= result.mean_kendall_rank_agreement <= 1.0


def test_rank_top_k_is_bounded_after_excluding_each_source():
    graph = _path(4)
    candidate = select_hop_budget_domain((0,), graph, maximum_nodes=3)
    reference = select_hop_budget_domain((0,), graph, maximum_nodes=5)
    with pytest.raises(ValueError, match="source-excluded protected set"):
        evaluate_bounded_domain_fidelity(
            candidate,
            reference,
            protected_nodes=(0, 1),
            intrinsic_leakage_conductance=0.2,
            rank_top_k=2,
        )


def test_protected_set_omission_is_a_distinct_coverage_failure():
    graph = _path(6)
    candidate = select_hop_budget_domain((0,), graph, maximum_nodes=3)
    reference = select_hop_budget_domain((0,), graph, maximum_nodes=7)
    with pytest.raises(ProtectedSetCoverageError) as error:
        evaluate_bounded_domain_fidelity(
            candidate,
            reference,
            protected_nodes=(0, 1, 4),
            intrinsic_leakage_conductance=0.2,
        )
    assert error.value.missing_candidate == (4,)
    assert error.value.missing_reference == ()


def test_fidelity_closure_is_opt_in_and_reference_remains_exact_dirichlet():
    graph, _, config, _, _ = _equal_series_closure_fixture()
    candidate = select_hop_budget_domain(("a", "b"), graph, maximum_nodes=2)
    reference = select_hop_budget_domain(("a", "b"), graph, maximum_nodes=3)
    result = evaluate_bounded_domain_fidelity(
        candidate,
        reference,
        protected_nodes=("a", "b"),
        intrinsic_leakage_conductance=0.2,
        boundary_closure_config=config,
        boundary_closure_pair_conductances=(("a", "b", 0.5),),
        include_effective_resistance=True,
    )

    assert result.closure_policy == "experimental_graph_derived_schur_closure"
    assert result.closure_edges == 1
    assert result.closure_pair_source == "two_terminal_series"
    assert result.closure_supplied_pairs == 1
    assert result.closure_filtered_pairs == 0
    assert result.closure_realized_pairs == 1
    assert result.closure_total_transfer_mass == pytest.approx(1.0)
    assert result.closure_total_self_return_mass == pytest.approx(1.0)
    assert result.closure_mass_fraction == 1.0
    assert result.rank_excludes_source is True
    assert len(result.closure_ledger_fingerprint) == 64


def test_alpha_mapping_must_cover_both_candidate_and_shared_reference():
    graph = _path(4)
    candidate = select_hop_budget_domain((0,), graph, maximum_nodes=3)
    reference = select_hop_budget_domain((0,), graph, maximum_nodes=5)
    with pytest.raises(ValueError, match="alpha mapping misses"):
        evaluate_bounded_domain_fidelity(
            candidate,
            reference,
            protected_nodes=(0, 1),
            intrinsic_leakage_conductance={0: 0.2, 1: 0.2, 2: 0.2},
        )


def test_union_hop_reference_preserves_all_candidate_domains_before_fill():
    graph = _neighbors(
        (("s", "a"), ("s", "b"), ("a", "d"), ("b", "c"))
    )
    embeddings = {
        "s": np.array([0.0]),
        "a": np.array([10.0]),
        "b": np.array([0.1]),
        "c": np.array([0.2]),
        "d": np.array([10.1]),
    }
    hop = select_hop_budget_domain(("s",), graph, maximum_nodes=3)
    semantic = select_semantic_resistance_domain(
        ("s",),
        graph,
        embeddings,
        maximum_nodes=3,
        length_scale=1.0,
        conductance_floor=0.1,
    )
    candidate_union = set(hop.domain.nodes).union(semantic.domain.nodes)

    reference = select_union_hop_reference(
        (hop, semantic), graph, maximum_nodes=5
    )

    assert candidate_union.issubset(reference.domain.nodes)
    assert reference.realized_nodes == 5
    assert reference.strategy == "candidate_union_hop_reference"
    with pytest.raises(ValueError, match="candidate-domain union exceeds"):
        select_union_hop_reference((hop, semantic), graph, maximum_nodes=3)


def test_fidelity_rejects_a_larger_but_nonnested_reference():
    candidate_graph = _neighbors(((0, 1), (1, 2)))
    reference_graph = _neighbors(((0, 3), (0, 4), (0, 5)))
    candidate = select_hop_budget_domain(
        (0,), candidate_graph, maximum_nodes=3
    )
    reference = select_hop_budget_domain(
        (0,), reference_graph, maximum_nodes=4
    )

    with pytest.raises(ValueError, match="subset of the reference"):
        evaluate_bounded_domain_fidelity(
            candidate,
            reference,
            protected_nodes=(0, 1),
            intrinsic_leakage_conductance=0.2,
        )


def test_bounded_exterior_schur_matches_reference_with_two_bath_boundaries():
    graph = _neighbors(
        (("p", "x"), ("q", "x"), ("x", "z"), ("p", "y"))
    )
    embeddings = {node: np.array([0.0]) for node in graph}
    candidate = select_hop_budget_domain(
        ("p", "q"), graph, maximum_nodes=2
    )
    reference = select_hop_budget_domain(
        ("p", "q"), graph, maximum_nodes=3
    )
    allowed = set(reference.domain.nodes).difference(candidate.domain.nodes)
    assert allowed == {"x"}

    pairs, self_return, provenance = _exact_two_port_exterior_dtn(
        candidate,
        graph,
        embeddings,
        intrinsic_leakage=0.2,
        length_scale=1.0,
        topology_conductance=1.0,
        maximum_pairs=1,
        maximum_component_nodes=8,
        allowed_exterior_nodes=allowed,
    )

    expected = 1.0 / 3.2
    assert pairs == (("p", "q", pytest.approx(expected)),)
    assert self_return == {
        "p": pytest.approx(expected),
        "q": pytest.approx(expected),
    }
    discovery = provenance["discovery"]
    assert discovery["outside_allowed_cut_edges"] == [[
        ["builtins", "str", "'p'"],
        ["builtins", "str", "'y'"],
    ]]
    assert discovery["components"][0]["outside_bath_edges"] == [[
        ["builtins", "str", "'x'"],
        ["builtins", "str", "'z'"],
    ]]

    config = ExperimentalBoundaryClosureConfig(
        maximum_edges=1,
        closure_mass_fraction=1.0,
        ordinary_branch_conductance=1.0,
        bridge_conductance_cap=0.25,
        pair_conductance_source="exact_component_schur",
        ledger_mode="explicit_self_return",
    )
    closure = build_experimental_boundary_closure(
        candidate.domain,
        intrinsic_leakage_conductance=0.2,
        pair_conductances=pairs,
        self_return_conductance=self_return,
        config=config,
    )
    reference_model = build_local_grounded_semantic_diffusion(
        reference.domain, intrinsic_leakage_conductance=0.2
    )
    reference_index = {
        node: row for row, node in enumerate(reference.domain.nodes)
    }
    schur = _schur_precision(
        reference_model,
        retained=(reference_index["p"], reference_index["q"]),
        exterior=(reference_index["x"],),
    )

    assert np.allclose(closure.model.precision, schur)


def test_exact_component_schur_rejects_approximate_mass_fraction():
    with pytest.raises(ValueError, match="requires closure_mass_fraction"):
        ExperimentalBoundaryClosureConfig(
            maximum_edges=1,
            closure_mass_fraction=0.5,
            ordinary_branch_conductance=1.0,
            bridge_conductance_cap=0.25,
            pair_conductance_source="exact_component_schur",
            ledger_mode="explicit_self_return",
        )


def test_exact_component_schur_requires_explicit_self_return_ledger():
    with pytest.raises(ValueError, match="requires explicit_self_return"):
        ExperimentalBoundaryClosureConfig(
            maximum_edges=1,
            closure_mass_fraction=1.0,
            ordinary_branch_conductance=1.0,
            bridge_conductance_cap=0.1,
            pair_conductance_source="exact_component_schur",
            ledger_mode="equal_return_transfer",
        )


def test_exterior_component_fingerprint_includes_canonical_internal_edges():
    first_graph = _neighbors(
        (("p", "x"), ("x", "y"), ("y", "z"), ("z", "q"))
    )
    second_graph = _neighbors(
        (("p", "x"), ("x", "z"), ("z", "y"), ("z", "q"))
    )
    first_domain = select_hop_budget_domain(
        ("p", "q"), first_graph, maximum_nodes=2
    ).domain
    second_domain = select_hop_budget_domain(
        ("p", "q"), second_graph, maximum_nodes=2
    ).domain

    first = discover_exterior_components(first_domain, first_graph).components[0]
    second = discover_exterior_components(
        second_domain, second_graph
    ).components[0]

    assert first.nodes == second.nodes
    assert first.cut_edges == second.cut_edges
    assert first.internal_edge_count == second.internal_edge_count == 2
    assert first.internal_edges != second.internal_edges
    assert first.component_fingerprint != second.component_fingerprint


def test_exterior_reciprocity_is_checked_on_already_discovered_cross_edge():
    graph = {
        "p": ("x",),
        "q": ("y",),
        "x": ("p", "y", "z"),
        "y": ("q", "x", "z"),
        "z": ("x",),
    }
    domain = select_hop_budget_domain(
        ("p", "q"), graph, maximum_nodes=2
    ).domain

    with pytest.raises(ValueError, match="reciprocal"):
        discover_exterior_components(domain, graph)


def test_exact_reduction_needs_no_embeddings_and_sorts_heterogeneous_ports():
    graph = _neighbors(
        ((0, "x"), ("a", "x"), (1, "y"), ("b", "y"))
    )
    selection = select_hop_budget_domain(
        (0, 1, "a", "b"), graph, maximum_nodes=4
    )

    pairs, self_return, provenance = _exact_two_port_exterior_dtn(
        selection,
        graph,
        None,
        intrinsic_leakage=0.0,
        length_scale=None,
        topology_conductance=1.0,
        maximum_pairs=2,
        maximum_component_nodes=8,
    )

    assert len(pairs) == 2
    assert set(self_return) == {0, 1, "a", "b"}
    assert provenance["semantic_role"] == "rank_graph_connected_pairs_only"


def test_fidelity_emits_protocol_per_anchor_top8_and_resistance_endpoints():
    graph = _path(12)
    candidate = select_hop_budget_domain((0,), graph, maximum_nodes=10)
    reference = select_hop_budget_domain((0,), graph, maximum_nodes=13)
    result = evaluate_bounded_domain_fidelity(
        candidate,
        reference,
        protected_nodes=tuple(range(10)),
        intrinsic_leakage_conductance=0.2,
        include_effective_resistance=True,
    )

    assert result.rank_top_k == 8
    assert len(result.per_anchor_raw_relative_l2_error) == 1
    assert len(result.per_anchor_maximum_h_absolute_error) == 1
    assert len(result.per_anchor_rank_inversion_fraction) == 1
    assert len(result.per_anchor_top_k_overlap) == 1
    assert len(result.per_anchor_source_diagonal_relative_error) == 1
    assert len(result.per_anchor_effective_resistance_relative_error) == 1
    assert result.effective_resistance_relative_error_90th_percentile >= 0.0


def test_exterior_bath_edge_requires_reciprocal_adjacency():
    graph = {
        "p": ("x",),
        "q": ("x",),
        "x": ("p", "q", "z"),
        "z": (),
    }
    domain = select_hop_budget_domain(
        ("p", "q"), graph, maximum_nodes=2
    ).domain

    with pytest.raises(ValueError, match="reciprocal"):
        discover_exterior_components(
            domain,
            graph,
            allowed_exterior_nodes=("x",),
        )


def test_topology_skeleton_rejects_traversed_parent_cycle():
    graph = _neighbors((("a", "b"),))
    parents = {"a": ("b",), "b": ("a",)}

    with pytest.raises(ValueError, match="cyclic parent data"):
        select_topology_skeleton_domain(
            ("a",),
            graph,
            parents,
            maximum_nodes=2,
            ancestor_depth=3,
        )


def test_fidelity_rejects_changed_shared_node_adjacency_snapshot():
    candidate_graph = _neighbors(((0, 1), (1, 2)))
    reference_graph = _neighbors(((0, 1), (0, 2)))
    candidate = select_hop_budget_domain(
        (0,), candidate_graph, maximum_nodes=2
    )
    reference = select_hop_budget_domain(
        (0,), reference_graph, maximum_nodes=3
    )

    with pytest.raises(ValueError, match="identical complete incident adjacency"):
        evaluate_bounded_domain_fidelity(
            candidate,
            reference,
            protected_nodes=(0, 1),
            intrinsic_leakage_conductance=0.2,
        )
