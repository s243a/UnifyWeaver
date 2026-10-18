#!/usr/bin/env python3
"""Tests for outcome-blind bounded-domain fidelity selectors and closure."""

from pathlib import Path
from dataclasses import replace
import math
import sys

import numpy as np
import pytest

import prototypes.mu_cosine.benchmark_bounded_diffusion_fidelity as fidelity_benchmark


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import unifyweaver.graph.bounded_diffusion_fidelity as bounded_fidelity  # noqa: E402
from unifyweaver.graph.bounded_diffusion_fidelity import (  # noqa: E402
    _conservative_lower_quantile,
    _conservative_upper_quantile,
    ExperimentalBoundaryClosure,
    ExperimentalBoundaryClosureConfig,
    ExactExteriorSchurReduction,
    ExteriorTraversalLimitError,
    ProtectedSetCoverageError,
    build_experimental_boundary_closure,
    discover_exterior_components,
    ensure_matched_budget,
    evaluate_bounded_domain_fidelity,
    evaluate_nested_bounded_domain_fidelity,
    reduce_exact_exterior_component,
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


def _exterior_component(graph, ports, *, allowed_exterior_nodes=None):
    ports = tuple(ports)
    domain = select_hop_budget_domain(
        ports, graph, maximum_nodes=len(ports)
    ).domain
    discovery = discover_exterior_components(
        domain,
        graph,
        allowed_exterior_nodes=allowed_exterior_nodes,
    )
    assert discovery.component_count == 1
    return discovery.components[0]


def test_exact_multiport_schur_matches_grounded_three_port_star_identity():
    graph = _neighbors((("p", "x"), ("q", "x"), ("r", "x")))
    component = _exterior_component(graph, ("p", "q", "r"))
    reduction = reduce_exact_exterior_component(
        component, intrinsic_leakage_conductance=1.0
    )
    expected_return = np.ones((3, 3)) / 4.0
    expected_reduced = np.eye(3) - expected_return

    assert isinstance(reduction, ExactExteriorSchurReduction)
    assert np.allclose(
        reduction.schur_return, expected_return, rtol=1e-10, atol=1e-12
    )
    assert np.allclose(
        reduction.reduced_boundary_precision,
        expected_reduced,
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.allclose(
        reduction.boundary_cut_conductance,
        np.ones(3),
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.allclose(
        reduction.self_return_conductance,
        np.full(3, 0.25),
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.allclose(
        reduction.transfer_degree,
        np.full(3, 0.5),
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.allclose(
        reduction.residual_ground_conductance,
        np.full(3, 0.25),
        rtol=1e-10,
        atol=1e-12,
    )
    assert reduction.pair_conductances == (
        ("p", "q", pytest.approx(0.25)),
        ("p", "r", pytest.approx(0.25)),
        ("q", "r", pytest.approx(0.25)),
    )
    assert np.allclose(
        reduction.boundary_cut_conductance,
        reduction.self_return_conductance
        + reduction.transfer_degree
        + reduction.residual_ground_conductance,
        rtol=1e-10,
        atol=1e-12,
    )

    source = np.array([1.0, -1.0, 0.0])
    effective_resistance = float(
        source
        @ np.linalg.solve(reduction.reduced_boundary_precision, source)
    )
    assert effective_resistance == pytest.approx(2.0)
    assert 1.0 / effective_resistance != pytest.approx(
        reduction.schur_return[0, 1]
    )


def test_exact_multiport_schur_matches_direct_full_graph_elimination():
    graph = _neighbors(
        (
            ("p", "a"),
            ("q", "a"),
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),
            ("p", "q"),
            ("r", "b"),
            ("s", "c"),
        )
    )
    ports = ("p", "q", "r", "s")
    exterior = ("a", "b", "c")
    component = _exterior_component(graph, ports)
    reduction = reduce_exact_exterior_component(
        component,
        intrinsic_leakage_conductance={
            "a": 0.2,
            "b": 0.2,
            "c": 0.2,
        },
    )
    nodes = ports + exterior
    full = build_grounded_semantic_diffusion(
        nodes,
        graph,
        leakage_conductance={
            "p": 0.3,
            "q": 0.3,
            "r": 0.3,
            "s": 0.3,
            "a": 0.2,
            "b": 0.2,
            "c": 0.2,
        },
    )
    expected = _schur_precision(
        full,
        retained=range(len(ports)),
        exterior=range(len(ports), len(nodes)),
    )
    public_boundary_precision = np.diag(np.full(len(ports), 0.3))
    public_boundary_precision[0, 0] += 1.0
    public_boundary_precision[1, 1] += 1.0
    public_boundary_precision[0, 1] -= 1.0
    public_boundary_precision[1, 0] -= 1.0
    observed = (
        public_boundary_precision + reduction.reduced_boundary_precision
    )
    assert np.allclose(observed, expected, rtol=1e-10, atol=1e-12)

    sources = np.array(
        (
            (1.0, 0.0, 0.5),
            (0.0, 1.0, 0.5),
            (0.5, 0.0, 1.0),
            (0.0, 0.5, 1.0),
        )
    )
    full_sources = np.zeros((len(nodes), sources.shape[1]))
    full_sources[: len(ports)] = sources
    assert np.allclose(
        np.linalg.solve(observed, sources),
        full.equilibrium_response(full_sources)[: len(ports)],
        rtol=1e-10,
        atol=1e-12,
    )


def test_exact_multiport_schur_does_not_cap_parallel_graph_return():
    hubs = ("x1", "x2", "x3", "x4")
    edges = [
        (port, hub)
        for port in ("p", "q", "r")
        for hub in hubs
    ]
    edges.extend(zip(hubs, hubs[1:]))
    graph = _neighbors(edges)
    component = _exterior_component(graph, ("p", "q", "r"))
    reduction = reduce_exact_exterior_component(component)

    assert np.allclose(
        reduction.schur_return,
        np.full((3, 3), 4.0 / 3.0),
        rtol=1e-10,
        atol=1e-12,
    )
    assert reduction.schur_return[0, 1] > reduction.topology_conductance
    assert np.allclose(
        reduction.residual_ground_conductance,
        np.zeros(3),
        rtol=1e-10,
        atol=1e-12,
    )
    assert reduction.minimum_reduced_precision_eigenvalue >= -1e-10


def test_exact_multiport_schur_is_invariant_to_input_and_identifier_order():
    ports = (1, "p", ("r",))
    edges = (
        (1, "x"),
        ("p", "x"),
        (("r",), "y"),
        ("x", "y"),
    )
    first_graph = _neighbors(edges)
    second_graph = _neighbors(reversed(edges))
    first_component = _exterior_component(first_graph, reversed(ports))
    second_component = _exterior_component(second_graph, ports)
    first = reduce_exact_exterior_component(
        first_component,
        intrinsic_leakage_conductance={"x": 0.1, "y": 0.3},
    )
    second = reduce_exact_exterior_component(
        second_component,
        intrinsic_leakage_conductance={"y": 0.3, "x": 0.1},
    )

    assert first.component_fingerprint == second.component_fingerprint
    assert first.reduction_fingerprint == second.reduction_fingerprint
    assert first.ports == second.ports
    assert np.array_equal(first.schur_return, second.schur_return)
    assert np.array_equal(
        first.reduced_boundary_precision,
        second.reduced_boundary_precision,
    )


def test_exact_multiport_schur_avoids_explicit_inverse_and_has_readonly_outputs(
    monkeypatch,
):
    graph = _neighbors((("p", "x"), ("q", "x"), ("r", "x")))
    component = _exterior_component(graph, ("p", "q", "r"))

    def reject_inverse(*_args, **_kwargs):
        raise AssertionError("the exact reducer must not form an inverse")

    original_cholesky = np.linalg.cholesky
    cholesky_calls = []

    def record_cholesky(*args, **kwargs):
        cholesky_calls.append(1)
        return original_cholesky(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "inv", reject_inverse)
    monkeypatch.setattr(np.linalg, "pinv", reject_inverse)
    monkeypatch.setattr(np.linalg, "cholesky", record_cholesky)
    reduction = reduce_exact_exterior_component(
        component, intrinsic_leakage_conductance=0.2
    )

    assert cholesky_calls
    assert reduction.schur_return.flags.writeable is False
    assert reduction.reduced_boundary_precision.flags.writeable is False
    assert reduction.bridge_conductance.flags.writeable is False
    assert reduction.solve_relative_residual <= 1e-10
    assert reduction.cholesky_reconstruction_relative_error <= 1e-11
    assert reduction.maximum_principle_violation <= 1e-10
    assert len(reduction.reduction_fingerprint) == 64


def test_exact_multiport_schur_accounts_for_outside_bath_grounding():
    graph = _neighbors(
        (("p", "x"), ("q", "x"), ("r", "x"), ("x", "z"))
    )
    component = _exterior_component(
        graph,
        ("p", "q", "r"),
        allowed_exterior_nodes=("x",),
    )
    reduction = reduce_exact_exterior_component(
        component, intrinsic_leakage_conductance=0.2
    )

    assert component.outside_bath_edges == (("x", "z"),)
    assert np.allclose(
        reduction.schur_return,
        np.ones((3, 3)) / 4.2,
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.allclose(
        reduction.residual_ground_conductance,
        np.full(3, 1.2 / 4.2),
        rtol=1e-10,
        atol=1e-12,
    )


def test_exact_exterior_schur_supports_one_port_and_fails_closed_on_inputs():
    one_port_graph = _neighbors((("p", "x"),))
    one_port = reduce_exact_exterior_component(
        _exterior_component(one_port_graph, ("p",)),
        intrinsic_leakage_conductance=0.5,
    )
    assert one_port.pair_conductances == ()
    assert one_port.schur_return[0, 0] == pytest.approx(2.0 / 3.0)
    assert one_port.residual_ground_conductance[0] == pytest.approx(
        1.0 / 3.0
    )

    graph = _neighbors((("p", "x"), ("x", "y"), ("y", "q")))
    component = _exterior_component(graph, ("p", "q"))
    for bad_topology in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="topology_conductance"):
            reduce_exact_exterior_component(
                component, topology_conductance=bad_topology
            )
    for bad_leakage in (-0.1, float("inf")):
        with pytest.raises(ValueError, match="leakage"):
            reduce_exact_exterior_component(
                component,
                intrinsic_leakage_conductance=bad_leakage,
            )
    with pytest.raises(ValueError, match="unknown exterior"):
        reduce_exact_exterior_component(
            component,
            intrinsic_leakage_conductance={"unknown": 0.2},
        )
    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        reduce_exact_exterior_component(
            component,
            minimum_reciprocal_condition=0.9,
        )
    with pytest.raises(np.linalg.LinAlgError, match="unrepresentable"):
        reduce_exact_exterior_component(
            _exterior_component(one_port_graph, ("p",)),
            topology_conductance=np.nextafter(0.0, 1.0),
        )


def test_exact_multiport_result_rejects_tampered_exposed_state():
    graph = _neighbors(
        (
            ("p", "x"),
            ("q", "x"),
            ("x", "y"),
            ("r", "y"),
        )
    )
    reduction = reduce_exact_exterior_component(
        _exterior_component(graph, ("p", "q", "r")),
        intrinsic_leakage_conductance={"x": 0.1, "y": 0.2},
    )

    with pytest.raises(ValueError, match="harmonic extension"):
        replace(
            reduction,
            harmonic_extension=np.full(
                len(reduction.exterior_nodes), -100.0
            ),
        )
    with pytest.raises(ValueError, match="Schur return|fingerprint"):
        replace(
            reduction,
            boundary_coupling=np.flip(
                reduction.boundary_coupling, axis=1
            ),
        )
    canonicalized = replace(
        reduction,
        boundary_cut_conductance=(
            reduction.boundary_cut_conductance + 5e-13
        ),
    )
    assert np.array_equal(
        canonicalized.boundary_cut_conductance,
        reduction.boundary_cut_conductance,
    )
    assert (
        canonicalized.reduction_fingerprint
        == reduction.reduction_fingerprint
    )


def test_exact_multiport_public_residual_shunt_is_roundoff_nonnegative():
    ports = tuple(f"p{index:03d}" for index in range(64))
    graph = _neighbors(((port, "x") for port in ports))
    reduction = reduce_exact_exterior_component(
        _exterior_component(graph, reversed(ports))
    )

    assert np.all(reduction.residual_ground_conductance >= 0.0)
    assert np.allclose(
        reduction.boundary_cut_conductance,
        reduction.self_return_conductance
        + reduction.transfer_degree
        + reduction.residual_ground_conductance,
        rtol=1e-10,
        atol=1e-12,
    )


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


def test_nested_fidelity_is_exactly_equal_to_repeated_evaluation_except_timings():
    graph = _path(10)
    candidates = (
        select_hop_budget_domain((0, 10), graph, maximum_nodes=6),
        select_hop_budget_domain((0, 10), graph, maximum_nodes=8),
    )
    reference = select_hop_budget_domain((0, 10), graph, maximum_nodes=11)
    arguments = {
        "protected_nodes": candidates[0].domain.nodes,
        "intrinsic_leakage_conductance": 0.2,
        "rank_top_k": 2,
        "include_effective_resistance": True,
    }

    repeated = tuple(
        evaluate_bounded_domain_fidelity(candidate, reference, **arguments)
        for candidate in candidates
    )
    batch = evaluate_nested_bounded_domain_fidelity(
        candidates, reference, **arguments
    )

    assert len(batch) == len(candidates)
    assert tuple(batch) == batch.results
    assert tuple(
        result.candidate_selection_fingerprint for result in batch
    ) == tuple(candidate.selection_fingerprint for candidate in candidates)
    for shared, separate in zip(batch, repeated):
        shared_values = shared.as_dict()
        separate_values = separate.as_dict()
        for name in tuple(shared_values):
            if name.endswith("_seconds"):
                shared_values.pop(name)
                separate_values.pop(name)
        assert shared_values == separate_values

    assert batch.reference_build_count == 1
    assert batch.reference_factorization_count == 1
    assert batch.candidate_build_count == len(candidates)
    assert batch.candidate_factorization_count == len(candidates)
    assert batch.reference_model_diagnostics.checks_passed
    assert all(
        diagnostic.checks_passed
        for diagnostic in batch.candidate_model_diagnostics
    )
    for diagnostic in (
        batch.reference_model_diagnostics,
        *batch.candidate_model_diagnostics,
    ):
        assert diagnostic.maximum_positive_off_diagonal == 0.0
        assert diagnostic.multi_rhs_solve_relative_residual <= 1e-10
        assert diagnostic.maximum_principle_violation <= 1e-10
        assert diagnostic.maximum_kirchhoff_relative_error <= 1e-10


def test_nested_fidelity_builds_and_factorizes_shared_reference_once(monkeypatch):
    graph = _path(9)
    candidates = (
        select_hop_budget_domain((0,), graph, maximum_nodes=4),
        select_hop_budget_domain((0,), graph, maximum_nodes=6),
    )
    reference = select_hop_budget_domain((0,), graph, maximum_nodes=9)
    built_domains = []
    factor_sizes = []
    reference_solve_calls = 0
    original_build = bounded_fidelity.build_local_grounded_semantic_diffusion
    original_cholesky = np.linalg.cholesky
    original_equilibrium = (
        bounded_fidelity.LocalGroundedSemanticDiffusion.equilibrium_response
    )

    def counted_build(domain, **kwargs):
        built_domains.append(domain)
        return original_build(domain, **kwargs)

    def counted_cholesky(matrix, *args, **kwargs):
        factor_sizes.append(np.asarray(matrix).shape[0])
        return original_cholesky(matrix, *args, **kwargs)

    def counted_equilibrium(model, source, *args, **kwargs):
        nonlocal reference_solve_calls
        if model.domain is reference.domain:
            reference_solve_calls += 1
        return original_equilibrium(model, source, *args, **kwargs)

    monkeypatch.setattr(
        bounded_fidelity,
        "build_local_grounded_semantic_diffusion",
        counted_build,
    )
    monkeypatch.setattr(np.linalg, "cholesky", counted_cholesky)
    monkeypatch.setattr(
        bounded_fidelity.LocalGroundedSemanticDiffusion,
        "equilibrium_response",
        counted_equilibrium,
    )

    batch = evaluate_nested_bounded_domain_fidelity(
        candidates,
        reference,
        protected_nodes=(0, 1, 2, 3),
        intrinsic_leakage_conductance=0.2,
        rank_top_k=2,
    )

    assert len(built_domains) == len(candidates) + 1
    assert sum(domain is reference.domain for domain in built_domains) == 1
    assert factor_sizes.count(reference.realized_nodes) == 1
    assert len(factor_sizes) == len(candidates) + 1
    # One multi-anchor solve, one source cut-current solve, and one boundary
    # harmonic solve are the only unique reference right-hand sides here.
    assert reference_solve_calls == 3
    assert batch.reference_build_count == 1
    assert batch.reference_factorization_count == 1


def test_nested_fidelity_reuses_node_identical_nominal_candidates(monkeypatch):
    graph = _path(6)
    first = select_hop_budget_domain((0,), graph, maximum_nodes=4)
    exhausted_domain = replace(first.domain, maximum_nodes=5)
    exhausted = replace(
        first,
        domain=exhausted_domain,
        requested_nodes=5,
        selection_fingerprint="1" * 64,
    )
    reference = select_hop_budget_domain((0,), graph, maximum_nodes=7)
    build_count = 0
    factor_count = 0
    original_build = bounded_fidelity.build_local_grounded_semantic_diffusion
    original_cholesky = np.linalg.cholesky

    def counted_build(domain, **kwargs):
        nonlocal build_count
        build_count += 1
        return original_build(domain, **kwargs)

    def counted_cholesky(matrix, *args, **kwargs):
        nonlocal factor_count
        factor_count += 1
        return original_cholesky(matrix, *args, **kwargs)

    monkeypatch.setattr(
        bounded_fidelity,
        "build_local_grounded_semantic_diffusion",
        counted_build,
    )
    monkeypatch.setattr(np.linalg, "cholesky", counted_cholesky)

    arguments = {
        "protected_nodes": first.domain.nodes,
        "intrinsic_leakage_conductance": 0.2,
        "rank_top_k": 2,
    }
    batch = evaluate_nested_bounded_domain_fidelity(
        (first, exhausted), reference, **arguments
    )
    repeated = (
        evaluate_bounded_domain_fidelity(first, reference, **arguments),
        evaluate_bounded_domain_fidelity(exhausted, reference, **arguments),
    )

    # The batch itself used one unique candidate factor and one reference
    # factor. The two calls above account for the four later factors.
    assert build_count == 6
    assert factor_count == 6
    assert batch.candidate_requested_result_count == 2
    assert batch.candidate_unique_model_count == 1
    assert batch.candidate_build_count == 1
    assert batch.candidate_factorization_count == 1
    assert batch.candidate_model_index == (0, 0)
    assert len(batch.candidate_model_diagnostics) == 1
    assert tuple(
        result.candidate_selection_fingerprint for result in batch
    ) == (first.selection_fingerprint, exhausted.selection_fingerprint)
    for shared, separate in zip(batch, repeated):
        shared_values = shared.as_dict()
        separate_values = separate.as_dict()
        for name in tuple(shared_values):
            if name.endswith("_seconds"):
                shared_values.pop(name)
                separate_values.pop(name)
        assert shared_values == separate_values


def test_nested_fidelity_reuses_reference_for_exhausted_nominal_candidate(monkeypatch):
    graph = _path(6)
    small = select_hop_budget_domain((0,), graph, maximum_nodes=4)
    reference = select_hop_budget_domain((0,), graph, maximum_nodes=7)
    exhausted = replace(
        reference,
        domain=replace(reference.domain, maximum_nodes=8),
        strategy="hop_budget_exhausted",
        requested_nodes=8,
        selection_fingerprint="2" * 64,
    )
    build_count = 0
    factor_count = 0
    original_build = bounded_fidelity.build_local_grounded_semantic_diffusion
    original_cholesky = np.linalg.cholesky

    def counted_build(domain, **kwargs):
        nonlocal build_count
        build_count += 1
        return original_build(domain, **kwargs)

    def counted_cholesky(matrix, *args, **kwargs):
        nonlocal factor_count
        factor_count += 1
        return original_cholesky(matrix, *args, **kwargs)

    monkeypatch.setattr(
        bounded_fidelity,
        "build_local_grounded_semantic_diffusion",
        counted_build,
    )
    monkeypatch.setattr(np.linalg, "cholesky", counted_cholesky)

    batch = evaluate_nested_bounded_domain_fidelity(
        (small, exhausted),
        reference,
        protected_nodes=small.domain.nodes,
        intrinsic_leakage_conductance=0.2,
        rank_top_k=2,
        include_effective_resistance=True,
    )

    assert build_count == 2
    assert factor_count == 2
    assert batch.candidate_model_index == (0, -1)
    assert batch.candidate_requested_result_count == 2
    assert batch.candidate_unique_model_count == 1
    assert batch.candidate_reference_reuse_count == 1
    assert batch.candidate_build_count == 1
    assert batch.candidate_factorization_count == 1
    exhausted_result = batch[1]
    assert exhausted_result.candidate_selection_fingerprint == "2" * 64
    assert exhausted_result.maximum_raw_absolute_error == 0.0
    assert exhausted_result.raw_relative_frobenius_error == 0.0
    assert exhausted_result.maximum_h_absolute_error == 0.0
    assert exhausted_result.maximum_effective_resistance_absolute_error == 0.0
    assert exhausted_result.maximum_effective_resistance_relative_error == 0.0

    all_exhausted = evaluate_nested_bounded_domain_fidelity(
        (exhausted,),
        reference,
        protected_nodes=small.domain.nodes,
        intrinsic_leakage_conductance=0.2,
        rank_top_k=2,
    )
    assert all_exhausted.candidate_model_index == (-1,)
    assert all_exhausted.candidate_model_diagnostics == ()
    assert all_exhausted.candidate_unique_model_count == 0
    assert all_exhausted.candidate_build_count == 0
    assert all_exhausted.candidate_factorization_count == 0


def test_nested_fidelity_reuses_reference_response_for_anchor_screening(monkeypatch):
    graph = _path(10)
    anchors = (0, 10)
    candidate = select_hop_budget_domain(anchors, graph, maximum_nodes=6)
    reference = select_hop_budget_domain(anchors, graph, maximum_nodes=11)
    reference_solve_calls = 0
    original_equilibrium = (
        bounded_fidelity.LocalGroundedSemanticDiffusion.equilibrium_response
    )

    def counted_equilibrium(model, source, *args, **kwargs):
        nonlocal reference_solve_calls
        if model.domain is reference.domain:
            reference_solve_calls += 1
        return original_equilibrium(model, source, *args, **kwargs)

    monkeypatch.setattr(
        bounded_fidelity.LocalGroundedSemanticDiffusion,
        "equilibrium_response",
        counted_equilibrium,
    )

    batch = evaluate_nested_bounded_domain_fidelity(
        (candidate,),
        reference,
        protected_nodes=candidate.domain.nodes,
        intrinsic_leakage_conductance=0.2,
        rank_top_k=2,
        screening_shell_nodes_by_anchor={0: (3,), 10: (7,)},
        screening_attenuation_threshold=math.exp(-1.0),
    )

    assert tuple(
        record.anchor for record in batch.reference_anchor_screening
    ) == anchors
    assert all(
        record.attenuation_threshold == pytest.approx(math.exp(-1.0))
        for record in batch.reference_anchor_screening
    )
    assert all(
        0.0 <= record.shell_attenuation <= 1.0
        for record in batch.reference_anchor_screening
    )
    assert batch.reference_build_count == 1
    assert batch.reference_factorization_count == 1
    # One shared two-anchor solve plus one Kirchhoff solve per anchor. The
    # screening records and fidelity metrics reuse those cached responses.
    assert reference_solve_calls == 3


def test_nested_fidelity_fails_closed_on_non_scalar_alpha_or_embeddings():
    graph = _path(5)
    candidate = select_hop_budget_domain((0,), graph, maximum_nodes=3)
    reference = select_hop_budget_domain((0,), graph, maximum_nodes=6)
    arguments = {
        "protected_nodes": (0, 1, 2),
        "intrinsic_leakage_conductance": 0.2,
    }

    with pytest.raises(ValueError, match="one frozen scalar alpha"):
        evaluate_nested_bounded_domain_fidelity(
            (candidate,),
            reference,
            protected_nodes=(0, 1, 2),
            intrinsic_leakage_conductance={node: 0.2 for node in graph},
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        evaluate_nested_bounded_domain_fidelity(
            (candidate,),
            reference,
            node_embeddings={node: np.array([float(node)]) for node in graph},
            **arguments,
        )


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
    assert provenance["semantic_role"] == "none"


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
    assert len(result.per_anchor_candidate_cut_current_fraction) == 1
    assert len(result.per_anchor_reference_cut_current_fraction) == 1
    assert all(
        0.0 <= value <= 1.0
        for value in result.per_anchor_candidate_cut_current_fraction
    )
    assert all(
        0.0 <= value <= 1.0
        for value in result.per_anchor_reference_cut_current_fraction
    )
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


def test_semantic_resistance_records_frozen_c_ref_without_changing_prefix():
    graph = _neighbors((("s", "a"), ("a", "b"), ("s", "c")))
    embeddings = {
        "s": np.array([0.0]),
        "a": np.array([0.1]),
        "b": np.array([0.2]),
        "c": np.array([3.0]),
    }
    unit = select_semantic_resistance_domain(
        ("s",),
        graph,
        embeddings,
        maximum_nodes=3,
        length_scale=1.0,
        conductance_floor=0.1,
        reference_conductance=1.0,
    )
    scaled = select_semantic_resistance_domain(
        ("s",),
        graph,
        embeddings,
        maximum_nodes=3,
        length_scale=1.0,
        conductance_floor=0.1,
        reference_conductance=2.0,
    )

    assert unit.domain.nodes == scaled.domain.nodes
    assert np.allclose(scaled.selection_distance, 2.0 * unit.selection_distance)
    assert dict(scaled.selector_parameters)["reference_conductance"] == 2.0
    assert scaled.selection_fingerprint != unit.selection_fingerprint


def test_fidelity_accepts_shared_zero_scalar_alpha_when_cuts_ground_both_models():
    graph = _path(3)
    candidate = select_hop_budget_domain((0,), graph, maximum_nodes=2)
    reference = select_hop_budget_domain((0,), graph, maximum_nodes=3)

    result = evaluate_bounded_domain_fidelity(
        candidate,
        reference,
        protected_nodes=(0, 1),
        intrinsic_leakage_conductance=0.0,
    )

    assert result.candidate_reciprocal_condition > 0.0
    assert result.reference_reciprocal_condition > 0.0


def test_fidelity_tail_summaries_use_conservative_observed_order_statistics():
    values = (0.0, 1.0, 2.0, 3.0)

    assert _conservative_upper_quantile(values, 0.9) == 3.0
    assert _conservative_lower_quantile(values, 0.1) == 0.0


def test_primary_exact_two_port_closure_is_exhaustive_and_embedding_free():
    graph = _neighbors(
        (
            ("p", "x"),
            ("x", "q"),
            ("r", "y"),
            ("y", "s"),
            ("p", "w"),
            ("r", "w"),
            ("s", "w"),
        )
    )
    selection = select_hop_budget_domain(
        ("p", "q", "r", "s"), graph, maximum_nodes=4
    )

    pairs, _, provenance = _exact_two_port_exterior_dtn(
        selection,
        graph,
        None,
        intrinsic_leakage=0.2,
        length_scale=1.0,
        topology_conductance=1.0,
        maximum_pairs=None,
        maximum_component_nodes=8,
    )

    assert {(left, right) for left, right, _ in pairs} == {
        ("p", "q"),
        ("r", "s"),
    }
    assert provenance["semantic_role"] == "none"
    assert provenance["eligible_two_port_components"] == 2
    assert provenance["joint_dtn_required_components"] == 1
    assert provenance["discarded_for_pair_budget"] == 0



def test_exhaustive_exact_two_port_output_ignores_embeddings_entirely():
    graph = _neighbors(
        (("p", "x"), ("x", "q"), ("r", "y"), ("y", "s"))
    )
    selection = select_hop_budget_domain(
        ("p", "q", "r", "s"), graph, maximum_nodes=4
    )
    first_embeddings = {
        "p": np.array([0.0]),
        "q": np.array([0.0]),
        "r": np.array([-10.0]),
        "s": np.array([10.0]),
    }
    second_embeddings = {
        "p": np.array([-10.0]),
        "q": np.array([10.0]),
        "r": np.array([0.0]),
        "s": np.array([0.0]),
    }
    arguments = {
        "intrinsic_leakage": 0.2,
        "length_scale": 1.0,
        "topology_conductance": 1.0,
        "maximum_pairs": None,
        "maximum_component_nodes": 8,
    }

    first = _exact_two_port_exterior_dtn(
        selection, graph, first_embeddings, **arguments
    )
    second = _exact_two_port_exterior_dtn(
        selection, graph, second_embeddings, **arguments
    )

    assert first == second
    assert first[2]["semantic_role"] == "none"


def test_traversal_limit_returns_stable_grounded_no_op():
    graph = _neighbors((("p", "x"), ("x", "y"), ("y", "q")))
    selection = select_hop_budget_domain(
        ("p", "q"), graph, maximum_nodes=2
    )

    first = _exact_two_port_exterior_dtn(
        selection,
        graph,
        None,
        intrinsic_leakage=0.2,
        length_scale=None,
        topology_conductance=1.0,
        maximum_pairs=None,
        maximum_component_nodes=1,
    )
    second = _exact_two_port_exterior_dtn(
        selection,
        graph,
        None,
        intrinsic_leakage=0.2,
        length_scale=None,
        topology_conductance=1.0,
        maximum_pairs=None,
        maximum_component_nodes=1,
    )

    assert first == second
    pairs, self_return, provenance = first
    assert pairs == ()
    assert self_return == {}
    assert provenance["status"] == "traversal_incomplete_grounded_no_op"
    assert provenance["no_op"] is True
    assert provenance["failure_count"] == 1
    assert provenance["failure_reasons"] == {"exterior_traversal_limit": 1}
    assert len(provenance["failure_fingerprint"]) == 64


def test_numerically_failed_two_port_stays_grounded_while_others_reduce(
    monkeypatch,
):
    graph = _neighbors(
        (("p", "x"), ("x", "q"), ("r", "y"), ("y", "s"))
    )
    selection = select_hop_budget_domain(
        ("p", "q", "r", "s"), graph, maximum_nodes=4
    )
    original = fidelity_benchmark._topology_two_port_schur

    def fail_one_component(component, *args, **kwargs):
        if component.nodes == ("x",):
            raise np.linalg.LinAlgError("synthetic numerical failure")
        return original(component, *args, **kwargs)

    monkeypatch.setattr(
        fidelity_benchmark,
        "_topology_two_port_schur",
        fail_one_component,
    )
    pairs, self_return, provenance = _exact_two_port_exterior_dtn(
        selection,
        graph,
        None,
        intrinsic_leakage=0.2,
        length_scale=None,
        topology_conductance=1.0,
        maximum_pairs=None,
        maximum_component_nodes=8,
    )

    assert {(left, right) for left, right, _ in pairs} == {("r", "s")}
    assert provenance["eligible_two_port_components"] == 2
    assert provenance["reduced_two_port_components"] == 1
    assert provenance["numerically_failed_two_port_components"] == 1
    assert provenance["failure_reasons"] == {
        "two_port_schur_numerical_failure": 1
    }
    assert len(provenance["failure_fingerprint"]) == 64

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
    index = {node: row for row, node in enumerate(closure.model.nodes)}
    assert closure.residual_ground_conductance[index["p"]] == pytest.approx(1.0)
    assert closure.residual_ground_conductance[index["q"]] == pytest.approx(1.0)
