#!/usr/bin/env python3
"""Tests for bounded grounded diffusion domains and bath boundaries."""

from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from unifyweaver.graph.leaky_diffusion import (  # noqa: E402
    build_grounded_semantic_diffusion,
)
from unifyweaver.graph.local_diffusion import (  # noqa: E402
    AnchorScreeningProvenance,
    build_local_grounded_semantic_diffusion,
    LocalDiffusionDomain,
    _tail_envelope_crossing,
    calibrate_uniform_leakage,
    compare_nested_domains,
    select_hop_local_domain,
)


def test_anchor_screening_provenance_requires_a_boolean_censor_flag():
    with pytest.raises(ValueError, match="right_censored must be boolean"):
        AnchorScreeningProvenance(
            anchor="a",
            shell_attenuation=0.25,
            attenuation_threshold=0.5,
            radius_lower=1.0,
            radius_upper=None,
            right_censored="false",
            maximum_observed_radius=1.0,
            distance_metric="hops",
        )


def _neighbors(edges, extra=()):
    output = {node: set() for node in extra}
    for left, right in edges:
        output.setdefault(left, set()).add(right)
        output.setdefault(right, set()).add(left)
    return {
        node: tuple(sorted(adjacent))
        for node, adjacent in output.items()
    }


def _provider(neighbors, calls=None):
    def incident(node):
        if calls is not None:
            calls.append(node)
        return neighbors.get(node, ())

    return incident


def _path(last):
    return _neighbors(tuple((node, node + 1) for node in range(last)))


def _source(nodes, node, magnitude=1.0):
    value = np.zeros(len(nodes), dtype=float)
    value[nodes.index(node)] = magnitude
    return value


def test_exact_k_hop_selection_is_deterministic_connected_and_lazy():
    graph = _neighbors(
        (
            ("a", "b"),
            ("a", "c"),
            ("b", "d"),
            ("c", "e"),
            ("d", "f"),
        )
    )
    calls = []
    domain = select_hop_local_domain(
        ("a",),
        _provider(graph, calls),
        maximum_nodes=4,
    )

    assert domain.nodes == ("a", "b", "c", "d")
    assert domain.anchors == ("a",)
    assert np.array_equal(domain.hop_distance, [0, 1, 1, 2])
    assert calls == ["a", "b", "c", "d"]
    assert set(domain.neighbor_mapping["a"]) == {"b", "c"}
    assert set(domain.neighbor_mapping["b"]) == {"a", "d"}
    assert set(domain.neighbor_mapping["c"]) == {"a", "e"}
    assert domain.cutoff_distance == 2
    assert domain.frontier_nodes == ("d",)


def test_mapping_adjacency_requires_explicit_entries_for_every_visited_node():
    with pytest.raises(
        ValueError,
        match="unable to read incident neighbors for 0",
    ):
        select_hop_local_domain((0,), {}, maximum_nodes=1)

    with pytest.raises(
        ValueError,
        match="unable to read incident neighbors for 1",
    ):
        select_hop_local_domain(
            (0,),
            {0: (1,)},
            maximum_nodes=2,
        )

    isolated = select_hop_local_domain(
        ("isolated",),
        {"isolated": ()},
        maximum_nodes=1,
    )
    assert isolated.nodes == ("isolated",)


def test_complete_distance_shell_may_exceed_k_while_truncation_breaks_ties():
    graph = _neighbors(
        (("a", "b"), ("a", "c"), ("a", "d"), ("a", "e"))
    )

    truncated = select_hop_local_domain(
        ("a",),
        _provider(graph),
        maximum_nodes=3,
    )
    full_shell = select_hop_local_domain(
        ("a",),
        _provider(graph),
        maximum_nodes=3,
        complete_distance_shell=True,
    )

    assert truncated.nodes == ("a", "b", "c")
    assert full_shell.nodes == ("a", "b", "c", "d", "e")
    assert np.array_equal(truncated.hop_distance, [0, 1, 1])
    assert np.array_equal(full_shell.hop_distance, [0, 1, 1, 1, 1])
    assert truncated.truncated_tie_count == 2
    assert full_shell.truncated_tie_count == 0


def test_provider_is_not_called_beyond_the_expanded_bfs_prefix():
    calls = []

    def infinite_path(node):
        calls.append(node)
        return (node - 1, node + 1) if node else (1,)

    domain = select_hop_local_domain(
        (0,),
        infinite_path,
        maximum_nodes=3,
    )

    assert domain.nodes == (0, 1, 2)
    assert calls == [0, 1, 2]


def test_severed_path_edge_becomes_an_exact_boundary_bath_shunt():
    graph = _path(3)
    domain = select_hop_local_domain(
        (0,),
        _provider(graph),
        maximum_nodes=3,
    )
    local = build_local_grounded_semantic_diffusion(domain)

    assert domain.nodes == (0, 1, 2)
    assert np.array_equal(local.cut_conductance, [0.0, 0.0, 1.0])
    assert np.array_equal(
        local.model.precision,
        np.array(
            [
                [1.0, -1.0, 0.0],
                [-1.0, 2.0, -1.0],
                [0.0, -1.0, 2.0],
            ]
        ),
    )
    assert np.array_equal(local.cut_conductance, [0.0, 0.0, 1.0])


def test_local_precision_is_the_full_grounded_precision_principal_block():
    graph = _path(4)
    alpha = 0.25
    domain = select_hop_local_domain(
        (0,),
        _provider(graph),
        maximum_nodes=3,
    )
    local = build_local_grounded_semantic_diffusion(
        domain,
        intrinsic_leakage_conductance=alpha,
    )
    full = build_grounded_semantic_diffusion(
        tuple(range(5)),
        graph,
        leakage_conductance=alpha,
    )
    indices = [full.nodes.index(node) for node in domain.nodes]

    assert np.allclose(
        local.model.precision,
        full.precision[np.ix_(indices, indices)],
    )
    full_green = full.green_kernel()[np.ix_(indices, indices)]
    local_green = local.model.green_kernel()
    assert not np.allclose(
        local_green,
        full_green,
    )


def test_naive_induced_subgraph_omits_the_cut_shunt_and_changes_response():
    graph = _path(3)
    alpha = 0.2
    domain = select_hop_local_domain(
        (0,),
        _provider(graph),
        maximum_nodes=3,
    )
    local = build_local_grounded_semantic_diffusion(
        domain,
        intrinsic_leakage_conductance=alpha,
    )
    naive = build_grounded_semantic_diffusion(
        domain.nodes,
        domain.neighbor_mapping,
        leakage_conductance=alpha,
    )
    source = _source(domain.nodes, 0)

    assert local.model.precision[-1, -1] == pytest.approx(
        naive.precision[-1, -1] + 1.0
    )
    assert not np.allclose(
        local.equilibrium_response(source),
        naive.equilibrium_response(source),
    )


def test_semantic_cut_conductance_requires_exterior_endpoint_embedding():
    graph = _path(2)
    domain = select_hop_local_domain(
        (0,),
        _provider(graph),
        maximum_nodes=2,
    )
    retained_only = {0: np.array([0.0]), 1: np.array([0.5])}

    with pytest.raises(ValueError, match="cut neighbor.*embeddings"):
        build_local_grounded_semantic_diffusion(
            domain,
            node_embeddings=retained_only,
            length_scale=1.0,
        )

    embeddings = {**retained_only, 2: np.array([2.0])}
    local = build_local_grounded_semantic_diffusion(
        domain,
        node_embeddings=embeddings,
        length_scale=1.0,
    )
    expected_cut = np.exp(-0.5 * (2.0 - 0.5) ** 2)
    assert local.cut_conductance[-1] == pytest.approx(expected_cut)


def test_nonzero_bath_temperature_is_an_affine_equilibrium_term():
    graph = _neighbors((("inside", "outside"),))
    domain = select_hop_local_domain(
        ("inside",),
        _provider(graph),
        maximum_nodes=1,
    )
    local = build_local_grounded_semantic_diffusion(
        domain,
        intrinsic_leakage_conductance=2.0,
        bath_temperature=7.5,
    )

    zero_source = np.zeros(1)
    assert local.equilibrium_response(zero_source) == pytest.approx([0.0])
    assert local.equilibrium_response(
        zero_source, absolute=True
    ) == pytest.approx([7.5])
    assert local.equilibrium_response(
        np.array([2.0]), absolute=True
    ) == pytest.approx([7.5 + 2.0 / 3.0])


def test_uniform_leakage_calibration_hits_chain_frontier_target():
    graph = _path(5)
    domain = select_hop_local_domain(
        (0,),
        _provider(graph),
        maximum_nodes=6,
    )
    target = float(np.exp(-1.0))
    calibration = calibrate_uniform_leakage(
        domain,
        anchors=(0,),
        shell_nodes=(5,),
        target_attenuation=target,
        relative_tolerance=1e-6,
    )

    assert calibration.leakage_conductance > 0.0
    assert calibration.leakage_resistance == pytest.approx(
        1.0 / calibration.leakage_conductance
    )
    assert calibration.achieved_attenuation <= target * (1.0 + 1e-6)
    assert calibration.achieved_attenuation >= target * (1.0 - 2e-6)
    source = _source(calibration.model.model.nodes, 0)
    response = calibration.model.equilibrium_response(source)
    assert response[-1] / response[0] == pytest.approx(
        calibration.achieved_attenuation
    )
    assert calibration.source_nodes == (0,)
    assert len(calibration.anchor_screening) == 1
    screening = calibration.anchor_screening[0]
    assert screening.anchor == 0
    assert screening.shell_attenuation == pytest.approx(
        calibration.achieved_attenuation
    )
    assert screening.attenuation_threshold == pytest.approx(target)
    assert screening.radius_lower == 4.0
    assert screening.radius_upper == 5.0
    assert not screening.right_censored
    assert screening.maximum_observed_radius == 5.0
    assert screening.distance_metric == "realized_positive_conductance_hops"


def test_multi_anchor_calibration_reports_tight_and_overgrounded_radii():
    graph = {
        "a0": ("a1",),
        "a1": ("a0", "a2"),
        "a2": ("a1",),
        "b0": ("b1",),
        "b1": ("b0", "b2"),
        "b2": ("b1",),
    }
    domain = select_hop_local_domain(
        ("a0", "b0"),
        graph,
        maximum_nodes=6,
    )
    embeddings = {
        "a0": np.array([0.0]),
        "a1": np.array([0.0]),
        "a2": np.array([0.0]),
        "b0": np.array([10.0]),
        "b1": np.array([13.5]),
        "b2": np.array([17.0]),
    }

    calibration = calibrate_uniform_leakage(
        domain,
        shell_nodes=("a2", "b2"),
        node_embeddings=embeddings,
        length_scale=1.0,
        relative_tolerance=1e-6,
    )

    tight, overgrounded = calibration.anchor_screening
    assert (tight.anchor, overgrounded.anchor) == ("a0", "b0")
    assert tight.shell_attenuation == pytest.approx(
        calibration.achieved_attenuation
    )
    assert tight.radius_upper == 2.0
    assert overgrounded.shell_attenuation < tight.shell_attenuation
    assert overgrounded.radius_upper == 1.0


def test_tail_envelope_rejects_an_early_nonmonotone_shell_crossing():
    lower, upper, censored, maximum = _tail_envelope_crossing(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([1.0, 0.2, 0.8, 0.1]),
        0.5,
    )

    assert (lower, upper, censored, maximum) == (2.0, 3.0, False, 3.0)
    lower, upper, censored, maximum = _tail_envelope_crossing(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 0.9, 0.8]),
        0.5,
    )
    assert (lower, upper, censored, maximum) == (2.0, None, True, 2.0)


def test_calibration_requires_a_reachable_shell_node_for_every_anchor():
    graph = {
        "a0": ("a1",),
        "a1": ("a0",),
        "b0": ("b1",),
        "b1": ("b0",),
    }
    domain = select_hop_local_domain(
        ("a0", "b0"),
        graph,
        maximum_nodes=4,
    )

    with pytest.raises(
        ValueError,
        match="reachable non-source node for anchor 'b0'",
    ):
        calibrate_uniform_leakage(
            domain,
            shell_nodes=("a1",),
        )


def test_non_e_fold_target_is_recorded_as_a_generic_threshold():
    graph = _path(3)
    domain = select_hop_local_domain(
        (0,),
        graph,
        maximum_nodes=4,
    )

    calibration = calibrate_uniform_leakage(
        domain,
        shell_nodes=(3,),
        target_attenuation=0.5,
        relative_tolerance=1e-6,
    )

    screening = calibration.anchor_screening[0]
    assert screening.attenuation_threshold == 0.5
    assert screening.radius_upper is not None


def test_leakage_calibration_reads_each_required_embedding_once():
    graph = _path(2)
    domain = select_hop_local_domain(
        (0,),
        graph,
        maximum_nodes=2,
    )
    calls = {}
    values = {
        0: np.array([0.0]),
        1: np.array([0.5]),
        2: np.array([1.0]),
    }

    class CountingEmbeddings:
        def __getitem__(self, node):
            calls[node] = calls.get(node, 0) + 1
            return values[node]

    calibration = calibrate_uniform_leakage(
        domain,
        shell_nodes=(1,),
        target_attenuation=float(np.exp(-1.0)),
        node_embeddings=CountingEmbeddings(),
        length_scale=1.0,
        relative_tolerance=1e-6,
    )

    assert calibration.achieved_attenuation <= float(np.exp(-1.0)) * (
        1.0 + 1e-6
    )
    assert calls == {0: 1, 1: 1, 2: 1}


def test_nested_domain_diagnostics_report_boundary_influence_and_cut_current():
    graph = _path(8)
    inner_domain = select_hop_local_domain(
        (0,),
        _provider(graph),
        maximum_nodes=4,
    )
    outer_domain = select_hop_local_domain(
        (0,),
        _provider(graph),
        maximum_nodes=8,
    )
    inner = build_local_grounded_semantic_diffusion(
        inner_domain,
        intrinsic_leakage_conductance=0.2,
    )
    outer = build_local_grounded_semantic_diffusion(
        outer_domain,
        intrinsic_leakage_conductance=0.2,
    )
    diagnostics = compare_nested_domains(
        inner,
        outer,
        source_nodes=(0,),
        protected_nodes=(0, 1),
    )

    source = _source(inner.model.nodes, 0)
    response = inner.equilibrium_response(source)
    drained = float(
        (inner.cut_conductance + inner.intrinsic_leakage_conductance)
        @ response
    )
    assert drained == pytest.approx(np.sum(source))
    harmonic = inner.boundary_harmonic_measure()
    protected_indices = [
        inner.model.nodes.index(node) for node in (0, 1)
    ]
    cut_fraction = inner.cut_current_fraction(source)
    assert diagnostics.maximum_protected_absolute_error > 0.0
    assert 0.0 < diagnostics.inner_boundary_harmonic_max < 1.0
    assert diagnostics.inner_boundary_harmonic_max == pytest.approx(
        np.max(harmonic[protected_indices])
    )
    assert 0.0 < cut_fraction < 1.0


def test_k_2k_4k_responses_converge_and_diagnostics_improve_on_a_path():
    graph = _path(16)
    models = []
    for size in (4, 8, 16):
        domain = select_hop_local_domain(
            (0,),
            _provider(graph),
            maximum_nodes=size,
        )
        models.append(
            build_local_grounded_semantic_diffusion(
                domain,
                intrinsic_leakage_conductance=0.2,
            )
        )

    anchor_responses = []
    for model in models:
        source = _source(model.model.nodes, 0)
        anchor_responses.append(model.equilibrium_response(source)[0])
    assert anchor_responses[0] < anchor_responses[1] < anchor_responses[2]

    k_to_4k = compare_nested_domains(
        models[0],
        models[2],
        source_nodes=(0,),
        protected_nodes=(0, 1),
    )
    two_k_to_4k = compare_nested_domains(
        models[1],
        models[2],
        source_nodes=(0,),
        protected_nodes=(0, 1),
    )
    assert (
        two_k_to_4k.maximum_protected_absolute_error
        < k_to_4k.maximum_protected_absolute_error
    )
    assert (
        two_k_to_4k.inner_boundary_harmonic_max
        < k_to_4k.inner_boundary_harmonic_max
    )
    assert k_to_4k.monotonic
    assert two_k_to_4k.monotonic
    assert k_to_4k.minimum_monotone_increment >= 0.0


def test_multiple_anchors_share_one_domain_and_one_precision_factorization():
    graph = _path(6)
    domain = select_hop_local_domain(
        (1, 5),
        _provider(graph),
        maximum_nodes=7,
    )
    local = build_local_grounded_semantic_diffusion(
        domain,
        intrinsic_leakage_conductance=0.1,
    )
    sources = np.column_stack(
        (
            _source(local.model.nodes, 1),
            _source(local.model.nodes, 5),
        )
    )
    response = local.equilibrium_response(sources)

    assert domain.anchors == (1, 5)
    assert domain.nodes == (1, 5, 0, 2, 4, 6, 3)
    assert response.shape == (7, 2)
    assert np.allclose(local.model.precision @ response, sources)
    assert local.model.precision_root.shape == (7, 7)


def test_conditioning_floor_has_roundoff_guard():
    graph = _path(1)
    domain = select_hop_local_domain(
        (0,),
        _provider(graph),
        maximum_nodes=2,
    )

    calibration = calibrate_uniform_leakage(
        domain,
        shell_nodes=(1,),
        target_attenuation=1.0,
    )

    assert calibration.leakage_conductance > 0.0
    assert (
        calibration.model.model.reciprocal_condition_number
        >= calibration.model.model.minimum_reciprocal_condition
    )


def test_leakage_calibration_fails_if_iteration_budget_cannot_converge():
    graph = _path(5)
    domain = select_hop_local_domain(
        (0,),
        _provider(graph),
        maximum_nodes=6,
    )

    with pytest.raises(
        np.linalg.LinAlgError,
        match="exhausted|did not converge",
    ):
        calibrate_uniform_leakage(
            domain,
            shell_nodes=(5,),
            target_attenuation=float(np.exp(-1.0)),
            maximum_iterations=2,
        )


def test_nested_diagnostics_reject_changed_realized_edge_weights():
    graph = _path(2)
    inner_domain = select_hop_local_domain(
        (0,),
        _provider(graph),
        maximum_nodes=2,
    )
    outer_domain = select_hop_local_domain(
        (0,),
        _provider(graph),
        maximum_nodes=3,
    )
    inner = build_local_grounded_semantic_diffusion(
        inner_domain,
        intrinsic_leakage_conductance=0.2,
        node_embeddings={
            0: np.array([0.0]),
            1: np.array([0.0]),
            2: np.array([0.0]),
        },
        length_scale=1.0,
    )
    outer = build_local_grounded_semantic_diffusion(
        outer_domain,
        intrinsic_leakage_conductance=0.2,
        node_embeddings={
            0: np.array([0.0]),
            1: np.array([3.0]),
            2: np.array([3.0]),
        },
        length_scale=1.0,
    )

    with pytest.raises(ValueError, match="same realized precision"):
        compare_nested_domains(inner, outer)


def test_absolute_common_bath_overflow_fails_closed():
    graph = _neighbors((("inside", "outside"),))
    domain = select_hop_local_domain(
        ("inside",),
        _provider(graph),
        maximum_nodes=1,
    )
    local = build_local_grounded_semantic_diffusion(
        domain,
        bath_temperature=1.0e308,
    )

    with pytest.raises(
        np.linalg.LinAlgError,
        match="absolute common-bath response is not finite",
    ):
        local.equilibrium_response(np.array([1.0e308]), absolute=True)


def test_semantic_embeddings_are_snapshotted_once_per_required_node():
    class SingleReadEmbeddings(dict):
        def __init__(self, values):
            super().__init__(values)
            self.reads = {}

        def __getitem__(self, key):
            count = self.reads.get(key, 0) + 1
            self.reads[key] = count
            if count > 1:
                raise AssertionError(f"embedding {key!r} was read more than once")
            return super().__getitem__(key)

    graph = _path(2)
    domain = select_hop_local_domain(
        (0,),
        _provider(graph),
        maximum_nodes=2,
    )
    embeddings = SingleReadEmbeddings(
        {
            0: np.array([0.0]),
            1: np.array([0.5]),
            2: np.array([1.0]),
        }
    )

    local = build_local_grounded_semantic_diffusion(
        domain,
        intrinsic_leakage_conductance=0.2,
        node_embeddings=embeddings,
        length_scale=1.0,
    )

    assert local.precision.shape == (2, 2)
    assert embeddings.reads == {0: 1, 1: 1, 2: 1}


def test_direct_domain_rejects_edges_that_skip_a_hop_shell():
    with pytest.raises(ValueError, match="cannot skip a hop shell"):
        LocalDiffusionDomain(
            nodes=("a", "b", "c"),
            anchors=("a",),
            hop_distance=np.array([0, 1, 2]),
            neighbors=(("b", "c"), ("a", "c"), ("a", "b")),
            maximum_nodes=3,
            complete_distance_shell=True,
            truncated_tie_count=0,
        )


def test_direct_domain_overrun_must_complete_only_the_final_shell():
    with pytest.raises(ValueError, match="must complete one shell"):
        LocalDiffusionDomain(
            nodes=(0, 1, 2),
            anchors=(0,),
            hop_distance=np.array([0, 1, 2]),
            neighbors=((1,), (0, 2), (1,)),
            maximum_nodes=2,
            complete_distance_shell=True,
            truncated_tie_count=0,
        )
