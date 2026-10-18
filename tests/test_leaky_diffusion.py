#!/usr/bin/env python3
"""Analytic tests for semantically weighted, grounded graph diffusion."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from unifyweaver.graph.leaky_diffusion import (  # noqa: E402
    build_grounded_semantic_diffusion,
    combinatorial_laplacian,
    semantic_conductance_matrix,
)


def _neighbors(edges, extra=()):
    output = {node: set() for node in extra}
    for left, right in edges:
        output.setdefault(left, set()).add(right)
        output.setdefault(right, set()).add(left)
    return {node: frozenset(values) for node, values in output.items()}


def _two_node(distance, *, leakage=0.2, length_scale=1.0):
    return build_grounded_semantic_diffusion(
        ("a", "b"),
        _neighbors((("a", "b"),)),
        leakage_conductance=leakage,
        node_embeddings={
            "a": np.array([0.0, 0.0]),
            "b": np.array([distance, 0.0]),
        },
        length_scale=length_scale,
    )


def test_semantics_modulate_only_existing_graph_edges():
    nodes = ("a", "b", "c")
    embeddings = {
        "a": np.array([0.0, 0.0]),
        "b": np.array([0.1, 0.0]),
        "c": np.array([3.0, 0.0]),
    }
    returned, conductance = semantic_conductance_matrix(
        nodes,
        _neighbors((("a", "b"), ("b", "c"))),
        embeddings,
        length_scale=1.0,
    )
    assert returned == nodes
    assert conductance[0, 1] > conductance[1, 2] > 0.0
    assert conductance[0, 2] == 0.0
    assert np.allclose(conductance, conductance.T)
    assert np.array_equal(np.diag(conductance), np.zeros(3))


def test_semantic_floor_retains_distant_topological_edges_without_shortcuts():
    _, conductance = semantic_conductance_matrix(
        ("a", "b", "c"),
        _neighbors((("a", "b"), ("b", "c"))),
        {
            "a": np.array([0.0]),
            "b": np.array([100.0]),
            "c": np.array([200.0]),
        },
        length_scale=1.0,
        conductance_floor=0.05,
    )
    assert conductance[0, 1] == pytest.approx(0.05)
    assert conductance[1, 2] == pytest.approx(0.05)
    assert conductance[0, 2] == 0.0


def test_topology_only_conductance_is_unit_weighted():
    _, conductance = semantic_conductance_matrix(
        ("a", "b", "c"),
        _neighbors((("a", "b"),), extra=("c",)),
    )
    assert np.array_equal(
        conductance,
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_two_node_precision_green_and_root_match_analytic_circuit():
    alpha = 0.2
    model = _two_node(0.0, leakage=alpha)
    expected_precision = np.array([[1.0 + alpha, -1.0], [-1.0, 1.0 + alpha]])
    denominator = alpha * (alpha + 2.0)
    expected_green = np.array(
        [[alpha + 1.0, 1.0], [1.0, alpha + 1.0]]
    ) / denominator

    assert np.allclose(model.precision, expected_precision)
    assert np.allclose(model.precision_root.T @ model.precision_root, expected_precision)
    assert np.allclose(model.green_kernel(), expected_green)
    assert np.allclose(model.precision @ model.green_kernel(), np.eye(2))
    assert model.minimum_precision_eigenvalue == pytest.approx(alpha)
    assert model.maximum_precision_eigenvalue == pytest.approx(alpha + 2.0)


def test_uniform_grounding_is_the_regularized_laplacian_resolvent():
    alpha = 0.4
    model = build_grounded_semantic_diffusion(
        ("a", "b", "c"),
        _neighbors((("a", "b"), ("b", "c"))),
        leakage_conductance=alpha,
    )
    reference = np.linalg.solve(
        np.eye(3) + model.laplacian / alpha,
        np.eye(3),
    ) / alpha
    assert np.allclose(model.green_kernel(), reference)


def test_equilibrium_satisfies_kcl_and_superposition_for_multiple_sources():
    model = build_grounded_semantic_diffusion(
        ("a", "b", "c"),
        _neighbors((("a", "b"), ("b", "c"))),
        leakage_conductance=0.1,
    )
    first = np.array([1.0, 0.0, 0.0])
    second = np.array([0.0, 0.0, 2.0])
    both = np.column_stack((first, second))
    response = model.equilibrium_response(both)

    assert np.allclose(model.precision @ response, both)
    assert np.allclose(response[:, 0], model.equilibrium_response(first))
    assert np.allclose(response[:, 1], model.equilibrium_response(second))
    assert np.all(response >= 0.0)


def test_sparse_boundary_grounding_requires_every_component_to_reach_ground():
    nodes = ("a", "b", "z")
    neighbors = _neighbors((("a", "b"),), extra=("z",))
    with pytest.raises(np.linalg.LinAlgError, match="every graph component"):
        build_grounded_semantic_diffusion(
            nodes,
            neighbors,
            leakage_conductance={"a": 0.1},
        )

    model = build_grounded_semantic_diffusion(
        nodes,
        neighbors,
        leakage_conductance={"a": 0.1, "z": 0.2},
    )
    assert model.minimum_precision_eigenvalue > 0.0
    assert np.allclose(model.leakage_conductance, [0.1, 0.0, 0.2])


def test_correlation_precision_root_whitens_normalized_green_kernel():
    model = build_grounded_semantic_diffusion(
        ("a", "b", "c"),
        _neighbors((("a", "b"), ("b", "c"))),
        leakage_conductance={"a": 0.2, "b": 0.1, "c": 0.4},
    )
    correlation = model.green_kernel(normalize=True)
    root = model.correlation_precision_root()
    assert np.allclose(np.diag(correlation), 1.0)
    assert np.min(np.linalg.eigvalsh(correlation)) > 0.0
    assert np.allclose(root.T @ root @ correlation, np.eye(3), atol=1e-11)


def test_heat_semigroup_and_step_relaxation_to_equilibrium():
    model = build_grounded_semantic_diffusion(
        ("a", "b", "c"),
        _neighbors((("a", "b"), ("b", "c"))),
        leakage_conductance=0.25,
    )
    source = np.array([1.0, 0.0, 0.0])
    assert np.allclose(model.heat_kernel(0.0), np.eye(3))
    assert np.allclose(model.step_response(source, time=0.0), np.zeros(3))
    assert np.allclose(
        model.heat_kernel(0.3) @ model.heat_kernel(0.7),
        model.heat_kernel(1.0),
        atol=1e-12,
    )
    early = model.step_response(source, time=1e-8)
    late = model.step_response(source, time=1e4)
    equilibrium = model.equilibrium_response(source)
    assert np.linalg.norm(early) < 1e-7
    assert np.allclose(late, equilibrium, atol=1e-12)
    assert np.allclose(model.impulse_response(source, time=0.5), model.heat_kernel(0.5) @ source)


def test_semantically_distant_edge_has_larger_effective_resistance():
    near = _two_node(0.1, leakage=0.1)
    far = _two_node(2.0, leakage=0.1)
    assert near.conductance[0, 1] > far.conductance[0, 1]
    assert near.resistance_distance()[0, 1] < far.resistance_distance()[0, 1]
    for distance in (near.resistance_distance(), far.resistance_distance()):
        assert np.allclose(distance, distance.T)
        assert np.array_equal(np.diag(distance), np.zeros(2))


def test_implementation_never_requires_explicit_matrix_inverse(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("explicit inverse is forbidden")

    monkeypatch.setattr(np.linalg, "inv", forbidden)
    model = _two_node(0.5)
    source = np.array([1.0, 0.0])
    assert np.all(np.isfinite(model.equilibrium_response(source)))
    assert np.all(np.isfinite(model.green_kernel()))
    assert np.all(np.isfinite(model.correlation_precision_root()))


def test_model_arrays_are_read_only():
    model = _two_node(0.0)
    with pytest.raises(ValueError):
        model.precision[0, 0] = 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"node_embeddings": {"a": [0.0], "b": [1.0]}}, "length_scale"),
        ({"length_scale": 1.0}, "requires node_embeddings"),
        ({"conductance_floor": 0.1}, "requires node_embeddings"),
        (
            {
                "node_embeddings": {"a": [0.0], "b": [1.0]},
                "length_scale": 1.0,
                "conductance_floor": 1.0,
            },
            r"\[0, 1\)",
        ),
    ],
)
def test_invalid_semantic_conductance_contract_fails_closed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        semantic_conductance_matrix(
            ("a", "b"),
            _neighbors((("a", "b"),)),
            **kwargs,
        )


def test_invalid_conductance_and_ungrounded_network_fail_closed():
    with pytest.raises(ValueError, match="symmetric"):
        combinatorial_laplacian(np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="nonnegative"):
        combinatorial_laplacian(np.array([[0.0, -1.0], [-1.0, 0.0]]))
    with pytest.raises(ValueError, match="at least one"):
        build_grounded_semantic_diffusion(
            ("a", "b"),
            _neighbors((("a", "b"),)),
            leakage_conductance=0.0,
        )

def test_semantic_rbf_is_stable_at_extreme_float64_scales():
    edge = _neighbors((("a", "b"),))
    _, opposite = semantic_conductance_matrix(
        ("a", "b"),
        edge,
        {"a": [1e308], "b": [-1e308]},
        length_scale=1e308,
    )
    assert opposite[0, 1] == pytest.approx(np.exp(-2.0))

    _, small_coordinate = semantic_conductance_matrix(
        ("a", "b"),
        edge,
        {"a": [1e308, 1.0], "b": [1e308, 0.0]},
        length_scale=1.0,
    )
    assert small_coordinate[0, 1] == pytest.approx(np.exp(-0.5))

    _, underflow = semantic_conductance_matrix(
        ("a", "b"),
        edge,
        {"a": [0.0], "b": [1.0]},
        length_scale=np.nextafter(0.0, 1.0),
    )
    assert np.isfinite(underflow).all()
    assert underflow[0, 1] == 0.0


def test_normalized_heat_is_stable_per_disconnected_component():
    nodes = ("a", "b", "c", "d", "z")
    model = build_grounded_semantic_diffusion(
        nodes,
        _neighbors((("a", "b"), ("c", "d")), extra=("z",)),
        leakage_conductance={
            "a": 1.0,
            "b": 1.0,
            "c": 1000.0,
            "d": 1000.0,
            "z": 10000.0,
        },
    )
    raw = model.heat_kernel(1.0)
    assert np.array_equal(raw[2:4, 2:4], np.zeros((2, 2)))

    normalized = model.heat_kernel(1.0, normalize=True)
    expected = np.eye(5)
    expected[0, 1] = expected[1, 0] = np.tanh(1.0)
    expected[2, 3] = expected[3, 2] = np.tanh(1.0)
    assert np.isfinite(normalized).all()
    assert np.allclose(normalized, expected, atol=1e-12)


def test_step_response_uses_small_time_stable_phi_function():
    singleton = build_grounded_semantic_diffusion(
        ("a",),
        _neighbors((), extra=("a",)),
        leakage_conductance=1.0,
    )
    tiny = singleton.step_response(np.array([1.0]), time=1e-20)
    assert tiny[0] > 0.0
    assert tiny[0] == pytest.approx(1e-20)

    weakly_grounded = build_grounded_semantic_diffusion(
        ("a",),
        _neighbors((), extra=("a",)),
        leakage_conductance=1e-308,
    )
    weak_tiny = weakly_grounded.step_response(np.array([1.0]), time=1e-20)
    assert weak_tiny[0] > 0.0
    assert weak_tiny[0] == pytest.approx(1e-20)

    model = build_grounded_semantic_diffusion(
        ("a", "b"),
        _neighbors((("a", "b"),)),
        leakage_conductance=1e-4,
    )
    source = np.array([1.0, 0.0])
    time = 1e-12
    assert np.allclose(
        model.step_response(source, time=time) / time,
        source,
        rtol=1e-10,
        atol=1e-10,
    )


def test_float64_condition_contract_fails_closed_and_records_override():
    with pytest.raises(np.linalg.LinAlgError, match="too ill-conditioned"):
        build_grounded_semantic_diffusion(
            ("a", "b"),
            _neighbors((("a", "b"),)),
            leakage_conductance=1e-9,
        )

    accepted = build_grounded_semantic_diffusion(
        ("a", "b"),
        _neighbors((("a", "b"),)),
        leakage_conductance=1e-7,
    )
    assert accepted.reciprocal_condition_number >= np.sqrt(np.finfo(float).eps)
    assert np.isfinite(accepted.green_kernel()).all()

    explicit = build_grounded_semantic_diffusion(
        ("a", "b"),
        _neighbors((("a", "b"),)),
        leakage_conductance=1e-9,
        minimum_reciprocal_condition=1e-10,
    )
    assert explicit.reciprocal_condition_number >= 1e-10
    assert explicit.minimum_reciprocal_condition == 1e-10


def test_unrepresentable_equilibrium_scale_fails_but_large_finite_green_survives():
    with pytest.raises(np.linalg.LinAlgError, match="unrepresentable"):
        build_grounded_semantic_diffusion(
            ("a",),
            _neighbors((), extra=("a",)),
            leakage_conductance=np.nextafter(0.0, 1.0),
        )

    model = build_grounded_semantic_diffusion(
        ("a",),
        _neighbors((), extra=("a",)),
        leakage_conductance=1e-308,
    )
    assert np.isfinite(model.green_kernel()).all()
    assert model.green_kernel()[0, 0] == pytest.approx(1e308)


def test_sparse_leakage_mapping_rejects_unknown_node_keys():
    with pytest.raises(ValueError, match="unknown nodes.*typo"):
        build_grounded_semantic_diffusion(
            ("a", "b"),
            _neighbors((("a", "b"),)),
            leakage_conductance={"a": 0.1, "typo": 0.2},
        )


def test_resistance_matches_unit_current_energy_without_green_cancellation():
    model = build_grounded_semantic_diffusion(
        ("a", "b", "c"),
        _neighbors((("a", "b"), ("b", "c"))),
        leakage_conductance={"a": 0.2, "b": 0.1, "c": 0.4},
    )
    squared = model.resistance_distance(squared=True)
    for left in range(3):
        for right in range(left):
            current = np.zeros(3)
            current[left] = 1.0
            current[right] = -1.0
            voltage = model.equilibrium_response(current)
            assert squared[left, right] == pytest.approx(current @ voltage)


def test_public_model_constructor_validates_precision_root_invariant():
    model = _two_node(0.0)
    with pytest.raises(ValueError, match="precision_root"):
        replace(model, precision_root=np.eye(2))
