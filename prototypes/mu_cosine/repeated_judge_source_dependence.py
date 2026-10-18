"""Topology-only source-region exposure and ESS diagnostics.

The module operates on full :class:`SourceRegionPartition` regions.  It never
forms a node-by-node Gram matrix, reads outcomes, or claims that the exposure
proxy is an empirical covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from repeated_judge_campaign import FROZEN_WALK_WEIGHTS
from repeated_judge_candidate_capacity import (
    ENDPOINTS_PER_COMPONENT,
    SOURCE_COMPONENT_CAP_FRACTION,
)
from repeated_judge_source_regions import (
    SourceRegionPartition,
    _canonical_source_graph,
    build_source_region_partition,
)


DEFAULT_RHO_GRID = (0.0, 0.025, 0.05, 0.10, 0.20)
DEFAULT_MIN_EFFECTIVE_REGIONS = 20


class SourceDependenceError(ValueError):
    """Raised when a source-dependence input or invariant fails closed."""


@dataclass(frozen=True)
class RegionExposure:
    """Canonical exposure and per-hop outside-region landing diagnostics."""

    region_ids: tuple[str, ...]
    matrix: np.ndarray
    walk_weights: tuple[float, ...]
    per_region_outside_landing_mass: tuple[tuple[float, ...], ...]
    mean_outside_landing_mass_by_hop: tuple[float, ...]


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SourceDependenceError(f"{name} must be a positive integer")
    return value


def _validated_weights(weights):
    try:
        values = tuple(float(value) for value in weights)
    except (TypeError, ValueError) as exc:
        raise SourceDependenceError("walk weights must be numeric") from exc
    if (
        not values
        or any(not math.isfinite(value) or value < 0.0 for value in values)
        or not any(value > 0.0 for value in values)
    ):
        raise SourceDependenceError(
            "walk weights must be finite, nonnegative, and have positive support"
        )
    return values


def _validated_partition(adjacency, partition):
    if not isinstance(partition, SourceRegionPartition):
        raise SourceDependenceError("partition must be a SourceRegionPartition")
    region_ids = tuple(sorted(partition.region_nodes))
    if (
        len(region_ids) != partition.target_region_count
        or not region_ids
        or any(not isinstance(region, str) or not region for region in region_ids)
    ):
        raise SourceDependenceError("partition has invalid stable region IDs")
    covered = set()
    expected_assignment = {}
    for region in region_ids:
        nodes = partition.region_nodes[region]
        if not nodes:
            raise SourceDependenceError("full source regions must be nonempty")
        if covered & set(nodes):
            raise SourceDependenceError("source regions must be disjoint")
        for node in nodes:
            expected_assignment[node] = region
        covered.update(nodes)
    if covered != set(adjacency):
        raise SourceDependenceError(
            "full source regions must cover the canonical graph exactly"
        )
    if partition.assignment != expected_assignment:
        raise SourceDependenceError("partition assignment disagrees with region nodes")
    return region_ids


def _advance_walk(distribution, adjacency):
    output = {}
    for node, mass in sorted(distribution.items()):
        neighbors = adjacency[node]
        if not neighbors:
            output[node] = output.get(node, 0.0) + mass
            continue
        share = mass / len(neighbors)
        for neighbor in sorted(neighbors):
            output[neighbor] = output.get(neighbor, 0.0) + share
    return output


def build_region_exposure(
    parents,
    children,
    partition,
    *,
    walk_weights=FROZEN_WALK_WEIGHTS,
):
    """Build ``E = Z_normalized Z_normalized.T`` without a node Gram."""
    try:
        adjacency = _canonical_source_graph(parents, children)
    except Exception as exc:
        if isinstance(exc, SourceDependenceError):
            raise
        raise SourceDependenceError(str(exc)) from exc
    region_ids = _validated_partition(adjacency, partition)
    weights = _validated_weights(walk_weights)
    features = []
    outside_landing = []
    for region in region_ids:
        nodes = partition.region_nodes[region]
        current = {node: 1.0 / len(nodes) for node in sorted(nodes)}
        accumulated = {}
        region_outside_landing = []
        for hop, weight in enumerate(weights):
            retained = math.fsum(
                mass for node, mass in current.items() if node in nodes
            )
            region_outside_landing.append(
                float(min(1.0, max(0.0, 1.0 - retained)))
            )
            scale = math.sqrt(weight)
            if scale:
                for node, mass in sorted(current.items()):
                    accumulated[node] = accumulated.get(node, 0.0) + scale * mass
            if hop + 1 < len(weights):
                current = _advance_walk(current, adjacency)
        norm = math.sqrt(math.fsum(value * value for value in accumulated.values()))
        if not math.isfinite(norm) or norm <= 0.0:
            raise SourceDependenceError("every cumulative-walk feature must be nonzero")
        features.append({node: value / norm for node, value in accumulated.items()})
        outside_landing.append(tuple(region_outside_landing))

    # Accumulate only the K x K Gram from sparse region features.  No V x V
    # node Gram is constructed.
    by_node = {}
    for region_index, feature in enumerate(features):
        for node, value in feature.items():
            by_node.setdefault(node, []).append((region_index, value))
    exposure = np.zeros((len(region_ids), len(region_ids)), dtype=float)
    for node in sorted(by_node):
        entries = by_node[node]
        for left_offset, (left, left_value) in enumerate(entries):
            for right, right_value in entries[left_offset:]:
                product = left_value * right_value
                exposure[left, right] += product
                if left != right:
                    exposure[right, left] += product
    exposure = 0.5 * (exposure + exposure.T)
    diagonal = np.diag(exposure)
    if np.any(diagonal <= 0.0) or not np.isfinite(exposure).all():
        raise SourceDependenceError("region exposure is not finite with positive diagonal")
    exposure /= np.sqrt(diagonal[:, None] * diagonal[None, :])
    exposure = 0.5 * (exposure + exposure.T)
    np.fill_diagonal(exposure, 1.0)
    if np.min(exposure) < -1e-12:
        raise SourceDependenceError("nonnegative walk features produced negative exposure")
    eigenvalues = np.linalg.eigvalsh(exposure)
    tolerance = 64.0 * np.finfo(float).eps * max(1, len(region_ids))
    if eigenvalues[0] < -tolerance:
        raise SourceDependenceError("region exposure is not numerically PSD")
    mean_outside_landing = tuple(
        float(np.mean([row[hop] for row in outside_landing]))
        for hop in range(len(weights))
    )
    exposure.setflags(write=False)
    return RegionExposure(
        region_ids,
        exposure,
        weights,
        tuple(outside_landing),
        mean_outside_landing,
    )


def compute_per_hop_outside_landing_mass(
    parents,
    children,
    partition,
    *,
    walk_weights=FROZEN_WALK_WEIGHTS,
):
    """Return hop-h mass landing outside each start region.

    This is not a first-exit probability: a walk may leave and return before
    hop ``h``.
    """
    exposure = build_region_exposure(
        parents, children, partition, walk_weights=walk_weights
    )
    return {
        "region_ids": list(exposure.region_ids),
        "per_region": {
            region: list(exposure.per_region_outside_landing_mass[index])
            for index, region in enumerate(exposure.region_ids)
        },
        "mean_by_hop": list(exposure.mean_outside_landing_mass_by_hop),
    }


def full_region_capacities(
    partition,
    components_per_corpus,
    *,
    endpoints_per_component=ENDPOINTS_PER_COMPONENT,
    cap_fraction=SOURCE_COMPONENT_CAP_FRACTION,
):
    """Return exact optimistic full-region capacities in stable ID order."""
    components_per_corpus = _positive_integer(
        components_per_corpus, "components_per_corpus"
    )
    endpoints_per_component = _positive_integer(
        endpoints_per_component, "endpoints_per_component"
    )
    if cap_fraction != SOURCE_COMPONENT_CAP_FRACTION:
        raise SourceDependenceError("cap_fraction must equal the frozen 0.10")
    if not isinstance(partition, SourceRegionPartition):
        raise SourceDependenceError("partition must be a SourceRegionPartition")
    cap = components_per_corpus // 10
    return {
        region: min(len(partition.region_nodes[region]) // endpoints_per_component, cap)
        for region in sorted(partition.region_nodes)
    }


def _validated_exposure(exposure):
    if not isinstance(exposure, RegionExposure):
        raise SourceDependenceError("exposure must be a RegionExposure")
    matrix = np.asarray(exposure.matrix, dtype=float)
    size = len(exposure.region_ids)
    if (
        tuple(exposure.region_ids) != tuple(sorted(set(exposure.region_ids)))
        or any(not isinstance(region, str) or not region for region in exposure.region_ids)
        or matrix.shape != (size, size)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0)
        or not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0)
        or np.min(matrix) < -1e-12
        or np.linalg.eigvalsh(matrix)[0] < -1e-10
    ):
        raise SourceDependenceError(
            "exposure must be finite, symmetric, nonnegative, PSD, and unit diagonal"
        )
    return matrix


def _validated_counts(exposure, values, name):
    if isinstance(values, Mapping) and "counts_by_region" in values:
        values = values["counts_by_region"]
    if not isinstance(values, Mapping) or set(values) != set(exposure.region_ids):
        raise SourceDependenceError(f"{name} must contain every canonical region ID")
    counts = []
    for region in exposure.region_ids:
        value = values[region]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SourceDependenceError(f"{name} must contain nonnegative integers")
        counts.append(value)
    return np.asarray(counts, dtype=int)


def exposure_aware_greedy_allocation(
    exposure,
    capacities,
    components_per_corpus,
):
    """Allocate components by the exact greedy quadratic-objective increment."""
    matrix = _validated_exposure(exposure)
    components_per_corpus = _positive_integer(
        components_per_corpus, "components_per_corpus"
    )
    capacity = _validated_counts(exposure, capacities, "capacities")
    if int(np.sum(capacity)) < components_per_corpus:
        raise SourceDependenceError("full-region capacities cannot allocate all components")
    counts = np.zeros(len(exposure.region_ids), dtype=int)
    matrix_counts = np.zeros(len(exposure.region_ids), dtype=float)
    assignments = []
    for _ in range(components_per_corpus):
        eligible = np.flatnonzero(counts < capacity)
        if not len(eligible):
            raise SourceDependenceError("allocation exhausted eligible regions")
        winner = min(
            map(int, eligible),
            key=lambda index: (
                float(2.0 * matrix_counts[index] + matrix[index, index]),
                exposure.region_ids[index],
            ),
        )
        counts[winner] += 1
        matrix_counts += matrix[:, winner]
        assignments.append(exposure.region_ids[winner])
    objective = float(counts @ matrix @ counts)
    return {
        "counts_by_region": {
            region: int(counts[index])
            for index, region in enumerate(exposure.region_ids)
        },
        "assignment_region_ids": assignments,
        "used_region_count": int(np.count_nonzero(counts)),
        "quadratic_exposure": objective,
    }


def _validated_rho(rho):
    if isinstance(rho, bool):
        raise SourceDependenceError("rho must be numeric, not boolean")
    try:
        rho = float(rho)
    except (TypeError, ValueError) as exc:
        raise SourceDependenceError("rho must be numeric") from exc
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise SourceDependenceError("rho must lie in [0,1]")
    return rho


def exact_mean_ess(exposure, allocation, rho):
    """Exact scalar-mean design effect and ESS for the stipulated path."""
    matrix = _validated_exposure(exposure)
    counts = _validated_counts(exposure, allocation, "allocation")
    total = int(np.sum(counts))
    if total < 1:
        raise SourceDependenceError("allocation must contain at least one component")
    rho = _validated_rho(rho)
    quadratic = float(counts @ matrix @ counts)
    variance_sum = float((1.0 - rho) * total + rho * quadratic)
    design_effect = variance_sum / total
    return {
        "components": total,
        "rho": rho,
        "quadratic_exposure": quadratic,
        "one_C_one": variance_sum,
        "design_effect": design_effect,
        "effective_components": total * total / variance_sum,
    }


def certified_ess_lower_bound(
    exposure,
    capacities,
    components_per_corpus,
    rho,
):
    """Allocation-free ESS floor for every cap-feasible count vector."""
    matrix = _validated_exposure(exposure)
    total = _positive_integer(components_per_corpus, "components_per_corpus")
    capacity = _validated_counts(exposure, capacities, "capacities")
    if int(np.sum(capacity)) < total:
        raise SourceDependenceError("capacities cannot supply the requested total")
    rho = _validated_rho(rho)
    remaining = total
    max_sum_squares = 0
    for index in sorted(
        range(len(capacity)),
        key=lambda item: (-int(capacity[item]), exposure.region_ids[item]),
    ):
        taken = min(remaining, int(capacity[index]))
        max_sum_squares += taken * taken
        remaining -= taken
        if not remaining:
            break
    # For a symmetric nonnegative matrix, the spectral radius is no greater
    # than its largest row sum.  ``nextafter`` guards the floating reductions
    # outward; unlike an eigensolver residual heuristic, this preserves the
    # analytic Perron/Gershgorin inequality for the stored float matrix.
    row_sums = [
        float(np.nextafter(math.fsum(map(float, row)), math.inf))
        for row in matrix
    ]
    lambda_upper = max(row_sums)
    spectral_upper = float(
        np.nextafter(lambda_upper * max_sum_squares, math.inf)
    )
    capacity_image_upper = []
    for row in matrix:
        products = [
            float(np.nextafter(float(value) * int(limit), math.inf))
            if value > 0.0 and limit > 0
            else 0.0
            for value, limit in zip(row, capacity)
        ]
        capacity_image_upper.append(
            float(np.nextafter(math.fsum(products), math.inf))
        )
    capacity_upper = float(
        np.nextafter(total * max(capacity_image_upper), math.inf)
    )
    quadratic_upper = max(float(total), min(spectral_upper, capacity_upper))
    variance_upper = float((1.0 - rho) * total + rho * quadratic_upper)
    return {
        "components": total,
        "rho": rho,
        "lambda_max_upper": lambda_upper,
        "lambda_max_upper_method": (
            "maximum nonnegative row sum with outward float guard"
        ),
        "maximum_feasible_sum_squares": int(max_sum_squares),
        "spectral_quadratic_upper_bound": float(spectral_upper),
        "capacity_quadratic_upper_bound": float(capacity_upper),
        "quadratic_exposure_upper_bound": quadratic_upper,
        "design_effect_upper_bound": variance_upper / total,
        "effective_components_lower_bound": total * total / variance_upper,
    }


def component_correlation_matrix(exposure, allocation, rho):
    """Materialize the stipulated component correlation for small diagnostics."""
    matrix = _validated_exposure(exposure)
    counts = _validated_counts(exposure, allocation, "allocation")
    rho = _validated_rho(rho)
    indices = np.repeat(np.arange(len(counts)), counts)
    if not len(indices):
        raise SourceDependenceError("allocation must contain at least one component")
    source = matrix[np.ix_(indices, indices)]
    return (1.0 - rho) * np.eye(len(indices)) + rho * source


def _exposure_diagnostics(exposure):
    matrix = _validated_exposure(exposure)
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = max(matrix.shape) * np.finfo(float).eps * max(
        1.0, float(eigenvalues[-1])
    )
    trace_square = float(np.sum(matrix * matrix))
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    quantile_levels = (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
    quantiles = (
        {str(level): float(np.quantile(off_diagonal, level)) for level in quantile_levels}
        if len(off_diagonal)
        else {str(level): 0.0 for level in quantile_levels}
    )
    return {
        "region_ids": list(exposure.region_ids),
        "matrix": matrix.tolist(),
        "eigenvalues_descending": list(map(float, eigenvalues[::-1])),
        "numerical_rank": int(np.sum(eigenvalues > tolerance)),
        "effective_rank": float(np.trace(matrix) ** 2 / trace_square),
        "off_diagonal_quantiles": quantiles,
        "per_region_outside_landing_mass_by_hop": {
            region: list(exposure.per_region_outside_landing_mass[index])
            for index, region in enumerate(exposure.region_ids)
        },
        "mean_outside_landing_mass_by_hop": list(
            exposure.mean_outside_landing_mass_by_hop
        ),
    }


def audit_source_dependence(
    parents,
    children,
    target_region_count,
    registered_sizes,
    *,
    rho_grid=DEFAULT_RHO_GRID,
):
    """Audit one deterministic full-region partition across registered G/rho."""
    target_region_count = _positive_integer(
        target_region_count, "target_region_count"
    )
    registered_sizes = tuple(registered_sizes)
    if (
        not registered_sizes
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in registered_sizes
        )
        or len(set(registered_sizes)) != len(registered_sizes)
    ):
        raise SourceDependenceError(
            "registered_sizes must contain unique positive integers"
        )
    rhos = tuple(_validated_rho(value) for value in rho_grid)
    if not rhos or len(set(rhos)) != len(rhos):
        raise SourceDependenceError("rho_grid must contain unique values")
    partition = build_source_region_partition(
        parents, children, target_region_count, halo_hops=0
    )
    exposure = build_region_exposure(parents, children, partition)
    matrix = _validated_exposure(exposure)
    size_results = {}
    for total in registered_sizes:
        capacities = full_region_capacities(partition, total)
        capacity_pass = sum(capacities.values()) >= total
        if capacity_pass:
            allocation = exposure_aware_greedy_allocation(
                exposure, capacities, total
            )
            exact = {
                str(rho): exact_mean_ess(exposure, allocation, rho)
                for rho in rhos
            }
            lower = {
                str(rho): certified_ess_lower_bound(
                    exposure, capacities, total, rho
                )
                for rho in rhos
            }
            used = allocation["used_region_count"]
            cap_respected = all(
                allocation["counts_by_region"][region] <= capacities[region]
                for region in exposure.region_ids
            )
        else:
            allocation = None
            exact = None
            lower = None
            used = 0
            cap_respected = False
        correlation_path_valid = bool(
            np.isfinite(matrix).all()
            and np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0)
            and np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0)
            and np.linalg.eigvalsh(matrix)[0] >= -1e-10
            and all(0.0 <= rho <= 1.0 for rho in rhos)
        )
        gates = {
            "capacity_supplies_all_components": capacity_pass,
            "allocation_supplies_exactly_G": bool(
                allocation is not None
                and sum(allocation["counts_by_region"].values()) == total
            ),
            "allocation_respects_cap": cap_respected,
            "uses_at_least_20_source_regions": used >= DEFAULT_MIN_EFFECTIVE_REGIONS,
            "exposure_and_correlation_path_valid": correlation_path_valid,
        }
        gates["all_topology_gates_pass"] = all(gates.values())
        size_results[str(total)] = {
            "components_per_corpus": total,
            "capacities_by_region": capacities,
            "optimistic_capacity_upper_bound": int(sum(capacities.values())),
            "allocation": allocation,
            "exact_mean_ess_by_rho": exact,
            "certified_mean_ess_lower_bound_by_rho": lower,
            "gates": gates,
        }
    return {
        "target_region_count": target_region_count,
        "actual_region_count": len(exposure.region_ids),
        "walk_weights": list(exposure.walk_weights),
        "rho_grid": list(rhos),
        "partition_assignment_record": partition.assignment_record,
        "exposure": _exposure_diagnostics(exposure),
        "registered_size_results": size_results,
        "gates": {
            "all_registered_sizes_pass": all(
                row["gates"]["all_topology_gates_pass"]
                for row in size_results.values()
            )
        },
        "authorization": {
            "attempted_input_inventory_authorized": False,
            "candidate_selection_authorized": False,
            "nomic_authorized": False,
            "live_scoring_authorized": False,
            "covariance_promotion_authorized": False,
            "independent_batching_authorized": False,
            "qr_specialization_authorized": False,
            "cuda_authorized": False,
        },
    }


# Readable aliases for callers that phrase the operation around components.
allocate_components_exposure_aware = exposure_aware_greedy_allocation
audit_source_region_dependence = audit_source_dependence
compute_per_hop_mean_escape = compute_per_hop_outside_landing_mass
