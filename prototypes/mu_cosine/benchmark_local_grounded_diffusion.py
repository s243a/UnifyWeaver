#!/usr/bin/env python3
"""Deterministic CPU scaling benchmark for query-local grounded diffusion.

The benchmark deliberately represents a potentially million-node graph through
an O(1) incident-neighbor provider.  Query-local BFS therefore touches only K
retained nodes and their fixed-degree adjacency.  The omitted exterior is not
discarded: every cut edge becomes a Dirichlet shunt to the common bath, exactly
as in :mod:`unifyweaver.graph.local_diffusion`.

The measured phases are kept distinct:

``domain_ms``
    deterministic multi-source BFS and local-domain validation;
``assembly_ms``
    unit-conductance dense precision assembly, including cut-edge shunts;
``cholesky_ms``
    raw dense Cholesky factorization of that precision;
``pipeline_build_ms``
    the complete validated reference builder (useful for measuring current
    implementation overhead beyond the raw phases); and
``solve_ms``
    one equilibrium solve through the stored reference precision root.

The implicit provider makes traversal independent of the total node universe
apart from integer identifier arithmetic.  Dense local algebra is still
quadratic in memory and cubic in K for factorization.  Locality consequently
makes a million-node *universe* feasible; it does not make a million-node dense
retained domain feasible.  For one million float64 nodes, one dense matrix is
about 7.28 TiB, while the current four-matrix reference model needs at least
about 29.1 TiB before temporaries.

This is a compute-only microbenchmark, not a deployment latency claim.  The
synthetic graph has constant degree, unit edge conductance, one anchor, and no
embedding lookup or storage cost.  Timings are median/MAD across independent
trials.  The current reference solve calls ``numpy.linalg.solve`` on triangular
roots; an optimized triangular BLAS path should have O(K^2) work, but the
reported solve time intentionally measures the implementation that exists.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unifyweaver.graph.local_diffusion import (  # noqa: E402
    build_local_grounded_semantic_diffusion,
    select_hop_local_domain,
)


_FLOAT64_BYTES = np.dtype(np.float64).itemsize
_MIB = 1024.0**2
_TIB = 1024.0**4


@dataclass(frozen=True)
class Timing:
    median_ms: float
    mad_ms: float


class ImplicitCirculantGraph:
    """A fixed-degree undirected graph with no O(N) materialized state."""

    def __init__(self, node_count: int, degree: int):
        if node_count < 3:
            raise ValueError("node_count must be at least 3")
        if degree < 2 or degree % 2:
            raise ValueError("degree must be a positive even integer")
        if degree >= node_count:
            raise ValueError("degree must be smaller than node_count")
        self.node_count = node_count
        self.degree = degree
        self.calls = 0

    def __call__(self, node: int) -> tuple[int, ...]:
        if not 0 <= node < self.node_count:
            raise KeyError(node)
        self.calls += 1
        half = self.degree // 2
        neighbors = {
            (node + offset) % self.node_count
            for offset in range(1, half + 1)
        }
        neighbors.update(
            (node - offset) % self.node_count
            for offset in range(1, half + 1)
        )
        return tuple(sorted(neighbors))


def _csv_positive_ints(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        )
    return values


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a nonnegative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a positive finite number"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return parsed


def _median_mad(values: list[float]) -> Timing:
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return Timing(median_ms=median, mad_ms=mad)


def _time_callable(
    function: Callable[[], object],
    *,
    warmups: int,
    trials: int,
    inner_repeats: int = 1,
) -> Timing:
    for _ in range(warmups):
        output = function()
        del output

    elapsed_ms: list[float] = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(inner_repeats):
            output = function()
            del output
        elapsed_ms.append(
            1000.0 * (time.perf_counter() - start) / inner_repeats
        )
    return _median_mad(elapsed_ms)


def _select_domain(universe_nodes: int, degree: int, retained_nodes: int):
    graph = ImplicitCirculantGraph(universe_nodes, degree)
    domain = select_hop_local_domain(
        (0,),
        graph,
        maximum_nodes=retained_nodes,
    )
    return domain, graph.calls


def _assemble_unit_precision(domain, intrinsic_leakage: float) -> np.ndarray:
    """Assemble the unit-edge Dirichlet precision without factorization.

    This is the no-embedding branch of the production local builder, split out
    here solely so assembly and factorization can be timed independently.
    """

    nodes = domain.nodes
    index = {node: row for row, node in enumerate(nodes)}
    conductance = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    cut_conductance = np.zeros(len(nodes), dtype=np.float64)
    for left, incident in zip(nodes, domain.neighbors):
        left_row = index[left]
        for right in incident:
            right_row = index.get(right)
            if right_row is None:
                cut_conductance[left_row] += 1.0
            else:
                conductance[left_row, right_row] = 1.0
                conductance[right_row, left_row] = 1.0

    precision = np.diag(np.sum(conductance, axis=1)) - conductance
    diagonal = np.diag_indices_from(precision)
    precision[diagonal] += intrinsic_leakage + cut_conductance
    return precision


def _environment_identity() -> str:
    keys = (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    values = [f"{key}={os.environ[key]}" for key in keys if key in os.environ]
    return ";".join(values) if values else "not_pinned"


def _run_size(
    *,
    retained_nodes: int,
    universe_nodes: int,
    degree: int,
    intrinsic_leakage: float,
    warmups: int,
    trials: int,
    solve_repeats: int,
) -> str:
    domain_timing = _time_callable(
        lambda: _select_domain(universe_nodes, degree, retained_nodes),
        warmups=warmups,
        trials=trials,
    )
    domain, provider_calls = _select_domain(
        universe_nodes, degree, retained_nodes
    )
    if len(domain.nodes) != retained_nodes:
        raise RuntimeError(
            f"implicit graph selected {len(domain.nodes)} nodes instead of K="
            f"{retained_nodes}"
        )

    assembly_timing = _time_callable(
        lambda: _assemble_unit_precision(domain, intrinsic_leakage),
        warmups=warmups,
        trials=trials,
    )
    precision = _assemble_unit_precision(domain, intrinsic_leakage)
    cholesky_timing = _time_callable(
        lambda: np.linalg.cholesky(precision),
        warmups=warmups,
        trials=trials,
    )

    pipeline_timing = _time_callable(
        lambda: build_local_grounded_semantic_diffusion(
            domain,
            intrinsic_leakage_conductance=intrinsic_leakage,
        ),
        warmups=warmups,
        trials=trials,
    )
    model = build_local_grounded_semantic_diffusion(
        domain,
        intrinsic_leakage_conductance=intrinsic_leakage,
    )
    if not np.array_equal(precision, model.precision):
        maximum_error = float(np.max(np.abs(precision - model.precision)))
        raise RuntimeError(
            "split assembly does not match the production local precision; "
            f"maximum absolute error={maximum_error:.3e}"
        )

    source = np.zeros(retained_nodes, dtype=np.float64)
    source[0] = 1.0
    solve_timing = _time_callable(
        lambda: model.equilibrium_response(source),
        warmups=warmups,
        trials=trials,
        inner_repeats=solve_repeats,
    )
    response = model.equilibrium_response(source)
    residual = float(np.linalg.norm(model.precision @ response - source, ord=np.inf))

    one_dense_mib = _FLOAT64_BYTES * retained_nodes**2 / _MIB
    reference_persistent_mib = 4.0 * one_dense_mib
    assembly_working_mib = 3.0 * one_dense_mib
    full_one_dense_tib = _FLOAT64_BYTES * universe_nodes**2 / _TIB
    full_reference_tib = 4.0 * full_one_dense_tib
    retained_fraction_ppm = 1.0e6 * retained_nodes / universe_nodes
    cholesky_flops = retained_nodes**3 / 3.0

    return (
        f"{retained_nodes},{universe_nodes},{degree},{provider_calls},"
        f"{retained_fraction_ppm:.6f},"
        f"{domain_timing.median_ms:.6f},{domain_timing.mad_ms:.6f},"
        f"{assembly_timing.median_ms:.6f},{assembly_timing.mad_ms:.6f},"
        f"{cholesky_timing.median_ms:.6f},{cholesky_timing.mad_ms:.6f},"
        f"{pipeline_timing.median_ms:.6f},{pipeline_timing.mad_ms:.6f},"
        f"{solve_timing.median_ms:.6f},{solve_timing.mad_ms:.6f},"
        f"{one_dense_mib:.6f},{assembly_working_mib:.6f},"
        f"{reference_persistent_mib:.6f},{full_one_dense_tib:.6f},"
        f"{full_reference_tib:.6f},{cholesky_flops:.3f},{residual:.9e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k-values",
        type=_csv_positive_ints,
        default=_csv_positive_ints("16,32,64,128"),
        help="comma-separated retained-domain sizes",
    )
    parser.add_argument(
        "--universe-nodes",
        type=_positive_int,
        default=1_000_000,
        help="implicit graph size; no O(N) graph is allocated",
    )
    parser.add_argument(
        "--degree",
        type=_positive_int,
        default=6,
        help="even degree of the implicit circulant graph",
    )
    parser.add_argument(
        "--intrinsic-leakage",
        type=_positive_float,
        default=0.1,
        help="uniform intrinsic conductance to the common ground",
    )
    parser.add_argument("--warmups", type=_nonnegative_int, default=1)
    parser.add_argument("--trials", type=_positive_int, default=5)
    parser.add_argument(
        "--solve-repeats",
        type=_positive_int,
        default=3,
        help="inner repeats per solve timing trial",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="override sizes/repeats with a quick 8,16,32-node validation run",
    )
    args = parser.parse_args()

    if args.smoke:
        args.k_values = [8, 16, 32]
        args.warmups = 0
        args.trials = 2
        args.solve_repeats = 1
    if args.degree < 2 or args.degree % 2:
        parser.error("--degree must be an even integer of at least 2")
    if args.degree >= args.universe_nodes:
        parser.error("--degree must be smaller than --universe-nodes")
    if max(args.k_values) >= args.universe_nodes:
        parser.error("every K must be smaller than --universe-nodes")

    print("# Query-local grounded diffusion CPU scaling benchmark")
    print("# compute-only synthetic microbenchmark; not deployment latency")
    print(
        "# implicit fixed-degree universe: traversal touches only retained nodes; "
        "the global graph is not materialized"
    )
    print(
        "# dense local factorization remains O(K^3) time and O(K^2) memory; "
        "million-node dense domains remain infeasible"
    )
    print(
        "# reference_persistent_mib counts conductance, Laplacian, precision, "
        "and precision-root arrays; temporaries are excluded"
    )
    print(
        "# assembly_working_mib is a three-dense-array estimate, not measured "
        "peak RSS"
    )
    print(
        "# solve_ms measures the current numpy.linalg.solve-based stored-root path; "
        "ideal triangular solves are O(K^2)"
    )
    print(f"# numpy={np.__version__}; blas_threads={_environment_identity()}")
    print(
        "k,universe_nodes,degree,provider_calls,retained_fraction_ppm,"
        "domain_median_ms,domain_mad_ms,assembly_median_ms,assembly_mad_ms,"
        "cholesky_median_ms,cholesky_mad_ms,pipeline_build_median_ms,"
        "pipeline_build_mad_ms,solve_median_ms,solve_mad_ms,one_dense_mib,"
        "assembly_working_mib,reference_persistent_mib,full_one_dense_tib,"
        "full_reference_tib,cholesky_flop_estimate,residual_inf"
    )
    for retained_nodes in args.k_values:
        print(
            _run_size(
                retained_nodes=retained_nodes,
                universe_nodes=args.universe_nodes,
                degree=args.degree,
                intrinsic_leakage=args.intrinsic_leakage,
                warmups=args.warmups,
                trials=args.trials,
                solve_repeats=args.solve_repeats,
            )
        )


if __name__ == "__main__":
    main()
