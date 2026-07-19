#!/usr/bin/env python3
"""Synthetic smoke harness for outcome-blind bounded-domain fidelity.

The harness freezes one larger exact-Dirichlet hop-union reference and one
plain-hop protected set before evaluating any selector.  Every family uses the
same intrinsic leakage and semantic conductance configuration.  A selector
that omits a protected node emits a distinct ``coverage_failure`` record; the
harness never intersects node sets post hoc or ranks missing nodes last.

This is a correctness/provenance smoke run, not a winner-selection procedure
or deployment benchmark.  Optional graph-derived Schur closure is reported as a
separate experimental variant and is never enabled by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unifyweaver.graph.bounded_diffusion_fidelity import (  # noqa: E402
    ExperimentalBoundaryClosureConfig,
    ProtectedSetCoverageError,
    discover_exterior_components,
    ensure_matched_budget,
    evaluate_bounded_domain_fidelity,
    select_hop_budget_domain,
    select_semantic_resistance_domain,
    select_topology_skeleton_domain,
    select_union_hop_reference,
)


def _positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return parsed


def _unit_interval(value):
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("expected a finite value in [0, 1)")
    return parsed


def _binary_tree(node_count):
    neighbors = {node: set() for node in range(node_count)}
    parents = {0: ()}
    for child in range(1, node_count):
        parent = (child - 1) // 2
        neighbors[parent].add(child)
        neighbors[child].add(parent)
        parents[child] = (parent,)
    return {
        node: tuple(sorted(values)) for node, values in neighbors.items()
    }, parents


def _embeddings(node_count):
    values = {}
    for node in range(node_count):
        depth = int(math.floor(math.log2(node + 1)))
        within = node - (2**depth - 1)
        scale = max(2**depth - 1, 1)
        values[node] = np.array([float(depth), within / scale], dtype=float)
    return values



def _stable_identifier_key(value):
    value_type = type(value)
    return (value_type.__module__, value_type.__qualname__, repr(value))


def _semantic_pair_score(left, right, embeddings, length_scale):
    if embeddings is None:
        return 0.0
    delta = np.asarray(embeddings[left]) - np.asarray(embeddings[right])
    squared_distance = float(delta @ delta)
    return math.exp(-0.5 * squared_distance / (length_scale * length_scale))


def _topology_two_port_schur(
    component,
    graph,
    *,
    intrinsic_leakage,
    topology_conductance,
):
    nodes = component.nodes
    ports = component.ports
    node_index = {node: row for row, node in enumerate(nodes)}
    port_index = {node: row for row, node in enumerate(ports)}
    precision = np.eye(len(nodes), dtype=float) * intrinsic_leakage
    coupling = np.zeros((2, len(nodes)), dtype=float)
    outside_bath_edges = set(component.outside_bath_edges)
    for node, row in node_index.items():
        for neighbor in graph[node]:
            if neighbor in node_index:
                neighbor_row = node_index[neighbor]
                if row < neighbor_row:
                    precision[row, row] += topology_conductance
                    precision[neighbor_row, neighbor_row] += topology_conductance
                    precision[row, neighbor_row] -= topology_conductance
                    precision[neighbor_row, row] -= topology_conductance
            elif neighbor in port_index:
                precision[row, row] += topology_conductance
                coupling[port_index[neighbor], row] += topology_conductance
            elif (node, neighbor) in outside_bath_edges:
                precision[row, row] += topology_conductance
            else:
                raise ValueError(
                    "bounded exterior component contains an unknown neighbor"
                )
    update = coupling @ np.linalg.solve(precision, coupling.T)
    if not np.isfinite(update).all() or not np.allclose(update, update.T):
        raise np.linalg.LinAlgError("two-port topology Schur response is invalid")
    if np.min(update) < -1e-12:
        raise np.linalg.LinAlgError("two-port topology Schur response is negative")
    return update


def _exact_two_port_exterior_dtn(
    selection,
    graph,
    embeddings,
    *,
    intrinsic_leakage,
    length_scale,
    topology_conductance,
    maximum_pairs,
    maximum_component_nodes,
    allowed_exterior_nodes=None,
):
    """Discover complete exterior components and reduce exact two-port cases.

    The graph alone licenses every pair and topology-only ``c0`` determines
    both transfer and self-return.  Optional frozen embeddings only rank already graph-connected pairs when the
    pair budget binds; exact reduction itself is embedding-free.
    """

    discovery = discover_exterior_components(
        selection.domain,
        graph,
        allowed_exterior_nodes=allowed_exterior_nodes,
        maximum_component_nodes=maximum_component_nodes,
    )
    grouped = {}
    one_port = 0
    multi_port = 0
    existing_edge = 0
    eligible_two_port = 0
    for component in discovery.components:
        if len(component.ports) == 1:
            one_port += 1
            continue
        if len(component.ports) > 2:
            multi_port += 1
            continue
        left, right = component.ports
        if right in graph[left]:
            existing_edge += 1
        eligible_two_port += 1
        update = _topology_two_port_schur(
            component,
            graph,
            intrinsic_leakage=intrinsic_leakage,
            topology_conductance=topology_conductance,
        )
        pair = (left, right)
        record = grouped.setdefault(
            pair,
            {
                "kappa": 0.0,
                "left_return": 0.0,
                "right_return": 0.0,
                "semantic_score": _semantic_pair_score(
                    left, right, embeddings, length_scale
                ),
                "components": [],
            },
        )
        record["kappa"] += float(update[0, 1])
        record["left_return"] += float(update[0, 0])
        record["right_return"] += float(update[1, 1])
        record["components"].append(component.component_fingerprint)

    candidates = sorted(
        grouped.items(),
        key=lambda item: (
            -item[1]["semantic_score"],
            -item[1]["kappa"],
            _stable_identifier_key(item[0][0]),
            _stable_identifier_key(item[0][1]),
        ),
    )
    chosen = candidates[:maximum_pairs]
    pairs = []
    self_return = {}
    for (left, right), record in chosen:
        pairs.append((left, right, float(record["kappa"])))
        self_return[left] = self_return.get(left, 0.0) + record["left_return"]
        self_return[right] = self_return.get(right, 0.0) + record["right_return"]
    provenance = {
        "source": "exact_component_schur",
        "strength_source": "topology_only_c0",
        "topology_conductance": topology_conductance,
        "semantic_role": "rank_graph_connected_pairs_only",
        "eligible_two_port_components": eligible_two_port,
        "grounded_one_port_components": one_port,
        "joint_dtn_required_components": multi_port,
        "parallel_retained_edge_pairs": existing_edge,
        "approximate_caps_applied": False,
        "aggregated_pair_candidates": len(candidates),
        "discarded_above_bridge_cap": 0,
        "discarded_for_pair_budget": max(len(candidates) - len(chosen), 0),
        "realized_pairs": len(pairs),
        "no_op": not pairs,
        "discovery": discovery.provenance_dict(),
    }
    return tuple(pairs), self_return, provenance


def _jsonable_result(result):
    output = result.as_dict()
    output["protected_nodes"] = list(output["protected_nodes"])
    output["source_nodes"] = list(output["source_nodes"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=_positive_integer, default=127)
    parser.add_argument("--k", type=_positive_integer, default=32)
    parser.add_argument("--reference-k", type=_positive_integer, default=96)
    parser.add_argument("--protected", type=_positive_integer, default=12)
    parser.add_argument("--alpha", type=_positive_float, default=0.2)
    parser.add_argument("--length-scale", type=_positive_float, default=2.0)
    parser.add_argument("--conductance-floor", type=_unit_interval, default=0.1)
    parser.add_argument("--effective-resistance", action="store_true")
    parser.add_argument("--experimental-closure", action="store_true")
    parser.add_argument(
        "--maximum-exterior-component-nodes",
        type=_positive_integer,
        default=4096,
    )
    args = parser.parse_args()
    if args.reference_k <= args.k:
        parser.error("--reference-k must be larger than --k")
    if args.reference_k > args.nodes:
        parser.error("--reference-k cannot exceed --nodes")
    if args.protected > args.k:
        parser.error("--protected cannot exceed --k")

    graph, parents = _binary_tree(args.nodes)
    embeddings = _embeddings(args.nodes)
    anchors = (args.nodes // 3, args.nodes // 3 + 1)
    protected_selector = select_hop_budget_domain(
        anchors, graph, maximum_nodes=args.protected
    )
    protected_nodes = protected_selector.domain.nodes
    selections = (
        select_hop_budget_domain(anchors, graph, maximum_nodes=args.k),
        select_topology_skeleton_domain(
            anchors,
            graph,
            parents,
            maximum_nodes=args.k,
            ancestor_depth=3,
        ),
        select_semantic_resistance_domain(
            anchors,
            graph,
            embeddings,
            maximum_nodes=args.k,
            length_scale=args.length_scale,
            conductance_floor=args.conductance_floor,
        ),
    )
    ensure_matched_budget(selections)
    reference = select_union_hop_reference(
        selections, graph, maximum_nodes=args.reference_k
    )
    manifest = {
        "record_type": "manifest",
        "outcome_blind": True,
        "winner_selection": False,
        "anchors": list(anchors),
        "requested_k": args.k,
        "reference_k": args.reference_k,
        "reference_fingerprint": reference.selection_fingerprint,
        "protected_nodes": list(protected_nodes),
        "alpha": args.alpha,
        "length_scale": args.length_scale,
        "conductance_floor": args.conductance_floor,
        "effective_resistance": args.effective_resistance,
        "experimental_closure_requested": args.experimental_closure,
        "maximum_exterior_component_nodes": (
            args.maximum_exterior_component_nodes
        ),
        "operator_regimes": ["topology_only_c0", "semantic_rbf"],
    }
    print(json.dumps(manifest, sort_keys=True))

    maximum_closure_edges = max(1, args.k // 8)
    topology_conductance = 1.0
    closure_cap = float(np.nextafter(topology_conductance, 0.0))
    for selection in selections:
        if selection.strategy == "semantic_resistance":
            operator_regime = "semantic_rbf"
            operator_embeddings = embeddings
            operator_length_scale = args.length_scale
            operator_floor = args.conductance_floor
        else:
            operator_regime = "topology_only_c0"
            operator_embeddings = None
            operator_length_scale = None
            operator_floor = 0.0
        variants = (
            (
                "exact_dirichlet",
                None,
                None,
                None,
                None,
            ),
        )
        if args.experimental_closure and operator_regime == "topology_only_c0":
            pairs, self_return, closure_provenance = _exact_two_port_exterior_dtn(
                selection,
                graph,
                embeddings,
                intrinsic_leakage=args.alpha,
                length_scale=args.length_scale,
                topology_conductance=topology_conductance,
                maximum_pairs=maximum_closure_edges,
                maximum_component_nodes=(
                    args.maximum_exterior_component_nodes
                ),
                allowed_exterior_nodes=set(reference.domain.nodes).difference(
                    selection.domain.nodes
                ),
            )
            if pairs:
                closure_config = ExperimentalBoundaryClosureConfig(
                    maximum_edges=maximum_closure_edges,
                    closure_mass_fraction=1.0,
                    ordinary_branch_conductance=topology_conductance,
                    bridge_conductance_cap=closure_cap,
                    pair_conductance_source="exact_component_schur",
                    ledger_mode="explicit_self_return",
                )
                variants += (
                    (
                        "experimental_graph_derived_closure",
                        closure_config,
                        pairs,
                        self_return,
                        closure_provenance,
                    ),
                )
            else:
                print(
                    json.dumps(
                        {
                            "record_type": "closure_discovery",
                            "status": "no_op",
                            "strategy": selection.strategy,
                            "operator_regime": operator_regime,
                            "selection": selection.provenance_dict(),
                            "closure_input": closure_provenance,
                        },
                        sort_keys=True,
                    )
                )
        elif args.experimental_closure:
            print(
                json.dumps(
                    {
                        "record_type": "closure_discovery",
                        "status": "not_applicable_semantic_operator",
                        "strategy": selection.strategy,
                        "operator_regime": operator_regime,
                        "selection": selection.provenance_dict(),
                    },
                    sort_keys=True,
                )
            )
        for variant, closure, pairs, self_return, closure_provenance in variants:
            base = {
                "record_type": "fidelity",
                "strategy": selection.strategy,
                "variant": variant,
                "operator_regime": operator_regime,
                "selection": selection.provenance_dict(),
            }
            if closure_provenance is not None:
                base["closure_input"] = closure_provenance
            try:
                result = evaluate_bounded_domain_fidelity(
                    selection,
                    reference,
                    protected_nodes=protected_nodes,
                    intrinsic_leakage_conductance=args.alpha,
                    node_embeddings=operator_embeddings,
                    length_scale=operator_length_scale,
                    conductance_floor=operator_floor,
                    boundary_closure_config=closure,
                    boundary_closure_pair_conductances=pairs,
                    boundary_closure_self_return=self_return,
                    include_effective_resistance=args.effective_resistance,
                )
            except ProtectedSetCoverageError as exc:
                base.update(
                    {
                        "status": "coverage_failure",
                        "missing_candidate": list(exc.missing_candidate),
                        "missing_reference": list(exc.missing_reference),
                    }
                )
            else:
                base.update({"status": "ok", "metrics": _jsonable_result(result)})
            print(json.dumps(base, sort_keys=True))


if __name__ == "__main__":
    main()
