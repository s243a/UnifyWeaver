#!/usr/bin/env python3
"""Synthetic smoke harness for outcome-blind bounded-domain fidelity.

The harness freezes separate topology-only and semantic exact-Dirichlet
hop-union references plus one plain-hop protected set before evaluating any
selector.  Each operator regime has its own explicitly recorded scalar
intrinsic leakage; candidate and reference share that scalar only within the
regime.  A selector
that omits a protected node emits a distinct ``coverage_failure`` record; the
harness never intersects node sets post hoc or ranks missing nodes last.

This is a correctness/provenance smoke run, not a winner-selection procedure
or deployment benchmark.  Optional graph-derived Schur closure is reported as a
separate experimental variant and is never enabled by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unifyweaver.graph.bounded_diffusion_fidelity import (  # noqa: E402
    ExperimentalBoundaryClosureConfig,
    ExteriorTraversalLimitError,
    ProtectedSetCoverageError,
    discover_exterior_components,
    ensure_matched_budget,
    evaluate_bounded_domain_fidelity,
    reduce_exact_exterior_component,
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


def _nonnegative_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("expected a nonnegative finite number")
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


def _stable_identifier_token(value):
    return list(_stable_identifier_key(value))


def _failure_fingerprint(records):
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


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
    """Return the nonnegative two-port Schur return B = C J_E^-1 C.T.

    Eliminating the exterior component E changes the retained precision by
    subtracting B from its Dirichlet principal block; B[0, 1] is the bridge
    conductance kappa and B[i, i] is the self-return at port i. Thus this matrix
    is not itself the reduced precision, whose off-diagonal bridge entries have
    the opposite sign.

    For one exterior node with port conductances c_l and c_r, leakage a, and
    d = c_l + c_r + a, this gives
    B = [[c_l**2, c_l*c_r], [c_l*c_r, c_r**2]] / d. The downstream closure
    ledger subtracts both transfer and self-return from the exact Dirichlet
    shunt before checking nonnegative residual ground, M-matrix signs, and SPD;
    it never adds this return on top of the shunt.

    ``graph`` remains in this private helper's signature for benchmark-call
    compatibility. The reusable reducer deliberately does not read it:
    ``component`` is the already-fingerprinted authoritative topology snapshot.
    """
    del graph
    return reduce_exact_exterior_component(
        component,
        intrinsic_leakage_conductance=intrinsic_leakage,
        topology_conductance=topology_conductance,
    ).schur_return


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
    both transfer and self-return.  With ``maximum_pairs=None`` every fully
    traversed exact two-port component is retained and embeddings have no role.
    A finite pair budget is a non-primary smoke sensitivity; optional frozen
    embeddings may rank only pairs already licensed by graph connectivity.
    """

    exhaustive = maximum_pairs is None
    try:
        discovery = discover_exterior_components(
            selection.domain,
            graph,
            allowed_exterior_nodes=allowed_exterior_nodes,
            maximum_component_nodes=maximum_component_nodes,
        )
    except ExteriorTraversalLimitError as exc:
        failure = {
            "reason": "exterior_traversal_limit",
            "component_start": _stable_identifier_token(exc.component_start),
            "maximum_component_nodes": exc.maximum_component_nodes,
            "visited_nodes": [
                _stable_identifier_token(node) for node in exc.visited_nodes
            ],
            "blocked_neighbor": _stable_identifier_token(exc.blocked_neighbor),
        }
        failures = [failure]
        return (), {}, {
            "source": "exact_component_schur",
            "strength_source": "topology_only_c0",
            "topology_conductance": topology_conductance,
            "semantic_role": "none" if exhaustive or embeddings is None else (
                "rank_graph_connected_pairs_only"
            ),
            "status": "traversal_incomplete_grounded_no_op",
            "eligible_two_port_components": 0,
            "reduced_two_port_components": 0,
            "numerically_failed_two_port_components": 0,
            "grounded_one_port_components": 0,
            "joint_dtn_required_components": 0,
            "parallel_retained_edge_pairs": 0,
            "approximate_caps_applied": False,
            "aggregated_pair_candidates": 0,
            "discarded_above_bridge_cap": 0,
            "discarded_for_pair_budget": 0,
            "realized_pairs": 0,
            "no_op": True,
            "failure_count": 1,
            "failure_reasons": {"exterior_traversal_limit": 1},
            "failure_fingerprint": _failure_fingerprint(failures),
            "failures": failures,
            "discovery": {
                "status": "traversal_incomplete",
                "maximum_component_nodes": maximum_component_nodes,
            },
        }
    grouped = {}
    one_port = 0
    multi_port = 0
    existing_edge = 0
    eligible_two_port = 0
    reduced_two_port = 0
    numerical_failures = []
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
        try:
            update = _topology_two_port_schur(
                component,
                graph,
                intrinsic_leakage=intrinsic_leakage,
                topology_conductance=topology_conductance,
            )
        except np.linalg.LinAlgError as exc:
            numerical_failures.append(
                {
                    "reason": "two_port_schur_numerical_failure",
                    "component_fingerprint": component.component_fingerprint,
                    "error_type": (
                        type(exc).__module__ + "." + type(exc).__qualname__
                    ),
                }
            )
            continue
        reduced_two_port += 1
        pair = (left, right)
        record = grouped.setdefault(
            pair,
            {
                "kappa": 0.0,
                "left_return": 0.0,
                "right_return": 0.0,
                "semantic_score": _semantic_pair_score(
                    left,
                    right,
                    None if exhaustive else embeddings,
                    length_scale,
                ),
                "components": [],
            },
        )
        record["kappa"] += float(update[0, 1])
        record["left_return"] += float(update[0, 0])
        record["right_return"] += float(update[1, 1])
        record["components"].append(component.component_fingerprint)

    if exhaustive:
        candidates = sorted(
            grouped.items(),
            key=lambda item: (
                -item[1]["kappa"],
                _stable_identifier_key(item[0][0]),
                _stable_identifier_key(item[0][1]),
            ),
        )
    else:
        candidates = sorted(
            grouped.items(),
            key=lambda item: (
                -item[1]["semantic_score"],
                -item[1]["kappa"],
                _stable_identifier_key(item[0][0]),
                _stable_identifier_key(item[0][1]),
            ),
        )
    if maximum_pairs is None:
        chosen = candidates
    else:
        maximum_pairs = _positive_integer(maximum_pairs)
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
        "semantic_role": "none" if exhaustive or embeddings is None else (
            "rank_graph_connected_pairs_only"
        ),
        "status": "ok_with_grounded_failures" if numerical_failures else "ok",
        "eligible_two_port_components": eligible_two_port,
        "reduced_two_port_components": reduced_two_port,
        "numerically_failed_two_port_components": len(numerical_failures),
        "grounded_one_port_components": one_port,
        "joint_dtn_required_components": multi_port,
        "parallel_retained_edge_pairs": existing_edge,
        "approximate_caps_applied": False,
        "aggregated_pair_candidates": len(candidates),
        "discarded_above_bridge_cap": 0,
        "discarded_for_pair_budget": max(len(candidates) - len(chosen), 0),
        "realized_pairs": len(pairs),
        "no_op": not pairs,
        "failure_count": len(numerical_failures),
        "failure_reasons": (
            {"two_port_schur_numerical_failure": len(numerical_failures)}
            if numerical_failures
            else {}
        ),
        "failure_fingerprint": (
            _failure_fingerprint(numerical_failures)
            if numerical_failures
            else None
        ),
        "failures": numerical_failures,
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
    parser.add_argument("--alpha", type=_nonnegative_float, default=0.2)
    parser.add_argument(
        "--semantic-alpha", type=_nonnegative_float, default=0.3
    )
    parser.add_argument("--length-scale", type=_positive_float, default=2.0)
    parser.add_argument("--conductance-floor", type=_unit_interval, default=0.1)
    parser.add_argument(
        "--reference-conductance", type=_positive_float, default=1.0
    )
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
    ancestor_depth = 3
    hop_selection = select_hop_budget_domain(
        anchors, graph, maximum_nodes=args.k
    )
    skeleton_selection = select_topology_skeleton_domain(
        anchors,
        graph,
        parents,
        maximum_nodes=args.k,
        ancestor_depth=ancestor_depth,
    )
    resistance_selection = select_semantic_resistance_domain(
        anchors,
        graph,
        embeddings,
        maximum_nodes=args.k,
        length_scale=args.length_scale,
        conductance_floor=args.conductance_floor,
        reference_conductance=args.reference_conductance,
    )
    topology_selections = (hop_selection, skeleton_selection)
    semantic_selections = (
        hop_selection,
        skeleton_selection,
        resistance_selection,
    )
    ensure_matched_budget(semantic_selections)
    references = {
        "topology_only_c0": select_union_hop_reference(
            topology_selections, graph, maximum_nodes=args.reference_k
        ),
        "semantic_rbf": select_union_hop_reference(
            semantic_selections, graph, maximum_nodes=args.reference_k
        ),
    }
    alpha_by_regime = {
        "topology_only_c0": args.alpha,
        "semantic_rbf": args.semantic_alpha,
    }
    manifest = {
        "record_type": "manifest",
        "outcome_blind": True,
        "winner_selection": False,
        "anchors": list(anchors),
        "requested_k": args.k,
        "reference_k": args.reference_k,
        "reference_fingerprints": {
            regime: reference.selection_fingerprint
            for regime, reference in references.items()
        },
        "protected_nodes": list(protected_nodes),
        "alpha_by_regime": alpha_by_regime,
        "ancestor_depth": ancestor_depth,
        "length_scale": args.length_scale,
        "conductance_floor": args.conductance_floor,
        "reference_conductance": args.reference_conductance,
        "effective_resistance": args.effective_resistance,
        "experimental_closure_requested": args.experimental_closure,
        "maximum_exterior_component_nodes": (
            args.maximum_exterior_component_nodes
        ),
        "operator_regimes": ["topology_only_c0", "semantic_rbf"],
    }
    print(json.dumps(manifest, sort_keys=True))

    topology_conductance = 1.0
    closure_cap = float(np.nextafter(topology_conductance, 0.0))
    regime_entries = tuple(
        (selection, "topology_only_c0")
        for selection in topology_selections
    ) + tuple(
        (selection, "semantic_rbf")
        for selection in semantic_selections
    )
    for selection, operator_regime in regime_entries:
        reference = references[operator_regime]
        intrinsic_leakage = alpha_by_regime[operator_regime]
        if operator_regime == "semantic_rbf":
            operator_embeddings = embeddings
            operator_length_scale = args.length_scale
            operator_floor = args.conductance_floor
        else:
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
                None,
                intrinsic_leakage=intrinsic_leakage,
                length_scale=args.length_scale,
                topology_conductance=topology_conductance,
                maximum_pairs=None,
                maximum_component_nodes=(
                    args.maximum_exterior_component_nodes
                ),
                allowed_exterior_nodes=set(reference.domain.nodes).difference(
                    selection.domain.nodes
                ),
            )
            if pairs:
                closure_config = ExperimentalBoundaryClosureConfig(
                    maximum_edges=len(pairs),
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
                "reference_fingerprint": reference.selection_fingerprint,
                "intrinsic_leakage": intrinsic_leakage,
                "selection": selection.provenance_dict(),
            }
            if closure_provenance is not None:
                base["closure_input"] = closure_provenance
            try:
                result = evaluate_bounded_domain_fidelity(
                    selection,
                    reference,
                    protected_nodes=protected_nodes,
                    intrinsic_leakage_conductance=intrinsic_leakage,
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
