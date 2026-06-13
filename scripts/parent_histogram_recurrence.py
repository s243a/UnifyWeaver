#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Parent-path histogram recurrence helpers.

For a parent-only DAG the recurrence is exact:

    H_root[0] = 1
    H_v[d] = sum(H_parent[d - 1] for parent in parents(v))

The helpers use unnormalized histograms: every bin stores path-count mass.  If
callers store normalized parent distributions instead, each shifted parent
distribution must be multiplied by its path count N_p before the sum, and N_v
must be stored separately.

This avoids enumerating every path from a target to root.  In cyclic graphs a
per-node histogram cannot enforce a unique-node visited set, so cycle encounters
are reported through `cycle_approximation`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time

from scripts.distribution_serialization import decode_distribution_payload, encode_selected_distribution


@dataclass
class RecurrenceHistogramStats:
    states_evaluated: int = 0
    memo_hits: int = 0
    edges_examined: int = 0
    cycle_edges: int = 0
    budget_cutoffs: int = 0
    path_cap_hit: bool = False
    expansion_cap_hit: bool = False
    cycle_approximation: bool = False


@dataclass
class PayloadParentRecurrenceStats:
    parent_payloads: int = 0
    payloads_decoded: int = 0
    payload_bytes_read: int = 0
    decode_ns: int = 0
    decoded_bins: int = 0
    output_bins: int = 0
    output_path_count: int = 0
    output_payload_bytes: int = 0
    path_cap_hit: bool = False


def shifted_add(out, parent_hist, remaining, path_cap, stats):
    for length, count in parent_hist.items():
        shifted = int(length) + 1
        if shifted <= remaining:
            out[shifted] += int(count)
            if path_cap is not None and sum(out.values()) >= path_cap:
                stats.path_cap_hit = True
                return


def recurrence_parent_histogram(parents_func, target, root, budget, path_cap=None, expansion_cap=None):
    """Compute a finite-horizon parent histogram by shifted parent sums.

    The returned histogram counts root-reaching walks under the recurrence.  It
    is exact for DAG parent cones.  If a cycle is encountered, the cyclic edge is
    skipped and `cycle_approximation` is set because a node-only recurrence does
    not carry the visited-set state required for exact simple-path semantics.
    """
    stats = RecurrenceHistogramStats()
    memo = {}
    visiting = set()

    def rec(node, remaining):
        key = (node, remaining)
        if key in memo:
            stats.memo_hits += 1
            return memo[key]
        if node == root:
            memo[key] = {0: 1}
            return memo[key]
        if remaining <= 0:
            stats.budget_cutoffs += 1
            memo[key] = {}
            return memo[key]
        if node in visiting:
            stats.cycle_edges += 1
            stats.cycle_approximation = True
            return {}
        if expansion_cap is not None and stats.states_evaluated >= expansion_cap:
            stats.expansion_cap_hit = True
            return {}

        visiting.add(node)
        stats.states_evaluated += 1
        out = Counter()
        for parent in parents_func(node):
            stats.edges_examined += 1
            if parent in visiting:
                stats.cycle_edges += 1
                stats.cycle_approximation = True
                continue
            parent_hist = rec(parent, remaining - 1)
            shifted_add(out, parent_hist, remaining, path_cap, stats)
            if stats.path_cap_hit or stats.expansion_cap_hit:
                break
        visiting.remove(node)
        memo[key] = dict(sorted(out.items()))
        return memo[key]

    return dict(sorted(rec(target, int(budget)).items())), stats


def scaled_distribution_histogram(probabilities, origin, total_count):
    """Convert a normalized finite distribution payload back to count mass."""
    if origin is None or total_count <= 0:
        return {}
    weighted = [(index, max(0.0, float(probability)) * int(total_count)) for index, probability in enumerate(probabilities)]
    floors = {int(origin) + index: int(value) for index, value in weighted if int(value) > 0}
    remainder = int(total_count) - sum(floors.values())
    if remainder > 0 and weighted:
        ranked = sorted(
            weighted,
            key=lambda item: (item[1] - int(item[1]), item[1], -item[0]),
            reverse=True,
        )
        for index, _value in ranked[:remainder]:
            floors[int(origin) + index] = floors.get(int(origin) + index, 0) + 1
    return dict(sorted((length, count) for length, count in floors.items() if count > 0))


def histogram_distribution(hist):
    if not hist:
        return [], 0, 0
    origin = min(hist)
    max_length = max(hist)
    total_count = sum(int(value) for value in hist.values())
    if total_count <= 0:
        return [], origin, 0
    probabilities = [
        int(hist.get(length, 0)) / total_count
        for length in range(origin, max_length + 1)
    ]
    return probabilities, origin, total_count


def payload_to_histogram(payload):
    """Decode one stored finite-distribution payload into count bins."""
    payload_bytes = getattr(payload, "payload", payload)
    started = time.perf_counter_ns()
    probabilities, metadata = decode_distribution_payload(payload_bytes)
    decode_ns = time.perf_counter_ns() - started
    total_count = int(round(float(metadata.get("total_mass", 0.0))))
    hist = scaled_distribution_histogram(probabilities, int(metadata.get("origin", 0)), total_count)
    return hist, metadata, decode_ns, len(payload_bytes)


def shifted_parent_payload_histogram(parent_payloads, remaining=None, path_cap=None):
    """Build a child histogram by summing shifted decoded parent payloads.

    This is the one-layer recurrence:

        H_child[d] = sum(H_parent[d - 1] for parent in parents(child))

    Payloads are decoded once each, shifted right by one, and summed as
    unnormalized path-count mass.  `remaining` is the optional finite horizon
    for the child histogram.
    """
    payloads = list(parent_payloads)
    stats = PayloadParentRecurrenceStats(parent_payloads=len(payloads))
    out = Counter()
    horizon = None if remaining is None else int(remaining)
    for payload in payloads:
        parent_hist, _metadata, decode_ns, payload_bytes = payload_to_histogram(payload)
        stats.payloads_decoded += 1
        stats.payload_bytes_read += payload_bytes
        stats.decode_ns += decode_ns
        stats.decoded_bins += len(parent_hist)
        for length, count in parent_hist.items():
            shifted = int(length) + 1
            if horizon is None or shifted <= horizon:
                out[shifted] += int(count)
                if path_cap is not None and sum(out.values()) >= path_cap:
                    stats.path_cap_hit = True
                    break
        if stats.path_cap_hit:
            break
    hist = dict(sorted(out.items()))
    stats.output_bins = len(hist)
    stats.output_path_count = sum(hist.values())
    return hist, stats


def serialize_shifted_parent_payload_histogram(parent_payloads, representation="packed_sparse_histogram", remaining=None, path_cap=None, cdf_bits=16):
    hist, stats = shifted_parent_payload_histogram(parent_payloads, remaining, path_cap)
    probabilities, origin, total_count = histogram_distribution(hist)
    payload, metadata = encode_selected_distribution(probabilities, representation, origin=origin, total_mass=total_count, cdf_bits=cdf_bits)
    stats.output_payload_bytes = metadata["payload_bytes"]
    return payload, metadata, hist, stats
