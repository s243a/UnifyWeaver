"""Reusable node-disjoint splitting and paired node-block bootstrap utilities.

The pair rows used by the mu-cosine experiments are not independent: a node can
occur in many rows.  This module keeps two different uncertainty questions
separate:

* :func:`node_disjoint_pair_split` makes an outcome-blind train/held partition
  by assigning *nodes*, then discarding pairs that cross the node partition.
* :func:`paired_node_bootstrap_ci` estimates uncertainty of a paired row metric
  on one fixed held partition by resampling held nodes.  A non-self pair gets
  the product of its two endpoint multiplicities, so dependence through either
  endpoint is retained (the two-endpoint/pigeonhole node bootstrap).

Variation across several split seeds is useful as a Monte Carlo stability
diagnostic.  It is not a sampling standard error and is deliberately not
computed here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class PartitionCounts:
    """Pair counts for one stratum after assigning nodes."""

    total: int
    train: int
    held: int
    cross: int


@dataclass
class NodeDisjointSplit:
    """Indices and coverage diagnostics for an outcome-blind node split."""

    train: np.ndarray
    held: np.ndarray
    cross: np.ndarray
    train_nodes: frozenset[Hashable]
    held_nodes: frozenset[Hashable]
    strata: dict[Hashable, PartitionCounts]
    seed: int
    held_node_fraction: float
    candidates: int
    selected_candidate: int

    @property
    def retained_fraction(self) -> float:
        total = len(self.train) + len(self.held) + len(self.cross)
        return (len(self.train) + len(self.held)) / total if total else float("nan")

    def missing_strata(self, minimum: int = 1) -> tuple[tuple[Hashable, ...], tuple[Hashable, ...]]:
        """Return strata below ``minimum`` in train and held, respectively."""
        train = tuple(k for k, v in self.strata.items() if v.train < minimum)
        held = tuple(k for k, v in self.strata.items() if v.held < minimum)
        return train, held


@dataclass(frozen=True)
class NodeBootstrapInterval:
    """Percentile interval for a mean paired gain on one fixed held split.

    ``estimate`` is the empirical plug-in row mean (equivalently, the
    pigeonhole ratio statistic when every observed node has multiplicity one).
    Bootstrap replicates use sampled endpoint multiplicities; their finite-
    sample mean can differ, so ``bootstrap_mean`` is retained for auditing.
    """

    estimate: float
    low: float
    high: float
    bootstrap_mean: float
    confidence: float
    n_resamples: int
    n_attempts: int


def _stable_node_key(node: Hashable) -> tuple[str, str]:
    """Sort heterogeneous hashable node IDs without relying on set order."""
    return type(node).__qualname__, repr(node)


def _partition_indices(pairs, held_nodes):
    train, held, cross = [], [], []
    for i, (left, right) in enumerate(pairs):
        left_held = left in held_nodes
        right_held = right in held_nodes
        if left_held and right_held:
            held.append(i)
        elif not left_held and not right_held:
            train.append(i)
        else:
            cross.append(i)
    return (
        np.asarray(train, dtype=int),
        np.asarray(held, dtype=int),
        np.asarray(cross, dtype=int),
    )


def _stratum_counts(labels, train, held, cross):
    total = Counter(labels)
    tr = Counter(labels[i] for i in train)
    he = Counter(labels[i] for i in held)
    cr = Counter(labels[i] for i in cross)
    return {
        label: PartitionCounts(total[label], tr[label], he[label], cr[label])
        for label in sorted(total, key=lambda x: (type(x).__qualname__, repr(x)))
    }


def _candidate_score(counts, train, held, cross, held_node_fraction, minimum):
    """Outcome-blind balance score; smaller is better.

    The expected held share among retained pairs is p²/(p²+(1-p)²) when a
    fraction p of nodes is held.  Candidate search uses that target plus
    per-stratum coverage.  Labels and graph incidence are allowed inputs;
    D/S outcomes are not.
    """
    empty_partitions = int(len(train) == 0) + int(len(held) == 0)
    coverage_deficit = sum(
        max(0, minimum - value.train) + max(0, minimum - value.held)
        for value in counts.values()
    )
    p = held_node_fraction
    target = p * p / (p * p + (1.0 - p) * (1.0 - p))
    retained = len(train) + len(held)
    observed = len(held) / retained if retained else 0.0
    pair_balance = abs(observed - target)

    total_pairs = sum(value.total for value in counts.values())
    stratum_balance = 0.0
    for value in counts.values():
        n = value.train + value.held
        share = value.held / n if n else 0.0
        stratum_balance += value.total / total_pairs * abs(share - target)

    return (
        empty_partitions,
        coverage_deficit,
        stratum_balance,
        pair_balance,
        -retained,
        len(cross),
    )


def node_disjoint_pair_split(
    pairs: Sequence[tuple[Hashable, Hashable]],
    seed: int,
    *,
    held_node_fraction: float = 0.40,
    strata: Sequence[Hashable] | None = None,
    candidates: int = 64,
    minimum_per_stratum: int = 1,
) -> NodeDisjointSplit:
    """Choose a deterministic, coverage-aware node-disjoint pair split.

    Exactly ``round(held_node_fraction * n_nodes)`` nodes are assigned to held
    (bounded to leave at least one node on each side).  The 0.40 default yields
    an expected held share of about 30.8% among retained pairs, because random
    node assignment gives p²/(p²+(1-p)²), rather than p, as the retained-pair
    share.  Pairs with one endpoint on each side are returned in ``cross`` and
    must not be used for fitting or evaluation.  If ``candidates > 1``, several
    seeded node assignments are scored using only pair incidence and optional
    stratum labels; outcomes are never inspected.
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("pairs must be non-empty")
    if not 0.0 < held_node_fraction < 1.0:
        raise ValueError("held_node_fraction must be strictly between 0 and 1")
    if candidates < 1:
        raise ValueError("candidates must be positive")
    if minimum_per_stratum < 0:
        raise ValueError("minimum_per_stratum must be non-negative")
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("every pair must contain exactly two endpoints")

    labels = list(strata) if strata is not None else ["all"] * len(pairs)
    if len(labels) != len(pairs):
        raise ValueError("strata must align one-for-one with pairs")
    nodes = sorted({node for pair in pairs for node in pair}, key=_stable_node_key)
    if len(nodes) < 2:
        raise ValueError("a node-disjoint split requires at least two distinct nodes")

    n_held = int(np.floor(held_node_fraction * len(nodes) + 0.5))
    n_held = min(len(nodes) - 1, max(1, n_held))
    rng = np.random.default_rng(seed)
    best = None
    best_score = None
    for candidate in range(candidates):
        order = rng.permutation(len(nodes))
        held_nodes = frozenset(nodes[i] for i in order[:n_held])
        train, held, cross = _partition_indices(pairs, held_nodes)
        counts = _stratum_counts(labels, train, held, cross)
        score = _candidate_score(
            counts,
            train,
            held,
            cross,
            held_node_fraction,
            minimum_per_stratum,
        ) + (candidate,)
        if best_score is None or score < best_score:
            best_score = score
            best = (candidate, held_nodes, train, held, cross, counts)

    candidate, held_nodes, train, held, cross, counts = best
    return NodeDisjointSplit(
        train=train,
        held=held,
        cross=cross,
        train_nodes=frozenset(nodes) - held_nodes,
        held_nodes=held_nodes,
        strata=counts,
        seed=int(seed),
        held_node_fraction=float(held_node_fraction),
        candidates=int(candidates),
        selected_candidate=int(candidate),
    )


def format_split_diagnostics(split: NodeDisjointSplit) -> str:
    """Return compact train/held/cross and per-stratum coverage diagnostics."""
    pair_total = len(split.train) + len(split.held) + len(split.cross)
    lines = [
        f"nodes train/held={len(split.train_nodes)}/{len(split.held_nodes)} "
        f"(held {len(split.held_nodes) / (len(split.train_nodes) + len(split.held_nodes)):.1%}); "
        f"pairs train/held/cross={len(split.train)}/{len(split.held)}/{len(split.cross)} "
        f"(retained {split.retained_fraction:.1%} of {pair_total})",
        "stratum                 total  train  held  cross",
    ]
    for label, value in split.strata.items():
        lines.append(
            f"{str(label):24.24s} {value.total:6d} {value.train:6d} {value.held:5d} {value.cross:6d}"
        )
    return "\n".join(lines)


def paired_node_bootstrap_ci(
    pairs: Sequence[tuple[Hashable, Hashable]],
    paired_values: Sequence[float],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> NodeBootstrapInterval:
    """Paired percentile CI for a mean using nodes as dependence blocks.

    ``paired_values`` should already be row-wise differences (for example,
    baseline NLL minus candidate NLL) on one fixed held split.  Each bootstrap
    draw samples the held nodes with replacement.  A pair's weight is the
    product of its endpoint multiplicities; self-pairs use the single endpoint
    multiplicity.  Draws containing none of the observed pairs are retried.
    The reported point is the empirical plug-in row mean; resampled intervals
    use multiplicity-weighted ratio means and agree with that estimand only in
    expectation under the bootstrap law.
    """
    pairs = list(pairs)
    values = np.asarray(paired_values, dtype=float)
    if len(pairs) == 0:
        raise ValueError("pairs must be non-empty")
    if values.ndim != 1 or len(values) != len(pairs):
        raise ValueError("paired_values must be a one-dimensional array aligned with pairs")
    if not np.all(np.isfinite(values)):
        raise ValueError("paired_values must all be finite")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")

    nodes = sorted({node for pair in pairs for node in pair}, key=_stable_node_key)
    node_index = {node: i for i, node in enumerate(nodes)}
    left = np.asarray([node_index[pair[0]] for pair in pairs], dtype=int)
    right = np.asarray([node_index[pair[1]] for pair in pairs], dtype=int)
    self_pair = left == right
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(nodes), 1.0 / len(nodes))
    replicates = []
    attempts = 0
    max_attempts = max(n_resamples * 20, n_resamples + 100)
    while len(replicates) < n_resamples and attempts < max_attempts:
        attempts += 1
        multiplicity = rng.multinomial(len(nodes), probabilities)
        weights = multiplicity[left] * multiplicity[right]
        weights[self_pair] = multiplicity[left[self_pair]]
        total_weight = int(weights.sum())
        if total_weight:
            replicates.append(float(np.dot(weights, values) / total_weight))
    if len(replicates) < n_resamples:
        raise RuntimeError(
            f"only {len(replicates)}/{n_resamples} non-empty node-bootstrap draws "
            f"after {attempts} attempts"
        )

    boot = np.asarray(replicates)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])
    return NodeBootstrapInterval(
        estimate=float(values.mean()),
        low=float(low),
        high=float(high),
        bootstrap_mean=float(boot.mean()),
        confidence=float(confidence),
        n_resamples=int(n_resamples),
        n_attempts=int(attempts),
    )
