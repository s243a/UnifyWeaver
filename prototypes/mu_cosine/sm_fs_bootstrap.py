#!/usr/bin/env python3
"""Frozen decision bootstrap (CAND2-6): SHA-256 rejection sampler, query-weighted block mean,
nearest-rank endpoints.

Implements REVIEW_sm_fs_ranking_candidate_lock.md §3 steps 3-7 exactly:
- each replicate draws exactly 82 block indices with replacement from the canonical UTF-8-sorted
  82-block list, identical multiplicities for both arms (the statistic consumes the paired
  per-query differences, so pairing is structural);
- replicate statistic = Σ_b m_b·Σ_{q∈b} d(q) / Σ_b m_b·n_b (query-weighted mean under block
  multiplicities);
- versioned SHA-256 rejection sampler; canonical-JSON key with exactly
  {schema, sampler_id, seed, replicate, draw, retry}; digest as unsigned big-endian 256-bit int,
  reject at floor(2^256/82)*82 before modulo;
- 9,999 resamples, seed 3997999; nearest-rank central endpoints = zero-based sorted replicate
  indices 249 and 9749;
- fail closed on nonfinite inputs or fewer than 20 eligible blocks in the OBSERVED population
  (a replicate may contain fewer than 20 unique blocks).

Versioned strings chosen by this candidate (bound fields, reviewed at step 3):
  KEY_SCHEMA = unifyweaver.sm-fs-ranking-bootstrap-key.v1
  SAMPLER_ID = sm-fs-ranking-bootstrap-v1
"""
import hashlib
import math

from sm_fs_sampler import canonical_json_bytes

KEY_SCHEMA = "unifyweaver.sm-fs-ranking-bootstrap-key.v1"
SAMPLER_ID = "sm-fs-ranking-bootstrap-v1"
RESAMPLES, SEED, N_BLOCKS = 9999, 3997999, 82
LO_INDEX, HI_INDEX = 249, 9749
MIN_OBSERVED_BLOCKS = 20


class BootstrapError(ValueError):
    pass


def _need(c, m):
    if not c:
        raise BootstrapError(m)


def block_draw(seed, replicate, draw):
    """One rejection-sampled block index in [0, 82)."""
    limit = ((1 << 256) // N_BLOCKS) * N_BLOCKS
    retry = 0
    while True:
        key = canonical_json_bytes({
            "draw": draw, "replicate": replicate, "retry": retry,
            "sampler_id": SAMPLER_ID, "schema": KEY_SCHEMA, "seed": seed,
        })
        value = int.from_bytes(hashlib.sha256(key).digest(), "big", signed=False)
        if value < limit:
            return value % N_BLOCKS, retry
        retry += 1


def decide(block_values):
    """block_values: {block_name: [d(q), ...]} over the canonical observed blocks.

    Returns the decision record: point estimate (query-weighted over observed blocks),
    nearest-rank CI, replicate stats, and gate booleans. Fails closed on nonfinite inputs
    or an observed population under 20 blocks."""
    blocks = sorted(block_values)                     # canonical UTF-8 sort
    _need(len(blocks) == N_BLOCKS,
          f"observed block list has {len(blocks)} entries, frozen spec requires {N_BLOCKS}")
    eligible = [b for b in blocks if block_values[b]]
    _need(len(eligible) >= MIN_OBSERVED_BLOCKS,
          f"only {len(eligible)} nonempty observed blocks (<{MIN_OBSERVED_BLOCKS})")
    for b in blocks:
        for v in block_values[b]:
            _need(isinstance(v, float) and math.isfinite(v), f"nonfinite d(q) in block {b}")
    total = sum(v for b in blocks for v in block_values[b])
    count = sum(len(block_values[b]) for b in blocks)
    _need(count > 0, "no paired differences")
    point = total / count
    sums = {b: sum(block_values[b]) for b in blocks}
    lens = {b: len(block_values[b]) for b in blocks}
    stats, attempts = [], 0
    for rep in range(RESAMPLES):
        num = den = 0.0
        for draw in range(N_BLOCKS):
            i, retry = block_draw(SEED, rep, draw)
            attempts += 1 + retry
            b = blocks[i]
            num += sums[b]
            den += lens[b]
        _need(den > 0, f"replicate {rep} drew only empty blocks")
        stats.append(num / den)
    stats.sort()
    return {
        "delta_mrr": point, "ci95": [stats[LO_INDEX], stats[HI_INDEX]],
        "resamples": RESAMPLES, "seed": SEED, "draws_per_replicate": N_BLOCKS,
        "endpoint_rule": "nearest-rank-zero-based-249-9749",
        "sampler": {"schema": KEY_SCHEMA, "sampler_id": SAMPLER_ID,
                    "rejection_limit": "floor(2^256/82)*82"},
        "bootstrap_mean": sum(stats) / len(stats), "sampler_attempts": attempts,
        "observed_blocks": len(blocks), "nonempty_blocks": len(eligible),
        "query_count": count,
        "passed_exploratory_gate": bool(point >= 0.010 and stats[LO_INDEX] > 0.0),
    }
