#!/usr/bin/env python3
"""Sampler obligations.

The load-bearing claim is not "the sampler runs" but "the sampler REACHES
the ruled coverage minimum, and the coverage it claims is the coverage the
extractor measures". Both are checked here, at a small `k` so the suite
stays fast — the mechanism is `k`-independent, and the headline `k = 100`
figure is reproduced by the module's own main().
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import process_cards as pc
import process_expression_enumerator as en
import process_expression_sampler as sam

K = 3


@pytest.fixture(scope="module")
def report():
    return sam.sample(coverage_minimum_k=K)


@pytest.fixture(scope="module")
def realizations():
    return sam.pair_realizations()


def test_every_required_identity_reaches_k(report):
    """The objective, stated as a test. A sampler that emitted rows without
    reaching `k` would satisfy every other check in this file."""
    assert report["starved_items"] == []
    assert report["starved_pairs"] == []
    assert report["rows"] > 0


def test_the_extractor_agrees_with_the_sampler(report):
    """The sampler counts coverage with the extractor's own helper, which
    is still the author checking their own arithmetic. This re-measures
    through the PUBLIC path — corpus policy, authoritative-universe
    containment, canonical_full dedup, semantic-once counting."""
    manifest = sam.verify_against_extractor(report)
    assert manifest["rows_kept"] == report["rows"]
    # Rows are constructed distinct, so the extractor must find nothing to
    # dedup; a duplicate here means the sampler paid for a row twice.
    assert manifest["duplicates_removed"] == 0
    assert manifest["authorizing"] is True


def test_every_realized_pair_is_witnessed_by_its_own_witness_row(realizations):
    """Realizability is proved by construction, so the proof must hold: the
    row recorded for a pair has to pass policy AND actually carry it."""
    realized, _ = realizations
    leaf_index = en._leaf_identity_index()
    for pair, (expression, _slot) in sorted(realized.items()):
        node = pc.parse(expression)
        assert en.corpus_policy_violation(node, "methodology-root-only") is None
        items: set = set()
        pairs: set = set()
        en._row_witnesses(node, items, pairs, leaf_index)
        assert pair in pairs, expression


def test_refused_candidates_carry_reasons_and_are_not_realized(realizations):
    """A candidate the grammar offers but no row can carry is recorded with
    its reason. Silence would let a generator defect shrink the universe."""
    realized, refused = realizations
    assert refused, "the caps make some compositions unreachable; none recorded"
    for candidate, reason in refused.items():
        assert candidate.startswith("candidate:")
        assert reason and isinstance(reason, str)
        # A refusal names a grammar or cap fact, never an unexplained miss.
        assert (reason.startswith("invalid:")
                or reason.startswith("policy:")
                or reason.startswith("unconstructible:")), reason
    assert not (set(refused) & set(realized))


def test_the_caps_make_some_grammatical_compositions_unreachable(realizations):
    """A measured consequence worth pinning rather than rediscovering:
    `blend/3` needs one weight per positional source, but the list cap is
    2, so no admissible row carries `blend/3` with an explicit `w`."""
    _realized, refused = realizations
    reasons = " ".join(refused.values())
    assert "one weight per positional source" in reasons
    assert en.MAX_LIST_LENGTH < 3


def test_variation_never_changes_the_identity_being_covered(realizations):
    """The freeze property. A stream opened for a pair must keep witnessing
    THAT pair: varying the child endpoint would silently redirect coverage
    to a neighbour, and the objective would be met on paper only."""
    realized, _ = realizations
    leaf_index = en._leaf_identity_index()
    sampled = sorted(realized)[::97][:12]
    for pair in sampled:
        expression, slot = realized[pair]
        rows = sam._stream(pc.parse(expression), "methodology-root-only",
                           8, slot)
        assert rows
        for row in rows:
            items: set = set()
            pairs: set = set()
            en._row_witnesses(pc.parse(row), items, pairs, leaf_index)
            assert pair in pairs, (pair, row)


def test_the_sampler_is_deterministic():
    """No seed exists because no randomness does — two runs are the same
    corpus, or the measurement is not reproducible."""
    first = sam.sample(coverage_minimum_k=2)
    second = sam.sample(coverage_minimum_k=2)
    assert first["expressions"] == second["expressions"]
    assert first["rows"] == second["rows"]


def test_mandatory_synthetic_coverage_is_reached(report):
    """§3 is not decorative: pins are required per operator HOST and all
    three string classes are required. Both were starved by construction
    defects that no other assertion in this file would have caught."""
    universe = set(en.required_witness_universe())
    starved = set(report["starved_items"])
    for item in universe:
        if item.startswith("synthetic:"):
            assert item not in starved, item


def test_manifest_is_hashed_into_the_report(report):
    assert report["sampler_manifest_sha256"] == sam.sampler_manifest_sha256()
    assert report["registry_version"] == pc.REGISTRY_VERSION
    assert report["coverage_minimum_k"] == K


def test_pair_universe_hash_is_stable():
    assert (sam.required_pair_universe_sha256()
            == sam.required_pair_universe_sha256())
    assert len(sam.required_pair_universe()) == len(
        set(sam.required_pair_universe()))


def test_a_nonpositive_minimum_fails_closed():
    with pytest.raises(sam.SamplerError):
        sam.sample(coverage_minimum_k=0)
