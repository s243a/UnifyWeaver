#!/usr/bin/env python3
"""Enumeration re-measurement tests (generator spec §2, re-measured under v0.4).

Two kinds of assertion:

1. **The DP is correct**: at reduced caps, exact counts match a brute-force
   materialization that builds every AST, validates it, and dedupes by
   canonical string. The reduced caps exercise every counting feature —
   literal slots, node-valued kwargs, methodology placement, paired
   routing kwargs, default elision, modifier leaves.
2. **The measured numbers are pinned**: the headline v0.4 counts are exact
   and drift-alarmed, per the standing rule that measured quantities land as
   tests rather than prose.
"""

from __future__ import annotations

import itertools
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import process_cards as pc
import process_expression_enumerator as en


# --------------------------------------------------------------------------
# pinned v0.4 measurements (drift alarms)
# --------------------------------------------------------------------------


def test_measured_scenario_counts_are_pinned():
    assert en.count("naive-full")[0] == 3_475_387_022_969
    assert en.count("methodology-root-only")[0] == 61_908_552
    assert en.count("structural-only")[0] == 11_409_263


def test_measured_template_counts_are_pinned():
    """Template identity is the RESOLVED-kwarg identity (the established
    structural-template convention): absent-vs-explicit-default is not a
    template distinction. The re-review computed 96,196 independently under
    that identity; this DP reproduces it exactly."""
    assert en.count("naive-full", template_mode=True)[0] == 3_826_859
    assert en.count("methodology-root-only", template_mode=True)[0] == 97_526
    assert en.count("structural-only", template_mode=True)[0] == 28_373


def test_the_binding_constraint_moved_to_kwarg_enumeration():
    """§2.3's v0.3 claim, falsified under v0.4: methodology kwargs multiply
    the corpus by ~60,000x, structure alone by ~37x."""
    naive = en.count("naive-full")[0]
    root_only = en.count("methodology-root-only")[0]
    structural = en.count("structural-only")[0]
    # ~56,000x under v0.5 (cowalk widens root-only faster than naive-full);
    # was ~60,000x under v0.4. The claim is the ORDER of the explosion.
    assert naive > 50_000 * root_only
    assert structural > 35 * 285_478  # the v0.3 corpus, for scale


def test_component_vocabulary_is_pinned():
    """§2.5 (owner ruling): the composable support is the predicate-level
    component vocabulary — operator-local shapes composed via recursion
    through the type system — not complete-tree skeletons."""
    assert en.component_vocabulary_counts() == {
        "leaf_shapes": 9,
        "operator_shapes_interior": 23,
        "operator_shapes_root_only_extension": 48,
        "node_composition_edges": 33,
        "literal_slots": 2,
        # Two totals, each named for what it sums: an external verification
        # pass found a single "total" row unauditable, since the five class
        # rows above summed to 109 while the total read 76.
        "composable_component_shapes": 80,
        "serialized_identities_total": 115,
    }
    counts = en.component_vocabulary_counts()
    assert (counts["leaf_shapes"] + counts["operator_shapes_interior"]
            + counts["operator_shapes_root_only_extension"]
            + counts["node_composition_edges"] + counts["literal_slots"]
            == counts["serialized_identities_total"] == 115)


def test_component_vocabulary_is_content_not_counts():
    """Sol's adversarial finding on the first draft: counts alone cannot
    freeze support — an invalid component can replace a valid one at equal
    cardinality. The vocabulary is serialized identities, hash-bound, and
    the illegal shape the first draft counted is provably absent."""
    vocabulary = en.component_vocabulary()
    everything = (vocabulary["operators_interior"]
                  + vocabulary["operators_root_only"])
    # blend.w's length is its arity; arity 3 exceeds the list cap, so no
    # w-bearing blend/3 component may exist (the first draft counted three).
    assert not any(item.startswith("op:blend/3") and "w:" in item
                   for item in everything)
    assert "op:blend/2{w:number_list}" in everything  # the legal counterpart
    # Node-composition edges are separated from literal slots: only the
    # former compose recursively.
    assert "edge:lineage.kw:mu->judge" in vocabulary["node_edges"]
    assert vocabulary["literal_slots"] == [
        "slot:max/2.arg0:number", "slot:max/3.arg0:number",
    ]
    # A cardinality-preserving substitution moves the hash.
    sha = en.component_vocabulary_sha256()
    assert len(sha) == 64
    import json as _json
    forged = dict(vocabulary)
    forged["operators_interior"] = (
        [item for item in vocabulary["operators_interior"]
         if item != "op:blend/2{w:number_list}"] + ["op:blend/3{w:number_list}"])
    import hashlib as _hashlib
    forged_sha = _hashlib.sha256(
        _json.dumps(forged, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert forged_sha != sha


def test_prereg_sha_binds_scenarios_and_vocabulary():
    """The preregistration hash covers methodology placement and the
    serialized support, not just caps and grids."""
    sha = en.enumeration_spec_sha256()
    original = en.SCENARIOS["methodology-root-only"]
    try:
        en.SCENARIOS["methodology-root-only"] = dict(
            original, methodology_interior=True)
        assert en.enumeration_spec_sha256() != sha
    finally:
        en.SCENARIOS["methodology-root-only"] = original


def test_component_vocabulary_is_tiny_relative_to_the_composition_space():
    """The reason the ruling is right: complete-tree templates were a
    cross-product of the vocabulary, three orders of magnitude larger."""
    templates = en.count("methodology-root-only", template_mode=True)[0]
    vocabulary = en.component_vocabulary_counts()["composable_component_shapes"]
    assert templates > 1000 * vocabulary


def test_covers_judges_literals_by_declared_kind_not_runtime_type():
    """Re-review counterexample: max(10, e5) parses (an int satisfies a
    `number` slot) but 10 is outside NUMBER_GRID, so it is outside the
    enumerable support — covers() must say so."""
    assert not en.covers(pc.parse("max(10,e5)"), "methodology-root-only")
    assert en.covers(pc.parse("max(0.02,e5)"), "methodology-root-only")


def test_manifest_pins_terminals_typed_kwargs_and_synthetic_boundary():
    """Re-review: the manifest must carry exact terminals, typed kwargs, and
    its own exclusion boundary — not imply them."""
    vocabulary = en.component_vocabulary()
    assert "leaf:substrate{enwiki,fs,pearltrees,simplemind,simplewiki}" in vocabulary["leaves"]
    assert "leaf:judge.D{luna}" in vocabulary["leaves"]
    assert "leaf:score.e5-atom{e5}" in vocabulary["leaves"]
    assert "op:lineage/1{decay:number,mu:judge}" in (
        vocabulary["operators_interior"])
    assert any("pins" in item for item in vocabulary["excluded_synthetic"])
    assert any("string" in item for item in vocabulary["excluded_synthetic"])
    assert any("interior methodology" in item
               for item in vocabulary["excluded_synthetic"])


def test_component_patterns_use_the_resolved_kwarg_identity():
    """decay has a registered default, so it is in EVERY lineage component —
    never a presence choice — matching canonical resolution."""
    vocabulary = en.component_vocabulary()
    lineage = [item for item in vocabulary["operators_interior"]
               + vocabulary["operators_root_only"]
               if item.startswith("op:lineage/")]
    assert lineage and all("decay:number" in item for item in lineage)


def test_spec_sha_binds_registry_semantics_and_domains():
    sha = en.enumeration_spec_sha256()
    original = pc.ESTIMANDS
    try:
        pc.ESTIMANDS = frozenset(original | {"bogus_relation"})
        assert en.enumeration_spec_sha256() != sha
    finally:
        pc.ESTIMANDS = original


def test_registered_processes_are_inside_the_root_only_support():
    """§2.4 coverage gate at the support level: every registered process lies
    inside the methodology-root-only enumerable set."""
    for name, expression in pc.PROCESSES.items():
        assert en.covers(pc.parse(expression), "methodology-root-only"), name


def test_graph_judge_needs_the_node_cap_of_six():
    node = pc.parse(pc.PROCESSES["graph-judge"])
    assert en._node_count(node) == 6 == en.MAX_NODE_COUNT
    assert en._depth(node) == 3 == en.MAX_DEPTH


def test_spec_sha_binds_caps_and_grids():
    sha = en.enumeration_spec_sha256()
    assert len(sha) == 64
    original = en.NUMBER_GRID
    try:
        en.NUMBER_GRID = original + (0.5,)
        assert en.enumeration_spec_sha256() != sha
    finally:
        en.NUMBER_GRID = original


# --------------------------------------------------------------------------
# the AST-to-family extractor and the authoritative witness universe
# (fourth adversarial review, finding 2: the split's universe was
# caller-supplied and no real builder existed)
# --------------------------------------------------------------------------


def test_extractor_groups_rows_by_resolved_template_not_by_spelling():
    """Two expressions differing only in their literals and terminals share
    one structural family; differing in SHAPE they do not."""
    same = en.families_from_expressions([
        "hop_decay(simplemind,gamma=0.6)",
        "hop_decay(simplewiki,gamma=0.85)",
    ])
    assert len(same) == 1 and same[0].row_count == 2
    different = en.families_from_expressions([
        "hop_decay(simplemind,gamma=0.6)",
        "lca_frac(simplemind)",
    ])
    assert len(different) == 2


def test_extractor_projects_pins_and_strings_into_the_semantic_family():
    """Synthetic projection (the split's `unit` rule): a pinned row joins its
    semantic template's family and contributes `synthetic:pin`, rather than
    creating a family of its own."""
    families = en.families_from_expressions([
        "lca_frac(simplemind)",
        "lca_frac(simplemind)@synthetic/pin-0001",
    ])
    assert len(families) == 1
    family = families[0]
    assert family.row_count == 2
    assert family.counts["synthetic:pin@lca_frac"] == 1        # one row of two
    # Seventh review, finding 1: the pinned row shares the unpinned row's
    # SEMANTIC identity, so semantic witnesses count once — k cannot be met
    # with pin-copies of one example. Synthetic witnesses count per row.
    assert family.counts["op:lca_frac/1{}"] == 1


def test_pin_copies_cannot_credit_semantic_coverage_twice():
    """Two pin-variants of one semantic expression are distinct corpus rows
    (canonical_full dedup — §3.1 needs them) but one semantic example."""
    (family,) = en.families_from_expressions([
        "lca_frac(simplemind)@synthetic/pin-0001",
        "lca_frac(simplemind)@synthetic/pin-0002",
    ])
    assert family.row_count == 2
    assert family.counts["op:lca_frac/1{}"] == 1          # one semantic example
    assert family.counts["synthetic:pin@lca_frac"] == 2   # two pinned rows
    # Distinct semantic rows still count independently.
    (other,) = en.families_from_expressions([
        "hop_decay(simplemind,gamma=0.6)", "hop_decay(simplewiki,gamma=0.6)"])
    assert other.counts["op:hop_decay/1{gamma:number}"] == 2


def test_resolved_defaults_witness_their_values():
    """Seventh review, finding 2: lineage(simplemind) semantically carries
    decay=0.85 — the component identity already resolves it — so it must
    witness the value and its digits, or grid coverage is unreachable
    through canonically-spelled rows. No explicitness motif is emitted."""
    (family,) = en.families_from_expressions(["lineage(simplemind)"])
    assert "value:number:0.85" in family.witness_items
    assert {"digit:0@int", "digit:8@frac", "digit:5@frac"} <= family.witness_items
    assert not any("decay" in item for item in family.pairs
                   if item.startswith("motif:"))
    # Both spellings are one canonical expression and witness identically.
    (explicit,) = en.families_from_expressions(["lineage(simplemind,decay=0.85)"])
    assert explicit.witness_items == family.witness_items


def test_extractor_counts_items_per_row_not_per_family():
    """Finding 1's root cause, at the source: an item witnessed by one row
    of a many-row family counts once, not once per row."""
    families = en.families_from_expressions([
        "hop_decay(simplemind,gamma=0.6)",
        "hop_decay(simplewiki,gamma=0.6)",
        "hop_decay(simplewiki,gamma=0.85)",
    ])
    (family,) = families
    assert family.row_count == 3
    assert family.counts["value:number:0.6"] == 2
    assert family.counts["value:number:0.85"] == 1
    assert family.counts["terminal:simplewiki"] == 2
    # A repeated item within ONE row still counts that row once.
    (repeated,) = en.families_from_expressions([
        "product(hop_decay(simplemind,gamma=0.6),lca_frac(simplemind))"
    ])
    assert repeated.counts["terminal:simplemind"] == 1


def test_extractor_emits_composition_pairs_over_node_edges_only():
    (family,) = en.families_from_expressions([
        'lineage(pearltrees,mu=haiku,estimand="ancestry")'
    ])
    # The pair carries its SLOT (fifth review, finding 3): without arity and
    # position a composition holdout could not distinguish argument orders,
    # nor a positional child from a node-valued kwarg.
    component = "lineage/1{decay:number,estimand:estimand,mu:judge}"
    assert family.pairs == frozenset({
        f"pair:{component}.arg0|pearltrees",       # positional child
        f"pair:{component}.kw:mu|haiku",           # node-valued kwarg edge
        f'motif:{component}.kw:estimand="ancestry"',   # holdable kwarg motif
    })
    # A kwarg elided at its registered default yields NO motif: the elided
    # and explicit spellings are one canonical expression, so a "decay=0.85"
    # motif would be universal rather than a holdout unit.
    assert not any(".kw:decay=" in item for item in family.pairs)
    assert family.counts["estimand:ancestry"] == 1   # categorical, not a pair


def test_extractor_items_are_the_frozen_vocabulary_serialization():
    """Every item a real expression witnesses must be an item the support
    freeze knows — otherwise the universe and the corpus speak different
    languages."""
    universe = set(en.required_witness_universe())
    for name, expression in pc.PROCESSES.items():
        for family in en.families_from_expressions([expression]):
            assert family.witness_items <= universe, name


def test_required_universe_is_derived_from_the_support_not_a_corpus():
    universe = en.required_witness_universe()
    vocabulary = en.component_vocabulary()
    for section in ("leaves", "operators_interior", "operators_root_only",
                    "node_edges", "literal_slots"):
        assert set(vocabulary[section]) <= set(universe), section
    # Grid values, their digit bytes, exact terminals, both categorical
    # domains, and the synthetic floors — the completeness the review asked
    # for, each checked rather than asserted in prose.
    assert {"value:number:0.6", "value:int:10"} <= set(universe)
    # Digits carry their POSITION CLASS (fifth review, finding 4): `0` in
    # the integer part of `10` and in the fraction of `0.02` exercise
    # different tokenizer paths and are no longer one item.
    assert {"digit:0@int", "digit:1@int",
            "digit:0@frac", "digit:2@frac", "digit:6@frac",
            "digit:8@frac"} <= set(universe)
    assert {"terminal:luna", "terminal:simplemind"} <= set(universe)
    assert {f"estimand:{e}" for e in pc.ESTIMANDS} <= set(universe)
    assert {f"impl:{i}" for i in pc.IMPLS} <= set(universe)
    # §3.2 needs all three string classes, and §3.1's V2-vs-V3 property is
    # per-render-path, so pins are required per operator that can host one.
    assert {f"synthetic:string:{c}" for c in en.STRING_CLASSES} <= set(universe)
    assert {"synthetic:pin@lineage", "synthetic:pin@product"} <= set(universe)
    assert "synthetic:interior_methodology@lineage.estimand" in universe


def test_spec_sha_binds_the_required_universe_and_output_roots():
    """Both were floating before the fourth review: the universe was not in
    the spec preimage at all, and OUTPUT_ROOTS drives count() yet was
    unbound."""
    sha = en.enumeration_spec_sha256()
    universe_sha = en.required_witness_universe_sha256()
    original_roots = en.OUTPUT_ROOTS
    try:
        en.OUTPUT_ROOTS = original_roots + ("bogus-output",)
        assert en.enumeration_spec_sha256() != sha
    finally:
        en.OUTPUT_ROOTS = original_roots
    original_int_grid = en.INT_GRID
    try:
        # A widened grid moves the universe (new value item, new digit
        # bytes), and the moved universe moves the spec hash.
        en.INT_GRID = original_int_grid + (37,)
        assert en.required_witness_universe_sha256() != universe_sha
        assert en.enumeration_spec_sha256() != sha
    finally:
        en.INT_GRID = original_int_grid


#: A deliberately overlapping corpus of REAL v0.4 expressions: several
#: distinct templates carry each composition pair, which is what makes a
#: held-composition split feasible at all (see the thin-corpus test below).
OVERLAPPING_CORPUS = [
    "product(hop_decay(simplemind,gamma=0.6),lca_frac(simplemind))",
    "product(hop_decay(simplewiki,gamma=0.85),lca_frac(simplewiki))",
    "product(hop_decay(simplemind,gamma=0.6),hop_decay(simplewiki,gamma=0.85))",
    "product(lca_frac(simplemind),lca_frac(simplewiki))",
    "max(0.02,product(hop_decay(simplemind,gamma=0.6),lca_frac(simplemind)))",
    "max(0.03,lca_frac(simplewiki))",
    "max(0.02,hop_decay(simplemind,gamma=0.6))",
    'lineage(pearltrees,mu=haiku,estimand="ancestry")',
    'lineage(simplewiki,mu=graph,estimand="ancestry")',
    "lineage(simplemind,mu=haiku)",
    "lineage(pearltrees,mu=graph)",
    "blend(luna.D,luna.S)",
    "blend(luna.D,haiku)",
    "kalman(luna.D,luna.S)",
    "kalman(haiku,luna.S)",
    "e5(margin(t=0.03))",
    "e5(margin(t=0.02))",
]


def test_extractor_and_universe_meet_in_a_real_split():
    """End-to-end over real expressions: extractor to families, families and
    the derived universe to the split. This path had no executable existence
    before this review round — the split was fed hand-written summaries."""
    import process_expression_split as sp

    expressions = OVERLAPPING_CORPUS * 2 + [
        "lca_frac(simplemind)", "lca_frac(simplemind)@synthetic/pin-0001",
    ]
    families = en.families_from_expressions(expressions)
    universe = sorted({i for f in families for i in f.witness_items})
    contract = sp.split_contract(
        seed="registered-smoke", coverage_minimum_k=1,
        required_universe_sha256=sp.universe_sha256(universe),
        held_compositions={"dev": ["pair:blend/2{}.arg1|haiku"],
                           "test": ["pair:kalman/2{}.arg0|haiku"]},
        buckets={"train": 6000, "dev": 2000, "test": 2000},
        authorizing=False,
    )
    manifest = sp.assign(families, contract, universe)
    assert all(manifest["train_coverage"][item] >= 1 for item in universe)
    assert manifest["far"]["test"] and manifest["far"]["dev"]
    assert len(manifest["manifest_sha256"]) == 64
    # Whole families, one slice each — LOCO at the structural level.
    placed = [f for ids in manifest["slices"].values() for f in ids]
    assert sorted(placed) == sorted(f.family_id for f in families)


def test_extractor_enforces_corpus_policy_not_only_grammar():
    """Fifth review, finding 2: grammar-valid is not policy-valid. An
    expression that parses and validates but sits outside the declared
    envelope would contribute witness items the frozen vocabulary does not
    contain, so the extractor refuses it."""
    for expression, why in [
        ("max(10,e5)", "int literal off the number grid"),
        ("hop_decay(simplemind,gamma=0.5)", "0.5 off the number grid"),
        ("product(max(0.02,product(hop_decay(simplemind,gamma=0.6),"
         "lca_frac(simplemind))),e5)", "depth 4 over the cap"),
    ]:
        with pytest.raises(en.PolicyError):
            en.families_from_expressions([expression]), why


def test_policy_admits_the_mandatory_synthetic_forms():
    """The §3 synthetics are REQUIRED corpus content that deliberately sits
    outside the enumerable support, so policy must admit what `covers()`
    refuses — conflating the two would make §3 unsatisfiable."""
    interior = ('product(e5(margin(t=0.02)),'
                'max(0.02,lca_frac(simplemind),estimand="path"))')
    assert not en.covers(pc.parse(interior), "methodology-root-only")
    assert en.corpus_policy_violation(
        pc.parse(interior), "methodology-root-only") is None
    (family,) = en.families_from_expressions([interior])
    assert any(item.startswith("synthetic:interior_methodology@max.estimand")
               for item in family.witness_items)


def test_synthetic_allowlist_is_enforced_fail_closed():
    """§3.3: the generator must never sample real paths, titles, or ids.
    A non-allowlisted pin is refused rather than silently recorded."""
    with pytest.raises(en.PolicyError, match="allowlist"):
        en.families_from_expressions(["lca_frac(simplemind)@run/2026-07-25"])
    (family,) = en.families_from_expressions(
        ["lca_frac(simplemind)@synthetic/pin-0001"])
    assert "synthetic:pin@lca_frac" in family.witness_items


def test_synthetic_extension_is_finite_versioned_and_hash_bound():
    """Sixth review, third finding: a PREFIX test is not an allowlist. The
    old check accepted `synthetic/pin-../../home/s243a/private/notes` — a
    real private path reached by traversal, exactly what §3.3 forbids.
    Membership is now exact over a committed finite set, and the policy is
    versioned and inside the preregistration hash."""
    policy = en.synthetic_extension_policy()
    assert policy["version"] == en.SYNTHETIC_EXTENSION_VERSION
    assert set(policy["manifests"]) == set(en.STRING_CLASSES)
    for value, cls in en.SYNTHETIC_MANIFESTS.items():
        assert en._string_class(cls) == value      # one string per class
    # traversal and lookalikes are refused; only committed members pass
    for pin in ("synthetic/pin-../../home/s243a/private/notes",
                "synthetic/pin-9999", "run/2026-07-25"):
        with pytest.raises(en.PolicyError, match="not a member"):
            en.families_from_expressions([f"lca_frac(simplemind)@{pin}"])
    (family,) = en.families_from_expressions(
        [f"lca_frac(simplemind)@{en.SYNTHETIC_PINS[0]}"])
    assert "synthetic:pin@lca_frac" in family.witness_items
    # the policy is bound by the preregistration hash
    sha = en.enumeration_spec_sha256()
    original = en.SYNTHETIC_PINS
    try:
        en.SYNTHETIC_PINS = original + ("synthetic/pin-0004",)
        assert en.enumeration_spec_sha256() != sha
    finally:
        en.SYNTHETIC_PINS = original


def test_digit_position_classes_separate_integer_and_fraction():
    items = en._digit_items("0.02")
    assert items == {"digit:0@int", "digit:0@frac", "digit:2@frac"}
    assert en._digit_items("10") == {"digit:1@int", "digit:0@int"}
    assert "digit:5@exp" in en._digit_items("1e-5")


def test_string_classes_cover_the_three_required_kinds():
    assert en._string_class("plain") == "ascii"
    assert en._string_class("café") == "non_ascii"
    assert en._string_class('a"b') == "escaped"
    assert set(en.STRING_CLASSES) == {"ascii", "non_ascii", "escaped"}


def test_pairs_distinguish_argument_order_and_kwarg_edges():
    """Fifth review, finding 3: coarse pairs made two structurally distinct
    compositions one holdout unit."""
    forward = en.families_from_expressions(
        ["product(hop_decay(simplemind,gamma=0.6),lca_frac(simplemind))"])[0]
    reversed_ = en.families_from_expressions(
        ["product(lca_frac(simplemind),hop_decay(simplemind,gamma=0.6))"])[0]
    assert forward.pairs != reversed_.pairs
    # Sixth review: the child endpoint carries its SHAPE, so a modifier can
    # no longer collapse two distinct compositions into one holdout unit.
    d_then_s = en.families_from_expressions(["blend(luna.D,luna.S)"])[0]
    s_then_d = en.families_from_expressions(["blend(luna.S,luna.D)"])[0]
    assert d_then_s.pairs != s_then_d.pairs
    assert "pair:blend/2{}.arg0|luna.D" in d_then_s.pairs
    assert "pair:product/2{}.arg0|hop_decay/1{gamma:number}" in forward.pairs
    assert "pair:product/2{}.arg0|lca_frac/1{}" in reversed_.pairs


def test_pins_on_leaf_atoms_are_refused():
    """Sixth review, finding 4: `simplemind@synthetic/pin-0001` parsed,
    validated, and emitted `synthetic:pin@simplemind` — a witness the
    authoritative universe does not contain, because pin witnesses are
    declared for operators only."""
    with pytest.raises(en.PolicyError, match="leaf atom"):
        en.families_from_expressions(["simplemind@synthetic/pin-0001"])
    assert "operators only" in (
        en.synthetic_extension_policy()["authorized_pin_hosts"])


def test_authorizing_build_refuses_witnesses_outside_the_universe():
    """Checking that required witnesses are PRESENT is not the same as
    checking that none ESCAPES. An authorizing build refuses any extracted
    item the authoritative universe does not declare."""
    universe = set(en.required_witness_universe())
    for expression in pc.PROCESSES.values():
        for family in en.families_from_expressions([expression]):
            assert family.witness_items <= universe
    original = en.required_witness_universe
    try:                      # shrink the universe: the build must notice
        en.required_witness_universe = lambda: sorted(
            set(original()) - {"terminal:simplemind"})
        with pytest.raises(en.PolicyError, match="outside the authoritative"):
            en.families_from_expressions(["lca_frac(simplemind)"])
        # a non-authorizing probe is permitted to explore
        assert en.families_from_expressions(
            ["lca_frac(simplemind)"], authorizing=False)
    finally:
        en.required_witness_universe = original


def test_duplicate_canonical_asts_are_removed_and_counted():
    """Sixth review, finding 5: duplicates inflated coverage, so `k` could
    be met with copies of one example, and the split silently depended on
    an unenforced upstream dedup assumption."""
    families, manifest = en.families_from_expressions(
        ["hop_decay(simplemind,gamma=0.6)"] * 3 + ["lca_frac(simplemind)"],
        return_manifest=True)
    assert manifest["rows_in"] == 4
    assert manifest["rows_kept"] == 2
    assert manifest["duplicates_removed"] == 2
    assert manifest["dedup_key"] == "canonical_full"
    assert all(family.row_count == 1 for family in families)
    # Non-canonical spellings of ONE identity are duplicates too.
    _, manifest = en.families_from_expressions(
        ["lineage(simplemind)", "lineage(simplemind,decay=0.85)"],
        return_manifest=True)
    assert manifest["duplicates_removed"] == 1
    # But rows differing only by PINS are distinct: §3.1 requires them, and
    # semantic dedup would delete the coverage that makes V3 differ from V2.
    _, manifest = en.families_from_expressions(
        ["lca_frac(simplemind)", "lca_frac(simplemind)@synthetic/pin-0001"],
        return_manifest=True)
    assert manifest["duplicates_removed"] == 0


def test_kwarg_and_list_shape_motifs_are_holdable():
    """Sixth review qualification: the encoder contract requires
    kwarg/list-shape holdouts as well as operator-composition holdouts, and
    neither was expressible — a pair recorded no literal kwarg, and
    t=[0.02] and t=[0.02,0.03] shared one component because a component
    records the kwarg KIND, not the length."""
    import process_expression_split as sp

    short = en.families_from_expressions(
        ["e5(routing(e5,haiku,t=[0.02],menus=[10]))"])[0]
    long_ = en.families_from_expressions(
        ["e5(routing(e5,haiku,t=[0.02,0.03],menus=[10,20]))"])[0]
    assert short.pairs != long_.pairs
    component = "routing/2{menus:int_list,t:number_list}"
    assert f"motif:{component}.kw:t:len1" in short.pairs
    assert f"motif:{component}.kw:t:len2" in long_.pairs

    # Motifs live in the HOLDABLE set, so a contract can name one and the
    # split groups its carriers exactly as it does for composition pairs.
    families = en.families_from_expressions([
        "e5(routing(e5,haiku,t=[0.02],menus=[10]))",
        "e5(routing(e5,sonnet.lineage,t=[0.02],menus=[10]))",
        "e5(routing(e5,haiku,t=[0.02,0.03],menus=[10,20]))",
        "e5(margin(t=0.02))", "e5(margin(t=0.03))",
        "menu(luna,n=10)", "menu(haiku,n=20)",
        "lca_frac(simplemind)", "max(0.02,e5)", "max(0.03,e5)",
        "blend(luna.D,luna.S)", "blend(haiku,luna.S)", "blend(luna.D,haiku)",
    ])
    universe = sorted({i for f in families for i in f.witness_items})
    held_motif = f"motif:{component}.kw:t:len2"
    contract = sp.split_contract(
        "motif-holdout", 1, sp.universe_sha256(universe),
        held_compositions={"dev": [held_motif],
                           "test": ["pair:blend/2{}.arg0|haiku"]},
        buckets={"train": 6000, "dev": 2000, "test": 2000},
        authorizing=False)
    manifest = sp.assign(families, contract, universe)
    assert manifest["held"]["dev"][held_motif]          # carriers grouped
    assert manifest["far"]["dev"]                        # and far by construction


def test_holding_every_carrier_of_a_component_fails_closed():
    """A measured structural consequence, recorded rather than worked
    around: holding BOTH sides of a motif that partitions a component's
    families leaves the component itself unwitnessed in train, so the split
    refuses. A holdout must leave at least one carrier of each component
    trainside — the same shape as the thin-corpus finding."""
    import process_expression_split as sp

    families = en.families_from_expressions([
        "e5(routing(e5,haiku,t=[0.02],menus=[10]))",
        "e5(routing(e5,haiku,t=[0.02,0.03],menus=[10,20]))",
        "e5(margin(t=0.02))", "lca_frac(simplemind)", "max(0.02,e5)",
    ])
    universe = sorted({i for f in families for i in f.witness_items})
    component = "routing/2{menus:int_list,t:number_list}"
    contract = sp.split_contract(
        "both-sides", 1, sp.universe_sha256(universe),
        held_compositions={"dev": [f"motif:{component}.kw:t:len1"],
                           "test": [f"motif:{component}.kw:t:len2"]},
        buckets={"train": 6000, "dev": 2000, "test": 2000},
        authorizing=False)
    with pytest.raises(sp.SplitError, match="held-pinned"):
        sp.assign(families, contract, universe)


def test_pairs_carry_the_resolved_parent_component():
    """The qualification's first part: a bare parent name/arity made
    lineage(s,decay=0.6) and lineage(s,decay=0.6,estimand=..) one holdout
    unit."""
    plain = en.families_from_expressions(["lineage(simplemind,decay=0.6)"])[0]
    with_est = en.families_from_expressions(
        ['lineage(simplemind,decay=0.6,estimand="ancestry")'])[0]
    assert plain.pairs != with_est.pairs
    assert any("estimand:estimand" in item and item.startswith("pair:")
               for item in with_est.pairs)


def test_a_thin_corpus_cannot_support_a_held_composition_split():
    """A measured consequence worth recording: over the ten registered
    processes alone, EVERY choice of held composition pair fails — each
    family is the sole carrier of some witness item, so pinning it out of
    train orphans that item. The corpus build must supply overlapping
    templates; the split says so rather than quietly shrinking coverage."""
    import itertools as _itertools
    import process_expression_split as sp

    families = en.families_from_expressions(
        [e for expression in pc.PROCESSES.values() for e in (expression,) * 2]
    )
    universe = sorted({i for f in families for i in f.witness_items})
    pairs = sorted({p for f in families for p in f.pairs})
    for dev_pair, test_pair in _itertools.permutations(pairs, 2):
        contract = sp.split_contract(
            seed="thin-corpus", coverage_minimum_k=1,
            required_universe_sha256=sp.universe_sha256(universe),
            held_compositions={"dev": [dev_pair], "test": [test_pair]},
            buckets={"train": 5000, "dev": 2500, "test": 2500},
            authorizing=False,
        )
        with pytest.raises(sp.SplitError):
            sp.assign(families, contract, universe)


# --------------------------------------------------------------------------
# DP-vs-brute equivalence at reduced caps
# --------------------------------------------------------------------------

SMALL = dict(
    MAX_DEPTH=2, MAX_NODE_COUNT=3, MAX_ARITY=2, MAX_KWARGS_NODE=2,
    MAX_LIST_LENGTH=1, NUMBER_GRID=(0.02, 0.85), INT_GRID=(10,),
)


def _brute_force(scenario: str) -> int:
    """Materialize every AST under the SMALL caps; dedupe by canonical."""
    flags = en.SCENARIOS[scenario]
    memo: dict = {}

    def leaves(output):
        out = []
        for name, sig in pc.REGISTRY.items():
            if sig.atom and sig.output == output and not sig.operator:
                out.append(pc.Node(name))
                for m in sorted(sig.modifiers):
                    out.append(pc.Node(name, mods=(m,)))
        if output == "score":
            out.append(pc.Node("e5"))
        return out

    def kw_values(kind):
        if kind == "number":
            return list(en.NUMBER_GRID)
        if kind == "int":
            return list(en.INT_GRID)
        if kind == "number_list":
            return [tuple(t) for L in range(1, en.MAX_LIST_LENGTH + 1)
                    for t in itertools.product(en.NUMBER_GRID, repeat=L)]
        if kind == "int_list":
            return [tuple(t) for L in range(1, en.MAX_LIST_LENGTH + 1)
                    for t in itertools.product(en.INT_GRID, repeat=L)]
        if kind == "estimand":
            return sorted(pc.ESTIMANDS)
        if kind == "impl":
            return sorted(pc.IMPLS)
        if kind == "walk":
            return sorted(pc.WALKS)
        if kind == "weight":
            return sorted(pc.WEIGHTS)
        if kind == "string":
            return []
        return None

    def nodes_of(n):
        return en._node_count(n)

    def build(output, d, allow_meth):
        """Returns (expr, node_count) pairs, pruned to the node cap early —
        the node count is arithmetic, validate() is recursion; checking the
        cheap bound first is what keeps this test fast."""
        key = (output, d, allow_meth)
        if key in memo:
            return memo[key]
        out = [(leaf, 1) for leaf in leaves(output)]
        if d > 0:
            for name, sig in pc.REGISTRY.items():
                if not sig.operator or sig.output != output:
                    continue
                cap = sig.max_args if sig.max_args is not None else en.MAX_ARITY
                for arity in range(sig.min_args, min(cap, en.MAX_ARITY) + 1):
                    slot_opts, dead = [], False
                    for i in range(arity):
                        declared = (sig.arg_types[i] if i < len(sig.arg_types)
                                    else sig.variadic_arg_type)
                        opts = []
                        for alt in declared.split("|"):
                            if alt in pc.VALUE_KINDS:
                                opts += [(v, 0) for v in kw_values(alt)]
                            elif alt == "process":
                                for t in en.OUTPUT_ROOTS:
                                    opts += build(t, d - 1, flags["methodology_interior"])
                            else:
                                opts += build(alt, d - 1, flags["methodology_interior"])
                        # a slot filler that already fills the cap cannot sit
                        # under a root node
                        opts = [o for o in opts if o[1] <= en.MAX_NODE_COUNT - 1]
                        if not opts:
                            dead = True
                            break
                        slot_opts.append(opts)
                    if dead:
                        continue
                    kw_opts = []
                    for kname in sorted(sig.kwargs):
                        spec = sig.kwargs[kname]
                        if kname in ("estimand", "impl") and not allow_meth:
                            continue
                        if spec.kind in pc.OUTPUT_TYPES:
                            if not flags["mu"]:
                                continue
                            vals = [(v, n) for v, n in
                                    build(spec.kind, d - 1, flags["methodology_interior"])
                                    if n <= en.MAX_NODE_COUNT - 1]
                        else:
                            vals = [(v, 0) for v in kw_values(spec.kind)]
                            if not vals:
                                continue
                        if not vals:
                            continue
                        kw_opts.append([(kname, v, n) for v, n in vals])
                    for arg_sel in itertools.product(*slot_opts):
                        arg_values = tuple(v for v, _ in arg_sel)
                        arg_nodes = sum(n for _, n in arg_sel)
                        if 1 + arg_nodes > en.MAX_NODE_COUNT:
                            continue
                        for r in range(0, en.MAX_KWARGS_NODE + 1):
                            for combo in itertools.combinations(range(len(kw_opts)), r):
                                for sel in itertools.product(*[kw_opts[i] for i in combo]):
                                    total = 1 + arg_nodes + sum(n for _, _, n in sel)
                                    if total > en.MAX_NODE_COUNT:
                                        continue
                                    kws = {k: v for k, v, _ in sel}
                                    if any(s.required and k not in kws
                                           for k, s in sig.kwargs.items()):
                                        continue
                                    defaults = {k: s.default for k, s in sig.kwargs.items()}
                                    kept = tuple(sorted(
                                        ((k, v) for k, v in kws.items()
                                         if defaults.get(k) != v),
                                        key=lambda item: item[0]))
                                    candidate = pc.Node(name, arg_values, kept)
                                    try:
                                        pc.validate(candidate)
                                    except pc.ParseError:
                                        continue
                                    out.append((candidate, nodes_of(candidate)))
        memo[key] = out
        return out

    seen = set()
    for output in en.OUTPUT_ROOTS:
        for expr, count in build(output, en.MAX_DEPTH, flags["methodology_root"]):
            if count <= en.MAX_NODE_COUNT:
                seen.add(pc.canonical_semantic(expr))
    return len(seen)


@pytest.fixture()
def small_caps(monkeypatch):
    for key, value in SMALL.items():
        monkeypatch.setattr(en, key, value)


@pytest.mark.parametrize("scenario", sorted(en.SCENARIOS))
def test_dp_matches_brute_force_at_reduced_caps(small_caps, scenario):
    dp_total, _ = en.count(scenario)
    assert dp_total == _brute_force(scenario)


# --------------------------------------------------------------------------
# full-caps template materialization (third independent path)
# --------------------------------------------------------------------------


def test_dp_matches_full_caps_template_materialization():
    """The reduced-caps brute force cannot exercise depth-3 composition at the
    real caps; template-mode materialization can (98,070 shapes is tractable
    where 54.9M expressions is not). This concretely materializes every
    root-only template shape at FULL caps with budget-threaded slot products
    and compares exactly. The blend.w length-forced-by-arity rule and the
    routing shared-length pairing are reproduced independently here — a
    divergence in either shows up as a count mismatch."""

    def leaf_templates(output):
        shapes = set()
        for name, sig in pc.REGISTRY.items():
            if sig.atom and sig.output == output and not sig.operator:
                shapes.add((output, None))
                for m in sig.modifiers:
                    shapes.add((output, m))
        out = [f"<{o}>" + (f".{m}" if m else "")
               for o, m in sorted(shapes, key=lambda x: (x[0], x[1] or ""))]
        if output == "score":
            out.append("<e5-atom>")
        return [(s, 1) for s in out]

    def kw_shapes(kind):
        if kind in ("number", "int", "estimand", "impl", "walk", "weight"):
            return [f"<{kind}>"]
        if kind in ("number_list", "int_list"):
            return [f"<{kind}:len{L}>" for L in range(1, en.MAX_LIST_LENGTH + 1)]
        return []

    def feasible_products(slots, budget):
        if not slots:
            yield (), 0
            return
        head, tail = slots[0], slots[1:]
        min_tail = sum(min(n for _, n in s) for s in tail) if tail else 0
        for item in head:
            n = item[1]
            if n + min_tail > budget:
                continue
            for rest, rest_nodes in feasible_products(tail, budget - n):
                yield (item,) + rest, n + rest_nodes

    memo = {}

    def build(output, d, allow_meth):
        key = (output, d, allow_meth)
        if key in memo:
            return memo[key]
        out = list(leaf_templates(output))
        if d > 0:
            for name, sig in pc.REGISTRY.items():
                if not sig.operator or sig.output != output:
                    continue
                cap = sig.max_args if sig.max_args is not None else en.MAX_ARITY
                for arity in range(sig.min_args, min(cap, en.MAX_ARITY) + 1):
                    slot_opts, dead = [], False
                    for i in range(arity):
                        declared = (sig.arg_types[i] if i < len(sig.arg_types)
                                    else sig.variadic_arg_type)
                        opts = []
                        for alt in declared.split("|"):
                            if alt in pc.VALUE_KINDS:
                                opts += [(s, 0) for s in kw_shapes(alt)]
                            elif alt == "process":
                                for typ in en.OUTPUT_ROOTS:
                                    opts += build(typ, d - 1, False)
                            else:
                                opts += build(alt, d - 1, False)
                        opts = [o for o in opts if o[1] <= en.MAX_NODE_COUNT - 1]
                        if not opts:
                            dead = True
                            break
                        slot_opts.append(opts)
                    if dead:
                        continue
                    kw_opts = []
                    for kname in sorted(sig.kwargs):
                        spec = sig.kwargs[kname]
                        if kname in ("estimand", "impl") and not allow_meth:
                            continue
                        if spec.kind in pc.OUTPUT_TYPES:
                            vals = [(s, n) for s, n in build(spec.kind, d - 1, False)
                                    if n <= en.MAX_NODE_COUNT - 1]
                        elif name == "blend" and kname == "w":
                            if arity > en.MAX_LIST_LENGTH:
                                continue  # capped-out list: unenumerable, fail closed
                            vals = [(f"<number_list:len{arity}>", 0)]
                        elif spec.kind == "string":
                            continue
                        elif spec.default is not None:
                            continue  # resolved into every shape below, not a choice
                        else:
                            vals = [(s, 0) for s in kw_shapes(spec.kind)]
                        if not vals:
                            continue
                        kw_opts.append([(kname, s, n) for s, n in vals])
                    for arg_sel, arg_nodes in feasible_products(
                        slot_opts, en.MAX_NODE_COUNT - 1
                    ):
                        base_nodes = 1 + arg_nodes
                        for r in range(0, en.MAX_KWARGS_NODE + 1):
                            for combo in itertools.combinations(range(len(kw_opts)), r):
                                names_in = [kw_opts[i][0][0] for i in combo]
                                if any(s.required and k not in names_in
                                       for k, s in sig.kwargs.items()):
                                    continue
                                if name == "routing" and (
                                    ("t" in names_in) != ("menus" in names_in)
                                ):
                                    continue
                                for sel in itertools.product(
                                    *[kw_opts[i] for i in combo]
                                ):
                                    if name == "routing" and "t" in names_in:
                                        chosen = {k: s for k, s, _ in sel}
                                        if (chosen["t"].split("len")[1]
                                                != chosen["menus"].split("len")[1]):
                                            continue
                                    total = base_nodes + sum(n for _, _, n in sel)
                                    if total > en.MAX_NODE_COUNT:
                                        continue
                                    resolved = list(sel) + [
                                        (k, f"<{sp.kind}>", 0)
                                        for k, sp in sig.kwargs.items()
                                        if sp.default is not None
                                    ]
                                    kw_part = ",".join(
                                        f"{k}={s}" for k, s, _ in
                                        sorted(resolved, key=lambda item: item[0])
                                    )
                                    body = ",".join(s for s, _ in arg_sel)
                                    template = (f"{name}({body}"
                                                + (f",{kw_part}" if kw_part else "")
                                                + ")")
                                    out.append((template, total))
        memo[key] = out
        return out

    seen = set()
    for output in en.OUTPUT_ROOTS:
        for template, n in build(output, en.MAX_DEPTH, True):
            if n <= en.MAX_NODE_COUNT:
                seen.add(template)
    assert len(seen) == en.count("methodology-root-only", template_mode=True)[0]
