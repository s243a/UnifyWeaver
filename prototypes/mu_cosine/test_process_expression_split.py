#!/usr/bin/env python3
"""split-v2 tests: the normative worked example, the focused second vector,
and the fail-closed behaviors, each traced to a fourth-review finding.

Two golden vectors are pinned as FULL manifests (every field, plus the
manifest's canonical-bytes hash), so the §2.5 narration, the executable
algorithm, and the recorded bytes must agree — a divergence fails CI.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import process_expression_split as sp


def _fam(fid, counts, pairs, rows=2):
    return sp.Family.make(fid, counts, pairs, rows)


# --------------------------------------------------------------------------
# vector 1: the normative worked example (§2.5)
# --------------------------------------------------------------------------


def worked_example_families():
    """Twelve toy families. Per-item counts matter (finding 1): G has two
    rows but only ONE witnesses `op:max`, and H's `digit:8` likewise counts
    one row of two. The dev-held pair is carried by H and K; the test-held
    pair by L alone."""
    spec = {
        "A": ({"op:e5": 2, "digit:0": 2, "op:routing": 2}, ["pair:e5|margin"]),
        "B": ({"op:e5": 2, "op:margin": 2, "synthetic:string": 2}, ["pair:e5|margin"]),
        "C": ({"item:rare": 2, "op:max": 2}, ["pair:max|product"]),
        "D": ({"op:margin": 2, "digit:0": 2}, ["pair:e5|margin"]),
        "E": ({"op:blend": 2, "terminal:luna": 2}, ["pair:blend|judge"]),
        "F": ({"op:blend": 2, "digit:8": 2}, ["pair:blend|judge"]),
        "G": ({"item:rare": 2, "op:max": 1}, ["pair:max|product"]),
        "H": ({"terminal:luna": 2, "digit:8": 1}, ["pair:kalman|judge"]),
        "I": ({"op:lineage": 2, "synthetic:pin": 2}, ["pair:lineage|substrate"]),
        "J": ({"op:lineage": 2, "estimand:ancestry": 2}, ["pair:lineage|judge"]),
        "K": ({"synthetic:pin": 2, "estimand:ancestry": 2}, ["pair:kalman|judge"]),
        "L": ({"op:routing": 2, "synthetic:string": 2}, ["pair:routing|score"]),
    }
    return [_fam(f"tmpl:{k}", v[0], v[1]) for k, v in sorted(spec.items())]


WORKED_UNIVERSE = sorted(
    {item for f in worked_example_families() for item in f.witness_items}
)

WORKED_CONTRACT = dict(
    seed="worked-v2-7",
    coverage_minimum_k=2,
    required_universe_sha256=sp.universe_sha256(WORKED_UNIVERSE),
    held_compositions={"dev": ["pair:kalman|judge"],
                       "test": ["pair:routing|score"]},
    buckets={"train": 5000, "dev": 2500, "test": 2500},
    authorizing=False,   # toy families: a feasibility probe, not a preregistration
)

#: The normative golden vector: §2.5 narrates exactly this, in full.
WORKED_GOLDEN = {
    "split_contract_sha256":
        "85cac58c07a620bee72c4c3b95364ea1639abc2174e2c25ad77567eff7a89fcd",
    "slices": {
        "train": ["tmpl:A", "tmpl:B", "tmpl:C", "tmpl:E", "tmpl:F",
                  "tmpl:G", "tmpl:I", "tmpl:J"],
        "dev": ["tmpl:H", "tmpl:K"],
        "test": ["tmpl:D", "tmpl:L"],
    },
    "held": {
        "dev": {"pair:kalman|judge": ["tmpl:H", "tmpl:K"]},
        "test": {"pair:routing|score": ["tmpl:L"]},
    },
    "moves": [
        {"family": "tmpl:A", "from": "dev", "forced_by": "digit:0"},
        {"family": "tmpl:F", "from": "test", "forced_by": "digit:8"},
        {"family": "tmpl:J", "from": "dev", "forced_by": "estimand:ancestry"},
        {"family": "tmpl:C", "from": "test", "forced_by": "op:max"},
    ],
    "far": {"dev": ["tmpl:H", "tmpl:K"], "test": ["tmpl:L"]},
    "train_coverage": {
        "digit:0": 2, "digit:8": 2, "estimand:ancestry": 2, "item:rare": 4,
        "op:blend": 4, "op:e5": 4, "op:lineage": 4, "op:margin": 2,
        "op:max": 3, "op:routing": 2, "synthetic:pin": 2,
        "synthetic:string": 2, "terminal:luna": 2,
    },
    "manifest_sha256":
        "8012e7ff5ac3adbe55cbdc574c55289fb8df5b5873fd63256215597fe3c8160a",
}


@pytest.fixture(scope="module")
def worked():
    contract = sp.split_contract(**WORKED_CONTRACT)
    return sp.assign(worked_example_families(), contract, WORKED_UNIVERSE)


def test_worked_example_is_the_normative_golden_vector_in_full(worked):
    """FULL manifest equality (fourth review: four fields was not a golden).
    The contract is compared against the constructor's own output, and the
    manifest's canonical-bytes hash is pinned alongside every field."""
    assert worked["contract"] == sp.split_contract(**WORKED_CONTRACT)
    for key, expected in WORKED_GOLDEN.items():
        assert worked[key] == expected, key


def test_worked_example_demonstrates_the_reviewed_behaviors(worked):
    # Whole families only: every family id sits in exactly one slice.
    all_ids = [f for ids in worked["slices"].values() for f in ids]
    assert sorted(all_ids) == sorted(set(all_ids))
    # Per-item row counts (finding 1): G is IN TRAIN yet witnesses op:max
    # from only one of its two rows, so k=2 still forced pulling C out of
    # test. Under split-v1's row_count crediting, G alone would have counted
    # 2 and this move would not exist.
    assert "tmpl:G" in worked["slices"]["train"]
    assert {"family": "tmpl:C", "from": "test", "forced_by": "op:max"} in worked["moves"]
    assert worked["train_coverage"]["op:max"] == 3   # C(2) + G(1), not 4
    # Held pins MOVED families (finding 3): H, K, and L were all
    # base-assigned to train and pinned out to their held pair's slice.
    contract = sp.split_contract(**WORKED_CONTRACT)
    for family_id in ("tmpl:H", "tmpl:K", "tmpl:L"):
        assert sp._base_slice(family_id, contract) == "train"
    # Held protection: digit:8's carriers are F and the pinned H; repair
    # took F and never touched a held group.
    assert not any(m["family"] in ("tmpl:H", "tmpl:K", "tmpl:L")
                   for m in worked["moves"])
    # Near vs far: the held pairs live only on their own side, so their
    # carriers are far; D stays test-near because pair:e5|margin is in train.
    assert "tmpl:D" in worked["slices"]["test"]
    assert "tmpl:D" not in worked["far"]["test"]


# --------------------------------------------------------------------------
# vector 2: the focused boundary/cascade vector (fourth review's request)
# --------------------------------------------------------------------------


def cascade_families():
    """Eight families exercising repeated repair for one item, a cross-item
    cascade, mixed held motifs, and the near/far distinction — compact by
    design rather than large for scale."""
    return [
        _fam("tmpl:T", {"x": 1, "y": 1, "z": 3, "h1": 3, "h2": 3,
                        "d": 3, "t": 3}, ["pair:core|core"], 3),
        _fam("tmpl:U", {"x": 1, "y": 1}, ["pair:core|core"], 1),
        _fam("tmpl:V", {"x": 1, "y": 1}, ["pair:core|core"], 1),
        _fam("tmpl:HD", {"h1": 2}, ["pair:held|dev"], 2),
        _fam("tmpl:HT", {"h2": 2}, ["pair:held|test"], 2),
        _fam("tmpl:DN", {"d": 2}, ["pair:seen|dev"], 2),
        _fam("tmpl:TN", {"t": 2}, ["pair:seen|dev"], 2),
        _fam("tmpl:TF", {"t": 2}, ["pair:unseen|anywhere"], 2),
    ]


CASCADE_UNIVERSE = sorted(
    {item for f in cascade_families() for item in f.witness_items}
)

CASCADE_CONTRACT = dict(
    seed="cascade-v2-1328",
    coverage_minimum_k=3,
    required_universe_sha256=sp.universe_sha256(CASCADE_UNIVERSE),
    held_compositions={"dev": ["pair:held|dev"], "test": ["pair:held|test"]},
    buckets={"train": 5000, "dev": 2500, "test": 2500},
    authorizing=False,
)

CASCADE_GOLDEN = {
    "split_contract_sha256":
        "aa97fa5dcc09437d61d3eda352dd36befe9d475aceaa599eb2fc6a9ce9b75930",
    "slices": {
        "train": ["tmpl:T", "tmpl:U", "tmpl:V"],
        "dev": ["tmpl:DN", "tmpl:HD"],
        "test": ["tmpl:HT", "tmpl:TF", "tmpl:TN"],
    },
    "held": {
        "dev": {"pair:held|dev": ["tmpl:HD"]},
        "test": {"pair:held|test": ["tmpl:HT"]},
    },
    "moves": [
        {"family": "tmpl:U", "from": "dev", "forced_by": "x"},
        {"family": "tmpl:V", "from": "test", "forced_by": "x"},
    ],
    "far": {"dev": ["tmpl:DN", "tmpl:HD"], "test": ["tmpl:HT", "tmpl:TF"]},
    "train_coverage": {"d": 3, "h1": 3, "h2": 3, "t": 3, "x": 3, "y": 3, "z": 3},
    "manifest_sha256":
        "bffb9af079e1257b7c8287b71c126c8b0749c620af69e10fe3412ec532a7ab70",
}


@pytest.fixture(scope="module")
def cascade():
    contract = sp.split_contract(**CASCADE_CONTRACT)
    return sp.assign(cascade_families(), contract, CASCADE_UNIVERSE)


def test_cascade_vector_full_manifest(cascade):
    assert cascade["contract"] == sp.split_contract(**CASCADE_CONTRACT)
    for key, expected in CASCADE_GOLDEN.items():
        assert cascade[key] == expected, key


def test_cascade_vector_focused_behaviors(cascade):
    # Repeated moves for ONE item: x totals exactly k=3 across three
    # one-row families, two of which started outside train, so x forced two
    # moves — and dev is searched before test, so U (dev) precedes V (test).
    assert [m for m in cascade["moves"] if m["forced_by"] == "x"] == [
        {"family": "tmpl:U", "from": "dev", "forced_by": "x"},
        {"family": "tmpl:V", "from": "test", "forced_by": "x"},
    ]
    # Cross-item cascade: y also needed 3 and was satisfied entirely by the
    # moves x forced (T, U, and V each carry y), so no move names y.
    assert not any(m["forced_by"] == "y" for m in cascade["moves"])
    # Mixed held motifs, pinning in both directions: HD was base-assigned to
    # TEST and pinned to dev; HT was base-assigned to TRAIN and pinned to test.
    contract = sp.split_contract(**CASCADE_CONTRACT)
    assert sp._base_slice("tmpl:HD", contract) == "test"
    assert sp._base_slice("tmpl:HT", contract) == "train"
    # Dev-observed means test-near: TN's only pair is carried by DN in dev,
    # so TN is near; TF's pair appears nowhere else, so TF is far.
    assert "tmpl:TN" not in cascade["far"]["test"]
    assert "tmpl:TF" in cascade["far"]["test"]
    # Dev-far is judged against train ALONE: DN's pair lives only in dev.
    assert "tmpl:DN" in cascade["far"]["dev"]


def test_exact_boundary_buckets():
    """Integer bucket boundaries at the exact edges (finding 4: round() over
    fractions left the dev/test boundary underspecified). Each family id was
    found by scan and hashes to the stated bucket under this seed."""
    contract = sp.split_contract(
        seed="boundary-vector", coverage_minimum_k=1,
        required_universe_sha256=sp.universe_sha256([]),
        held_compositions={"dev": ["p"], "test": ["q"]},
        buckets={"train": 5000, "dev": 2500, "test": 2500},
        authorizing=False,
    )
    edges = {
        "tmpl:bnd-3085": (4999, "train"),    # last train bucket
        "tmpl:bnd-7460": (5000, "dev"),      # first dev bucket
        "tmpl:bnd-17084": (7499, "dev"),     # last dev bucket
        "tmpl:bnd-7736": (7500, "test"),     # first test bucket
    }
    for family_id, (bucket, side) in edges.items():
        digest = hashlib.sha256(
            family_id.encode() + b"|" + b"boundary-vector").hexdigest()
        assert int(digest, 16) % sp.MODULUS == bucket
        assert sp._base_slice(family_id, contract) == side


# --------------------------------------------------------------------------
# determinism, fail-closed behaviors, and binding
# --------------------------------------------------------------------------


def test_assignment_is_deterministic_and_order_independent():
    contract = sp.split_contract(**WORKED_CONTRACT)
    forward = sp.assign(worked_example_families(), contract, WORKED_UNIVERSE)
    backward = sp.assign(
        list(reversed(worked_example_families())), contract,
        list(reversed(WORKED_UNIVERSE)),
    )
    assert forward == backward


def test_universe_is_hash_bound_not_caller_shaped():
    """Finding 2: a universe that does not hash to the contract's binding is
    refused, so an extractor's omission cannot vanish silently."""
    contract = sp.split_contract(**WORKED_CONTRACT)
    shrunk = [item for item in WORKED_UNIVERSE if item != "synthetic:pin"]
    with pytest.raises(sp.SplitError, match="required_universe_sha256"):
        sp.assign(worked_example_families(), contract, shrunk)


def test_required_item_missing_from_corpus_fails_closed():
    """An item the universe requires but no family witnesses is a corpus
    bug, not something a split may repair."""
    universe = WORKED_UNIVERSE + ["item:never-built"]
    contract = dict(WORKED_CONTRACT,
                    required_universe_sha256=sp.universe_sha256(universe))
    with pytest.raises(sp.SplitError, match="item:never-built"):
        sp.assign(worked_example_families(),
                  sp.split_contract(**contract), universe)


def test_repair_never_consumes_held_pinned_families():
    """A scarce item whose only carrier is held-pinned cannot be repaired;
    the split refuses rather than consuming a held group."""
    families = [
        _fam("tmpl:AA", {"common": 2}, ["p"]),
        _fam("tmpl:AB", {"common": 2}, ["p"]),
        _fam("tmpl:AC", {"common": 2, "scarce": 2}, ["q"]),
    ]
    universe = sorted({i for f in families for i in f.witness_items})
    contract = sp.split_contract(
        seed="held-trap", coverage_minimum_k=2,
        required_universe_sha256=sp.universe_sha256(universe),
        held_compositions={"dev": ["q"], "test": ["p"]},
        buckets={"train": 5000, "dev": 2500, "test": 2500},
        authorizing=False,
    )
    with pytest.raises(sp.SplitError, match="held-pinned"):
        sp.assign(families, contract, universe)


def test_conflicting_held_pairs_on_one_family_fail_closed():
    families = [_fam("tmpl:AA", {"a": 2}, ["p", "q"]),
                _fam("tmpl:AB", {"a": 2}, ["r"])]
    universe = ["a"]
    contract = sp.split_contract(
        seed="conflict", coverage_minimum_k=1,
        required_universe_sha256=sp.universe_sha256(universe),
        held_compositions={"dev": ["p"], "test": ["q"]},
        authorizing=False,
    )
    with pytest.raises(sp.SplitError, match="both a dev-held and"):
        sp.assign(families, contract, universe)


def test_held_pair_with_no_carrier_fails_closed():
    families = [_fam("tmpl:AA", {"a": 2}, ["p"]),
                _fam("tmpl:AB", {"a": 2}, ["q"])]
    universe = ["a"]
    contract = sp.split_contract(
        seed="ghost", coverage_minimum_k=1,
        required_universe_sha256=sp.universe_sha256(universe),
        held_compositions={"dev": ["p"], "test": ["pair:not|present"]},
        authorizing=False,
    )
    with pytest.raises(sp.SplitError, match="no carrier"):
        sp.assign(families, contract, universe)


def test_far_floor_fails_closed_when_unmet():
    """Finding 3: a far slice thinner than the contract requires is a
    failure, not a quiet success. Held pinning makes one test family far;
    a contract asking for three refuses."""
    families = [_fam("tmpl:AA", {"a": 2}, ["p"]),
                _fam("tmpl:AB", {"a": 2}, ["q"]),
                _fam("tmpl:AC", {"a": 2}, ["q"]),
                _fam("tmpl:AD", {"a": 2}, ["r"])]
    universe = ["a"]
    contract = sp.split_contract(
        seed="thin-far", coverage_minimum_k=1,
        required_universe_sha256=sp.universe_sha256(universe),
        held_compositions={"dev": ["p"], "test": ["q"]},
        far_floors={"dev": 1, "test": 3},
        authorizing=False,
    )
    with pytest.raises(sp.SplitError, match="far floor unmet"):
        sp.assign(families, contract, universe)


def test_contract_validation_fails_closed():
    """Fifth review, finding 5: several malformed values reached the
    algorithm. Every probe here bypassed validation before this round —
    note especially the bools, since `bool` IS an `int` in Python."""
    usha = sp.universe_sha256([])
    held = {"dev": ["p"], "test": ["q"]}

    def probe(**kwargs):
        base = dict(seed="s", coverage_minimum_k=1,
                    required_universe_sha256=usha, held_compositions=held,
                    authorizing=False)
        return sp.split_contract(**{**base, **kwargs})

    for label, kwargs in [
        ("k not positive",       dict(coverage_minimum_k=0)),
        ("k is a bool",          dict(coverage_minimum_k=True)),
        ("empty held side",      dict(held_compositions={"dev": [], "test": ["q"]})),
        ("pair held both sides", dict(held_compositions={"dev": ["p"], "test": ["p"]})),
        ("held pair not a str",  dict(held_compositions={"dev": [7], "test": ["q"]})),
        ("held side not a list", dict(held_compositions={"dev": "p", "test": ["q"]})),
        ("buckets miss modulus", dict(buckets={"train": 9000, "dev": 900, "test": 99})),
        ("bucket is a float",    dict(buckets={"train": 8000.0, "dev": 1000, "test": 1000})),
        ("bucket is a bool",     dict(buckets={"train": True, "dev": 4999, "test": 5000})),
        ("sha not a digest",     dict(required_universe_sha256="not-a-sha")),
        ("sha not hex",          dict(required_universe_sha256="z" * 64)),
        ("far floor zero",       dict(far_floors={"dev": 0, "test": 1})),
        ("far floor a float",    dict(far_floors={"dev": 1.5, "test": 1})),
        ("far floor key missing", dict(far_floors={"dev": 1})),
        ("seed not a string",    dict(seed=None)),
        ("seed empty",           dict(seed="")),
        ("authorizing not bool", dict(authorizing="yes")),
    ]:
        with pytest.raises(sp.SplitError):
            probe(**kwargs)
        assert True, label


# --------------------------------------------------------------------------
# finding 1: the universe binding must be AUTHORITATIVE, not self-consistent
# --------------------------------------------------------------------------


def test_authorizing_contract_must_bind_the_authoritative_universe():
    """The fifth review's central finding: the caller supplied BOTH the
    universe and its hash, so a one-item universe validated against its own
    hash and the binding proved nothing."""
    import process_expression_enumerator as en

    held = {"dev": ["p"], "test": ["q"]}
    with pytest.raises(sp.SplitError, match="AUTHORIZING"):
        sp.split_contract("s", 1, sp.universe_sha256(["op:e5"]), held)
    # The authoritative hash is accepted, and it is the one the enumerator
    # derives from the support rather than from any corpus.
    contract = sp.split_contract(
        "s", 1, en.required_witness_universe_sha256(), held)
    assert contract["authorizing"] is True
    assert contract["required_universe_sha256"] == (
        sp.universe_sha256(en.required_witness_universe()))


def test_a_mutated_contract_is_refused_at_every_use_site():
    """Sixth review, principal finding: `split_contract` returns a plain
    dict, so a validated contract could be mutated before use and both
    `assign` and the witness accepted it — reopening the one-item-universe
    exploit in full. Validation at construction proves nothing about a
    value used later, so every use site re-derives and compares."""
    import process_expression_enumerator as en

    held = {"dev": ["p"], "test": ["q"]}
    families = [_fam(f"tmpl:{fid}", {"op:e5": 2}, [pair], 2)
                for fid, pair in (("W", "r"), ("X", "r"), ("Y", "p"), ("Z", "q"))]

    # (a) swap the bound universe for a one-item one, post-validation
    tampered = sp.split_contract("s", 1, en.required_witness_universe_sha256(), held)
    tampered["required_universe_sha256"] = sp.universe_sha256(["op:e5"])
    with pytest.raises(sp.SplitError):
        sp.assign(families, tampered, ["op:e5"])

    # (b) promote a probe to authorizing, post-validation
    promoted = sp.split_contract("s", 1, sp.universe_sha256(["x"]), held,
                                 authorizing=False)
    promoted["authorizing"] = True
    with pytest.raises(sp.SplitError):
        sp.preregistration_witness_sha256(promoted)

    # (c) any structural mutation, not merely the two known ones
    bucketed = sp.split_contract("s", 1, en.required_witness_universe_sha256(), held)
    bucketed["buckets"]["train"] = 9999
    with pytest.raises(sp.SplitError):
        sp.assign(families, bucketed, en.required_witness_universe())

    # A valid-but-different mutation (k) re-derives cleanly — it is caught
    # instead by the preregistration witness, which moves with it.
    original = sp.split_contract("s", 2, en.required_witness_universe_sha256(), held)
    changed = sp.split_contract("s", 3, en.required_witness_universe_sha256(), held)
    assert sp.revalidate(changed) == changed
    assert (sp.preregistration_witness_sha256(original)
            != sp.preregistration_witness_sha256(changed))


def test_probe_contracts_are_allowed_but_cannot_be_witnessed():
    """Warning-free small universes remain possible for feasibility work —
    but the flag is inside the contract, so it is inside the contract hash,
    and a probe can never be mistaken for a preregistration."""
    held = {"dev": ["p"], "test": ["q"]}
    probe = sp.split_contract("s", 1, sp.universe_sha256(["x"]), held,
                              authorizing=False)
    assert probe["authorizing"] is False
    with pytest.raises(sp.SplitError, match="non-authorizing"):
        sp.preregistration_witness_sha256(probe)
    # Authorizing and probe contracts never share a hash.
    import process_expression_enumerator as en
    real = sp.split_contract("s", 1, en.required_witness_universe_sha256(), held)
    assert sp.split_contract_sha256(real) != sp.split_contract_sha256(probe)


def test_family_make_validates_counts():
    with pytest.raises(sp.SplitError, match="row count"):
        sp.Family.make("tmpl:bad", {"a": 3}, [], 2)   # more carriers than rows
    with pytest.raises(sp.SplitError, match="row count"):
        sp.Family.make("tmpl:bad", {"a": 0}, [], 2)   # an item nothing witnesses


def test_preregistration_witness_binds_both_halves():
    """Witnessing requires an AUTHORIZING contract, so this uses the real
    authoritative universe rather than the toy one the vectors probe with."""
    import process_expression_enumerator as en

    base = dict(WORKED_CONTRACT,
                required_universe_sha256=en.required_witness_universe_sha256(),
                authorizing=True)
    contract_a = sp.split_contract(**base)
    variants = [
        sp.split_contract(**dict(base, coverage_minimum_k=3)),
        sp.split_contract(**dict(base, seed="other-seed")),
        sp.split_contract(**dict(base, held_compositions={
            "dev": ["pair:blend|judge"], "test": ["pair:routing|score"]})),
        sp.split_contract(**dict(base, buckets={
            "train": 6000, "dev": 2000, "test": 2000})),
    ]
    witness = sp.preregistration_witness_sha256(contract_a)
    assert len(witness) == 64
    for variant in variants:
        assert witness != sp.preregistration_witness_sha256(variant)


def _flat(text: str) -> str:
    """Collapse whitespace so a claim that wraps across lines still matches.
    Prose wraps legitimately; a line break must not defeat the guard."""
    return " ".join(text.split())


def _once(spec: str, needle: str) -> None:
    """The narrated claim must appear EXACTLY once.

    An external verification pass showed that `needle in spec` tolerates
    outright contradiction: appending a paragraph stating the opposite
    slices left the guard green, because the correct string still existed
    somewhere. Presence is not the same as the prose saying it.
    """
    count = _flat(spec).count(_flat(needle))
    assert count == 1, f"expected exactly one occurrence of {needle!r}, found {count}"


def test_specification_narrates_the_pinned_vectors(worked, cascade):
    """Every normative §2.5 claim is projected from the manifest and read
    back out of the prose — not just the final slices.

    An external verification pass mutation-tested the previous guard and
    found it covered only the final-slice sentences: corrupting the step-1
    base assignment, the repair moves, the coverage arithmetic, the far
    membership, or the cascade's cross-item claim all left CI green. Those
    are exactly the sentences that teach a reader WHY the slices are what
    they are, and they were the ones free to rot. The code side was never
    at risk — the full-manifest goldens catch any behavior change — so the
    hole was prose-only drift, which is precisely what this guard exists
    for.
    """
    spec = (ROOT / "DESIGN_process_expression_generator.md").read_text(
        encoding="utf-8")
    contract = sp.split_contract(**WORKED_CONTRACT)

    # -- contract surface --------------------------------------------------
    for needle in (WORKED_CONTRACT["seed"], CASCADE_CONTRACT["seed"],
                   "buckets 5000/2500/2500",
                   "pair:kalman|judge", "pair:routing|score"):
        assert needle in spec, needle

    # -- step 1: the base assignment, re-derived, not restated -------------
    base = {}
    for family in worked_example_families():
        base.setdefault(sp._base_slice(family.family_id, contract), []).append(
            family.family_id.removeprefix("tmpl:"))
    for side, phrase in (("train", "`{%s}` in train"),
                         ("dev", "`{%s}` in dev"),
                         ("test", "and `{%s}` in test")):
        _once(spec, phrase % ",".join(sorted(base[side])))

    # -- steps 3-4: every repair move, and the coverage arithmetic ---------
    for move in worked["moves"]:
        family = move["family"].removeprefix("tmpl:")
        _once(spec, f"moves `{family}` from {move['from']}")
    _once(spec, "3 = C(2) + G(1)")
    assert worked["train_coverage"]["op:max"] == 3

    # -- step 5: far membership, both sides --------------------------------
    for side in ("dev", "test"):
        narrated = "`{%s}` %s-far" % (
            ",".join(f.removeprefix("tmpl:") for f in worked["far"][side]), side)
        _once(spec, narrated)
    assert "tmpl:D" not in worked["far"]["test"]
    near = sorted(set(worked["slices"]["test"]) - set(worked["far"]["test"]))
    for family_id in near:
        _once(spec, f"`{family_id.removeprefix('tmpl:')}` shares "
                    "`pair:e5|margin` with train and is test-near")

    # -- final slices: every slice-shaped claim in the file must be right --
    # Uniqueness of the CORRECT string does not stop a contradicting
    # restatement elsewhere (external verification, finding 2), so instead
    # of asking "is the right sentence present", collect every claim of
    # this SHAPE anywhere in the spec and require each to match the
    # manifest. A stale or contradicting restatement is caught by being a
    # match that disagrees.
    import re

    for side in ("train", "dev", "test"):
        expected = "{" + ",".join(
            f.removeprefix("tmpl:") for f in worked["slices"][side]) + "}"
        claims = re.findall(rf"{side} `(\{{[A-Z,]+\}})`", _flat(spec))
        assert claims, f"no {side} slice claim found in the spec"
        for claim in claims:
            assert claim == expected, (
                f"spec states {side} `{claim}` but the manifest says "
                f"`{expected}`")

    # -- the cascade vector's claims ---------------------------------------
    _once(spec, "two moves for one item")
    assert len([m for m in cascade["moves"] if m["forced_by"] == "x"]) == 2
    _once(spec, "no move names `y`")
    assert not any(m["forced_by"] == "y" for m in cascade["moves"])
    assert "4999/5000/7499/7500" in spec
    assert len(cascade["moves"]) == 2


def test_specification_records_the_vocabulary_counts():
    """The §2.5 vocabulary table is measured content, so it is checked
    against the measurement rather than transcribed once and trusted."""
    import process_expression_enumerator as en

    spec = (ROOT / "DESIGN_process_expression_generator.md").read_text(
        encoding="utf-8")
    counts = en.component_vocabulary_counts()
    # Both totals are named for what they sum, and both are auditable: the
    # five class rows add to serialized_identities_total, and the component
    # subtotal is leaves + operator shapes.
    assert (counts["leaf_shapes"] + counts["operator_shapes_interior"]
            + counts["operator_shapes_root_only_extension"]
            + counts["node_composition_edges"] + counts["literal_slots"]
            == counts["serialized_identities_total"])
    assert (counts["leaf_shapes"] + counts["operator_shapes_interior"]
            + counts["operator_shapes_root_only_extension"]
            == counts["composable_component_shapes"])
    for label, key in (
        ("leaf shapes", "leaf_shapes"),
        ("operator shapes, interior", "operator_shapes_interior"),
        ("operator shapes, root-only extension",
         "operator_shapes_root_only_extension"),
        ("node-composition edges", "node_composition_edges"),
        ("literal slots", "literal_slots"),
        ("composable component shapes", "composable_component_shapes"),
        ("serialized identities, total", "serialized_identities_total"),
    ):
        _once(spec, f"| {label} | {counts[key]} |")
    _once(spec, f"{len(en.required_witness_universe())} required witness items")


def test_spec_sha_binds_the_whole_algorithm_manifest():
    """Finding 4: the enumeration spec binds the complete machine-readable
    manifest, so changing any rule's text moves the spec hash."""
    import process_expression_enumerator as en
    sha = en.enumeration_spec_sha256()
    original = sp.SPLIT_ALGORITHM_MANIFEST["repair_rule"]
    try:
        sp.SPLIT_ALGORITHM_MANIFEST["repair_rule"] = original + " (changed)"
        assert en.enumeration_spec_sha256() != sha
    finally:
        sp.SPLIT_ALGORITHM_MANIFEST["repair_rule"] = original
