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
)

#: The normative golden vector: §2.5 narrates exactly this, in full.
WORKED_GOLDEN = {
    "split_contract_sha256":
        "ea9d3bddac2b0a1f2be9985e30b9585beaf33ec5fd280b77fc6df88bc79af0cc",
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
        "3cae8ee8dae5d23e06a3e09dcb107270819fb8f26fdebf2b5ae6c328372d52f5",
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
)

CASCADE_GOLDEN = {
    "split_contract_sha256":
        "d015687fa0a334d03fd5b6b8f6799b83951d1f5a18ad02d68bb5e6e1db9f359d",
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
        "164e015904fae7d4ce24ddcdd11b32b38306e49de9fa9600be9fe1dd1d968ca8",
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
    )
    with pytest.raises(sp.SplitError, match="far floor unmet"):
        sp.assign(families, contract, universe)


def test_contract_validation_fails_closed():
    usha = sp.universe_sha256([])
    held = {"dev": ["p"], "test": ["q"]}
    with pytest.raises(sp.SplitError):
        sp.split_contract("s", 0, usha, held)                     # k positive
    with pytest.raises(sp.SplitError):
        sp.split_contract("s", 1, usha, {"dev": [], "test": ["q"]})   # empty side
    with pytest.raises(sp.SplitError):
        sp.split_contract("s", 1, usha, {"dev": ["p"], "test": ["p"]})  # both sides
    with pytest.raises(sp.SplitError):
        sp.split_contract("s", 1, usha, held,
                          buckets={"train": 9000, "dev": 900, "test": 99})
    with pytest.raises(sp.SplitError):
        sp.split_contract("s", 1, usha, held,
                          buckets={"train": 8000.0, "dev": 1000, "test": 1000})
    with pytest.raises(sp.SplitError):
        sp.split_contract("s", 1, "not-a-sha", held)
    with pytest.raises(sp.SplitError):
        sp.split_contract("s", 1, usha, held, far_floors={"dev": 0, "test": 1})


def test_family_make_validates_counts():
    with pytest.raises(sp.SplitError, match="row count"):
        sp.Family.make("tmpl:bad", {"a": 3}, [], 2)   # more carriers than rows
    with pytest.raises(sp.SplitError, match="row count"):
        sp.Family.make("tmpl:bad", {"a": 0}, [], 2)   # an item nothing witnesses


def test_preregistration_witness_binds_both_halves():
    base = dict(WORKED_CONTRACT)
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


def test_specification_narrates_the_pinned_vectors(worked, cascade):
    """The §2.5 narration and the executable algorithm must not drift apart:
    the seeds, the contract shape, and the narrated final slices are read
    back out of the spec text and checked against the manifests."""
    spec = (ROOT / "DESIGN_process_expression_generator.md").read_text(
        encoding="utf-8")
    assert WORKED_CONTRACT["seed"] in spec
    assert CASCADE_CONTRACT["seed"] in spec
    assert "buckets 5000/2500/2500" in spec
    for pair in ("pair:kalman|judge", "pair:routing|score"):
        assert pair in spec
    narrated = "train `{" + ",".join(
        f.removeprefix("tmpl:") for f in worked["slices"]["train"]) + "}`"
    assert narrated in spec, narrated
    for side in ("dev", "test"):
        narrated = side + " `{" + ",".join(
            f.removeprefix("tmpl:") for f in worked["slices"][side]) + "}`"
        assert narrated in spec, narrated
    # The cascade vector's headline behaviors are named in the prose.
    assert "two moves for one item" in spec
    assert "4999/5000/7499/7500" in spec
    assert len(cascade["moves"]) == 2


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
