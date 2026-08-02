#!/usr/bin/env python3
"""Family-level split algorithm tests, including the normative worked example.

The §2.5 worked example is not prose: the exact manifest narrated in the spec
is pinned here against the executable algorithm, so the example is a golden
vector — a divergence between spec text and algorithm behavior fails CI.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import process_expression_split as sp


def _fam(fid, items, pairs, rows=2):
    return sp.Family(fid, frozenset(items), frozenset(pairs), rows)


def worked_example_families():
    """Twelve toy families; every witness item has exactly two carrier
    families, so no item is hostage to a single held family. `item:rare` is
    carried by C and G only; family L carries the only `pair:routing|score`."""
    spec = {
        "A": (["op:e5", "digit:0", "op:routing"], ["pair:e5|margin"]),
        "B": (["op:e5", "op:margin", "synthetic:string"], ["pair:e5|margin"]),
        "C": (["item:rare", "op:max"], ["pair:max|product"]),
        "D": (["op:margin", "digit:0"], ["pair:e5|margin"]),
        "E": (["op:blend", "terminal:luna"], ["pair:blend|judge"]),
        "F": (["op:blend", "digit:8"], ["pair:blend|judge"]),
        "G": (["item:rare", "op:max"], ["pair:max|product"]),
        "H": (["terminal:luna", "digit:8"], ["pair:kalman|judge"]),
        "I": (["op:lineage", "synthetic:pin"], ["pair:lineage|substrate"]),
        "J": (["op:lineage", "estimand:ancestry"], ["pair:lineage|judge"]),
        "K": (["synthetic:pin", "estimand:ancestry"], ["pair:kalman|judge"]),
        "L": (["op:routing", "synthetic:string"], ["pair:routing|score"]),
    }
    return [_fam(f"tmpl:{k}", v[0], v[1]) for k, v in sorted(spec.items())]


WORKED_CONTRACT = dict(
    seed="worked-example-v5",
    coverage_minimum_k=2,
    fractions={"train": 0.5, "dev": 0.25, "test": 0.25},
)

#: The normative golden vector: the §2.5 worked example narrates exactly this.
WORKED_GOLDEN = {
    "slices": {
        "train": ["tmpl:A", "tmpl:D", "tmpl:E", "tmpl:G", "tmpl:H",
                  "tmpl:I", "tmpl:K", "tmpl:L"],
        "dev": ["tmpl:C"],
        "test": ["tmpl:B", "tmpl:F", "tmpl:J"],
    },
    "held": {"dev": ["tmpl:C"], "test": ["tmpl:B"]},
    "moves": [
        {"family": "tmpl:H", "from": "dev", "forced_by": "digit:8"},
        {"family": "tmpl:G", "from": "test", "forced_by": "item:rare"},
        {"family": "tmpl:I", "from": "dev", "forced_by": "op:lineage"},
    ],
    "far": {"dev": [], "test": ["tmpl:J"]},
}


@pytest.fixture(scope="module")
def worked():
    contract = sp.split_contract(**WORKED_CONTRACT)
    return sp.assign(worked_example_families(), contract)


def test_worked_example_is_the_normative_golden_vector(worked):
    for key in WORKED_GOLDEN:
        assert worked[key] == WORKED_GOLDEN[key], key


def test_worked_example_demonstrates_all_required_behaviors(worked):
    """The five behaviors the third adversarial review required the example
    to cover, each read off the golden manifest rather than asserted."""
    # Grouping: whole families only — every family id sits in exactly one slice.
    all_ids = [f for ids in worked["slices"].values() for f in ids]
    assert sorted(all_ids) == sorted(set(all_ids))
    # Repair to k: after repair, every witness item reaches k=2 in train.
    assert all(v >= 2 for v in worked["train_coverage"].values())
    # Dev-preferred repair with test fallback: digit:8's carriers were F (test)
    # and H (dev); repair took H from dev first. item:rare's only movable
    # carrier was in test; repair fell back to test for G.
    assert {"family": "tmpl:H", "from": "dev", "forced_by": "digit:8"} in worked["moves"]
    assert {"family": "tmpl:G", "from": "test", "forced_by": "item:rare"} in worked["moves"]
    # Held-family protection: tmpl:C carries item:rare and is dev-held; repair
    # took the OTHER carrier and left C exactly where the base assignment and
    # holding rule put it.
    assert "tmpl:C" in worked["held"]["dev"]
    assert "tmpl:C" in worked["slices"]["dev"]
    assert not any(m["family"] == "tmpl:C" for m in worked["moves"])
    # Mixed motifs / far: tmpl:J's pair:lineage|judge appears nowhere in
    # train or dev, so J is far; tmpl:B and tmpl:F share every pair with
    # train and are near.
    assert worked["far"]["test"] == ["tmpl:J"]
    # Floors survived repair: dev kept its held family, >= floor 1.
    assert len(worked["slices"]["dev"]) >= 1


def test_assignment_is_deterministic_and_order_independent():
    contract = sp.split_contract(**WORKED_CONTRACT)
    forward = sp.assign(worked_example_families(), contract)
    backward = sp.assign(list(reversed(worked_example_families())), contract)
    assert forward == backward


def test_repair_fails_closed_when_only_held_families_carry_an_item():
    """A scarce item whose every carrier is held cannot be repaired; the
    split refuses rather than consuming a held family."""
    families = [
        _fam("tmpl:AA", ["common"], ["p1"]),
        _fam("tmpl:AB", ["common"], ["p1"]),
        _fam("tmpl:AC", ["common", "scarce"], ["p2"]),
    ]
    # Find a seed placing AC (the only scarce carrier) into a held slot.
    for i in range(200):
        contract = sp.split_contract(
            f"held-trap-{i}", 1, fractions={"train": 0.34, "dev": 0.33, "test": 0.33})
        try:
            sp.assign(families, contract)
        except sp.SplitError as error:
            if "only witnessed by held families" in str(error):
                return
    pytest.fail("no seed produced the held-trap configuration")


def test_split_fails_closed_when_total_coverage_is_below_k():
    families = [_fam("tmpl:solo", ["item:x"], ["p"], rows=1)]
    contract = sp.split_contract("s", 2, fractions={"train": 1.0, "dev": 0.0, "test": 0.0},
                                 floors={"dev": 1, "test": 1})
    with pytest.raises(sp.SplitError):
        sp.assign(families, contract)


def test_contract_validation_fails_closed():
    with pytest.raises(sp.SplitError):
        sp.split_contract("s", 0)                       # k must be positive
    with pytest.raises(sp.SplitError):
        sp.split_contract("s", 1, floors={"dev": 0, "test": 1})  # nonempty floors
    with pytest.raises(sp.SplitError):
        sp.split_contract("s", 1, fractions={"train": 0.9, "dev": 0.2, "test": 0.1})


def test_preregistration_witness_binds_both_halves():
    """The witness moves when either the enumeration spec or the value-level
    split contract moves — binding either half alone was the review's gap."""
    contract_a = sp.split_contract("seed-a", 2)
    contract_b = sp.split_contract("seed-a", 3)   # different k
    contract_c = sp.split_contract("seed-b", 2)   # different seed
    w_a = sp.preregistration_witness_sha256(contract_a)
    assert w_a != sp.preregistration_witness_sha256(contract_b)
    assert w_a != sp.preregistration_witness_sha256(contract_c)
    assert len(w_a) == 64


def test_spec_sha_binds_the_split_algorithm():
    import process_expression_enumerator as en
    sha = en.enumeration_spec_sha256()
    original = sp.HELD_SELECTION_RULE
    try:
        sp.HELD_SELECTION_RULE = original + " (changed)"
        assert en.enumeration_spec_sha256() != sha
    finally:
        sp.HELD_SELECTION_RULE = original
