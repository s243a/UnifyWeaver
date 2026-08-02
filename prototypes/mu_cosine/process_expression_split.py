#!/usr/bin/env python3
"""Family-level composition-aware split for the v0.4 corpus (§2.5 / §8).

The third adversarial review of the corpus re-measurement found the first
split algorithm violated the standing whole-template LOCO contract: it hashed
individual semantic expressions, so rows sharing one structural template could
cross train/test. This module is the corrected, EXECUTABLE algorithm — the
worked example in the spec is enforced by tests against this code, so the
example is a normative golden vector rather than prose.

Corrections, review finding by review finding:

1. **The split unit is the structural family** — all rows sharing one
   template identity (the resolved-kwarg structural template). Base
   assignment hashes the FAMILY identity, so whole templates land on one
   side by construction.
2. **Held families**: disjoint dev-held and test-held family sets with
   nonempty floors are derived deterministically at split time and recorded;
   they are protected — repair may never consume them, and a repair that
   could only be satisfied from a held family fails closed.
3. **Repair moves whole families and repairs to the coverage minimum k**,
   not merely from zero to one witness.
4. **The witness-item universe is complete**: components, node edges,
   literal slots, grid values, exact terminal atoms, categorical values
   (every estimand and impl), digit bytes, and the synthetic-form floors
   (pinned rows, string rows) — not just the component list.
5. **Synthetic projection is defined**: a pinned or string-bearing row
   belongs to the family of its SEMANTIC template (pins and strings never
   create families), while contributing its synthetic witness items.
6. **Exact hash bytes are defined**: base assignment hashes
   ``utf-8(family_id) + b"|" + utf-8(seed)`` with SHA-256, taking the
   integer value modulo 10_000 against cumulative fraction boundaries.
7. **The whole split contract is bound by the preregistration witness**:
   ``split_contract_sha256()`` covers seed, fractions, floors, k, the
   held-family selection rule, and the hash-byte definition, and
   ``enumeration_spec_sha256()`` includes it.

Far-slice semantics: a TEST family is *far* iff its pair set (union of
members' edge-context pairs) contains at least one pair absent from the union
over train AND dev — a composition seen in dev is not far, because dev
steers model selection. A DEV family's far flag is computed against train
alone. Both are computed, never sampled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterable, Mapping

#: The preregistered split contract. `coverage_minimum_k` is the §2.5 owner
#: input; it is a *parameter* of the contract and its chosen value enters the
#: contract hash at preregistration.
SPLIT_CONTRACT_VERSION = "split-v1"
DEFAULT_FRACTIONS = {"train": 0.8, "dev": 0.1, "test": 0.1}
MODULUS = 10_000

#: Held-family selection rule (versioned prose, hashed): among each of dev
#: and test after base assignment, the `floor` lexicographically smallest
#: family ids become held. Deterministic, curation-free, disjoint by
#: construction (a family has one base slice).
HELD_SELECTION_RULE = (
    "after base assignment, the floor-many lexicographically smallest family "
    "ids within dev and within test are held; held families are immovable"
)

HASH_BYTES_RULE = "sha256(utf8(family_id) + '|' + utf8(seed)) % 10000"


class SplitError(ValueError):
    """The split cannot satisfy its contract; failing closed."""


@dataclass(frozen=True)
class Family:
    """One structural family: a template identity and its rows' facts.

    ``witness_items``: every support item any member row witnesses —
    component identities, edge identities, literal-slot identities, grid
    values (``value:number:0.6``), terminal atoms (``terminal:luna``),
    categorical values (``estimand:path``, ``impl:structural``), digit bytes
    (``digit:6``), and synthetic markers (``synthetic:pin``,
    ``synthetic:string``). Item identity strings are the same serialized
    vocabulary the support freeze pins.
    ``pairs``: the union of members' edge-context pair identities.
    ``row_count``: members in the materialized corpus (pinned/string rows
    project into their semantic family and count here).
    """

    family_id: str
    witness_items: frozenset
    pairs: frozenset
    row_count: int


def split_contract(seed: str, coverage_minimum_k: int,
                   fractions: Mapping[str, float] | None = None,
                   floors: Mapping[str, int] | None = None) -> dict:
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    if sorted(fractions) != ["dev", "test", "train"]:
        raise SplitError("fractions must cover exactly train/dev/test")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise SplitError("fractions must sum to 1")
    floors = dict(floors or {"dev": 1, "test": 1})
    if any(f < 1 for f in floors.values()):
        raise SplitError("held-family floors must be nonempty (>= 1)")
    if not isinstance(coverage_minimum_k, int) or coverage_minimum_k < 1:
        raise SplitError("coverage_minimum_k must be a positive integer")
    return {
        "version": SPLIT_CONTRACT_VERSION,
        "seed": seed,
        "fractions": fractions,
        "floors": {"dev": floors["dev"], "test": floors["test"]},
        "coverage_minimum_k": coverage_minimum_k,
        "held_selection_rule": HELD_SELECTION_RULE,
        "hash_bytes_rule": HASH_BYTES_RULE,
        "modulus": MODULUS,
    }


def split_contract_sha256(contract: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(dict(contract), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _base_slice(family_id: str, contract: Mapping) -> str:
    digest = hashlib.sha256(
        family_id.encode("utf-8") + b"|" + contract["seed"].encode("utf-8")
    ).hexdigest()
    bucket = int(digest, 16) % contract["modulus"]
    train_edge = round(contract["fractions"]["train"] * contract["modulus"])
    dev_edge = train_edge + round(contract["fractions"]["dev"] * contract["modulus"])
    if bucket < train_edge:
        return "train"
    if bucket < dev_edge:
        return "dev"
    return "test"


def _coverage(families: Iterable[Family]) -> dict:
    counts: dict = {}
    for family in families:
        for item in family.witness_items:
            counts[item] = counts.get(item, 0) + family.row_count
    return counts


def assign(families: list[Family], contract: Mapping) -> dict:
    """Deterministic family-level assignment with recorded repair.

    Returns the split manifest: slices, held sets, every repair move with
    the item that forced it, far-slice membership, the coverage report, and
    the contract hash. Fails closed rather than degrading.
    """

    by_id = {f.family_id: f for f in families}
    if len(by_id) != len(families):
        raise SplitError("duplicate family ids")

    slices: dict = {"train": [], "dev": [], "test": []}
    for family_id in sorted(by_id):
        slices[_base_slice(family_id, contract)].append(family_id)

    # Held families: floor-many lexicographically smallest ids in dev/test.
    held: dict = {}
    for name in ("dev", "test"):
        floor = contract["floors"][name]
        if len(slices[name]) < floor:
            raise SplitError(
                f"{name} received {len(slices[name])} families under base "
                f"assignment; its held floor is {floor} — fail closed"
            )
        held[name] = sorted(slices[name])[:floor]
    held_all = set(held["dev"]) | set(held["test"])

    # Repair: every witness item must reach coverage k in train. Moves take
    # whole families, smallest id first, from dev then test, never held.
    k = contract["coverage_minimum_k"]
    moves = []
    universe: set = set()
    for family in families:
        universe |= family.witness_items
    for item in sorted(universe):
        while True:
            train_cov = _coverage(by_id[f] for f in slices["train"]).get(item, 0)
            if train_cov >= k:
                break
            candidates = [
                f for name in ("dev", "test") for f in sorted(slices[name])
                if f not in held_all and item in by_id[f].witness_items
            ]
            if not candidates:
                total = _coverage(families).get(item, 0)
                if total < k:
                    raise SplitError(
                        f"witness item {item!r} has total coverage {total} < k={k}; "
                        "the corpus build must add rows — a split cannot repair it"
                    )
                raise SplitError(
                    f"witness item {item!r} is only witnessed by held families; "
                    "repair may not consume held families — fail closed"
                )
            chosen = candidates[0]
            source = "dev" if chosen in slices["dev"] else "test"
            slices[source].remove(chosen)
            slices["train"].append(chosen)
            moves.append({"family": chosen, "from": source, "forced_by": item})

    for name in ("dev", "test"):
        if len(slices[name]) < contract["floors"][name]:
            raise SplitError(f"repair emptied {name} below its floor — fail closed")

    # Far membership: computed, never sampled. Test-far is judged against
    # train ∪ dev (dev steers selection); dev-far against train alone.
    def pair_union(names: Iterable[str]) -> frozenset:
        out: set = set()
        for family_id in names:
            out |= by_id[family_id].pairs
        return frozenset(out)

    seen_by_train = pair_union(slices["train"])
    seen_by_train_dev = seen_by_train | pair_union(slices["dev"])
    far = {
        "dev": sorted(f for f in slices["dev"]
                      if by_id[f].pairs - seen_by_train),
        "test": sorted(f for f in slices["test"]
                       if by_id[f].pairs - seen_by_train_dev),
    }

    return {
        "contract": dict(contract),
        "split_contract_sha256": split_contract_sha256(contract),
        "slices": {name: sorted(ids) for name, ids in slices.items()},
        "held": held,
        "moves": moves,
        "far": far,
        "train_coverage": dict(sorted(
            _coverage(by_id[f] for f in slices["train"]).items()
        )),
    }


def preregistration_witness_sha256(contract: Mapping) -> str:
    """The single hash a preregistration pins: the enumeration spec (support,
    registry semantics, split algorithm) combined with the instantiated split
    contract (seed, fractions, floors, k). Binding either half alone is the
    gap the third adversarial review closed."""

    from process_expression_enumerator import enumeration_spec_sha256

    payload = {
        "enumeration_spec_sha256": enumeration_spec_sha256(),
        "split_contract_sha256": split_contract_sha256(contract),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
