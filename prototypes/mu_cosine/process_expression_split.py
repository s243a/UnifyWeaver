#!/usr/bin/env python3
"""Family-level composition-aware split for the v0.4 corpus (§2.5 / §8).

``split-v2`` — the fourth adversarial review of the corpus re-measurement
confirmed the family-level correction (split-v1) but found four
preregistration gaps. Each is closed here, finding by finding:

1. **Coverage counts are per-item row counts.** split-v1 credited a family's
   total ``row_count`` to every witness item it carried anywhere, so one
   pinned row in a 100-row family counted as 100 pin witnesses. A ``Family``
   now carries ``witness_counts`` — for each item, the number of member ROWS
   witnessing it (a row witnesses an item once, however many times the item
   occurs inside it) — and coverage sums those.

2. **The required witness universe is authoritative, not caller-supplied.**
   split-v1 derived the universe from whatever families the caller passed,
   so an extractor that silently omitted digits, terminals, or synthetic
   forms still "covered everything". The contract now binds
   ``required_universe_sha256``; ``assign()`` takes the universe list,
   verifies it against the contract hash, and fails closed when any required
   item is absent from every family or cannot reach coverage k. The
   deterministic AST-to-family extractor and the authoritative universe both
   live in ``process_expression_enumerator``
   (``families_from_expressions`` / ``required_witness_universe``) — the
   module that owns the serialized support.

3. **Held units are preregistered COMPOSITION pairs, and far has floors.**
   split-v1 held lexicographically-smallest structural families, which
   protected nothing compositional: the far slice could be empty and the
   held families' pairs could still cross splits. The contract now names
   explicit held composition pair ids per side; every structural family
   carrying a held pair is grouped and pinned to that pair's slice,
   immovable during repair (a family carrying both a dev-held and a
   test-held pair is a contract error, fail closed). Far floors are
   enforced: a test-held pair's carriers all sit in test, so the pair is
   unseen by train ∪ dev and its carriers are far by construction — and
   ``assign()`` still checks the floor rather than trusting the
   construction.

4. **The executable behavior is bound machine-readably, and buckets are
   integers.** ``SPLIT_ALGORITHM_MANIFEST`` states every rule the algorithm
   executes — coverage semantics, repair order, far classification, pair
   extraction, synthetic projection, held grouping, bucket boundaries — and
   the whole manifest is part of the contract and its hash. Fractions are
   gone: the contract takes integer ``buckets`` summing exactly to the
   modulus, so no rounding ambiguity exists (``round()`` left the boundary
   between dev and test underspecified at fractional inputs).

Far-slice semantics (unchanged from split-v1): a TEST family is *far* iff
its pair set contains at least one pair absent from the union over train AND
dev — a composition seen in dev is not far, because dev steers model
selection. A DEV family's far flag is computed against train alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping

SPLIT_CONTRACT_VERSION = "split-v2"
MODULUS = 10_000
DEFAULT_BUCKETS = {"train": 8_000, "dev": 1_000, "test": 1_000}

#: Every rule the algorithm executes, stated machine-readably and hashed as
#: part of the contract (fourth-review finding 4: binding four constants
#: while the executable behavior floated was not a preregistration).
SPLIT_ALGORITHM_MANIFEST = {
    "version": SPLIT_CONTRACT_VERSION,
    "unit": (
        "structural family: all rows sharing one resolved-kwarg template "
        "identity; pinned and string-bearing rows project into their "
        "SEMANTIC template's family (pins and strings never create "
        "families) while contributing their synthetic witness items"
    ),
    "hash_bytes_rule": "sha256(utf8(family_id) + '|' + utf8(seed)) % 10000",
    "bucket_rule": (
        "integer bucket boundaries over the modulus: train [0, b_train), "
        "dev [b_train, b_train + b_dev), test [b_train + b_dev, modulus); "
        "buckets are exact integers summing to the modulus, so no fraction "
        "rounding exists anywhere in the assignment"
    ),
    "coverage_rule": (
        "witness counts are per-item ROW counts: a row witnesses an item "
        "once, however many times the item occurs inside it; an item is "
        "covered when the sum of its row counts over train families reaches "
        "coverage_minimum_k"
    ),
    "required_universe_rule": (
        "the contract binds required_universe_sha256 = sha256(canonical "
        "json of the sorted universe list); an AUTHORIZING contract must "
        "bind the AUTHORITATIVE hash (required_witness_universe_sha256), "
        "so a caller-shaped universe hashed against itself is refused; "
        "assign() verifies the supplied universe against the contract and "
        "fails closed when a required item is absent from every family or "
        "cannot reach coverage k in train; a non-authorizing contract is a "
        "feasibility probe, carries authorizing=false inside its own hash, "
        "and cannot be witnessed for preregistration; every use site "
        "RE-VALIDATES the contract by re-deriving it from its own fields, "
        "because construction-time validation proves nothing about a value "
        "mutated afterwards"
    ),
    "held_rule": (
        "held units are composition PAIR ids, preregistered explicitly in "
        "the contract per side; every structural family whose pair set "
        "contains a held pair is grouped with it and pinned to that pair's "
        "slice before repair, immovable; a held pair with no carriers is a "
        "contract error; a family carrying both a dev-held and a test-held "
        "pair is a contract error"
    ),
    "repair_rule": (
        "witness items in ascending lexicographic order; while an item is "
        "under-covered in train, move its lexicographically smallest "
        "non-pinned carrier family, searching dev before test, whole "
        "families only; a repair satisfiable only from pinned families "
        "fails closed"
    ),
    "far_rule": (
        "a test family is far iff its pair set minus pairs(train union dev) "
        "is nonempty; a dev family is far iff its pair set minus "
        "pairs(train) is nonempty; len(far[side]) must reach the contract's "
        "far floor for that side, else fail closed"
    ),
    "holdable_rule": (
        "holdable identities are (a) composition pairs "
        "'pair:<resolved parent component>.<slot>|<resolved child component>' "
        "over node-valued edges — positional children and node-valued kwargs "
        "such as mu= — where BOTH endpoints are resolved components carrying "
        "arity, typed kwarg pattern, and modifiers, and (b) kwarg/list-shape "
        "motifs 'motif:<resolved component>.kw:<key>=<value>' and "
        "'motif:<resolved component>.kw:<key>:len<N>'. Endpoint names alone "
        "collapsed argument order, modifiers, and parent kwarg patterns into "
        "single holdout units, and list LENGTH was invisible because a "
        "component records the kwarg KIND; the encoder contract requires "
        "operator-composition AND kwarg/list-shape holdouts, so both are "
        "expressible and both may be named in held_compositions"
    ),
    "row_identity_rule": (
        "corpus rows are deduplicated by canonical_full before family "
        "counting, and the duplicate count is recorded in the corpus "
        "manifest; canonical_FULL rather than canonical_semantic because "
        "rows differing only by pins are distinct rows (§3.1: V3 differs "
        "from V2 only by pins), so semantic dedup would delete required "
        "coverage. Duplicates previously inflated coverage, letting k be "
        "satisfied by copies of one example"
    ),
    "witness_containment_rule": (
        "an authorizing corpus build refuses any extracted witness OUTSIDE "
        "the authoritative universe, not merely checking that required "
        "witnesses are present; a synthetic form the universe does not "
        "declare (a pin hosted on a leaf atom) cannot enter the corpus"
    ),
    "family_invariants_rule": (
        "every family is validated at construction AND re-validated inside "
        "assign(), since the dataclass constructor bypasses the factory: "
        "non-empty canonical family id, strict positive integer row_count "
        "rejecting bool, non-empty witness map, every witness count a strict "
        "integer in [1, row_count] rejecting bool, and non-empty string "
        "witness and holdable identities"
    ),
}


class SplitError(ValueError):
    """The split cannot satisfy its contract; failing closed."""


@dataclass(frozen=True)
class Family:
    """One structural family: a template identity and its rows' facts.

    ``witness_counts``: sorted tuple of ``(item, rows_witnessing_it)`` pairs
    — item identity strings use the same serialized vocabulary the support
    freeze pins (components, edges, slots, ``value:number:0.6``,
    ``terminal:luna``, ``estimand:path``, ``digit:6``, ``synthetic:pin``, …).
    ``pairs``: the union of members' composition pair identities.
    ``row_count``: members in the materialized corpus.
    """

    family_id: str
    witness_counts: tuple
    pairs: frozenset
    row_count: int

    @staticmethod
    def make(family_id: str, witness_counts: Mapping[str, int],
             pairs: Iterable[str], row_count: int) -> "Family":
        return Family(family_id, tuple(sorted(witness_counts.items())),
                      frozenset(pairs), row_count).validated()

    def validated(self) -> "Family":
        """Check every invariant, fail closed.

        Sixth review, finding 6: `Family.make("zero", {}, [], 0)` and
        `Family.make("bool", {"x": True}, [], True)` both succeeded, and the
        dataclass constructor bypassed `make()` entirely. Validation now
        lives on the instance and `assign()` re-checks every family it is
        handed, so a directly-constructed one cannot slip past."""
        if not isinstance(self.family_id, str) or not self.family_id:
            raise SplitError(f"family id {self.family_id!r} must be a "
                             "non-empty string")
        if (isinstance(self.row_count, bool)
                or not isinstance(self.row_count, int) or self.row_count < 1):
            raise SplitError(
                f"family {self.family_id!r}: row_count must be a positive "
                f"integer, got {self.row_count!r}"
            )
        if not self.witness_counts:
            raise SplitError(f"family {self.family_id!r} witnesses nothing")
        for item, count in self.witness_counts:
            if not isinstance(item, str) or not item:
                raise SplitError(f"family {self.family_id!r}: witness item "
                                 f"{item!r} must be a non-empty string")
            if (isinstance(count, bool) or not isinstance(count, int)
                    or not 1 <= count <= self.row_count):
                raise SplitError(
                    f"family {self.family_id!r}: item {item!r} has row count "
                    f"{count!r}, outside [1, row_count={self.row_count}]"
                )
        for pair in self.pairs:
            if not isinstance(pair, str) or not pair:
                raise SplitError(f"family {self.family_id!r}: holdable id "
                                 f"{pair!r} must be a non-empty string")
        return self

    @property
    def counts(self) -> dict:
        return dict(self.witness_counts)

    @property
    def witness_items(self) -> frozenset:
        return frozenset(item for item, _ in self.witness_counts)


def universe_sha256(universe: Iterable[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(universe), separators=(",", ":")).encode()
    ).hexdigest()


def _positive_int(value, what: str) -> int:
    """Strict integer check. `bool` is an `int` in Python, so `k=True` and
    `buckets={"train": True}` passed the old guards (fifth review, finding
    5); every numeric field is checked the same way."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SplitError(f"{what} must be a positive integer, got {value!r}")
    return value


def split_contract(seed: str, coverage_minimum_k: int,
                   required_universe_sha256: str,
                   held_compositions: Mapping[str, list],
                   buckets: Mapping[str, int] | None = None,
                   far_floors: Mapping[str, int] | None = None,
                   authorizing: bool = True) -> dict:
    """Build a split contract, validating every field fail-closed.

    ``authorizing`` (default True) is the fifth review's central
    correction: an authorizing contract's ``required_universe_sha256`` must
    equal the AUTHORITATIVE universe hash. Previously the caller supplied
    both the universe and its hash, so a one-item universe validated
    against its own hash and the binding proved nothing. A non-authorizing
    contract may carry any universe — it is a feasibility PROBE, the flag
    is inside the contract and therefore inside its hash, and
    ``preregistration_witness_sha256`` refuses to witness one, so a probe
    can never be mistaken for a preregistration.
    """
    if not isinstance(seed, str) or not seed:
        raise SplitError("seed must be a non-empty string")
    buckets = dict(buckets or DEFAULT_BUCKETS)
    if sorted(buckets) != ["dev", "test", "train"]:
        raise SplitError("buckets must cover exactly train/dev/test")
    for name, value in buckets.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SplitError(f"bucket {name} must be a non-negative integer, "
                             f"got {value!r}")
    if sum(buckets.values()) != MODULUS:
        raise SplitError(f"buckets must sum to the modulus {MODULUS}")
    _positive_int(coverage_minimum_k, "coverage_minimum_k")
    if not isinstance(held_compositions, Mapping):
        raise SplitError("held_compositions must be a mapping of side -> pairs")
    held = {}
    for side in ("dev", "test"):
        pairs = held_compositions.get(side, [])
        if isinstance(pairs, str) or not isinstance(pairs, (list, tuple)):
            raise SplitError(f"held_compositions[{side!r}] must be a list of "
                             "composition pair id strings")
        for pair in pairs:
            if not isinstance(pair, str) or not pair:
                raise SplitError(f"held composition {pair!r} must be a "
                                 "non-empty string")
        held[side] = sorted(pairs)
    if not held["dev"] or not held["test"]:
        raise SplitError(
            "held_compositions must name at least one composition pair per "
            "side — held units are pairs, and empty held sides were the "
            "fourth review's finding"
        )
    if set(held["dev"]) & set(held["test"]):
        raise SplitError("a composition pair cannot be held on both sides")
    far_floors = dict(far_floors or {"dev": 1, "test": 1})
    if sorted(far_floors) != ["dev", "test"]:
        raise SplitError("far_floors must cover exactly dev and test")
    for side in ("dev", "test"):
        _positive_int(far_floors[side], f"far_floors[{side!r}]")
    if not (isinstance(required_universe_sha256, str)
            and len(required_universe_sha256) == 64
            and all(ch in "0123456789abcdef" for ch in required_universe_sha256)):
        raise SplitError("required_universe_sha256 must be a 64-character "
                         "lowercase hex sha256 digest")
    if not isinstance(authorizing, bool):
        raise SplitError("authorizing must be a bool")
    if authorizing:
        from process_expression_enumerator import required_witness_universe_sha256

        authoritative = required_witness_universe_sha256()
        if required_universe_sha256 != authoritative:
            raise SplitError(
                "an AUTHORIZING contract must bind the authoritative witness "
                f"universe ({authoritative}); got {required_universe_sha256}. "
                "A caller-shaped universe hashed against itself proves "
                "nothing. Pass authorizing=False for a feasibility probe."
            )
    return {
        "version": SPLIT_CONTRACT_VERSION,
        "seed": seed,
        "authorizing": authorizing,
        "buckets": buckets,
        "modulus": MODULUS,
        "coverage_minimum_k": coverage_minimum_k,
        "held_compositions": held,
        "far_floors": {"dev": far_floors["dev"], "test": far_floors["test"]},
        "required_universe_sha256": required_universe_sha256,
        "algorithm": SPLIT_ALGORITHM_MANIFEST,
    }


def revalidate(contract: Mapping) -> dict:
    """Re-derive the contract from its own fields and require equality.

    Sixth review, principal finding: `split_contract()` returns a plain
    dict, so a validated contract could be MUTATED before use — flipping
    `required_universe_sha256` to a one-item universe, or `authorizing`
    from False to True — and both `assign()` and the witness accepted it.
    Validation at construction time proves nothing about a value used
    later, so every use site re-validates instead of trusting.

    Reconstruction catches any mutation, not merely the two known ones: a
    tampered field either fails `split_contract`'s own checks (an
    authorizing contract must still bind the authoritative universe) or
    produces a dict that differs from the one presented.
    """
    try:
        rebuilt = split_contract(
            seed=contract["seed"],
            coverage_minimum_k=contract["coverage_minimum_k"],
            required_universe_sha256=contract["required_universe_sha256"],
            held_compositions=contract["held_compositions"],
            buckets=contract["buckets"],
            far_floors=contract["far_floors"],
            authorizing=contract["authorizing"],
        )
    except KeyError as missing:
        raise SplitError(f"contract is missing {missing}") from None
    if rebuilt != dict(contract):
        raise SplitError(
            "contract does not re-derive from its own fields — it was "
            "mutated after validation; fail closed"
        )
    return rebuilt


def split_contract_sha256(contract: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(dict(contract), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _base_slice(family_id: str, contract: Mapping) -> str:
    digest = hashlib.sha256(
        family_id.encode("utf-8") + b"|" + contract["seed"].encode("utf-8")
    ).hexdigest()
    bucket = int(digest, 16) % contract["modulus"]
    train_edge = contract["buckets"]["train"]
    dev_edge = train_edge + contract["buckets"]["dev"]
    if bucket < train_edge:
        return "train"
    if bucket < dev_edge:
        return "dev"
    return "test"


def _train_coverage(by_id: Mapping[str, Family], train_ids: Iterable[str]) -> dict:
    counts: dict = {}
    for family_id in train_ids:
        for item, count in by_id[family_id].witness_counts:
            counts[item] = counts.get(item, 0) + count
    return counts


def assign(families: list[Family], contract: Mapping,
           required_universe: Iterable[str]) -> dict:
    """Deterministic family-level assignment with recorded repair.

    Returns the split manifest: slices, held pair groups, every repair move
    with the item that forced it, far-slice membership, the coverage report,
    the contract hash, and the manifest's own canonical-bytes hash. Fails
    closed rather than degrading.
    """

    revalidate(contract)          # never trust a contract handed to us
    universe = sorted(required_universe)
    if universe_sha256(universe) != contract["required_universe_sha256"]:
        raise SplitError(
            "supplied witness universe does not hash to the contract's "
            "required_universe_sha256 — the universe is authoritative and "
            "hash-bound, not caller-shaped"
        )

    for family in families:
        family.validated()        # direct construction bypasses make()
    by_id = {f.family_id: f for f in families}
    if len(by_id) != len(families):
        raise SplitError("duplicate family ids")

    # The universe is REQUIRED: every item must be reachable at coverage k
    # from the corpus as a whole before any assignment happens.
    total: dict = {}
    for family in families:
        for item, count in family.witness_counts:
            total[item] = total.get(item, 0) + count
    k = contract["coverage_minimum_k"]
    for item in universe:
        if total.get(item, 0) < k:
            raise SplitError(
                f"required witness item {item!r} has total coverage "
                f"{total.get(item, 0)} < k={k}; the corpus build must add "
                "rows — a split cannot repair it"
            )

    slices: dict = {"train": [], "dev": [], "test": []}
    for family_id in sorted(by_id):
        slices[_base_slice(family_id, contract)].append(family_id)

    # Held composition groups: every carrier of a held pair is pinned to
    # that pair's slice, wherever base assignment put it.
    pinned: dict = {}   # family_id -> side
    held_groups: dict = {"dev": {}, "test": {}}
    for side in ("dev", "test"):
        for pair in contract["held_compositions"][side]:
            carriers = sorted(
                f.family_id for f in families if pair in f.pairs
            )
            if not carriers:
                raise SplitError(
                    f"held composition pair {pair!r} has no carrier family "
                    "— a held pair must exist in the corpus"
                )
            held_groups[side][pair] = carriers
            for family_id in carriers:
                if pinned.get(family_id, side) != side:
                    raise SplitError(
                        f"family {family_id!r} carries both a dev-held and "
                        "a test-held composition pair — the held sets "
                        "conflict; fix the contract"
                    )
                pinned[family_id] = side
    for family_id, side in sorted(pinned.items()):
        for name in ("train", "dev", "test"):
            if family_id in slices[name] and name != side:
                slices[name].remove(family_id)
                slices[side].append(family_id)

    # Repair: every required item must reach coverage k in train. Moves take
    # whole families, smallest id first, from dev then test, never pinned.
    moves = []
    for item in universe:
        while True:
            if _train_coverage(by_id, slices["train"]).get(item, 0) >= k:
                break
            candidates = [
                f for name in ("dev", "test") for f in sorted(slices[name])
                if f not in pinned and item in by_id[f].witness_items
            ]
            if not candidates:
                raise SplitError(
                    f"witness item {item!r} is only witnessed by held-pinned "
                    "families; repair may not consume held groups — fail closed"
                )
            chosen = candidates[0]
            source = "dev" if chosen in slices["dev"] else "test"
            slices[source].remove(chosen)
            slices["train"].append(chosen)
            moves.append({"family": chosen, "from": source, "forced_by": item})

    for name in ("dev", "test"):
        if not slices[name]:
            raise SplitError(f"repair emptied {name} — fail closed")

    # Far membership: computed, never sampled; floors enforced.
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
    for side in ("dev", "test"):
        if len(far[side]) < contract["far_floors"][side]:
            raise SplitError(
                f"far floor unmet: {side} has {len(far[side])} far families, "
                f"floor is {contract['far_floors'][side]} — fail closed"
            )

    manifest = {
        "contract": dict(contract),
        "split_contract_sha256": split_contract_sha256(contract),
        "slices": {name: sorted(ids) for name, ids in slices.items()},
        "held": held_groups,
        "moves": moves,
        "far": far,
        "train_coverage": dict(sorted(
            _train_coverage(by_id, slices["train"]).items()
        )),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest


def preregistration_witness_sha256(contract: Mapping) -> str:
    """The single hash a preregistration pins: the enumeration spec (support,
    registry semantics, split algorithm manifest, required universe) combined
    with the instantiated split contract (seed, buckets, k, held pairs, far
    floors, universe hash)."""

    from process_expression_enumerator import enumeration_spec_sha256

    revalidate(contract)
    if not contract.get("authorizing"):
        raise SplitError(
            "a non-authorizing feasibility probe cannot be witnessed; "
            "preregistration requires an authorizing contract bound to the "
            "authoritative witness universe"
        )
    payload = {
        "enumeration_spec_sha256": enumeration_spec_sha256(),
        "split_contract_sha256": split_contract_sha256(contract),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
