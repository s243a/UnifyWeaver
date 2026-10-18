#!/usr/bin/env python3
"""Exact enumeration counts for the process-expression corpus under registry v0.4.

``DESIGN_process_expression_generator.md`` §2 froze its corpus posture against
registry v0.3 and paused enumeration on the v0.4 gate. v0.4 has shipped, and
this module is the re-measurement: exact counts of the enumerable support
under declared caps and value grids, computed by dynamic programming over the
live registry — never by materialization, because the headline finding is that
materialization is no longer possible:

    naive v0.4 enumeration ~= 3.3e15 expressions.

§2.3's claim that "the type system is the binding constraint" was true of
v0.3 and is FALSE under v0.4: the binding constraint moved to kwarg
enumeration (`estimand=` x9, `impl=` x2, `mu=` x judge-expressions, on eleven
operators). The counts here are the evidence base for the corpus decision;
this module deliberately does not build a corpus.

Counting is exact and was verified against brute-force materialization at
reduced caps (see ``test_process_expression_enumerator.py``): every distinct
AST has a unique structural decomposition (root choice, slot fillings, kwarg
selection), so the recursion counts each expression exactly once.

Scope declarations, all versioned in ``ENUMERATION_SPEC``:

- Node count includes every ``Node`` in the tree, including node-valued kwarg
  values (``mu=haiku`` costs a node); depth likewise nests through both args
  and node-valued kwargs.
- Value grids are the v0.4 witnessed literal set. Strings and pins are
  excluded: §3 makes them sampler coverage, not enumeration support.
- A kwarg spelled at its registered default is the same canonical expression
  as its elided form, so defaults are excluded from value grids (counting
  them would double-count identities).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from typing import Callable, Iterator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import process_cards as pc
from process_cards import REGISTRY
from registry_v04_migration import registry_content_sha256 as _registry_content_sha256
import process_expression_split as _split

ENUMERATION_VERSION = "enum-v1"

MAX_DEPTH = 3
#: 5 under v0.3; the v0.4 envelope moved — `graph-judge` has exactly 6 nodes,
#: so a cap of 5 fails the §2.4 coverage gate.
MAX_NODE_COUNT = 6
MAX_ARITY = 3
MAX_KWARGS_NODE = 2
MAX_LIST_LENGTH = 2

#: The v0.4 witnessed literal set (measured: envelope tests).
NUMBER_GRID = (0.02, 0.03, 0.6, 0.85)
INT_GRID = (10, 20)

OUTPUT_ROOTS = ("substrate", "judge", "score", "target-set", "pick")

#: The three measured scenarios. Methodology placement is the dominant lever:
#: `estimand=`/`impl=` are deployment metadata (`require_deployable` checks
#: the ROOT), so restricting their enumeration to the root is semantically
#: motivated, not merely convenient.
SCENARIOS = {
    "naive-full": dict(methodology_interior=True, methodology_root=True, mu=True),
    "methodology-root-only": dict(methodology_interior=False, methodology_root=True, mu=True),
    "structural-only": dict(methodology_interior=False, methodology_root=False, mu=False),
}


def enumeration_spec_sha256() -> str:
    """Content witness for preregistration: version, caps, grids, the scenario
    definitions (root/interior methodology placement is part of the spec, not
    folklore), and the serialized component vocabulary's own hash — so the
    preimage pins WHAT the support contains, not merely how big it is."""

    payload = {
        "version": ENUMERATION_VERSION,
        "registry_version": pc.REGISTRY_VERSION,
        "caps": {
            "max_depth": MAX_DEPTH,
            "max_node_count": MAX_NODE_COUNT,
            "max_arity": MAX_ARITY,
            "max_kwargs_node": MAX_KWARGS_NODE,
            "max_list_length": MAX_LIST_LENGTH,
        },
        "number_grid": list(NUMBER_GRID),
        "int_grid": list(INT_GRID),
        "excluded": ["string", "pins"],
        # The root output-type universe is executable behavior (count() and
        # the vocabulary's `process` expansion both iterate it), so it is
        # bound here — the fourth review found it floating.
        "output_roots": list(OUTPUT_ROOTS),
        "scenarios": {name: dict(flags) for name, flags in sorted(SCENARIOS.items())},
        "component_vocabulary_sha256": component_vocabulary_sha256(),
        # The §3 synthetic extension is a separate, finite, versioned
        # policy (sixth review ruling: the support stays exhaustive).
        "synthetic_extension_sha256": synthetic_extension_sha256(),
        # The authoritative REQUIRED witness universe (fourth review: a
        # caller-supplied universe lets omissions vanish silently). The split
        # contract binds the same hash, and assign() verifies against it.
        "required_witness_universe_sha256": _split.universe_sha256(
            required_witness_universe()
        ),
        # Registry semantics: the signature-table content witness (kinds,
        # defaults, outputs, modifiers) plus the categorical value domains
        # the signatures reference by kind name only.
        "registry_content_sha256": _registry_content_sha256(),
        "estimands": sorted(pc.ESTIMANDS),
        "impls": sorted(pc.IMPLS),
        # v0.5: walk values with their DECLARED shapes (the family function
        # depends on the shape, so it is bound, not inferred), and weights.
        "walks": dict(sorted(pc.WALKS.items())),
        "weights": sorted(pc.WEIGHTS),
        # The split ALGORITHM is bound here as its complete machine-readable
        # manifest — coverage semantics, repair, far classification, pair
        # extraction, held grouping, bucket boundaries — not a four-constant
        # summary (the fourth review's finding). The value-level contract
        # (seed, buckets, k, held pairs, floors) is chosen at preregistration
        # and enters through preregistration_witness_sha256(), which combines
        # this hash with split_contract_sha256.
        "split_algorithm": dict(_split.SPLIT_ALGORITHM_MANIFEST),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------


def _dist_mul(a: dict, b: dict, cap: int) -> dict:
    out: dict = {}
    for n1, c1 in a.items():
        for n2, c2 in b.items():
            if n1 + n2 <= cap:
                out[n1 + n2] = out.get(n1 + n2, 0) + c1 * c2
    return out


def _dist_add(a: dict, b: dict) -> dict:
    out = dict(a)
    for n, c in b.items():
        out[n] = out.get(n, 0) + c
    return out


def _value_count(kind: str, template_mode: bool) -> int:
    if template_mode:
        # A template keeps the slot's typed shape; lists keep their length.
        return {
            "number": 1, "int": 1, "estimand": 1, "impl": 1,
            "walk": 1, "weight": 1,
            "number_list": MAX_LIST_LENGTH, "int_list": MAX_LIST_LENGTH,
        }[kind]
    if kind == "number":
        return len(NUMBER_GRID)
    if kind == "int":
        return len(INT_GRID)
    if kind == "number_list":
        return sum(len(NUMBER_GRID) ** length for length in range(1, MAX_LIST_LENGTH + 1))
    if kind == "int_list":
        return sum(len(INT_GRID) ** length for length in range(1, MAX_LIST_LENGTH + 1))
    if kind == "estimand":
        return len(pc.ESTIMANDS)
    if kind == "impl":
        return len(pc.IMPLS)
    if kind == "walk":
        return len(pc.WALKS)
    if kind == "weight":
        return len(pc.WEIGHTS)
    raise AssertionError(f"unexpected value kind: {kind}")


def _leaf_dist(output: str, template_mode: bool) -> dict:
    if template_mode:
        shapes: set = set()
        for name, sig in REGISTRY.items():
            if sig.atom and sig.output == output and not sig.operator:
                shapes.add(None)
                shapes.update(sig.modifiers)
        total = len(shapes)
        if output == "score":
            total += 1  # the bare `e5` dual atom is its own leaf shape
        return {1: total} if total else {}
    total = 0
    for name, sig in REGISTRY.items():
        if sig.atom and sig.output == output and not sig.operator:
            total += 1 + len(sig.modifiers)
    if output == "score":
        total += 1
    return {1: total} if total else {}


def _kwarg_selections(
    name: str, sig, arity: int, allow_methodology: bool, template_mode: bool
) -> Iterator[tuple[int, list[str]]]:
    """Yield (scalar multiplicity, node-valued kwarg kinds) per legal shape.

    Enumerates kwarg-presence subsets up to MAX_KWARGS_NODE with the
    operator-specific constraints the validator enforces: routing's t/menus
    are paired with equal lengths (and jointly count as two kwargs), and
    blend.w has one weight per positional source.
    """

    per_key: dict = {}
    for key in sorted(sig.kwargs):
        spec = sig.kwargs[key]
        if key in ("estimand", "impl") and not allow_methodology:
            per_key[key] = None
            continue
        if spec.kind == "string":
            per_key[key] = None  # §3.2: strings are sampler coverage
            continue
        if spec.kind in pc.OUTPUT_TYPES or spec.kind == "process":
            per_key[key] = "NODE"
            continue
        if template_mode and spec.default is not None:
            # The template identity is computed over RESOLVED kwargs — the
            # established structural-template convention: a defaulted kwarg is
            # always present after canonical resolution, so absent-vs-explicit
            # is not a template distinction and contributes no presence choice.
            # (Re-review finding: counting it split templates the canonical
            # identity merges.)
            per_key[key] = None
            continue
        count = _value_count(spec.kind, template_mode)
        if not template_mode and spec.default is not None:
            if spec.kind == "number" and spec.default in NUMBER_GRID:
                count -= 1  # explicit default == elided form: one identity
            elif spec.kind == "int" and spec.default in INT_GRID:
                count -= 1
            elif spec.kind == "weight" and spec.default in pc.WEIGHTS:
                count -= 1  # weight="uniform" == elided: one identity
            elif spec.kind == "walk" and spec.default in pc.WALKS:
                count -= 1
        per_key[key] = count
    usable = [key for key in sorted(sig.kwargs) if per_key[key] is not None]

    for r in range(0, MAX_KWARGS_NODE + 1):
        for subset_keys in itertools.combinations(usable, r):
            subset = set(subset_keys)
            if any(
                spec.required and key not in subset
                for key, spec in sig.kwargs.items()
            ):
                continue
            if name == "routing" and ("t" in subset) != ("menus" in subset):
                continue
            multiplicity, node_kinds = 1, []
            if name == "routing" and "t" in subset:
                if template_mode:
                    multiplicity *= MAX_LIST_LENGTH  # one shape per shared length
                else:
                    multiplicity *= sum(
                        (len(NUMBER_GRID) * len(INT_GRID)) ** length
                        for length in range(1, MAX_LIST_LENGTH + 1)
                    )
                subset -= {"t", "menus"}
            if name == "blend" and "w" in subset:
                # w has one weight per positional source, so its length is the
                # arity — and the list cap applies to it like any list. An
                # arity above the cap makes the w-bearing shape unenumerable
                # (fail-closed), not a longer list. (Caught by the brute-force
                # equivalence test; the first draft counted capped-out lists.)
                if arity > MAX_LIST_LENGTH:
                    continue
                multiplicity *= 1 if template_mode else len(NUMBER_GRID) ** arity
                subset -= {"w"}
            dead = False
            for key in subset:
                option = per_key[key]
                if option == "NODE":
                    node_kinds.append(sig.kwargs[key].kind)
                elif isinstance(option, int) and option > 0:
                    multiplicity *= option
                else:
                    dead = True
                    break
            if not dead:
                yield multiplicity, node_kinds


def make_counter(scenario: str, template_mode: bool = False) -> Callable:
    """A memoized counter: f(output, depth_budget, is_root) -> {node_count: n}.

    Counts DISTINCT canonical expressions of depth <= budget: each AST has a
    unique decomposition into root, slots, and kwarg selection, so nothing is
    counted twice, and default-elision is handled by excluding defaults from
    the grids.
    """

    flags = SCENARIOS[scenario]
    memo: dict = {}

    def slot_dist(declared: str, d: int) -> dict:
        out: dict = {}
        for alternative in declared.split("|"):
            if alternative in pc.VALUE_KINDS:
                out = _dist_add(out, {0: _value_count(alternative, template_mode)})
            elif alternative == "process":
                for output in OUTPUT_ROOTS:
                    out = _dist_add(out, f(output, d))
            else:
                out = _dist_add(out, f(alternative, d))
        return out

    def f(output: str, d: int, is_root: bool = False) -> dict:
        allow_methodology = (
            flags["methodology_root"] if is_root else flags["methodology_interior"]
        )
        key = (output, d, allow_methodology)
        if key in memo:
            return memo[key]
        total = _leaf_dist(output, template_mode)
        if d > 0:
            for name, sig in REGISTRY.items():
                if not sig.operator or sig.output != output:
                    continue
                cap = sig.max_args if sig.max_args is not None else MAX_ARITY
                for arity in range(sig.min_args, min(cap, MAX_ARITY) + 1):
                    body = {0: 1}
                    for index in range(arity):
                        declared = (
                            sig.arg_types[index]
                            if index < len(sig.arg_types)
                            else sig.variadic_arg_type
                        )
                        body = _dist_mul(body, slot_dist(declared, d - 1), MAX_NODE_COUNT)
                        if not body:
                            break
                    if not body:
                        continue
                    combos: dict = {0: 0}
                    for multiplicity, node_kinds in _kwarg_selections(
                        name, sig, arity, allow_methodology, template_mode
                    ):
                        if node_kinds and not flags["mu"]:
                            continue
                        part = {0: multiplicity}
                        for kind in node_kinds:
                            part = _dist_mul(part, f(kind, d - 1), MAX_NODE_COUNT)
                        combos = _dist_add(combos, part)
                    contribution = _dist_mul(body, combos, MAX_NODE_COUNT)
                    contribution = _dist_mul(contribution, {1: 1}, MAX_NODE_COUNT)
                    total = _dist_add(total, contribution)
        memo[key] = total
        return total

    return f


def count(scenario: str, template_mode: bool = False) -> tuple[int, dict]:
    """(grand total, {node_count: count}) over every root output type."""

    counter = make_counter(scenario, template_mode)
    by_nodes: dict = {}
    for output in OUTPUT_ROOTS:
        for n, c in counter(output, MAX_DEPTH, is_root=True).items():
            by_nodes[n] = by_nodes.get(n, 0) + c
    return sum(by_nodes.values()), dict(sorted(by_nodes.items()))


# --------------------------------------------------------------------------
# coverage predicate (§2.4 gate at the support level)
# --------------------------------------------------------------------------


def _node_count(node) -> int:
    total = 1
    for child in node.args:
        if isinstance(child, pc.Node):
            total += _node_count(child)
    for _, value in node.kwargs:
        if isinstance(value, pc.Node):
            total += _node_count(value)
    return total


def _depth(node) -> int:
    children = [c for c in node.args if isinstance(c, pc.Node)]
    children += [v for _, v in node.kwargs if isinstance(v, pc.Node)]
    return 1 + max((_depth(c) for c in children), default=-1)


def _values_in_grid(node) -> bool:
    """Every literal must come from the declared grids, judged by the
    registry-DECLARED kind — never the runtime type. `estimand`/`impl` are
    enumerations (validate already holds them to their sets), `string` fields
    are excluded from enumeration entirely (§3.2), and numeric kinds must sit
    on the declared kind's grid: `max(10, e5)` parses (int satisfies a
    `number` slot) but 10 is outside NUMBER_GRID, so it is outside the
    enumerable support — the adversarial re-review's counterexample."""
    signature = REGISTRY[node.name]
    for index, child in enumerate(node.args):
        if isinstance(child, pc.Node):
            if not _values_in_grid(child):
                return False
            continue
        declared = (
            signature.arg_types[index]
            if index < len(signature.arg_types)
            else signature.variadic_arg_type
        )
        kinds = [alt for alt in declared.split("|") if alt in pc.VALUE_KINDS]
        if len(kinds) != 1:
            return False
        grid = NUMBER_GRID if kinds[0] == "number" else (
            INT_GRID if kinds[0] == "int" else ())
        if child not in grid:
            return False
    for key, value in node.kwargs:
        kind = signature.kwargs[key].kind
        if isinstance(value, pc.Node):
            if not _values_in_grid(value):
                return False
        elif kind in ("estimand", "impl", "walk", "weight"):
            continue
        elif kind == "string":
            return False  # strings are sampler coverage, never enumeration
        elif kind in ("number_list", "int_list"):
            if len(value) > MAX_LIST_LENGTH:
                return False
            grid = NUMBER_GRID if kind == "number_list" else INT_GRID
            if any(item not in grid for item in value):
                return False
        elif kind == "number":
            if value not in NUMBER_GRID:
                return False
        elif kind == "int":
            if value not in INT_GRID:
                return False
    return True


def _methodology_placement_ok(node, flags, is_root: bool) -> bool:
    allowed = flags["methodology_root"] if is_root else flags["methodology_interior"]
    present = {key for key, _ in node.kwargs} & {"estimand", "impl"}
    if present and not allowed:
        return False
    for child in node.args:
        if isinstance(child, pc.Node) and not _methodology_placement_ok(child, flags, False):
            return False
    for key, value in node.kwargs:
        if isinstance(value, pc.Node):
            if not flags["mu"]:
                return False
            if not _methodology_placement_ok(value, flags, False):
                return False
    return True


def _max_kwargs(node) -> int:
    best = len(node.kwargs)
    for child in node.args:
        if isinstance(child, pc.Node):
            best = max(best, _max_kwargs(child))
    for _, value in node.kwargs:
        if isinstance(value, pc.Node):
            best = max(best, _max_kwargs(value))
    return best


def _max_arity(node) -> int:
    best = len(node.args)
    for child in node.args:
        if isinstance(child, pc.Node):
            best = max(best, _max_arity(child))
    for _, value in node.kwargs:
        if isinstance(value, pc.Node):
            best = max(best, _max_arity(value))
    return best


def covers(node, scenario: str) -> bool:
    """True when the expression lies inside the scenario's enumerable support.

    Pins are ignored (they are outside semantic identity and outside
    enumeration); strings exclude an expression, matching the grids.
    """

    pc.validate(node)
    if _depth(node) > MAX_DEPTH or _node_count(node) > MAX_NODE_COUNT:
        return False
    if _max_arity(node) > MAX_ARITY or _max_kwargs(node) > MAX_KWARGS_NODE:
        return False
    if not _values_in_grid(node):
        return False
    return _methodology_placement_ok(node, SCENARIOS[scenario], True)


def _component_kwarg_patterns(name: str, sig, arity: int, allow_methodology: bool):
    """Legal RESOLVED kwarg patterns for one operator shape — the SAME
    legality rules `_kwarg_selections` enforces (routing pairing, blend.w
    length forced by arity and capped by the list limit, strings excluded),
    so a pattern is a component if and only if it is enumerable. The pattern
    is the canonical *resolved* key set: a defaulted kwarg is always present
    after resolution, so it belongs to every pattern and is never a presence
    choice (re-review alignment with the structural-template identity).
    Counts alone cannot freeze support; patterns serialize into identities."""

    constant, usable = [], []
    for key in sorted(sig.kwargs):
        spec = sig.kwargs[key]
        if key in ("estimand", "impl") and not allow_methodology:
            continue
        if spec.kind == "string":
            continue
        if spec.default is not None:
            constant.append(key)  # resolved into every pattern
        else:
            usable.append(key)
    patterns = set()
    for r in range(0, MAX_KWARGS_NODE + 1):
        for subset in itertools.combinations(usable, r):
            chosen = set(subset)
            if any(
                spec.required and key not in chosen and key not in constant
                for key, spec in sig.kwargs.items()
            ):
                continue
            if name == "routing" and (("t" in chosen) != ("menus" in chosen)):
                continue
            if name == "blend" and "w" in chosen and arity > MAX_LIST_LENGTH:
                continue  # w's length is the arity; capped out = unenumerable
            patterns.add(frozenset(chosen | set(constant)))
    return patterns


def component_vocabulary() -> dict:
    """The composable support of §2.5, as SERIALIZED IDENTITIES.

    The owner's ruling on the corpus posture: templates must be general,
    composable components — one operator with one arity and one legal
    kwarg-presence pattern (or one typed leaf shape) — composed via recursion
    through the type system. A support freeze needs content, not counts: each
    component and each edge has a canonical string identity here, and
    ``component_vocabulary_sha256`` binds the whole set, so a component
    swapped for an equal-cardinality impostor moves the hash.

    Node-composition edges (parent slot -> child OUTPUT type, including the
    ``mu=`` kwarg edge) are serialized separately from literal slots (parent
    slot -> value kind): only the former compose recursively; the latter
    terminate in grid values.
    """

    # Leaf identities carry their exact terminal atoms: `leaf:judge.D{luna}`.
    # Without the terminals, a leaf shape's realizing set could change (an
    # atom added or renamed) while the identity — and any count — stayed
    # fixed (re-review finding: the manifest must pin terminals).
    terminals: dict = {}
    for name, sig in REGISTRY.items():
        if sig.atom and not sig.operator:
            terminals.setdefault((sig.output, None), set()).add(name)
            for modifier in sig.modifiers:
                terminals.setdefault((sig.output, modifier), set()).add(name)
    terminals[("score", "e5-atom")] = {"e5"}  # the dual atom's own leaf shape
    leaf_ids = set()
    for (output, modifier), atoms in terminals.items():
        shape = f"{output}.{modifier}" if modifier else output
        leaf_ids.add(f"leaf:{shape}{{{','.join(sorted(atoms))}}}")

    def _typed(pattern, sig):
        # kwarg keys carry their registry-declared kinds: `mu:judge`.
        return ",".join(f"{key}:{sig.kwargs[key].kind}" for key in sorted(pattern))

    interior_ids: set = set()
    root_only_ids: set = set()
    node_edge_ids: set = set()
    literal_slot_ids: set = set()

    for name, sig in REGISTRY.items():
        if not sig.operator:
            continue
        cap = sig.max_args if sig.max_args is not None else MAX_ARITY
        for arity in range(sig.min_args, min(cap, MAX_ARITY) + 1):
            interior_patterns = _component_kwarg_patterns(name, sig, arity, False)
            for pattern in interior_patterns:
                interior_ids.add(f"op:{name}/{arity}{{{_typed(pattern, sig)}}}")
            for pattern in _component_kwarg_patterns(name, sig, arity, True):
                identity = f"op:{name}/{arity}{{{_typed(pattern, sig)}}}"
                if pattern not in interior_patterns:
                    root_only_ids.add(identity)
            for index in range(arity):
                declared = (
                    sig.arg_types[index]
                    if index < len(sig.arg_types)
                    else sig.variadic_arg_type
                )
                for alternative in declared.split("|"):
                    if alternative == "process":
                        for output in OUTPUT_ROOTS:
                            node_edge_ids.add(f"edge:{name}/{arity}.arg{index}->{output}")
                    elif alternative in pc.OUTPUT_TYPES:
                        node_edge_ids.add(f"edge:{name}/{arity}.arg{index}->{alternative}")
                    elif alternative in pc.VALUE_KINDS:
                        literal_slot_ids.add(f"slot:{name}/{arity}.arg{index}:{alternative}")
        for key in sorted(sig.kwargs):
            kind = sig.kwargs[key].kind
            if kind in pc.OUTPUT_TYPES:
                node_edge_ids.add(f"edge:{name}.kw:{key}->{kind}")

    return {
        "leaves": sorted(leaf_ids),
        "operators_interior": sorted(interior_ids),
        "operators_root_only": sorted(root_only_ids),
        "node_edges": sorted(node_edge_ids),
        "literal_slots": sorted(literal_slot_ids),
        # Synthetic forms the support deliberately excludes, named so the
        # manifest states its own boundary instead of implying it: each is
        # sampler coverage with its owning section.
        "excluded_synthetic": [
            "pins (sampler coverage, generator spec 3.1)",
            "string kwargs (sampler coverage, generator spec 3.2)",
            "interior methodology kwargs (sampler coverage under"
            " methodology-root-only, generator spec 2.5)",
        ],
    }


def component_vocabulary_counts() -> dict:
    """Two totals, each named for what it sums.

    An external verification pass found the old table unauditable: the five
    class rows sum to 109 while the row labelled as the total read
    76 — the count of COMPOSABLE COMPONENT shapes (leaves plus operator
    shapes) only. Edges and literal slots are relations between components
    rather than components, so both numbers are meaningful, but a row named
    "total" sitting under five rows that sum to something else cannot be
    checked by a reader. Both are now named for their contents, and
    ``serialized_identities_total`` is the count the vocabulary hash binds.
    """
    vocabulary = component_vocabulary()
    sections = ("leaves", "operators_interior", "operators_root_only",
                "node_edges", "literal_slots")
    return {
        "leaf_shapes": len(vocabulary["leaves"]),
        "operator_shapes_interior": len(vocabulary["operators_interior"]),
        "operator_shapes_root_only_extension": len(vocabulary["operators_root_only"]),
        "node_composition_edges": len(vocabulary["node_edges"]),
        "literal_slots": len(vocabulary["literal_slots"]),
        "composable_component_shapes": (
            len(vocabulary["leaves"])
            + len(vocabulary["operators_interior"])
            + len(vocabulary["operators_root_only"])
        ),
        "serialized_identities_total": sum(
            len(vocabulary[section]) for section in sections),
    }


def component_vocabulary_sha256() -> str:
    """Content hash of the serialized support — what a preregistration pins.
    Cardinality-preserving substitutions move this hash."""

    return hashlib.sha256(
        json.dumps(component_vocabulary(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()



# --------------------------------------------------------------------------
# deterministic AST-to-family extractor + the authoritative witness universe
# --------------------------------------------------------------------------
#
# The fourth adversarial review's second finding: the split's witness
# universe was whatever the caller summarized, and no real AST-to-family
# builder existed — an extractor that silently dropped digits, terminals, or
# synthetic forms would still "cover everything". This section is the
# builder. `families_from_expressions` maps real v0.4 expressions (parsed by
# the sealed registry, never re-lexed here) to `Family` records whose item
# identity strings are the SAME serialized vocabulary the support freeze
# pins, and `required_witness_universe` derives — from the vocabulary, the
# grids, and the registry's categorical domains, never from any corpus — the
# authoritative item list whose hash both the enumeration spec and the split
# contract bind.


#: The §3 SYNTHETIC EXTENSION POLICY — finite, versioned, hash-bound.
#:
#: The sixth review's ruling on the seam: the enumerable support stays
#: exhaustive, and §3's forms enter through this separate policy rather than
#: by widening the support. It also closes the review's third finding: a
#: PREFIX test is not an allowlist. `synthetic/pin-` accepted
#: `synthetic/pin-../../home/s243a/private/notes` — a real private path,
#: reachable by traversal, which is exactly what §3.3 forbids. Membership is
#: now exact set membership over a committed finite set.
SYNTHETIC_EXTENSION_VERSION = "synth-v1"

#: Committed pin ids. §3.1 needs pins at all (V2 and V3 are byte-identical
#: without them); the count is a sampler choice, the SET is policy.
SYNTHETIC_PINS = (
    "synthetic/pin-0001",
    "synthetic/pin-0002",
    "synthetic/pin-0003",
)

#: §3.2's allowlist: exactly one string per required class, so the three
#: classes cannot be satisfied by three strings of the same class.
SYNTHETIC_MANIFESTS = {
    "ascii": "synthetic-manifest-0001",
    "non_ascii": "synthetic-manifest-0002-café",
    "escaped": 'synthetic-manifest-0003-"quoted"',
}

#: §3.2 requires three string classes, not one: an ASCII string, a
#: non-ASCII UTF-8 string, and one containing an escape. A single
#: `synthetic:string` item could be satisfied by three ASCII rows and leave
#: the tokenizer's byte-fallback path untrained (fifth review, finding 4).
STRING_CLASSES = ("ascii", "non_ascii", "escaped")


def synthetic_extension_policy() -> dict:
    """The policy as data — what a preregistration pins alongside the
    support. Separate from the support by ruling: the support stays
    exhaustive, and §3 forms are a declared, bounded extension to it."""
    return {
        "version": SYNTHETIC_EXTENSION_VERSION,
        "pins": list(SYNTHETIC_PINS),
        "manifests": {cls: SYNTHETIC_MANIFESTS[cls] for cls in STRING_CLASSES},
            # Sixth review, finding 4: `simplemind@synthetic/pin-0001` parsed,
        # validated, and emitted `synthetic:pin@simplemind` — a witness the
        # authoritative universe does not contain, because pin witnesses are
        # declared for OPERATORS only. The policy now states its authorized
        # hosts rather than leaving them implicit in the universe builder.
    "authorized_pin_hosts": "operators only (leaf atoms may not host pins)",
    "membership_rule": (
            "exact set membership over the committed finite sets above; a "
            "prefix test is NOT an allowlist and admitted traversal into "
            "real private paths"
        ),
    }


def synthetic_extension_sha256() -> str:
    return hashlib.sha256(
        json.dumps(synthetic_extension_policy(), sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()


class PolicyError(ValueError):
    """A grammar-valid expression that violates corpus policy; fail closed."""


def _spell(value) -> str:
    """Canonical literal spelling — identical to the canonical renderer's."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _string_class(value: str) -> str:
    """The §3.2 coverage class of a string literal."""
    if any(ord(ch) > 127 for ch in value):
        return "non_ascii"
    if json.dumps(value)[1:-1] != value:
        return "escaped"          # the canonical spelling had to escape something
    return "ascii"


def _digit_items(spelling: str) -> set:
    """Digit-byte witnesses carrying their POSITION CLASS.

    §7's digit-holdout split trains on a held-out digit, so a digit's
    position matters: `0` in the integer part of `10` and `0` in the
    fractional part of `0.02` exercise different tokenizer paths. A single
    `digit:0` item conflated them (fifth review, finding 4)."""
    mantissa, _, exponent = spelling.partition("e") if "e" in spelling else (
        spelling.partition("E") if "E" in spelling else (spelling, "", ""))
    integer, _, fraction = mantissa.partition(".")
    items = set()
    for part, where in ((integer, "int"), (fraction, "frac"), (exponent, "exp")):
        items.update(f"digit:{ch}@{where}" for ch in part if ch.isdigit())
    return items


def _leaf_identity_index() -> dict:
    """(output, modifier|None) -> the vocabulary's leaf identity string.

    Built by the same construction as `component_vocabulary` so the
    extractor cannot drift from the frozen support serialization."""
    terminals: dict = {}
    for name, sig in REGISTRY.items():
        if sig.atom and not sig.operator:
            terminals.setdefault((sig.output, None), set()).add(name)
            for modifier in sig.modifiers:
                terminals.setdefault((sig.output, modifier), set()).add(name)
    terminals[("score", "e5-atom")] = {"e5"}
    return {
        (output, modifier): (
            f"leaf:{output + '.' + modifier if modifier else output}"
            f"{{{','.join(sorted(atoms))}}}"
        )
        for (output, modifier), atoms in terminals.items()
    }


def _component_identity(node) -> str:
    """The node's resolved component identity — the same string the frozen
    vocabulary carries for operators, or the terminal-with-modifiers for a
    leaf. Pairs use it for BOTH endpoints (sixth review qualification: the
    parent must be the resolved component, not merely name/arity, or
    lineage(s,decay=0.6) and lineage(s,decay=0.6,estimand=..) are one
    holdout unit)."""
    sig = REGISTRY[node.name]
    if _is_leaf(node):
        return node.name + "".join("." + modifier for modifier in node.mods)
    typed = ",".join(
        f"{key}:{sig.kwargs[key].kind}" for key in sorted(_resolved_pattern(node))
    )
    shape = f"{node.name}/{len(node.args)}{{{typed}}}"
    return shape + "".join("." + modifier for modifier in node.mods)


def _motifs(node) -> set:
    """Holdable literal-kwarg and list-shape motifs.

    The encoder contract requires kwarg/list-shape holdouts alongside
    operator-composition holdouts, and neither was expressible: a pair
    recorded no literal kwarg at all, and `t=[0.02]` and `t=[0.02,0.03]`
    shared one component because the component records the KIND, not the
    length. These motif ids live alongside pairs in the holdable set, so a
    contract can hold them."""
    sig = REGISTRY[node.name]
    if _is_leaf(node):
        return set()
    out = set()
    component = _component_identity(node)
    # EXPLICIT kwargs only: a kwarg elided at its registered default is the
    # same canonical expression as its explicit spelling, so its motif would
    # hold every row rather than a distinguishable subset.
    for key, value in node.kwargs:
        kind = sig.kwargs[key].kind
        if isinstance(value, pc.Node):
            continue
        if kind in ("number_list", "int_list"):
            out.add(f"motif:{component}.kw:{key}:len{len(value)}")
        elif kind == "string":
            out.add(f"motif:{component}.kw:{key}:synthetic-string")
        else:
            out.add(f"motif:{component}.kw:{key}={_spell(value)}")
    return out


def _child_shape(node) -> str:
    """The child endpoint of a composition pair, as a SHAPE.

    Sixth review: `pair:blend/2.arg0|luna` dropped the modifier, so
    blend(luna.D, luna.S) and blend(luna.S, luna.D) produced identical pair
    sets — two structurally distinct compositions collapsed into one holdout
    unit. A leaf endpoint now carries its modifiers, and an operator
    endpoint its arity and resolved kwarg shape, so a pair determines both
    of its endpoints."""
    return _component_identity(node)


def _is_leaf(node) -> bool:
    return bool(REGISTRY[node.name].atom) and not node.args and not node.kwargs


def _resolved_pattern(node) -> frozenset:
    """The resolved kwarg key set: explicit non-string keys plus every
    defaulted key (a default is always present after canonical resolution).
    String kwargs are synthetic coverage, never part of the component."""
    sig = REGISTRY[node.name]
    keys = {key for key, _ in node.kwargs if sig.kwargs[key].kind != "string"}
    keys |= {key for key, spec in sig.kwargs.items() if spec.default is not None}
    return frozenset(keys)


def _row_witnesses(node, items: set, pairs: set, leaf_index: dict,
                   is_root: bool = True) -> None:
    sig = REGISTRY[node.name]
    pairs |= _motifs(node)          # holdable kwarg/list-shape motifs
    for pin in node.pins:
        if not sig.operator:
            raise PolicyError(
                f"pin {pin!r} hosted on the leaf atom {node.name!r}: "
                f"{SYNTHETIC_EXTENSION_VERSION} authorizes operators only, and "
                "the authoritative universe carries no leaf pin witness"
            )
        if pin not in SYNTHETIC_PINS:
            raise PolicyError(
                f"pin {pin!r} is not a member of the committed §3.3 synthetic "
                f"pin set ({SYNTHETIC_EXTENSION_VERSION}) — a prefix test is "
                "not an allowlist and admitted traversal into real paths"
            )
        # Pins are the only thing separating V2 from V3 (§3.1), so the pin's
        # HOST matters: pins on distinct nodes exercise distinct render paths.
        items.add(f"synthetic:pin@{node.name}")
    if not is_root:
        present = {key for key, _ in node.kwargs} & {"estimand", "impl"}
        for key in sorted(present):
            # Interior methodology is sampler coverage under the ruled
            # scenario (§2.5 excluded_synthetic). It was NAMED as excluded
            # but had no witness item, so nothing required the corpus to
            # contain any (fifth review, finding 4).
            items.add(f"synthetic:interior_methodology@{node.name}.{key}")
    if _is_leaf(node):
        items.add(f"terminal:{node.name}")
        shapes = (
            [("score", "e5-atom")] if node.name == "e5"
            else [(sig.output, modifier) for modifier in node.mods]
            or [(sig.output, None)]
        )
        for shape in shapes:
            items.add(leaf_index[shape])
        return
    arity = len(node.args)
    typed = ",".join(
        f"{key}:{sig.kwargs[key].kind}" for key in sorted(_resolved_pattern(node))
    )
    items.add(f"op:{node.name}/{arity}{{{typed}}}")
    for index, child in enumerate(node.args):
        declared = (
            sig.arg_types[index]
            if index < len(sig.arg_types)
            else sig.variadic_arg_type
        )
        if isinstance(child, pc.Node):
            child_output = REGISTRY[child.name].output
            items.add(f"edge:{node.name}/{arity}.arg{index}->{child_output}")
            # The pair carries its SLOT. Without arity and position,
            # product(hop_decay(..), X) and product(X, hop_decay(..)) shared
            # one pair, so a composition holdout could not distinguish them
            # (fifth review, finding 3).
            pairs.add(f"pair:{_component_identity(node)}.arg{index}"
                      f"|{_child_shape(child)}")
            _row_witnesses(child, items, pairs, leaf_index, False)
        else:
            kinds = [a for a in declared.split("|") if a in pc.VALUE_KINDS]
            (kind,) = kinds  # validate() admitted the literal, so exactly one
            items.add(f"slot:{node.name}/{arity}.arg{index}:{kind}")
            items.add(f"value:{kind}:{_spell(child)}")
            items.update(_digit_items(_spell(child)))
    for key, value in node.kwargs:
        kind = sig.kwargs[key].kind
        if isinstance(value, pc.Node):
            items.add(f"edge:{node.name}.kw:{key}->{kind}")
            pairs.add(f"pair:{_component_identity(node)}.kw:{key}"
                      f"|{_child_shape(value)}")
            _row_witnesses(value, items, pairs, leaf_index, False)
        elif kind in ("estimand", "impl", "walk", "weight"):
            items.add(f"{kind}:{value}")
        elif kind == "string":
            if value not in SYNTHETIC_MANIFESTS.values():
                raise PolicyError(
                    f"string {value!r} is not a member of the committed §3.3 "
                    f"synthetic manifest set ({SYNTHETIC_EXTENSION_VERSION})"
                )
            items.add(f"synthetic:string:{_string_class(value)}")
        elif kind in ("number_list", "int_list"):
            element_kind = "number" if kind == "number_list" else "int"
            for element in value:
                items.add(f"value:{element_kind}:{_spell(element)}")
                items.update(_digit_items(_spell(element)))
        else:
            items.add(f"value:{kind}:{_spell(value)}")
            items.update(_digit_items(_spell(value)))
    # Resolved defaults witness their VALUES (seventh review, finding 2). An
    # elided default is canonically identical to its explicit spelling, and
    # the component identity already resolves it — so lineage(simplemind)
    # semantically carries decay=0.85 and must witness value:number:0.85 and
    # its digits, or the universe's grid coverage is unreachable through
    # rows spelled canonically. No MOTIF is emitted (the motif would be
    # universal — that rationale stands for holdouts); the VALUE is counted.
    explicit_keys = {key for key, _ in node.kwargs}
    for key in sorted(sig.kwargs):
        spec = sig.kwargs[key]
        if key in explicit_keys or spec.default is None:
            continue
        if spec.kind in ("number", "int"):
            items.add(f"value:{spec.kind}:{_spell(spec.default)}")
            items.update(_digit_items(_spell(spec.default)))
        elif spec.kind in ("estimand", "impl", "walk", "weight"):
            items.add(f"{spec.kind}:{spec.default}")


def _template_identity(node) -> str:
    """The resolved-kwarg structural template — the family unit.

    Leaves collapse to their SHAPE (the template convention `_leaf_dist`
    counts: `luna` and `haiku` share `leaf:judge`); literals collapse to
    typed placeholders with lists keeping their length; defaulted kwargs are
    resolved present; strings and pins are stripped (synthetic projection:
    they never create families)."""
    sig = REGISTRY[node.name]
    if _is_leaf(node):
        if node.name == "e5":
            return "leaf:score.e5-atom"
        shape = sig.output + "".join("." + modifier for modifier in node.mods)
        return f"leaf:{shape}"
    arity = len(node.args)
    parts = []
    for child in node.args:
        if isinstance(child, pc.Node):
            parts.append(_template_identity(child))
        else:
            index = len(parts)
            declared = (
                sig.arg_types[index]
                if index < len(sig.arg_types)
                else sig.variadic_arg_type
            )
            (kind,) = [a for a in declared.split("|") if a in pc.VALUE_KINDS]
            parts.append(f"<{kind}>")
    explicit = dict(node.kwargs)
    for key in sorted(_resolved_pattern(node)):
        kind = sig.kwargs[key].kind
        value = explicit.get(key, REGISTRY[node.name].kwargs[key].default)
        if isinstance(value, pc.Node):
            parts.append(f"{key}={_template_identity(value)}")
        elif kind in ("number_list", "int_list"):
            parts.append(f"{key}=<{kind}*{len(value)}>")
        else:
            parts.append(f"{key}=<{kind}>")
    rendered = f"{node.name}/{arity}({','.join(parts)})"
    return rendered + "".join("." + modifier for modifier in node.mods)


def _grid_violation(node, path: str = "") -> str | None:
    """Grid/cap check that TOLERATES the §3 synthetic extensions.

    `covers()` answers "is this inside the enumerable support" and therefore
    refuses string kwargs and interior methodology — but §3 makes exactly
    those MANDATORY corpus content. Corpus policy is the support's caps and
    numeric grids, extended by the declared synthetic forms; conflating the
    two would make §3 unsatisfiable."""
    signature = REGISTRY[node.name]
    where = path or node.name
    for index, child in enumerate(node.args):
        if isinstance(child, pc.Node):
            inner = _grid_violation(child, f"{where}.arg{index}")
            if inner:
                return inner
            continue
        declared = (
            signature.arg_types[index]
            if index < len(signature.arg_types)
            else signature.variadic_arg_type
        )
        kinds = [alt for alt in declared.split("|") if alt in pc.VALUE_KINDS]
        if len(kinds) != 1:
            return f"{where}.arg{index}: ambiguous declared value kind"
        grid = NUMBER_GRID if kinds[0] == "number" else (
            INT_GRID if kinds[0] == "int" else ())
        if child not in grid:
            return f"{where}.arg{index}: literal {child!r} is off the {kinds[0]} grid"
    for key, value in node.kwargs:
        kind = signature.kwargs[key].kind
        if isinstance(value, pc.Node):
            inner = _grid_violation(value, f"{where}.kw:{key}")
            if inner:
                return inner
        elif kind in ("estimand", "impl", "walk", "weight", "string"):
            continue                      # enumerated domains / §3.2 synthetics
        elif kind in ("number_list", "int_list"):
            grid = NUMBER_GRID if kind == "number_list" else INT_GRID
            if len(value) > MAX_LIST_LENGTH:
                return f"{where}.kw:{key}: list longer than the cap"
            if any(item not in grid for item in value):
                return f"{where}.kw:{key}: list value off the {kind} grid"
        elif value not in (NUMBER_GRID if kind == "number" else INT_GRID):
            return f"{where}.kw:{key}: literal {value!r} is off the {kind} grid"
    return None


def corpus_policy_violation(node, scenario: str) -> str | None:
    """None when the row is admissible corpus content; else why it is not.

    Enforced: the declared caps and value grids. Tolerated because §3 makes
    them mandatory: pins, allowlisted string kwargs, and interior
    methodology under the ruled scenario (the allowlist itself is enforced
    in `_row_witnesses`, which fails closed on a non-synthetic literal)."""
    if scenario not in SCENARIOS:
        raise PolicyError(f"unknown scenario {scenario!r}")
    if _depth(node) > MAX_DEPTH:
        return f"depth {_depth(node)} exceeds the cap {MAX_DEPTH}"
    if _node_count(node) > MAX_NODE_COUNT:
        return f"node count {_node_count(node)} exceeds the cap {MAX_NODE_COUNT}"
    if _max_arity(node) > MAX_ARITY:
        return f"arity {_max_arity(node)} exceeds the cap {MAX_ARITY}"
    if _max_kwargs(node) > MAX_KWARGS_NODE:
        return f"kwargs/node {_max_kwargs(node)} exceeds the cap {MAX_KWARGS_NODE}"
    return _grid_violation(node)


def families_from_expressions(expressions,
                              scenario: str = "methodology-root-only",
                              authorizing: bool = True,
                              return_manifest: bool = False):
    """Group real v0.4 expression strings into split `Family` records.

    Deterministic: parse with the sealed registry, validate, ENFORCE CORPUS
    POLICY, project each row into its SEMANTIC template's family (pins and
    strings contribute synthetic items but never families), and count each
    witness item once per row that witnesses it.

    Policy enforcement is not the same as grammar validation, and conflating
    them was the fifth review's second finding: `max(10,e5)` parses and
    validates but sits outside the declared support, so admitting it would
    let witness items appear that the frozen vocabulary does not contain.
    Rows outside the scenario's caps, grids, or methodology placement are
    refused here, as are pins and strings outside the §3.3 allowlist.
    """
    leaf_index = _leaf_identity_index()
    grouped: dict = {}
    seen: set = set()
    seen_semantic: set = set()
    duplicates = 0
    universe = set(required_witness_universe())
    for expression in expressions:
        node = pc.parse(expression)
        pc.validate(node)
        violation = corpus_policy_violation(node, scenario)
        if violation:
            raise PolicyError(f"{expression!r} violates corpus policy: {violation}")
        # Sixth review, finding 5: duplicate canonical ASTs inflated
        # coverage — two identical inputs counted as two witnesses, so `k`
        # could be satisfied by copies of one example, and the split relied
        # on an UNENFORCED upstream dedup assumption. The row identity is
        # `canonical_full`, not `canonical_semantic`: rows differing only by
        # pins are genuinely distinct rows (§3.1 — V3 differs from V2 only
        # by pins), so semantic dedup would delete required coverage.
        identity = pc.canonical_full(node)
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        family_id = "tmpl:" + _template_identity(node)
        items: set = set()
        pairs: set = set()
        _row_witnesses(node, items, pairs, leaf_index)
        # Seventh review, finding 1: pin-copies of one SEMANTIC example are
        # distinct corpus rows (canonical_full dedup is ruled correct — §3.1
        # needs them), but they must not credit semantic witnesses twice
        # toward k, or the coverage minimum can be met with pinned copies of
        # one example. Semantic items count once per distinct semantic
        # identity; synthetic items (the pins' own coverage) count per row.
        semantic_identity = pc.canonical_semantic(node)
        if semantic_identity in seen_semantic:
            items = {item for item in items if item.startswith("synthetic:")}
        seen_semantic.add(semantic_identity)
        # Sixth review, finding 4: checking that required witnesses are
        # PRESENT is not the same as checking that no witness escapes. An
        # authorizing build refuses any extracted item outside the
        # authoritative universe, so a synthetic form the universe does not
        # declare cannot enter the corpus.
        if authorizing:
            escaped = sorted(items - universe)
            if escaped:
                raise PolicyError(
                    f"{expression!r} emits witnesses outside the authoritative "
                    f"universe: {escaped}"
                )
        counts, family_pairs, rows = grouped.setdefault(family_id, ({}, set(), [0]))
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        family_pairs |= pairs
        rows[0] += 1
    families = [
        _split.Family.make(family_id, counts, family_pairs, rows[0])
        for family_id, (counts, family_pairs, rows) in sorted(grouped.items())
    ]
    if return_manifest:
        return families, {
            "rows_in": len(expressions),
            "rows_kept": len(seen),
            "duplicates_removed": duplicates,
            "families": len(families),
            "dedup_key": "canonical_full",
            "scenario": scenario,
            "authorizing": authorizing,
        }
    return families


def required_witness_universe() -> list:
    """The authoritative required item list — derived from the serialized
    vocabulary, the grids, and the registry's categorical domains; never
    from any corpus. Its hash is bound by both the enumeration spec and the
    split contract, so an extractor omission cannot vanish silently: a
    required item no family witnesses fails the split closed."""
    vocabulary = component_vocabulary()
    universe: set = set()
    for section in ("leaves", "operators_interior", "operators_root_only",
                    "node_edges", "literal_slots"):
        universe.update(vocabulary[section])
    for value in NUMBER_GRID:
        universe.add(f"value:number:{_spell(value)}")
        universe.update(_digit_items(_spell(value)))
    for value in INT_GRID:
        universe.add(f"value:int:{_spell(value)}")
        universe.update(_digit_items(_spell(value)))
    for name, sig in REGISTRY.items():
        if sig.atom:
            universe.add(f"terminal:{name}")
    for estimand in pc.ESTIMANDS:
        universe.add(f"estimand:{estimand}")
    for impl in pc.IMPLS:
        universe.add(f"impl:{impl}")
    for walk in pc.WALKS:
        universe.add(f"walk:{walk}")
    for weight in pc.WEIGHTS:
        universe.add(f"weight:{weight}")
    # §3 synthetic floors, specified rather than gestured at (fifth review,
    # finding 4). Pins are required on every operator that can host one,
    # because §3.1's V2-vs-V3 collapse is a per-render-path property, not a
    # global one; strings require all three §3.2 classes; interior
    # methodology needs a witness per (operator, methodology kwarg) it can
    # legally carry, or nothing obliges the sampler to emit any.
    for name, sig in REGISTRY.items():
        if sig.operator:
            universe.add(f"synthetic:pin@{name}")
            for key in ("estimand", "impl"):
                if key in sig.kwargs:
                    universe.add(f"synthetic:interior_methodology@{name}.{key}")
    universe.update(f"synthetic:string:{cls}" for cls in STRING_CLASSES)
    return sorted(universe)


def required_witness_universe_sha256() -> str:
    return _split.universe_sha256(required_witness_universe())


def main(argv=None) -> int:
    print(f"enumeration {ENUMERATION_VERSION} over registry {pc.REGISTRY_VERSION}")
    print(f"spec sha {enumeration_spec_sha256()[:16]}…")
    print(f"caps: depth {MAX_DEPTH}, nodes {MAX_NODE_COUNT}, arity {MAX_ARITY}, "
          f"kwargs/node {MAX_KWARGS_NODE}, list {MAX_LIST_LENGTH}")
    print(f"grids: number {NUMBER_GRID}, int {INT_GRID}; strings and pins excluded")
    for scenario in SCENARIOS:
        expressions, by_nodes = count(scenario)
        templates, _ = count(scenario, template_mode=True)
        print(f"  {scenario:22s} expressions {expressions:>16,}  templates {templates:>10,}")
        print(f"  {'':22s} by node count {by_nodes}")
    counts = component_vocabulary_counts()
    print("component vocabulary (§2.5 — the composable support, serialized):")
    for key, value in counts.items():
        print(f"  {key:40s} {value}")
    print(f"  vocabulary sha {component_vocabulary_sha256()[:16]}…")
    universe = required_witness_universe()
    print(f"required witness universe: {len(universe)} items, "
          f"sha {required_witness_universe_sha256()[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
