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
    """Content witness binding version, caps, and grids for preregistration."""

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
        count = _value_count(spec.kind, template_mode)
        if not template_mode and spec.default is not None:
            if spec.kind == "number" and spec.default in NUMBER_GRID:
                count -= 1  # explicit default == elided form: one identity
            elif spec.kind == "int" and spec.default in INT_GRID:
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
    registry-declared kind — `estimand`/`impl` are enumerations (validate
    already holds them to their sets), `string` fields are excluded from
    enumeration entirely (§3.2), and numeric kinds must sit on the grid."""
    signature = REGISTRY[node.name]
    for child in node.args:
        if isinstance(child, pc.Node):
            if not _values_in_grid(child):
                return False
        elif isinstance(child, float):
            if child not in NUMBER_GRID:
                return False
        elif isinstance(child, int) and not isinstance(child, bool):
            if child not in NUMBER_GRID + INT_GRID:
                return False
    for key, value in node.kwargs:
        kind = signature.kwargs[key].kind
        if isinstance(value, pc.Node):
            if not _values_in_grid(value):
                return False
        elif kind in ("estimand", "impl"):
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


def component_vocabulary() -> dict:
    """The composable support of §2.5: operator-local shapes and typed leaves.

    The owner's ruling on the corpus posture: templates must be general,
    composable components — not complete-tree skeletons. A component is one
    operator with one arity and one kwarg-presence pattern (or one typed leaf
    shape); complete trees are compositions of components through the type
    system, and the legal composition edges are part of the frozen support.
    """

    leaves: set = set()
    for name, sig in REGISTRY.items():
        if sig.atom and not sig.operator:
            leaves.add((sig.output, None))
            for modifier in sig.modifiers:
                leaves.add((sig.output, modifier))
    leaves.add(("score", "e5-atom"))  # the dual atom is its own leaf shape

    def kwarg_patterns(name: str, sig, allow_methodology: bool) -> set:
        usable = []
        for key in sorted(sig.kwargs):
            spec = sig.kwargs[key]
            if key in ("estimand", "impl") and not allow_methodology:
                continue
            if spec.kind == "string":
                continue
            usable.append(key)
        patterns: set = set()
        for r in range(0, MAX_KWARGS_NODE + 1):
            for subset in itertools.combinations(usable, r):
                chosen = set(subset)
                if any(
                    spec.required and key not in chosen
                    for key, spec in sig.kwargs.items()
                ):
                    continue
                if name == "routing" and (("t" in chosen) != ("menus" in chosen)):
                    continue
                patterns.add(frozenset(chosen))
        return patterns

    interior: set = set()
    with_root: set = set()
    edges = 0
    for name, sig in REGISTRY.items():
        if not sig.operator:
            continue
        cap = sig.max_args if sig.max_args is not None else MAX_ARITY
        for arity in range(sig.min_args, min(cap, MAX_ARITY) + 1):
            for pattern in kwarg_patterns(name, sig, False):
                interior.add((name, arity, pattern))
            for pattern in kwarg_patterns(name, sig, True):
                with_root.add((name, arity, pattern))
            for index in range(arity):
                declared = (
                    sig.arg_types[index]
                    if index < len(sig.arg_types)
                    else sig.variadic_arg_type
                )
                for alternative in declared.split("|"):
                    if alternative == "process":
                        edges += len(OUTPUT_ROOTS)
                    elif alternative in pc.OUTPUT_TYPES or alternative in pc.VALUE_KINDS:
                        edges += 1
        for key in sorted(sig.kwargs):
            if sig.kwargs[key].kind in pc.OUTPUT_TYPES:
                edges += 1  # node-valued kwarg edge (mu -> judge)

    return {
        "leaf_shapes": len(leaves),
        "operator_shapes_interior": len(interior),
        "operator_shapes_with_root_methodology": len(with_root),
        "composition_edges": edges,
        "vocabulary_total": len(leaves) + len(with_root),
    }


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
    vocabulary = component_vocabulary()
    print("component vocabulary (§2.5 — the composable support):")
    for key, value in vocabulary.items():
        print(f"  {key:40s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
