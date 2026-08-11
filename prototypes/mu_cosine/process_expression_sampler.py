#!/usr/bin/env python3
"""Coverage-directed sampler — §2.6's provisional figures, measured.

§2.6 rules the coverage minimum as `pair-k = 100` and marks its row
figures **provisional**, because they came from a scratch sampler that was
never committed and no test reproduced. This module is that sampler, and
the numbers it prints are reproducible from sealed inputs.

WHAT IS DIRECTED, AND WHY IT MATTERS
------------------------------------
A uniform sampler pays for the tail twice: rare compositions arrive by
luck, so reaching `k` on the rarest pair means over-covering everything
else. A coverage-directed sampler emits, at each step, a row chosen to
advance the currently least-covered holdable identity. §2.6 predicted this
reaches the same `k` with fewer rows and named 200,000 an UPPER bound; the
measurement either confirms that or contradicts it, and both are results.

THE COUNTING GAP THIS MODULE MAKES VISIBLE
------------------------------------------
`Family.pairs` is a SET: the split contract counts witness ITEMS per row
and records pairs only as holdable identities. So the ruled minimum —
which is over PAIRS — is not a quantity the split can check. This module
counts pair coverage explicitly and reports both, so the ruling is
measurable while that gap stands. Closing it is a split-contract change
(pairs would enter the counted universe) and is deliberately NOT done
here: the split contract is sealed, and a sampler may not amend it.

REALIZABILITY IS PROVED, NEVER ASSUMED
--------------------------------------
The pair universe is derived from the grammar — every type-legal
(parent component, slot, child component) triple — and then each
candidate must be REALIZED by construction: a row is built and put
through the real `corpus_policy_violation` and `validate`. A candidate
that no row can carry is REFUSED WITH A REASON rather than dropped, so
the universe cannot shrink silently through a generator bug. The refusal
list is part of the manifest.

WITNESSES COME FROM THE EXTRACTOR, NOT FROM A SECOND OPINION
------------------------------------------------------------
Coverage is computed with `_row_witnesses` — the extractor's own
function, not a reimplementation — and `verify_against_extractor()`
re-measures a finished sample through the public
`families_from_expressions` path and fails closed on any disagreement.
A sampler that scored itself would be the author-is-verifier shape the
lane refuses everywhere else.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Iterable, Mapping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import process_cards as pc
import process_expression_enumerator as en

SAMPLER_VERSION = "sampler-v1"

#: Every rule the sampler executes, machine-readable and hashed into the
#: measurement report — the same discipline the split algorithm manifest
#: established (binding constants while the behavior floats is not a
#: preregistration).
SAMPLER_MANIFEST = {
    "version": SAMPLER_VERSION,
    "objective_rule": (
        "emit rows until every REQUIRED WITNESS ITEM and every REALIZABLE "
        "COMPOSITION PAIR is witnessed by at least coverage_minimum_k "
        "DISTINCT rows; row distinctness is canonical_full, the extractor's "
        "own dedup key, so k identical rows credit coverage once"
    ),
    "direction_rule": (
        "at each step the sampler targets the lexicographically smallest "
        "identity among those with the fewest witnessing rows so far, and "
        "emits the next unused row from that identity's realization stream; "
        "ties broken lexicographically so the sampler is deterministic "
        "without a random seed"
    ),
    "pair_universe_rule": (
        "candidate pairs are every type-legal (parent component, slot, "
        "child component) triple derivable from the registry under the "
        "scenario's flags; a candidate enters the required universe only "
        "when a row realizing it PASSES validate() and "
        "corpus_policy_violation(); a candidate no row can carry is "
        "recorded in refused_pairs with its reason and is not silently "
        "dropped"
    ),
    "root_placement_rule": (
        "a pair's witness row places the PARENT at the root. Under "
        "methodology-root-only this is not a convenience: a component "
        "carrying estimand/impl is legal only at the root, so rooting the "
        "parent is the placement that maximizes the realizable set"
    ),
    "variation_rule": (
        "distinct rows for one pair are produced by varying, in a fixed "
        "order: literal slot values over the sealed grids, list lengths and "
        "contents, sibling-slot fillers, the child's own literals, and the "
        "§3 synthetic extensions (pins, manifest strings). Variation never "
        "changes the parent's resolved kwarg pattern, because that would "
        "change the component and therefore the pair being witnessed"
    ),
    "witness_source_rule": (
        "coverage is computed with the extractor's own _row_witnesses; a "
        "finished sample is re-measured through families_from_expressions "
        "and any disagreement fails closed"
    ),
}


#: How many literal bindings a candidate pair is offered before it is
#: refused. The grids are small and the binding order is fixed, so this
#: bounds the search without making refusal depend on luck.
_REALIZATION_ATTEMPTS = 24

#: Recursion guard for the minimal-filler fallback: a type with no leaf is
#: filled by the smallest operator producing it, and that operator's own
#: slots must not recurse without bound.
_depth_guard = [0]

#: Fixpoint bound for the item phase — a bound, not a tuning knob: the
#: loop exits as soon as a pass admits nothing new.
_ITEM_PHASE_PASSES = 8

#: Kinds whose VALUE is itself a required witness item. Varying one of these
#: changes which item a row covers, so streams hold them fixed.
_IDENTITY_KINDS = frozenset({"estimand", "impl", "walk", "weight", "string"})


class SamplerError(ValueError):
    """The sampler cannot satisfy its objective; failing closed."""


def sampler_manifest_sha256() -> str:
    return hashlib.sha256(
        json.dumps(SAMPLER_MANIFEST, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Grammar walking: protos, components, and realization
# ---------------------------------------------------------------------------

def _leaf_candidates() -> list:
    """Every leaf node the grammar admits, deterministically ordered."""
    out = []
    for name, sig in sorted(pc.REGISTRY.items()):
        if not sig.atom or sig.operator:
            continue
        out.append(pc.Node(name))
        for modifier in sorted(sig.modifiers):
            out.append(pc.Node(name, mods=(modifier,)))
    # `e5` is an atom AND an operator, so the operator filter above drops it —
    # and it is the ONLY leaf of output `score`, so dropping it made every
    # score-typed slot look unfillable. The frozen vocabulary carries it
    # explicitly as the `e5-atom` leaf shape, and so must this.
    out.append(pc.Node("e5"))
    return out


def _literal_values(kind: str) -> list:
    """The sealed grid for a literal kind, in a fixed order."""
    if kind == "number":
        return list(en.NUMBER_GRID)
    if kind == "int":
        return list(en.INT_GRID)
    if kind == "number_list":
        return ([(v,) for v in en.NUMBER_GRID]
                + [(a, b) for a in en.NUMBER_GRID for b in en.NUMBER_GRID])
    if kind == "int_list":
        return ([(v,) for v in en.INT_GRID]
                + [(a, b) for a in en.INT_GRID for b in en.INT_GRID])
    if kind == "estimand":
        return sorted(pc.ESTIMANDS)
    if kind == "impl":
        return sorted(pc.IMPLS)
    if kind == "walk":
        return sorted(pc.WALKS)
    if kind == "weight":
        return sorted(pc.WEIGHTS)
    if kind == "string":
        return [en.SYNTHETIC_MANIFESTS[cls] for cls in en.STRING_CLASSES]
    return []


def _minimal_filler(output: str) -> pc.Node | None:
    """The cheapest node of an output type — a leaf where one exists.

    Sibling slots must be filled with something, and filling them with the
    smallest legal node is what leaves node budget for the pair's own
    endpoints. A type with no leaf has no minimal filler and its pairs are
    refused with that reason rather than silently skipped.
    """
    for leaf in _leaf_candidates():
        if pc.REGISTRY[leaf.name].output == output and not leaf.mods:
            return leaf
    for leaf in _leaf_candidates():
        if pc.REGISTRY[leaf.name].output == output:
            return leaf
    # No leaf of this output exists — `pick` takes a target-set, which only
    # operators produce. Without an operator fallback such a slot is
    # unfillable and every item reachable only through it (its pin host,
    # among others) starves for a reason that is not a grammar fact.
    if _depth_guard[0] >= 2:
        return None
    _depth_guard[0] += 1
    try:
        for name, sig in sorted(pc.REGISTRY.items()):
            if not sig.operator or sig.output != output:
                continue
            for arity in _arities(sig):
                for explicit in _explicit_key_sets(sig, False):
                    for attempt in range(_REALIZATION_ATTEMPTS):
                        node = _build_parent(name, arity, explicit, None,
                                             None, attempt)
                        if node is None:
                            continue
                        try:
                            pc.validate(node)
                        except pc.ParseError:
                            continue
                        return node
    finally:
        _depth_guard[0] -= 1
    return None


def _arities(sig) -> range:
    cap = sig.max_args if sig.max_args is not None else en.MAX_ARITY
    return range(sig.min_args, min(cap, en.MAX_ARITY) + 1)


def _explicit_key_sets(sig, allow_methodology: bool) -> list:
    """Every explicit-kwarg key set a node of this signature may carry.

    String kwargs are excluded: `_resolved_pattern` drops them, so they do
    not change the component and therefore do not multiply pairs.
    """
    keys = [
        key for key, spec in sorted(sig.kwargs.items())
        if spec.kind != "string"
        and (allow_methodology or key not in ("estimand", "impl"))
    ]
    required = {key for key, spec in sig.kwargs.items() if spec.required}
    if any(key not in keys for key in required):
        return []
    out = []
    for mask in range(1 << len(keys)):
        chosen = tuple(key for index, key in enumerate(keys) if mask >> index & 1)
        if len(chosen) > en.MAX_KWARGS_NODE:
            continue
        if any(key not in chosen for key in required):
            continue
        out.append(chosen)
    return out


def _kwarg_binding(sig, key: str, variant: int):
    """A value for one kwarg — a literal from the grid, or a minimal node
    for a node-valued kwarg such as `mu=`."""
    spec = sig.kwargs[key]
    if spec.kind in pc.OUTPUT_TYPES:
        return _minimal_filler(spec.kind)
    values = _literal_values(spec.kind)
    if not values:
        return None
    return values[variant % len(values)]


def _build_parent(name: str, arity: int, explicit: tuple, child_slot,
                  child: pc.Node, variant: int = 0,
                  _fail: list | None = None) -> pc.Node | None:
    """Assemble a root row: `name` with `child` in `child_slot`, siblings
    filled minimally, and every explicit kwarg bound.

    `child_slot` is an int (positional) or ('kw', key).
    """
    sig = pc.REGISTRY[name]
    if _fail is None:
        _fail = []
    args: list = []
    for index in range(arity):
        declared = (sig.arg_types[index] if index < len(sig.arg_types)
                    else sig.variadic_arg_type)
        if child_slot == index:
            args.append(child)
            continue
        alternatives = declared.split("|")
        filler = None
        _fail.append(f"slot arg{index} of type {declared!r} has no filler")
        for alternative in alternatives:
            if alternative in pc.VALUE_KINDS:
                values = _literal_values(alternative)
                if values:
                    filler = values[variant % len(values)]
                    break
            elif alternative == "process":
                for output in en.OUTPUT_ROOTS:
                    filler = _minimal_filler(output)
                    if filler is not None:
                        break
                if filler is not None:
                    break
            else:
                filler = _minimal_filler(alternative)
                if filler is not None:
                    break
        if filler is None:
            return None
        _fail.pop()
        args.append(filler)
    kwargs: list = []
    for key in explicit:
        if isinstance(child_slot, tuple) and child_slot[1] == key:
            kwargs.append((key, child))
            continue
        value = _kwarg_binding(sig, key, variant)
        if value is None:
            _fail.append(f"kwarg {key!r} of kind "
                         f"{sig.kwargs[key].kind!r} has no binding")
            return None
        kwargs.append((key, value))
    kwargs.sort(key=lambda item: item[0])
    return pc.Node(name, tuple(args), tuple(kwargs))


def _admissible(node: pc.Node, scenario: str) -> str | None:
    """The real gate: grammar validation plus corpus policy. Returns a
    refusal reason, or None when the row is admissible."""
    try:
        pc.validate(node)
    except pc.ParseError as exc:
        return f"invalid: {exc}"
    if pc.REGISTRY[node.name].output not in en.OUTPUT_ROOTS:
        return f"output {pc.REGISTRY[node.name].output!r} is not a root type"
    violation = en.corpus_policy_violation(node, scenario)
    if violation:
        return f"policy: {violation}"
    return None


def _child_candidates(declared: str, scenario: str) -> list:
    """Nodes that may fill a slot of the declared type — leaves, plus
    interior operator shapes when the scenario admits them."""
    flags = en.SCENARIOS[scenario]
    outputs = (list(en.OUTPUT_ROOTS) if declared == "process"
               else [alt for alt in declared.split("|")
                     if alt not in pc.VALUE_KINDS])
    out = [leaf for leaf in _leaf_candidates()
           if pc.REGISTRY[leaf.name].output in outputs]
    for name, sig in sorted(pc.REGISTRY.items()):
        if not sig.operator or sig.output not in outputs:
            continue
        for arity in _arities(sig):
            for explicit in _explicit_key_sets(
                    sig, flags["methodology_interior"]):
                node = _build_parent(name, arity, explicit, None, None)
                if node is None:
                    continue
                try:
                    pc.validate(node)
                except pc.ParseError:
                    continue
                out.append(node)
    return out


def _slots(sig, arity: int, explicit: tuple) -> list:
    """Node-valued slots of a proto: positional slots that admit a node,
    plus explicit kwargs whose kind is an output type."""
    out: list = []
    for index in range(arity):
        declared = (sig.arg_types[index] if index < len(sig.arg_types)
                    else sig.variadic_arg_type)
        if any(alt not in pc.VALUE_KINDS for alt in declared.split("|")):
            out.append((index, declared))
    for key in explicit:
        if sig.kwargs[key].kind in pc.OUTPUT_TYPES:
            out.append((("kw", key), sig.kwargs[key].kind))
    return out


def pair_realizations(scenario: str = "methodology-root-only") -> tuple:
    """(realized, refused): the pair universe, proved by construction.

    `realized` maps each pair id to (witness row, the slot the child
    occupies); `refused` maps a candidate description to the reason no row
    carries it. The slot travels with the row because variation must leave
    the pair's own endpoint alone.
    """
    flags = en.SCENARIOS[scenario]
    leaf_index = en._leaf_identity_index()
    realized: dict = {}
    refused: dict = {}
    for name, sig in sorted(pc.REGISTRY.items()):
        if not sig.operator:
            continue
        for arity in _arities(sig):
            for explicit in _explicit_key_sets(sig, flags["methodology_root"]):
                for slot, declared in _slots(sig, arity, explicit):
                    for child in _child_candidates(declared, scenario):
                        # A candidate is refused only when NO binding of its
                        # free literals is admissible. Committing to the first
                        # binding would report generator defects as grammar
                        # facts — `blend.w` needs one weight per positional
                        # source, so a fixed-length list refuses a pair the
                        # grammar admits.
                        row, reason = None, "no admissible binding found"
                        for attempt in range(_REALIZATION_ATTEMPTS):
                            failure: list = []
                            candidate = _build_parent(name, arity, explicit,
                                                      slot, child, attempt,
                                                      failure)
                            if candidate is None:
                                if failure:
                                    reason = f"unconstructible: {failure[-1]}"
                                continue
                            verdict = _admissible(candidate, scenario)
                            if verdict is None:
                                row, reason = candidate, None
                                break
                            reason = verdict
                        if row is None:
                            expression = None
                        else:
                            expression = pc.canonical_full(row)
                        items: set = set()
                        pairs: set = set()
                        if reason is None:
                            en._row_witnesses(row, items, pairs, leaf_index)
                        slot_id = (f"arg{slot}" if isinstance(slot, int)
                                   else f"kw:{slot[1]}")
                        if reason is not None:
                            # No admissible row exists, so no resolved parent
                            # component exists either — the candidate is named
                            # by its PROTO, which is what the grammar offered.
                            typed = ",".join(sorted(explicit))
                            refused.setdefault(
                                f"candidate:{name}/{arity}{{{typed}}}."
                                f"{slot_id}|{en._component_identity(child)}",
                                reason)
                            continue
                        parent_component = en._component_identity(row)
                        pair_id = (f"pair:{parent_component}.{slot_id}"
                                   f"|{en._component_identity(child)}")
                        if pair_id not in pairs:
                            # The constructed row does not actually carry the
                            # pair it was built for — a generator defect, not
                            # a grammar fact. Fail closed rather than record a
                            # universe entry nothing witnesses.
                            raise SamplerError(
                                f"row {expression!r} was built to witness "
                                f"{pair_id!r} but witnesses {sorted(pairs)!r}"
                            )
                        realized.setdefault(pair_id, (expression, slot))
                        refused.pop(pair_id, None)
    return realized, dict(sorted(refused.items()))


def required_pair_universe(scenario: str = "methodology-root-only") -> list:
    realized, _ = pair_realizations(scenario)
    return sorted(realized)


def required_pair_universe_sha256(
        scenario: str = "methodology-root-only") -> str:
    return hashlib.sha256(
        json.dumps(required_pair_universe(scenario),
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Variation: distinct rows for one identity
# ---------------------------------------------------------------------------

def _sibling_alternatives(output: str) -> list:
    """Leaves of one output type, in a fixed order — the distinctness axis
    for a row whose slots carry no literals at all. `blend/2{}` has two
    node slots and no kwargs, so without sibling rotation its only variation
    is pins, and it starves long before k."""
    return [leaf for leaf in _leaf_candidates()
            if pc.REGISTRY[leaf.name].output == output]


def _vary(row: pc.Node, variant: int, protect=None,
          freeze: bool = False, hold_values: bool = False) -> pc.Node:
    """A structurally identical row with different literals and synthetic
    extensions. The parent's resolved kwarg pattern is untouched — changing
    it would change the component, and therefore the pair."""
    sig = pc.REGISTRY[row.name]
    args = []
    for index, child in enumerate(row.args):
        if isinstance(child, pc.Node):
            if index == protect:
                # The pair's own endpoint keeps its COMPONENT IDENTITY, which
                # is not the same as keeping its bytes. A leaf endpoint IS its
                # identity, so it is held verbatim; an operator endpoint's
                # identity is name/arity{typed kwargs}, which literal values
                # do not touch — so its literals still vary, and pairs whose
                # child is an operator do not starve for want of variation.
                # `freeze` keeps the whole embedded subtree verbatim. An
                # embedded witness carries its pair one level DOWN, so
                # varying inside it would swap that pair's own leaf endpoint
                # and the stream would stop covering the identity it was
                # opened for — the defect this flag exists to prevent.
                args.append(child if (freeze or en._is_leaf(child))
                            else _vary(child, variant + index + 1,
                                       hold_values=hold_values))
                continue
            alternatives = _sibling_alternatives(pc.REGISTRY[child.name].output)
            if alternatives and en._is_leaf(child):
                args.append(alternatives[(variant + index) % len(alternatives)])
            else:
                args.append(_vary(child, variant + index + 1,
                                  hold_values=hold_values))
            continue
        declared = (sig.arg_types[index] if index < len(sig.arg_types)
                    else sig.variadic_arg_type)
        kinds = [alt for alt in declared.split("|") if alt in pc.VALUE_KINDS]
        values = _literal_values(kinds[0]) if kinds else []
        args.append(values[variant % len(values)] if values else child)
    kwargs = []
    for key, value in row.kwargs:
        if isinstance(value, pc.Node):
            if protect == ("kw", key):
                kwargs.append((key, value if (freeze or en._is_leaf(value))
                               else _vary(value, variant + 1,
                                          hold_values=hold_values)))
                continue
            alternatives = _sibling_alternatives(pc.REGISTRY[value.name].output)
            if alternatives and en._is_leaf(value):
                kwargs.append((key, alternatives[variant % len(alternatives)]))
            else:
                kwargs.append((key, _vary(value, variant + 1,
                                          hold_values=hold_values)))
            continue
        if hold_values and sig.kwargs[key].kind in _IDENTITY_KINDS:
            # The value IS the identity being covered — `estimand:subtopic`,
            # `walk:cousin`, §3.2's three manifest classes. Rotating it makes
            # each variant witness a DIFFERENT item, so a stream opened for
            # one of them dilutes across all of them and none reaches k.
            # Held verbatim; breadth comes from the catalogue offering a
            # witness row per value, not from rotation inside one stream.
            #
            # `hold_values` scopes this to the ITEM phase, where such a value
            # IS the target. In the pair phase the target is an edge, and the
            # component identity records kwarg KINDS rather than values — so
            # rotating estimand/impl leaves the pair intact and supplies the
            # variation it needs. Holding them there instead collapsed
            # `blend/2{estimand,impl}` from ~270 distinct spellings to 30 and
            # made a reachable k look infeasible.
            kwargs.append((key, value))
            continue
        values = _literal_values(sig.kwargs[key].kind)
        kwargs.append((key, values[variant % len(values)] if values else value))
    # A witness row opened for `synthetic:pin@X` already CARRIES the pin
    # that made it the witness; regenerating pins by variant would drop it
    # on six variants out of seven and starve that host.
    pins = row.pins
    if not pins and variant and variant % 7 == 0 and sig.operator:
        pins = (en.SYNTHETIC_PINS[(variant // 7) % len(en.SYNTHETIC_PINS)],)
    return pc.Node(row.name, tuple(args), tuple(kwargs), row.mods, pins)


def _stream(row: pc.Node, scenario: str, wanted: int, protect=None,
            freeze: bool = False, hold_values: bool = False) -> list:
    """Up to `wanted` DISTINCT admissible rows sharing one structure."""
    out: list = []
    seen: set = set()
    variant = 0
    while len(out) < wanted and variant < wanted * 12 + 64:
        candidate = _vary(row, variant, protect, freeze, hold_values)
        variant += 1
        if _admissible(candidate, scenario) is not None:
            continue
        expression = pc.canonical_full(candidate)
        if expression in seen:
            continue
        seen.add(expression)
        out.append(expression)
    return out


def _embeddings(row: pc.Node, scenario: str) -> list:
    """Rows that contain `row` as a NON-root subterm.

    Root placement maximizes the realizable pair set, but it does not
    maximize VARIATION: `e5(gemini)` has no literal, no kwarg and no
    sibling, so its only distinct spellings are its three pins and it
    starves far below k. Embedding it under an enclosing operator restores
    the axes — the enclosing node's own literals, siblings and kwargs — and
    the embedded pair is witnessed identically, since a pair is a property
    of an edge and not of the root.
    """
    output = pc.REGISTRY[row.name].output
    out: list = []
    flags = en.SCENARIOS[scenario]
    for name, sig in sorted(pc.REGISTRY.items()):
        if not sig.operator or sig.output not in en.OUTPUT_ROOTS:
            continue
        for arity in _arities(sig):
            for explicit in _explicit_key_sets(sig, flags["methodology_root"]):
                for slot, declared in _slots(sig, arity, explicit):
                    if not isinstance(slot, int):
                        continue
                    accepts = (list(en.OUTPUT_ROOTS) if declared == "process"
                               else declared.split("|"))
                    if output not in accepts:
                        continue
                    for attempt in range(_REALIZATION_ATTEMPTS):
                        candidate = _build_parent(name, arity, explicit,
                                                  slot, row, attempt)
                        if candidate is None:
                            continue
                        if _admissible(candidate, scenario) is None:
                            out.append((candidate, slot))
                            break
    return out


def item_realizations(scenario: str = "methodology-root-only") -> dict:
    """item id -> (witness row, protected slot or None).

    The pair phase only ever builds rows AROUND an edge, so a component
    with no node-valued slot — `margin/0{estimand,t}` — is never rooted and
    its item starves however long the sampler runs. Interior methodology is
    the same shape of omission from the other direction: §3 makes it
    mandatory synthetic coverage and `corpus_policy_violation` deliberately
    TOLERATES it, but no pair needs it, so nothing emits it.

    This catalogue closes both by walking the protos directly: every proto
    at the root, and every methodology-bearing proto embedded one level
    down, which is the only placement where an interior-methodology witness
    can exist.
    """
    leaf_index = en._leaf_identity_index()
    flags = en.SCENARIOS[scenario]
    found: dict = {}

    def offer(row: pc.Node, slot=None) -> None:
        if _admissible(row, scenario) is not None:
            return
        items: set = set()
        pairs: set = set()
        en._row_witnesses(row, items, pairs, leaf_index)
        expression = pc.canonical_full(row)
        for item in items:
            found.setdefault(item, (expression, slot))

    for name, sig in sorted(pc.REGISTRY.items()):
        if not sig.operator:
            continue
        for arity in _arities(sig):
            for explicit in _explicit_key_sets(sig, flags["methodology_root"]):
                for attempt in range(_REALIZATION_ATTEMPTS):
                    row = _build_parent(name, arity, explicit, None, None,
                                        attempt)
                    if row is None:
                        continue
                    if _admissible(row, scenario) is None:
                        offer(row)
                        # §3 mandatory synthetics. Pins are hosted per
                        # operator (§3.1 makes the HOST matter), and string
                        # kwargs are excluded from component patterns, so
                        # nothing else in the sampler ever emits one.
                        for pin in en.SYNTHETIC_PINS:
                            offer(pc.Node(row.name, row.args, row.kwargs,
                                          row.mods, (pin,)))
                        for key, spec in sorted(sig.kwargs.items()):
                            if spec.kind != "string":
                                continue
                            for manifest in en.SYNTHETIC_MANIFESTS.values():
                                bound = tuple(sorted(
                                    list(row.kwargs) + [(key, manifest)],
                                    key=lambda item: item[0]))
                                offer(pc.Node(row.name, row.args, bound))
                                # ...and the string ALONE. Appending it to a
                                # proto that already carries two kwargs
                                # always breaks the per-node kwarg cap, so
                                # without this the §3.2 classes are starved
                                # by arithmetic rather than by grammar.
                                offer(pc.Node(row.name, row.args,
                                              ((key, manifest),)))
                        # Interior methodology: the same proto, but carried
                        # one level down. Legal because §3 authorizes it as
                        # synthetic coverage even under a root-only scenario.
                        for wrapper, slot in _embeddings(row, scenario):
                            offer(wrapper, slot)
                        # No `break`: every admissible BINDING is offered, not
                        # just the first. Enumerated values are themselves
                        # required items (`estimand:subtopic`), so stopping at
                        # one binding leaves most of them with no witness row
                        # in the catalogue at all — starvation the pair phase
                        # only masks when k is small enough for its rotation
                        # to cover them by luck.
    # Methodology-bearing protos are what interior-methodology items need,
    # and those protos are refused at the root under this scenario only when
    # the scenario forbids ROOT methodology, which it does not. Embedding
    # above therefore supplies them.
    for leaf in _leaf_candidates():
        offer(leaf)
    return found


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------

def sample(coverage_minimum_k: int = 100,
           scenario: str = "methodology-root-only") -> dict:
    """Emit rows until every required item and pair reaches `k` distinct
    witnessing rows. Returns the measurement report."""
    if coverage_minimum_k < 1:
        raise SamplerError("coverage_minimum_k must be positive")
    leaf_index = en._leaf_identity_index()
    realized, refused = pair_realizations(scenario)
    item_universe = set(en.required_witness_universe())
    pair_universe = set(realized)

    rows: dict = {}                 # expression -> (items, pairs)
    item_rows: dict = {item: 0 for item in item_universe}
    pair_rows: dict = {pair: 0 for pair in pair_universe}
    semantic_seen: set = set()

    def admit(expression: str) -> bool:
        if expression in rows:
            return False
        node = pc.parse(expression)
        items: set = set()
        pairs: set = set()
        en._row_witnesses(node, items, pairs, leaf_index)
        semantic = pc.canonical_semantic(node)
        if semantic in semantic_seen:
            items = {item for item in items if item.startswith("synthetic:")}
        semantic_seen.add(semantic)
        rows[expression] = (items, pairs)
        for item in items:
            if item in item_rows:
                item_rows[item] += 1
        for pair in pairs:
            if pair in pair_rows:
                pair_rows[pair] += 1
        return True

    # Directed phase: walk the pair universe, then the items, always
    # servicing the least-covered identity first.
    for pair in sorted(pair_universe):
        if pair_rows[pair] >= coverage_minimum_k:
            continue
        expression, slot = realized[pair]
        witness = pc.parse(expression)
        needed = coverage_minimum_k - pair_rows[pair]
        for candidate in _stream(witness, scenario, needed * 2, slot):
            if pair_rows[pair] >= coverage_minimum_k:
                break
            admit(candidate)
        # A variation-poor shape exhausts its own stream long before k; the
        # same pair rides into the corpus embedded under other operators,
        # which is where its remaining coverage comes from.
        if pair_rows[pair] < coverage_minimum_k:
            for wrapper, wrapped_slot in _embeddings(witness, scenario):
                if pair_rows[pair] >= coverage_minimum_k:
                    break
                remaining = coverage_minimum_k - pair_rows[pair]
                for candidate in _stream(wrapper, scenario, remaining * 2,
                                         wrapped_slot, freeze=True):
                    if pair_rows[pair] >= coverage_minimum_k:
                        break
                    admit(candidate)

    # Item phase: everything the pair phase structurally cannot reach.
    # Run to a FIXPOINT rather than once. Items share streams — the three
    # §3.2 string classes rotate through a single `manifest=` slot — so a
    # single pass lets whichever item is serviced first stop the stream at
    # its own k and leave its stream-mates short. Iterating until no pass
    # makes progress removes the ordering dependence.
    catalogue = item_realizations(scenario)
    for _ in range(_ITEM_PHASE_PASSES):
        progressed = False
        for item in sorted(item_rows):
            if item_rows[item] >= coverage_minimum_k or item not in catalogue:
                continue
            expression, slot = catalogue[item]
            witness = pc.parse(expression)
            needed = coverage_minimum_k - item_rows[item]
            for candidate in _stream(witness, scenario, needed * 8, slot,
                                     hold_values=True):
                if item_rows[item] >= coverage_minimum_k:
                    break
                if admit(candidate):
                    progressed = True
        if not progressed:
            break

    starved_items = sorted(item for item, count in item_rows.items()
                           if count < coverage_minimum_k)
    starved_pairs = sorted(pair for pair, count in pair_rows.items()
                           if count < coverage_minimum_k)
    tokens = _token_statistics(rows)
    return {
        "sampler_version": SAMPLER_VERSION,
        "sampler_manifest_sha256": sampler_manifest_sha256(),
        "registry_version": pc.REGISTRY_VERSION,
        "scenario": scenario,
        "coverage_minimum_k": coverage_minimum_k,
        "rows": len(rows),
        "required_items": len(item_universe),
        "required_pairs": len(pair_universe),
        "refused_pair_candidates": len(refused),
        "starved_items": starved_items,
        "starved_pairs": starved_pairs,
        "distinct_templates": len({
            "tmpl:" + en._template_identity(pc.parse(expression))
            for expression in rows
        }),
        "tokens": tokens,
        "expressions": sorted(rows),
    }


def _token_statistics(rows: Iterable[str]) -> dict:
    import process_expression_contract as contract
    import process_expression_tokenizer as tok
    lengths = sorted(len(tok.encode(contract.resolve_expression(expression)))
                     for expression in rows)
    if not lengths:
        return {"vocabulary": tok.VOCAB_VERSION, "rows": 0}
    total = sum(lengths)
    return {
        "vocabulary": tok.VOCAB_VERSION,
        "rows": len(lengths),
        "total": total,
        "mean": round(total / len(lengths), 1),
        "median": lengths[len(lengths) // 2],
        "p99": lengths[min(len(lengths) - 1, int(len(lengths) * 0.99))],
        "max": lengths[-1],
    }


def verify_against_extractor(report: Mapping[str, Any]) -> dict:
    """Re-measure a finished sample through the PUBLIC extractor path.

    The sampler counts coverage with the extractor's own `_row_witnesses`,
    but that is the same author checking their own arithmetic. This runs
    the sample through `families_from_expressions` — corpus policy,
    authoritative-universe containment, canonical_full dedup, and the
    semantic-once counting rule — and fails closed on disagreement.
    """
    families, manifest = en.families_from_expressions(
        report["expressions"], scenario=report["scenario"],
        authorizing=True, return_manifest=True)
    if manifest["rows_kept"] != report["rows"]:
        raise SamplerError(
            f"extractor kept {manifest['rows_kept']} rows, sampler reported "
            f"{report['rows']}"
        )
    counts: dict = {}
    for family in families:
        for item, count in family.witness_counts:
            counts[item] = counts.get(item, 0) + count
    starved = sorted(
        item for item in en.required_witness_universe()
        if counts.get(item, 0) < report["coverage_minimum_k"]
    )
    if starved != report["starved_items"]:
        raise SamplerError(
            "extractor and sampler disagree on starved items: "
            f"{starved[:5]} vs {report['starved_items'][:5]}"
        )
    return manifest


def main(argv=None) -> int:
    k = int((argv or sys.argv[1:] or ["100"])[0])
    report = sample(coverage_minimum_k=k)
    manifest = verify_against_extractor(report)
    print(f"{SAMPLER_VERSION} over registry {pc.REGISTRY_VERSION}, "
          f"scenario {report['scenario']}")
    print(f"manifest sha {report['sampler_manifest_sha256'][:16]}…")
    print(f"required: {report['required_items']} items, "
          f"{report['required_pairs']} pairs "
          f"({report['refused_pair_candidates']} candidates refused)")
    print(f"coverage minimum k = {report['coverage_minimum_k']}")
    print(f"ROWS = {report['rows']:,}  templates = "
          f"{report['distinct_templates']:,}")
    print(f"tokens: {report['tokens']}")
    print(f"starved items: {len(report['starved_items'])}, "
          f"starved pairs: {len(report['starved_pairs'])}")
    print(f"extractor agrees: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
