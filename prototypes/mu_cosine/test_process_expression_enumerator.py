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
    assert en.count("naive-full")[0] == 3_303_413_185_358
    assert en.count("methodology-root-only")[0] == 54_871_574
    assert en.count("structural-only")[0] == 10_756_382


def test_measured_template_counts_are_pinned():
    assert en.count("naive-full", template_mode=True)[0] == 3_835_821
    assert en.count("methodology-root-only", template_mode=True)[0] == 98_070
    assert en.count("structural-only", template_mode=True)[0] == 28_521


def test_the_binding_constraint_moved_to_kwarg_enumeration():
    """§2.3's v0.3 claim, falsified under v0.4: methodology kwargs multiply
    the corpus by ~60,000x, structure alone by ~37x."""
    naive = en.count("naive-full")[0]
    root_only = en.count("methodology-root-only")[0]
    structural = en.count("structural-only")[0]
    assert naive > 60_000 * root_only
    assert structural > 35 * 285_478  # the v0.3 corpus, for scale


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
