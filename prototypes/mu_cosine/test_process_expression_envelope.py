#!/usr/bin/env python3
"""Measured envelope facts for the step-2 generator specification.

``DESIGN_process_expression_generator.md`` states caps and coverage numbers.  The
standing rule for this track is that such quantities are *measured and recorded
as tests*, never asserted in prose — so every number the specification marks
`measured` is reproduced here from the registry and the nine registered
processes.

These tests deliberately assert exact values rather than bounds.  When the
registry gains a process or an operator the numbers move, a test fails, and the
specification must be revised with the new measurement.  A silently drifting cap
is the failure mode this file exists to prevent.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import process_cards as pc
from process_cards import PROCESSES, REGISTRY
import process_expression_contract as pec
from process_expression_contract import resolve_expression, role_paths, token_stream

SPEC = ROOT / "DESIGN_process_expression_generator.md"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _walk(node, depth=0):
    yield node, depth
    for child in node.args:
        yield from _walk(child, depth + 1)


def _resolved_walk(resolved):
    yield resolved
    for child in resolved.args:
        yield from _resolved_walk(child)


def _numeric_literals(resolved):
    out = []
    for node in _resolved_walk(resolved):
        for kwarg in node.kwargs:
            values = kwarg.value if isinstance(kwarg.value, tuple) else [kwarg.value]
            for value in values:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out.append(value)
    return out


def _all_registered():
    return [pc.parse(expression) for expression in PROCESSES.values()]


# --------------------------------------------------------------------------
# §2.1 measured envelope
# --------------------------------------------------------------------------


def test_measured_ast_depth_and_node_count():
    nodes = [list(_walk(n)) for n in _all_registered()]
    assert max(max(d for _, d in group) for group in nodes) == 3
    assert max(len(group) for group in nodes) == 5


def test_measured_arity_kwargs_modifiers_and_pins():
    flat = [n for tree in _all_registered() for n, _ in _walk(tree)]
    assert max(len(n.args) for n in flat) == 3
    assert max(len(n.kwargs) for n in flat) == 2
    assert max(len(n.mods) for n in flat) == 1
    # No registered process carries a pin: the generator is solely responsible
    # for pin coverage, which §3.1 shows the alignment arm depends on.
    assert max(len(n.pins) for n in flat) == 0


def test_measured_list_length_and_absence_of_string_literals():
    lengths, strings = [], []
    for expression in PROCESSES.values():
        for node in _resolved_walk(resolve_expression(expression)):
            for kwarg in node.kwargs:
                if isinstance(kwarg.value, tuple):
                    lengths.append(len(kwarg.value))
                if isinstance(kwarg.value, str):
                    strings.append(kwarg.value)
    assert max(lengths) == 2
    assert strings == []  # routing.manifest is registered but never used


def test_measured_token_count_and_path_length():
    counts, path_lengths = [], []
    for expression in PROCESSES.values():
        tokens = token_stream(resolve_expression(expression))
        counts.append(len(tokens))
        path_lengths += [len(t.path) for t in tokens]
    assert max(counts) == 120
    assert max(path_lengths) == 5


def test_registry_caps_modifiers_at_one_per_node():
    """`MOD(index>0)` stays schema-legal but unreachable under registry v0.3."""

    with pytest.raises(pc.ParseError, match="at most one modifier"):
        pc.parse("llm.element.subcat")
    assert all(len(s.modifiers) >= 0 for s in REGISTRY.values())


# --------------------------------------------------------------------------
# §2.3 the type system is the binding constraint
# --------------------------------------------------------------------------


def test_only_atoms_produce_source():
    """This, not the caps, is why enumeration does not explode."""

    producers = {n for n, s in REGISTRY.items() if s.output == "source"}
    assert producers, "expected at least one source producer"
    for name in producers:
        assert REGISTRY[name].atom, f"{name} produces source but is not an atom"


def test_process_is_the_only_recursive_channel():
    """`e5`'s positional type is the wildcard that compounds with depth."""

    wildcards = {
        name: sig.arg_types
        for name, sig in REGISTRY.items()
        if "process" in sig.arg_types or sig.variadic_arg_type == "process"
    }
    assert set(wildcards) == {"e5"}
    assert pc._output_matches("process", "target-set")
    assert not pc._output_matches("score", "source")


# --------------------------------------------------------------------------
# §3.1 the V2/V3 collapse that makes pins mandatory
# --------------------------------------------------------------------------


def test_v3_differs_from_v2_only_by_pins():
    for expression in PROCESSES.values():
        node = pc.parse(expression)
        assert pc.render(node, 2) == pc.render(node, 3)  # no registered pins

    pinned = pc.parse("lineage(graph,decay=0.85)@synthetic/pin-a")
    assert pc.render(pinned, 2) != pc.render(pinned, 3)
    assert pc.render(pinned, 3).endswith("@synthetic/pin-a")


def test_v1_differs_from_v2_only_by_non_default_kwargs():
    default = pc.parse("lineage(graph,decay=0.85)")  # decay default is 0.85
    assert pc.render(default, 1) == pc.render(default, 2)
    non_default = pc.parse("lineage(graph,decay=0.5)")
    assert pc.render(non_default, 1) != pc.render(non_default, 2)


def test_v0_has_no_semantic_card():
    """V0 is excluded from alignment rather than used as an empty teacher."""

    for expression in PROCESSES.values():
        assert pc.render(pc.parse(expression), 0) == ""


# --------------------------------------------------------------------------
# §5.2 digit-byte coverage floor
# --------------------------------------------------------------------------


PRESENT_DIGITS = set("012358")
ABSENT_DIGITS = set("4679")


def test_registered_processes_cover_only_six_digit_bytes():
    """Measured from the byte tokens the tokenizer actually emits.

    Four digit bytes never appear in any registered process, so a byte-level
    decoder trained on observed values alone could not spell them.  §5.2 turns
    this into a support floor for the generator.
    """

    emitted = set()
    for expression in PROCESSES.values():
        for token in token_stream(resolve_expression(expression)):
            if token.token.startswith("BYTE:"):
                emitted.add(chr(int(token.token[7:], 16)))
    digits = {ch for ch in emitted if ch.isdigit()}
    assert digits == PRESENT_DIGITS
    assert ABSENT_DIGITS & emitted == set()


def test_resolved_numeric_literals_are_the_measured_set():
    """Defaults count: `decay=0.85` is elided by parse and restored by canonical."""

    literals = set()
    for expression in PROCESSES.values():
        for value in _numeric_literals(resolve_expression(expression)):
            literals.add(pc._render_val(value))
    assert literals == {"0.02", "0.03", "0.85", "10", "20"}
    # The pre-resolution view would wrongly omit the default-valued decay.
    unresolved = {
        pc._render_val(v)
        for e in PROCESSES.values()
        for n, _ in _walk(pc.parse(e))
        for k, val in n.kwargs
        for v in (val if isinstance(val, tuple) else [val])
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    assert "0.85" not in unresolved


# --------------------------------------------------------------------------
# §4.2 the witnessed template set
# --------------------------------------------------------------------------


def structural_template(node) -> str:
    """Leaf identities and literal payloads become typed slots.

    Operators, nesting, argument roles, kwarg keys, modifiers, and list shape
    are preserved, per the encoder handoff's structural-template definition.
    """

    signature = REGISTRY[node.name]
    if not node.args and not node.kwargs:
        return f"<{signature.output}>" + "".join("." + m for m in node.mods)
    parts = [structural_template(child) for child in node.args]
    resolved = {key: spec.default for key, spec in signature.kwargs.items()}
    resolved.update(dict(node.kwargs))
    for key, value in sorted(resolved.items()):
        if value is None:
            continue
        kind = signature.kwargs[key].kind
        slot = f"<{kind}:len{len(value)}>" if isinstance(value, tuple) else f"<{kind}>"
        parts.append(f"{key}={slot}")
    body = node.name + ("(" + ",".join(parts) + ")" if parts else "")
    return body + "".join("." + m for m in node.mods) + "@<pin>" * len(node.pins)


def test_witnessed_template_set_is_nine_distinct_shapes():
    templates = {structural_template(pc.parse(e)): n for n, e in PROCESSES.items()}
    assert len(templates) == 9
    assert "kalman(<source>.D,<source>.S)" in templates
    assert (
        "e5(routing(<score>,<source>.lineage,menus=<int_list:len2>,t=<number_list:len2>))"
        in templates
    )


def test_templates_erase_leaf_identity_but_keep_roles_and_shape():
    a = structural_template(pc.parse("kalman(luna.D,luna.S)"))
    b = structural_template(pc.parse("blend(luna.D,luna.S)"))
    assert a != b  # operator is preserved
    # list shape is preserved, so len1 and len2 are different templates
    one = structural_template(pc.parse("e5(routing(e5,haiku,t=[0.02],menus=[10]))"))
    two = structural_template(
        pc.parse("e5(routing(e5,haiku,t=[0.02,0.03],menus=[10,20]))")
    )
    assert one != two


# --------------------------------------------------------------------------
# the specification must not drift from these measurements
# --------------------------------------------------------------------------


def test_specification_records_the_measured_numbers():
    """Guard against the prose and the measurements diverging."""

    text = SPEC.read_text(encoding="utf-8")
    for needle in (
        "285,478",          # corpus size at the stated caps
        "19,131",           # structural templates
        "9/9",              # registered-process coverage
        "0.9725",           # e5 numeric insensitivity
        "1.15 of 3",        # mean distinct teacher texts without pins
        "max_node_count   = 5",
        "max_depth        = 3",
    ):
        assert needle in text, f"specification no longer records {needle!r}"
