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
        if isinstance(child, pc.Node):
            yield from _walk(child, depth + 1)


def _resolved_walk(resolved):
    yield resolved
    for child in resolved.args:
        if isinstance(child, pec.ResolvedNode):
            yield from _resolved_walk(child)
    for kwarg in resolved.kwargs:
        if isinstance(kwarg.value, pec.ResolvedNode):
            yield from _resolved_walk(kwarg.value)


def _numeric_literals(resolved):
    out = []
    for node in _resolved_walk(resolved):
        for child in node.args:
            if isinstance(child, pec.ResolvedLiteral):
                out.append(child.value)
        for kwarg in node.kwargs:
            if isinstance(kwarg.value, pec.ResolvedNode):
                continue
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
    assert max(len(group) for group in nodes) == 6


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
                if kwarg.value_type == "string":
                    strings.append(kwarg.value)
    assert max(lengths) == 2
    assert strings == []  # routing.manifest is registered but never used
    # The enumerated methodology kinds are strings lexically but never free
    # text; they are measured as their own kinds, not as `string`.


def test_measured_token_count_and_path_length():
    counts, path_lengths = [], []
    for expression in PROCESSES.values():
        tokens = token_stream(resolve_expression(expression))
        counts.append(len(tokens))
        path_lengths += [len(t.path) for t in tokens]
    assert max(counts) == 120
    assert max(path_lengths) == 5


def test_registry_caps_modifiers_at_one_per_node():
    """`MOD(index>0)` stays schema-legal but unreachable under registry v0.4."""

    with pytest.raises(pc.ParseError, match="at most one modifier"):
        pc.parse("llm.element.subcat")
    # The cap actually bites: several names declare more than one modifier, so
    # the restriction is a real one rather than vacuously satisfied.
    multi = {n for n, s in REGISTRY.items() if len(s.modifiers) > 1}
    assert multi == {"luna", "llm"}


# --------------------------------------------------------------------------
# §2.3 the type system is the binding constraint
# --------------------------------------------------------------------------


def test_only_atoms_produce_substrates_and_judges():
    """This, not the caps, is why enumeration does not explode.

    v0.4 split the old `source` role into `substrate` and `judge` (R2); both
    remain atom-only producer roles, so the argument carries over unchanged.
    """

    assert not any(s.output == "source" for s in REGISTRY.values())  # retired
    for role in ("substrate", "judge"):
        producers = {n for n, s in REGISTRY.items() if s.output == role}
        assert producers, f"expected at least one {role} producer"
        for name in producers:
            assert REGISTRY[name].atom, f"{name} produces {role} but is not an atom"


def test_process_is_the_only_recursive_channel():
    """`e5`'s positional type is the wildcard that compounds with depth."""

    wildcards = {
        name: sig.arg_types
        for name, sig in REGISTRY.items()
        if "process" in sig.arg_types or sig.variadic_arg_type == "process"
    }
    assert set(wildcards) == {"e5"}
    assert pc._output_matches("process", "target-set")
    assert not pc._output_matches("score", "judge")


# --------------------------------------------------------------------------
# §3.1 the V2/V3 collapse that makes pins mandatory
# --------------------------------------------------------------------------


def test_v3_differs_from_v2_only_by_pins():
    for expression in PROCESSES.values():
        node = pc.parse(expression)
        assert pc.render(node, 2) == pc.render(node, 3)  # no registered pins

    pinned = pc.parse("lineage(pearltrees,decay=0.85)@synthetic/pin-a")
    assert pc.render(pinned, 2) != pc.render(pinned, 3)
    assert pc.render(pinned, 3).endswith("@synthetic/pin-a")


def test_v1_differs_from_v2_only_by_non_default_kwargs():
    default = pc.parse("lineage(pearltrees,decay=0.85)")  # decay default is 0.85
    assert pc.render(default, 1) == pc.render(default, 2)
    non_default = pc.parse("lineage(pearltrees,decay=0.5)")
    assert pc.render(non_default, 1) != pc.render(non_default, 2)


def test_v0_has_no_semantic_card():
    """V0 is excluded from alignment rather than used as an empty teacher."""

    for expression in PROCESSES.values():
        assert pc.render(pc.parse(expression), 0) == ""


# --------------------------------------------------------------------------
# §5.2 digit-byte coverage floor
# --------------------------------------------------------------------------


PRESENT_DIGITS = set("0123568")
ABSENT_DIGITS = set("479")


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
    assert literals == {"0.02", "0.03", "0.6", "0.85", "10", "20"}
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
    parts = []
    for child in node.args:
        if isinstance(child, pc.Node):
            parts.append(structural_template(child))
        else:
            # v0.4: a positional numeric literal is a typed slot too.
            parts.append("<number>")
    resolved = {key: spec.default for key, spec in signature.kwargs.items()}
    resolved.update(dict(node.kwargs))
    for key, value in sorted(resolved.items()):
        if value is None:
            continue
        if isinstance(value, pc.Node):
            # v0.4: a node-valued kwarg (mu=…) keeps its structural shape.
            parts.append(f"{key}={structural_template(value)}")
            continue
        kind = signature.kwargs[key].kind
        slot = f"<{kind}:len{len(value)}>" if isinstance(value, tuple) else f"<{kind}>"
        parts.append(f"{key}={slot}")
    body = node.name + ("(" + ",".join(parts) + ")" if parts else "")
    return body + "".join("." + m for m in node.mods) + "@<pin>" * len(node.pins)


def test_witnessed_template_set_is_ten_distinct_shapes():
    templates = {structural_template(pc.parse(e)): n for n, e in PROCESSES.items()}
    assert len(templates) == 10
    assert "kalman(<judge>.D,<judge>.S)" in templates
    assert (
        "e5(routing(<score>,<judge>.lineage,menus=<int_list:len2>,t=<number_list:len2>))"
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
        "3,475,387,022,969",  # naive v0.5 enumeration — the explosion finding
        "61,908,552",         # methodology-root-only expressions (v0.5)
        "97,526",             # methodology-root-only structural templates
        # External review (M1): the structural-only row and template counts
        # were transcribed but unguarded — exactly the drift class this test
        # exists to catch.
        "11,409,263",         # structural-only expressions (v0.5)
        "28,373",             # structural-only structural templates
        "3,826,859",          # naive-full structural templates
        "285,478",            # v0.3 corpus, retained as history
        "10/10",              # registered-process coverage (10 registered processes)
        "0.9725",             # e5 numeric insensitivity
        "1.15 of 3",          # mean distinct teacher texts without pins
        "max_node_count   = 6",
        "max_depth        = 3",
    ):
        assert needle in text, f"specification no longer records {needle!r}"


# --------------------------------------------------------------------------
# §5.0 the closeness objective — its numbers are spec, so they are tests too
# --------------------------------------------------------------------------


def test_specification_records_the_closeness_numbers():
    text = SPEC.read_text(encoding="utf-8")
    for needle in (
        "p = 0.75",                      # exact-grading fraction
        "{0, 0.5, 0.75, 1}",             # its ablation grid
    ):
        assert needle in text, f"specification no longer records {needle!r}"


def _spec_tolerances() -> dict[str, float]:
    """Parse the whole tolerance table out of the specification."""

    import re

    text = SPEC.read_text(encoding="utf-8")
    return {
        m.group(1): float(m.group(2))
        for m in re.finditer(r"^tolerance\.(\S+)\s*=\s*([0-9.]+)", text, re.M)
    }


def test_tolerance_table_covers_every_number_field_and_nothing_else():
    """The whole mapping is pinned, not a sampled pair of entries.

    Checking only two rows would let `margin.t` or `blend.w` change or vanish
    while the suite stayed green, contrary to the rule that every specification
    number is reproduced by a test.
    """

    declared = _spec_tolerances()
    number_fields = {
        f"{name}.{key}"
        for name, sig in REGISTRY.items()
        for key, spec in sig.kwargs.items()
        if spec.kind in ("number", "number_list")
    }
    assert set(declared) == number_fields
    assert declared == {
        "routing.t": 0.002,
        "margin.t": 0.002,
        "lineage.decay": 0.01,
        "blend.w": 0.01,
        "hop_decay.gamma": 0.01,
    }


def test_threshold_tolerance_is_below_half_the_operational_gap():
    """§5.0.2: independent per-element tolerance can collapse a routing tier.

    Registered thresholds are 0.01 apart.  A tolerance of half that gap admits
    [0.025, 0.025] — both elements "within tolerance" and the two-tier policy
    destroyed — so the tolerance must be strictly below half the gap.
    """

    registered = [0.02, 0.03]
    gap = min(b - a for a, b in zip(registered, registered[1:]))
    assert gap == pytest.approx(0.01)
    tolerance = _spec_tolerances()["routing.t"]
    assert tolerance < gap / 2

    # The collapse the tolerance alone cannot prevent: the parser accepts both
    # an unordered and a degenerate threshold list, so §5.0.2 additionally
    # requires a structural ordering/separation predicate.
    for collapsed in ("routing(e5,haiku,t=[0.025,0.025],menus=[10,20])",
                      "routing(e5,haiku,t=[0.03,0.02],menus=[10,20])"):
        pc.parse(collapsed)  # parses today: no ordering or separation enforced


def test_authoritative_bundle_pointers_all_name_the_current_bundle():
    """§0: the pointer moves with the bundle, or consumers fail closed.

    A step-2 implementer following the parent handoff or the README must be
    directed at the bundle the current loader accepts.  Pointing at a
    superseded bundle would either fail closed or silently miss the coverage
    the new contract added.
    """

    from process_expression_contract import (
        CURRENT_GOLDEN_BUNDLE,
        SUPERSEDED_GOLDEN_BUNDLES,
    )

    handoff = (ROOT / "DESIGN_expression_encoder_future.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for document, label in ((handoff, "handoff"), (readme, "README")):
        assert CURRENT_GOLDEN_BUNDLE in document, f"{label} does not name the current bundle"

    # A superseded bundle may still be *mentioned* as provenance, but never
    # without the current one alongside it.
    for stale in SUPERSEDED_GOLDEN_BUNDLES:
        for document, label in ((handoff, "handoff"), (readme, "README")):
            if stale in document:
                assert CURRENT_GOLDEN_BUNDLE in document, (
                    f"{label} names superseded {stale} without the current bundle"
                )


def test_specification_records_the_supersession_procedure():
    text = SPEC.read_text(encoding="utf-8")
    assert "Bundle supersession procedure" in text
    assert "CURRENT_GOLDEN_BUNDLE" in text
    assert "SUPERSEDED_GOLDEN_BUNDLES" in text


def test_specification_fixes_the_grading_unit_and_schedule():
    """§5.3: one unit, one schedule — the earlier draft implied both."""

    text = SPEC.read_text(encoding="utf-8")
    assert "The unit is the **numeric field**, and the draw is **resampled every step**" in text
    assert "Per field, not per row." in text
    assert "Resampled per step, not fixed." in text


def test_specification_states_precedence_over_the_parent_handoff():
    """§5.0.1: the handoff's exact-AST gate stays primary until amended."""

    text = SPEC.read_text(encoding="utf-8")
    assert "does **not** silently supersede its parent" in text
    assert "exact-AST reconstruction gate remains primary" in text


def test_integer_fields_are_typed_int_and_therefore_exempt_from_tolerance():
    """§5.0.2: menus=10 vs 11 is a different process, not a near miss."""

    int_fields = {
        (name, key)
        for name, sig in REGISTRY.items()
        for key, spec in sig.kwargs.items()
        if spec.kind in ("int", "int_list")
    }
    assert ("routing", "menus") in int_fields
    assert ("menu", "n") in int_fields
    assert ("lineage", "depth") in int_fields
    number_fields = {
        (name, key)
        for name, sig in REGISTRY.items()
        for key, spec in sig.kwargs.items()
        if spec.kind in ("number", "number_list")
    }
    # The tolerance table covers exactly the number-valued fields.
    assert number_fields == {
        ("routing", "t"),
        ("margin", "t"),
        ("lineage", "decay"),
        ("blend", "w"),
        ("hop_decay", "gamma"),
    }
    assert not (int_fields & number_fields)


def test_missing_digits_cap_resolution_below_the_stated_tolerance():
    """§5.2: the digit floor is about resolution, not catastrophic failure.

    A decoder can render 0.47 as 0.45 using only trained glyphs, so an unseen
    digit is not fatal.  But in the production threshold range the forced error
    exceeds the routing tolerance, which is why the grid still widens.
    """

    nearest_trained = 0.06  # 0.07 unreachable: digit 7 is absent
    forced_error = abs(0.07 - nearest_trained)
    # Read the live tolerance rather than a literal: hard-coding the superseded
    # 0.005 left the inequality passing under either value, so the test no
    # longer demonstrated what its name claims.
    routing_tolerance = _spec_tolerances()["routing.t"]
    assert forced_error > routing_tolerance
    assert ABSENT_DIGITS == {"4", "7", "9"}


def test_identity_is_derived_from_retained_bytes_not_decoded_text():
    """§5.0: what this proves, and what it deliberately does not.

    It proves identity is derived from a validated AST and its retained
    canonical bytes, so an approximate decode cannot corrupt an existing
    identity.  It does *not* prove decoder output can never become an identity:
    the helpers accept any parsed node and cannot tell where it came from.  That
    boundary needs an origin-tagged wrapper, recorded in §5.0 as a step-3
    obligation blocking on the decoder — see the companion test below.
    """

    from process_identity import deployed_identity, verify_identity_record

    node = pc.parse("lineage(pearltrees,decay=0.85)")
    record = deployed_identity(node, factory_fingerprint="f").as_record()
    # The record carries the canonical bytes, so verification never needs the
    # decoder to have spelled anything.
    assert record["canonical_identity_string"] == "lineage(pearltrees,decay=0.85)"
    verify_identity_record(record)

    # A near-miss decode is a different process identity, not a tolerated one:
    # nothing in the identity layer knows about tolerance.
    near = pc.parse("lineage(pearltrees,decay=0.84)")
    from process_identity import full_ast_digest

    assert full_ast_digest(near) != full_ast_digest(node)


def test_identity_boundary_is_currently_a_discipline_not_an_invariant():
    """Records the gap honestly so the step-3 obligation is not forgotten.

    The identity APIs accept any parsed node, so text that *happened* to come
    from a decoder can still mint an identity today.  Asserting the gap keeps
    the specification's claim and the code in agreement, and this test is the
    one to invert once the origin-tagged wrapper lands.
    """

    from process_identity import deployed_identity, full_ast_digest_for_expression

    decoded_looking_text = "lineage(pearltrees,decay=0.84)"  # imagine a decode
    # Nothing distinguishes it from an authored expression:
    assert len(full_ast_digest_for_expression(decoded_looking_text)) == 64
    identity = deployed_identity(
        pc.parse(decoded_looking_text), factory_fingerprint="f"
    )
    assert identity.full_digest == full_ast_digest_for_expression(decoded_looking_text)

    # The specification must therefore not claim the boundary is enforced.
    text = SPEC.read_text(encoding="utf-8")
    assert "discipline, not an invariant" in text
    assert "step-3 obligation" in text
