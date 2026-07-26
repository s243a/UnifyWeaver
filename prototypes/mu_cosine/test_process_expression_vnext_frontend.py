#!/usr/bin/env python3
"""Frontend-kernel tests for the experimental vNext process-expression package.

Covers the numbered semantics required of this milestone: the parse/elaborate
seam, registry-driven typing, indexed ownership, exact numerics, and
phase-appropriate diagnostics.

Two boundaries are asserted rather than assumed:

* nothing in the package exports hashing, identity, resolution, or deployment;
* registry v0.3, ``pec-v2``, and both sealed golden bundles are untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from process_expression_vnext import (
    ElaborationError,
    NotImplementedInMilestone,
    ParseError,
    RegistryError,
    elaborate,
    load_registry,
    parse_functional,
)
from process_expression_vnext.ast import (
    Atom,
    Call,
    ReferenceIndex,
    TermIndex,
    Float64,
    IndexedType,
    Int,
    ListTerm,
    ListType,
    Real,
    Reference,
    String,
    TypeName,
)
from process_expression_vnext.numerics import NumberLexeme, NumericError, to_float64

FIXTURE = ROOT / "process_expression_vnext" / "testdata" / "frontend_registry_fixture.json"

SEALED = {
    "PROCESS_EXPRESSION_GOLDEN_v1.json":
        "b053351a2a419ac58b7ab644afe15c60543846ce8b9d5a3d9bcbc332ca24db29",
    "PROCESS_EXPRESSION_GOLDEN_v2.json":
        "85e6421f5a1347fca5937d1243dc01500a9aa5b7221571b4918248e57ece6344",
}


@pytest.fixture(scope="module")
def reg():
    return load_registry(FIXTURE)


def ela(text, reg):
    return elaborate(parse_functional(text, reg), reg)


# --------------------------------------------------------------------------
# 1-4  surface forms, collisions, lexing, order
# --------------------------------------------------------------------------


def test_reference_and_call_are_distinct_node_kinds(reg):
    assert isinstance(ela("pearltrees", reg), Reference)
    assert isinstance(ela("principal_tree(pearltrees)", reg), Call)


def test_reference_atom_and_string_cannot_collide(reg):
    ref = ela("haiku", reg)
    atom = ela("'haiku'", reg)
    text = ela('"haiku"', reg)
    assert isinstance(ref, Reference) and ref.inferred_type == TypeName("judge")
    assert isinstance(atom, Atom) and atom.inferred_type == TypeName("atom")
    assert isinstance(text, String) and text.inferred_type == TypeName("string")
    assert atom.value == text.value == "haiku"
    assert atom != text  # same characters, different semantic nodes


def test_dotted_and_hyphenated_names_use_longest_match(reg):
    """`gpt-5.5-low` must lex as one token, not as arithmetic on `gpt`."""

    assert ela("gpt-5.5-low", reg).name == "gpt-5.5-low"
    assert ela("gpt-5.5", reg).name == "gpt-5.5"
    assert ela("haiku.fast", reg).name == "haiku.fast"
    assert ela("haiku", reg).name == "haiku"


def test_positional_argument_order_is_preserved(reg):
    term = ela(
        "cross_substrate(principal_tree(pearltrees),full_dag(pearltrees))", reg
    )
    assert [a.name for a in term.args] == ["principal_tree", "full_dag"]


# --------------------------------------------------------------------------
# 5-8  named fields
# --------------------------------------------------------------------------


def test_reordered_named_fields_are_structurally_equal(reg):
    a = ela("lineage_op(principal_tree(pearltrees),estimand='hop_decay',decay=0.5)", reg)
    b = ela("lineage_op(principal_tree(pearltrees),decay=0.5,estimand='hop_decay')", reg)
    assert a == b


def test_duplicate_named_fields_fail_before_normalization(reg):
    with pytest.raises(ElaborationError, match="duplicate named field 'decay'"):
        ela("lineage_op(principal_tree(pearltrees),decay=0.5,decay=0.5)", reg)


def test_unknown_missing_and_ignored_fields_fail_closed(reg):
    with pytest.raises(ElaborationError, match="does not consume field 'nope'"):
        ela("lineage_op(principal_tree(pearltrees),nope=1)", reg)
    with pytest.raises(ElaborationError, match="missing required field 't'"):
        ela("margin()", reg)
    # `margin` does not consume `decay`; a plausible-looking field is still ignored.
    with pytest.raises(ElaborationError, match="does not consume field 'decay'"):
        ela("margin(t=0.01,decay=0.85)", reg)


def test_explicit_and_elided_defaults_elaborate_identically(reg):
    elided = ela("lineage_op(principal_tree(pearltrees))", reg)
    explicit = ela(
        "lineage_op(principal_tree(pearltrees),decay=0.85,depth=unbounded_depth)", reg
    )
    assert elided == explicit
    fields = dict(elided.fields)
    assert isinstance(fields["decay"], Real)
    assert fields["decay"].inferred_type == TypeName("real")
    assert isinstance(fields["depth"], Reference)


# --------------------------------------------------------------------------
# 9-11  annotations assert, never convert
# --------------------------------------------------------------------------


def test_redundant_annotation_leaves_no_semantic_trace(reg):
    assert ela("pearltrees", reg) == ela("pearltrees::corpus", reg)


def test_conflicting_annotation_fails_before_semantic_output(reg):
    with pytest.raises(ElaborationError, match="expected judge, found corpus"):
        ela("pearltrees::judge", reg)


def test_a_judge_is_not_a_substrate(reg):
    with pytest.raises(ElaborationError, match="expects substrate"):
        ela("lineage_op(haiku)", reg)


# --------------------------------------------------------------------------
# 12-14  nesting and indexed ownership
# --------------------------------------------------------------------------


def test_nested_calls_and_expression_valued_fields(reg):
    term = ela(
        "lineage_op(principal_tree(pearltrees),depth=bounded_depth(4))", reg
    )
    assert isinstance(term.args[0], Call)
    depth = dict(term.fields)["depth"]
    assert isinstance(depth, Call) and depth.name == "bounded_depth"
    assert depth.args[0] == Int(TypeName("int"), 4)


def test_indexed_type_substitution(reg):
    assert ela("principal_tree(pearltrees)", reg).inferred_type == IndexedType(
        "substrate", (ReferenceIndex("pearltrees"),)
    )
    assert ela("full_dag(simplewiki)", reg).inferred_type == IndexedType(
        "substrate", (ReferenceIndex("simplewiki"),)
    )
    # The annotation in the specification's own example type-checks.
    ela("principal_tree(pearltrees)::substrate[pearltrees]", reg)


def test_incompatible_indexed_use_fails(reg):
    """Both substrates must view the same corpus; `C` binds once."""

    ela("cross_substrate(principal_tree(pearltrees),full_dag(pearltrees))", reg)
    with pytest.raises(ElaborationError, match="expected substrate\\[pearltrees\\]"):
        ela(
            "cross_substrate(principal_tree(pearltrees),principal_tree(simplewiki))",
            reg,
        )


def test_value_indices_are_not_types(reg):
    """`substrate[pearltrees]` indexes by a corpus *reference*, not a type.

    Representing the index as ``TypeName("pearltrees")`` would conflate the
    value and type namespaces.
    """

    substrate = ela("principal_tree(pearltrees)", reg).inferred_type
    assert isinstance(substrate, IndexedType)
    (index,) = substrate.indices
    assert isinstance(index, ReferenceIndex)
    assert not isinstance(index, TypeName)
    assert index.name == "pearltrees"


def test_punctuated_reference_parses_inside_a_type_index(reg):
    """`judge_scope[gpt-5.5-low]` must lex the index as one registered name."""

    typed = ela("judge_view(gpt-5.5-low)::judge_scope[gpt-5.5-low]", reg)
    (index,) = typed.inferred_type.indices
    assert isinstance(index, ReferenceIndex) and index.name == "gpt-5.5-low"


def test_expression_valued_index_is_tagged_experimental(reg):
    """`lineage_op(S,...) :: abstract_lineage_process[S]` where S is a call.

    The wire form of an expression-valued index is an open specification
    decision, so it is carried by an explicitly experimental in-memory node
    rather than faked as a reference.
    """

    typed = ela("lineage_op(principal_tree(pearltrees))", reg)
    assert typed.inferred_type.name == "abstract_lineage_process"
    (index,) = typed.inferred_type.indices
    assert isinstance(index, TermIndex)
    assert "unresolved" in TermIndex.__doc__ and "specification decision" in (
        " ".join(TermIndex.__doc__.split())
    )
    # It cannot be written in an annotation; that would presume the wire form.
    with pytest.raises(NotImplementedInMilestone, match="open .*specification"):
        parse_functional(
            "lineage_op(principal_tree(pearltrees))"
            "::abstract_lineage_process[principal_tree(pearltrees)]",
            reg,
        )


def test_fixture_does_not_conflate_lineage_at_with_lineage_op(reg):
    """The specification separates the family reference from the operator."""

    assert ela("lineage_at", reg).inferred_type == TypeName("relation_family")
    op = ela("lineage_op(principal_tree(pearltrees))", reg)
    assert op.inferred_type != TypeName("relation_family")
    assert op.inferred_type.name == "abstract_lineage_process"


def test_a_bind_cannot_overwrite_an_established_ownership_constraint(reg):
    """Argument unification binds `C` first; `binds` may not silently change it."""

    ela("rebinding_probe(principal_tree(pearltrees),pearltrees)", reg)
    with pytest.raises(ElaborationError, match="would rebind index C"):
        ela("rebinding_probe(principal_tree(pearltrees),simplewiki)", reg)


def test_defaults_are_type_checked_like_authored_values(tmp_path, reg):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["entries"]["bad_default"] = {
        "kind": "call",
        "type": "cross_view",
        "args": [],
        "fields": {"n": {"type": "int", "default": "'not_an_int'"}},
    }
    path = tmp_path / "bad_default.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    bad = load_registry(path)
    with pytest.raises(ElaborationError, match="ill-typed|expected int"):
        elaborate(parse_functional("bad_default()", bad), bad)


def test_a_default_with_an_unresolved_type_cannot_enter_the_typed_ast(tmp_path):
    """Rather than a dependency solver, refuse order-dependent defaults."""

    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["entries"]["unresolved_default"] = {
        "kind": "call",
        "type": "cross_view",
        "args": [],
        "fields": {"e": {"type": "entity[S]", "default": "pearltrees"}},
    }
    path = tmp_path / "unresolved_default.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    registry = load_registry(path)
    with pytest.raises(ElaborationError, match="unresolved type"):
        elaborate(parse_functional("unresolved_default()", registry), registry)


def test_a_reference_may_not_declare_an_unresolvable_type_variable(tmp_path):
    """A reference has no arguments, so nothing could ever bind `S`."""

    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["entries"]["dangling_entity"] = {"kind": "reference", "type": "entity[S]"}
    path = tmp_path / "dangling.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RegistryError, match="unresolved type variable"):
        load_registry(path)


def test_no_type_variable_survives_into_an_elaborated_result(reg):
    def walk(term):
        yield term
        if isinstance(term, Call):
            for a in term.args:
                yield from walk(a)
            for _, v in term.fields:
                yield from walk(v)

    for text in [
        "principal_tree(pearltrees)",
        "lineage_op(principal_tree(pearltrees))",
        "hop_decay_targets(principal_tree(pearltrees),decay=0.85)",
    ]:
        for term in walk(ela(text, reg)):
            assert "TypeVar" not in repr(term.inferred_type)


# --------------------------------------------------------------------------
# 15  lists
# --------------------------------------------------------------------------


def test_lists_are_homogeneous_per_declared_element_type(reg):
    term = ela(
        "hop_decay_targets(principal_tree(pearltrees),decay=0.85,weights=[0.4,0.6])",
        reg,
    )
    weights = dict(term.fields)["weights"]
    assert isinstance(weights, ListTerm)
    assert weights.inferred_type == ListType(TypeName("real"))
    assert all(isinstance(i, Real) for i in weights.items)

    with pytest.raises(ElaborationError):
        ela(
            "hop_decay_targets(principal_tree(pearltrees),decay=0.85,"
            "weights=[0.4,'six'])",
            reg,
        )


# --------------------------------------------------------------------------
# 16-20  numerics
# --------------------------------------------------------------------------


def test_integral_lexeme_in_a_real_field_becomes_real(reg):
    """The declared type wins over the surface spelling (§3.6)."""

    term = ela("margin(t=1)", reg)
    value = dict(term.fields)["t"]
    assert isinstance(value, Real) and not isinstance(value, Int)
    assert value.inferred_type == TypeName("real")


def test_equivalent_real_spellings_normalize_identically(reg):
    one = ela("margin(t=1)", reg)
    assert one == ela("margin(t=1.0)", reg) == ela("margin(t=1e0)", reg)


def test_decimal_beyond_binary64_precision_survives(reg):
    precise = "0.1234567890123456789012345"
    term = ela(f"margin(t={precise})", reg)
    value = dict(term.fields)["t"]
    assert value.value.plain_string() == precise
    # Going through float would have rounded it; prove the rounded form differs.
    assert str(float(precise)) != precise
    assert ela(f"margin(t={precise})", reg) != ela("margin(t=0.1234567890123457)", reg)


def test_real_negative_zero_normalizes_to_zero(reg):
    assert ela("margin(t=-0.0)", reg) == ela("margin(t=0.0)", reg)
    assert ela("margin(t=-0.0)", reg) == ela("margin(t=0)", reg)


def test_float64_preserves_signed_zero_as_distinct_bits(reg):
    negative = ela("margin(t=1,epsilon=-0.0)", reg)
    positive = ela("margin(t=1,epsilon=0.0)", reg)
    neg_bits = dict(negative.fields)["epsilon"]
    pos_bits = dict(positive.fields)["epsilon"]
    assert isinstance(neg_bits, Float64) and isinstance(pos_bits, Float64)
    assert neg_bits.value.hex16() == "8000000000000000"
    assert pos_bits.value.hex16() == "0000000000000000"
    assert negative != positive  # distinct, unlike Python float equality
    assert neg_bits.value.as_float == pos_bits.value.as_float == 0.0


# --------------------------------------------------------------------------
# 21  phase-appropriate diagnostics
# --------------------------------------------------------------------------


def test_fractional_value_cannot_satisfy_int(reg):
    with pytest.raises(ElaborationError, match="not an exact integer"):
        ela("bounded_depth(3.5)", reg)
    with pytest.raises(ElaborationError, match="not an exact integer"):
        ela("0.85::int", reg)


@pytest.mark.parametrize("text", ["margin(t=1e)", "margin(t=1e+)", "margin(t=1.)"])
def test_malformed_numeric_lexemes_fail_at_parse(text, reg):
    with pytest.raises(ParseError):
        parse_functional(text, reg)


def test_leading_zeros_are_accepted_as_in_v0(reg):
    """`01` is valid; rejecting it would be an unapproved break from v0.

    The vNext grammar says only "a finite decimal numeric token", and v0.3
    accepts `01` and canonicalizes it to `1`.  Whether to forbid it is a
    specification decision, not something an experimental frontend should
    freeze.
    """

    import process_cards as pc

    assert pc.canonical(pc.parse("margin(t=01)")) == "margin(t=1)"
    assert ela("margin(t=01)", reg) == ela("margin(t=1)", reg)


@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "margin(t=nan)"])
def test_nan_and_infinity_are_not_numbers(text, reg):
    """No spelling of a non-finite value reaches a numeric type.

    They fail at *parse*, as unregistered names or stray characters, because
    the numeric grammar admits finite decimals only — so `float('inf')` is
    never constructed anywhere on the path.
    """

    with pytest.raises(ParseError, match="unregistered name|malformed|unexpected"):
        parse_functional(text, reg)


def test_trailing_input_and_trailing_commas_fail_at_parse(reg):
    with pytest.raises(ParseError, match="trailing input"):
        parse_functional("pearltrees pearltrees", reg)
    with pytest.raises(ParseError, match="trailing comma"):
        parse_functional("bounded_depth(4,)", reg)
    with pytest.raises(ParseError, match="trailing comma"):
        parse_functional("hop_decay_targets(principal_tree(pearltrees),)", reg)


def test_unregistered_name_is_not_guessed_to_be_an_atom(reg):
    with pytest.raises(ParseError, match="unregistered name"):
        parse_functional("attention", reg)
    assert ela("'attention'", reg).value == "attention"


def test_deferred_constructs_are_rejected_precisely_not_misparsed(reg):
    for text in ["lineage_op(S)", "haiku.D", "haiku@rev/abc", "X"]:
        with pytest.raises((NotImplementedInMilestone, ParseError)) as excinfo:
            parse_functional(text, reg)
        assert "not implemented in this milestone" in str(excinfo.value) or (
            "unregistered" in str(excinfo.value)
        )


def test_arity_errors_name_the_call(reg):
    with pytest.raises(ElaborationError, match="expects 1 positional"):
        ela("principal_tree(pearltrees,simplewiki)", reg)
    with pytest.raises(ElaborationError, match="takes no arguments"):
        ela("pearltrees(simplewiki)", reg)
    with pytest.raises(ElaborationError, match="must be applied"):
        ela("principal_tree", reg)


def test_errors_carry_a_source_location_where_applicable(reg):
    with pytest.raises(ParseError) as excinfo:
        parse_functional("pearltrees pearltrees", reg)
    assert excinfo.value.position is not None


def test_lone_surrogates_are_rejected_but_valid_pairs_survive(reg):
    """A typed String must always be UTF-8 encodable."""

    paired = parse_functional('"\\ud83d\\ude00"', reg)   # a valid surrogate pair
    assert paired.value == "\U0001f600"
    paired.value.encode("utf-8")

    for lone in ['"\\ud800"', '"\\udc00"', '"a\\ud800b"']:
        with pytest.raises(ParseError, match="surrogate"):
            parse_functional(lone, reg)


def test_oversized_integers_fail_as_controlled_numeric_errors(reg):
    """Neither CPython's digit limit nor a huge exponent may escape raw."""

    from process_expression_vnext.numerics import MAX_INT_DIGITS

    long_literal = "9" * (MAX_INT_DIGITS + 10)
    with pytest.raises((ElaborationError, ParseError)) as excinfo:
        ela(f"bounded_depth({long_literal})", reg)
    assert "limit" in str(excinfo.value)

    # Short lexeme, catastrophic exponent: rejected without allocating it.
    with pytest.raises(ElaborationError, match="limit"):
        ela("bounded_depth(1e1000000000)", reg)


def test_spans_cover_the_expression_and_exclude_trailing_trivia(reg):
    bare = parse_functional("pearltrees", reg)
    assert (bare.span.start, bare.span.end) == (0, len("pearltrees"))

    grouped = parse_functional("(pearltrees)", reg)
    assert (grouped.span.start, grouped.span.end) == (0, len("(pearltrees)"))

    annotated = parse_functional("(pearltrees)::corpus", reg)
    assert annotated.span.start == 0
    assert annotated.span.end == len("(pearltrees)::corpus")

    padded = parse_functional("  pearltrees  ", reg)
    assert padded.span.end == len("  pearltrees")  # trailing trivia excluded


def test_source_tree_is_documented_as_elaboration_preserving_not_lossless():
    """The public claim must match what the implementation delivers."""

    from process_expression_vnext import ast as ast_module
    from process_expression_vnext import functional_parser as parser_module

    for doc in (ast_module.__doc__, parser_module.__doc__):
        flat = " ".join(doc.split())
        assert "elaboration-preserving" in flat
        assert "exact source reconstruction" in flat
        assert "lossless source AST" not in flat


# --------------------------------------------------------------------------
# 22  determinism
# --------------------------------------------------------------------------


def test_parsing_and_elaboration_are_deterministic(reg):
    text = (
        "lineage_op(principal_tree(pearltrees),estimand='hop_decay',"
        "decay=0.85,depth=bounded_depth(4),note=\"n\")"
    )
    results = [ela(text, reg) for _ in range(5)]
    assert all(r == results[0] for r in results)
    assert len({repr(r) for r in results}) == 1


# --------------------------------------------------------------------------
# registry isolation
# --------------------------------------------------------------------------


def test_registry_is_injected_and_never_falls_back_to_v0_3(reg):
    """v0.3 names must not resolve through the vNext fixture."""

    import process_cards as pc

    assert "luna" in pc.REGISTRY and "luna" not in reg
    with pytest.raises(ParseError, match="unregistered name"):
        parse_functional("luna", reg)
    # And the fixture's own names are not in v0.3.
    assert "pearltrees" not in pc.REGISTRY


def test_fixture_declares_itself_experimental_and_non_release():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert document["status"] == "experimental-test-fixture"
    assert document["production"] is False
    assert "registry_version" not in document


def test_loader_rejects_anything_resembling_a_release_registry(tmp_path):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for mutation in (
        {"status": "release"},
        {"production": True},
        {"registry_version": "v0.4"},
    ):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({**document, **mutation}), encoding="utf-8")
        with pytest.raises(RegistryError):
            load_registry(path)


def test_loader_rejects_duplicate_keys_at_every_level(tmp_path):
    """`json.loads` keeps the last duplicate, so a fixture could lie."""

    path = tmp_path / "dupe.json"
    path.write_text(
        '{"status":"experimental-test-fixture","production":false,'
        '"entries":{"a":{"kind":"reference","type":"corpus"},'
        '"a":{"kind":"reference","type":"judge"}}}',
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="duplicate key"):
        load_registry(path)


@pytest.mark.parametrize(
    "path_keys,mutation,pattern",
    [
        ([], {"surprise": 1}, "unknown key"),
        (["entries", "pearltrees"], {"surprise": 1}, "unknown key"),
        (["entries", "margin", "fields", "t"], {"surprise": 1}, "unknown key"),
        (["entries", "margin", "fields", "t"], {"required": "false"}, "must be a JSON boolean"),
        (["entries", "pearltrees"], {"type": 7}, "must be a string"),
        (["entries", "margin"], {"args": "nope"}, "must be a list"),
    ],
)
def test_loader_fails_closed_on_malformed_declarations(
    tmp_path, path_keys, mutation, pattern
):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    target = document
    for key in path_keys:
        target = target[key]
    target.update(mutation)
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RegistryError, match=pattern):
        load_registry(path)


def test_a_numeric_json_default_is_rejected_before_it_can_round(tmp_path):
    """A JSON number would be parsed as a float and rounded by `str()`."""

    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["entries"]["lineage_op"]["fields"]["decay"]["default"] = (
        0.1234567890123456789012345
    )
    path = tmp_path / "numeric_default.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RegistryError, match="must be a JSON string"):
        load_registry(path)


# --------------------------------------------------------------------------
# 23  the frozen contracts are untouched
# --------------------------------------------------------------------------


def test_sealed_golden_bundles_are_byte_identical():
    for name, digest in SEALED.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} changed"


def test_v0_3_behavior_is_unchanged():
    import process_cards as pc

    assert pc.REGISTRY_VERSION == "v0.3"
    node = pc.parse("lineage(graph,decay=0.85)")
    assert pc.canonical(node) == "lineage(graph,decay=0.85)"
    assert len(pc.ast_sha(node)) == 16


def test_package_exports_no_identity_or_deployment_surface():
    import process_expression_vnext as vnext

    forbidden = (
        "digest", "hash", "sha", "identity", "deploy", "resolve", "verify",
        "receipt", "canonical_bytes", "pe_typed_ast",
    )
    for name in dir(vnext):
        if name.startswith("_"):
            continue
        assert not any(f in name.lower() for f in forbidden), name


def test_debug_repr_is_labelled_noncanonical(reg):
    from process_expression_vnext.ast import debug_repr

    assert "NONCANONICAL" in debug_repr.__doc__
    payload = debug_repr(ela("margin(t=1)", reg))
    assert payload["fields"][0]["value"]["kind"] == "real"
    assert "schema" not in payload  # never claims pe-typed-ast-v1
