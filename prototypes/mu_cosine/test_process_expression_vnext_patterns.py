#!/usr/bin/env python3
"""Milestone-2 tests: typed variables, ``PatternAST``, and checked grounding.

Covers the one state-machine edge this milestone implements
(``DESIGN_process_expression_patterns.md`` §1)::

    PatternAST --ground(bindings)--> GroundAST

and the acceptance criteria that edge is responsible for: §16.3 (``_`` fresh per
occurrence, repeated named variables unify) and the *equality-structure* half of
§16.4 (alpha-equivalent patterns preserve equality structure).  The other half of
§16.4 — a shared pattern *digest* — is deliberately not implemented and not
tested, because a digest would have to fix the wire form of an expression-valued
``TermIndex``, which is an unresolved specification decision.

Two boundaries are asserted rather than assumed:

* grounding produces no identity, no digest, and no canonical bytes;
* the ground surface from milestone 1 is unchanged.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from process_expression_vnext import (
    ElaborationError,
    GroundAST,
    GroundingError,
    ParseError,
    PatternAST,
    alpha_equivalent,
    elaborate,
    elaborate_ground,
    elaborate_pattern,
    ground,
    ground_surface,
    is_ground,
    load_registry,
    parse_functional,
)
from process_expression_vnext.ast import (
    Call,
    IndexedType,
    ListTerm,
    ListType,
    PatternIndex,
    PatternVariable,
    Reference,
    ReferenceIndex,
    TermIndex,
    TypeName,
    TypeVar,
    TypedTerm,
    debug_repr,
)
from process_expression_vnext.numerics import NumberLexeme, to_real
from process_expression_vnext.patterns import index_of, make_ground

FIXTURE = ROOT / "process_expression_vnext" / "testdata" / "frontend_registry_fixture.json"

SEALED = {
    "PROCESS_EXPRESSION_GOLDEN_v1.json":
        "b053351a2a419ac58b7ab644afe15c60543846ce8b9d5a3d9bcbc332ca24db29",
    "PROCESS_EXPRESSION_GOLDEN_v2.json":
        "85e6421f5a1347fca5937d1243dc01500a9aa5b7221571b4918248e57ece6344",
}

#: The §13.1 worked example, used by several tests.
HOP_DECAY_PATTERN = """hop_decay_targets(
  S::substrate[C],
  decay=D::real,
  hop_origin=H::int,
  direction='ancestor',
  depth=Limit::depth_limit
)"""

HOP_DECAY_GROUND = (
    "hop_decay_targets(principal_tree(pearltrees),decay=0.85,hop_origin=1,"
    "direction='ancestor',depth=bounded_depth(4))"
)

HOP_DECAY_BINDINGS = {
    "S": "principal_tree(pearltrees)",
    "C": "pearltrees",
    "D": "0.85",
    "H": "1",
    "Limit": "bounded_depth(4)",
}


@pytest.fixture(scope="module")
def reg():
    return load_registry(FIXTURE)


def pat(text, registry) -> PatternAST:
    return elaborate_pattern(parse_functional(text, registry), registry)


def gnd(text, registry) -> GroundAST:
    return elaborate_ground(parse_functional(text, registry), registry)


# --------------------------------------------------------------------------
# the §13.1 worked example
# --------------------------------------------------------------------------


def test_typed_pattern_grounds_to_the_expected_term(reg):
    pattern = pat(HOP_DECAY_PATTERN, reg)
    result = ground_surface(pattern, reg, HOP_DECAY_BINDINGS)
    assert isinstance(result, GroundAST)
    assert result.term == gnd(HOP_DECAY_GROUND, reg).term


def test_grounded_and_directly_elaborated_terms_are_structurally_identical(reg):
    """The two routes to the same process must not diverge in any node.

    Structural equality of the roots is the headline claim, but it would also
    hold if both sides were equally wrong, so the inferred types are compared
    node by node as well.
    """

    grounded = ground_surface(pat(HOP_DECAY_PATTERN, reg), reg, HOP_DECAY_BINDINGS).term
    direct = gnd(HOP_DECAY_GROUND, reg).term
    assert grounded == direct
    assert debug_repr(grounded) == debug_repr(direct)
    assert grounded.inferred_type == direct.inferred_type
    assert str(grounded.inferred_type) == "target_factory[principal_tree(pearltrees),hop_decay]"
    assert [n for n, _ in grounded.fields] == [
        "decay", "depth", "direction", "hop_origin"
    ]


def test_inferring_an_index_does_not_require_supplying_it(reg):
    """`C` is determined by `S`, so binding it is optional — but must agree."""

    pattern = pat(HOP_DECAY_PATTERN, reg)
    without_c = {k: v for k, v in HOP_DECAY_BINDINGS.items() if k != "C"}
    assert ground_surface(pattern, reg, without_c).term == gnd(HOP_DECAY_GROUND, reg).term

    conflicting = {**HOP_DECAY_BINDINGS, "C": "simplewiki"}
    with pytest.raises(GroundingError, match="does not satisfy"):
        ground_surface(pattern, reg, conflicting)


def test_an_index_only_variable_is_not_a_required_binding(reg):
    pattern = pat(HOP_DECAY_PATTERN, reg)
    assert pattern.variable("C").in_term_position is False
    assert pattern.variable("C").constraint is None
    assert "C" not in {v.name for v in pattern.required_variables}
    assert {v.name for v in pattern.required_variables} == {"S", "D", "H", "Limit"}


# --------------------------------------------------------------------------
# repeated ownership constraints
# --------------------------------------------------------------------------


CROSS = "cross_substrate(S::substrate[C], T::substrate[C])"


def test_repeated_ownership_accepts_one_corpus(reg):
    result = ground_surface(
        pat(CROSS, reg),
        reg,
        {"S": "principal_tree(pearltrees)", "T": "full_dag(pearltrees)"},
    )
    assert result.term == gnd(
        "cross_substrate(principal_tree(pearltrees),full_dag(pearltrees))", reg
    ).term


def test_repeated_ownership_rejects_two_corpora(reg):
    with pytest.raises(GroundingError, match="does not satisfy"):
        ground_surface(
            pat(CROSS, reg),
            reg,
            {"S": "principal_tree(pearltrees)", "T": "full_dag(simplewiki)"},
        )


def test_binding_order_cannot_change_the_result(reg):
    """Both acceptance and rejection must be order-independent.

    Dict order is the caller's, not the pattern's, so a solver that consumed
    bindings in insertion order could accept one spelling of a binding set and
    reject the other.
    """

    pattern = pat(CROSS, reg)
    forward = {"S": "principal_tree(pearltrees)", "T": "full_dag(pearltrees)"}
    reverse = {"T": "full_dag(pearltrees)", "S": "principal_tree(pearltrees)"}
    assert ground_surface(pattern, reg, forward).term == (
        ground_surface(pattern, reg, reverse).term
    )

    bad_forward = {"S": "principal_tree(pearltrees)", "T": "full_dag(simplewiki)"}
    bad_reverse = {"T": "full_dag(simplewiki)", "S": "principal_tree(pearltrees)"}
    messages = []
    for bindings in (bad_forward, bad_reverse):
        with pytest.raises(GroundingError) as excinfo:
            ground_surface(pattern, reg, bindings)
        messages.append(str(excinfo.value))
    assert messages[0] == messages[1]


def test_the_full_hop_decay_pattern_is_also_order_independent(reg):
    pattern = pat(HOP_DECAY_PATTERN, reg)
    reversed_bindings = dict(reversed(list(HOP_DECAY_BINDINGS.items())))
    assert ground_surface(pattern, reg, HOP_DECAY_BINDINGS).term == (
        ground_surface(pattern, reg, reversed_bindings).term
    )


# --------------------------------------------------------------------------
# repeated named term variables
# --------------------------------------------------------------------------


def test_one_binding_substitutes_every_occurrence_of_a_named_variable(reg):
    pattern = pat("cross_substrate(S, S)", reg)
    assert len(pattern.variables) == 1
    result = ground_surface(pattern, reg, {"S": "principal_tree(pearltrees)"})
    assert result.term == gnd(
        "cross_substrate(principal_tree(pearltrees),principal_tree(pearltrees))", reg
    ).term


def test_repeated_occurrences_retain_equality_structure_in_the_pattern(reg):
    """The two occurrences must be *the same* variable, not two equal ones."""

    pattern = pat("cross_substrate(S, S)", reg)
    first, second = pattern.term.args
    assert isinstance(first, PatternVariable) and isinstance(second, PatternVariable)
    assert first.var == second.var
    assert first.var.serial == second.var.serial
    assert first == second

    distinct = pat("cross_substrate(S::substrate[C], T::substrate[C])", reg)
    left, right = distinct.term.args
    assert left.var != right.var


def test_a_named_variable_cannot_carry_two_different_constraints(reg):
    with pytest.raises(ElaborationError, match="constrained as both"):
        pat("cross_substrate(S::substrate[C], S::substrate[OtherC])", reg)


# --------------------------------------------------------------------------
# anonymous variables
# --------------------------------------------------------------------------


ANON = "cross_substrate(_::substrate[C], _::substrate[C])"


def test_each_underscore_is_a_distinct_variable(reg):
    pattern = pat(ANON, reg)
    handles = pattern.anonymous_variables
    assert len(handles) == 2
    assert handles[0].var != handles[1].var
    assert handles[0].var.serial != handles[1].var.serial
    # Same spelling, different identity — which is exactly why the display name
    # must not be the identity.
    assert handles[0].name == handles[1].name == "_"
    assert pattern.term.args[0] != pattern.term.args[1]


def test_a_string_binding_named_underscore_binds_nothing(reg):
    pattern = pat(ANON, reg)
    with pytest.raises(GroundingError, match="unknown binding '_'"):
        ground_surface(pattern, reg, {"_": "principal_tree(pearltrees)"})


def test_anonymous_variables_are_bindable_through_their_handles(reg):
    pattern = pat(ANON, reg)
    left, right = pattern.anonymous_variables
    result = ground_surface(
        pattern,
        reg,
        {left.var: "principal_tree(pearltrees)", right.var: "full_dag(pearltrees)"},
    )
    assert result.term == gnd(
        "cross_substrate(principal_tree(pearltrees),full_dag(pearltrees))", reg
    ).term


def test_anonymous_occurrences_still_share_a_named_ownership_index(reg):
    """`_` is fresh, but the `C` they both mention is not."""

    pattern = pat(ANON, reg)
    left, right = pattern.anonymous_variables
    with pytest.raises(GroundingError, match="does not satisfy"):
        ground_surface(
            pattern,
            reg,
            {left.var: "principal_tree(pearltrees)", right.var: "full_dag(simplewiki)"},
        )


def test_anonymous_diagnostics_identify_which_occurrence_failed(reg):
    pattern = pat(ANON, reg)
    left, _right = pattern.anonymous_variables
    with pytest.raises(GroundingError) as excinfo:
        ground_surface(pattern, reg, {left.var: "principal_tree(pearltrees)"})
    # Not just "_": two occurrences share that spelling.
    assert "_#" in str(excinfo.value)


def test_a_handle_from_another_pattern_is_refused(reg):
    """Variable identity is global, so a foreign handle cannot silently bind."""

    first = pat(ANON, reg)
    second = pat(ANON, reg)
    foreign = second.anonymous_variables[0].var
    with pytest.raises(GroundingError, match="does not belong to this pattern"):
        ground_surface(
            first,
            reg,
            {
                first.anonymous_variables[0].var: "principal_tree(pearltrees)",
                foreign: "full_dag(pearltrees)",
            },
        )


# --------------------------------------------------------------------------
# term variables and value-index variables share one binding
# --------------------------------------------------------------------------


#: `C` is a term here (the corpus `principal_tree` views) *and* an index there
#: (the ownership index of `S`).  One spelling, and it must be one variable.
OWNERSHIP = "cross_substrate(principal_tree(C), S::substrate[C])"


def test_a_variable_in_term_and_index_position_is_one_logical_binding(reg):
    pattern = pat(OWNERSHIP, reg)
    corpus_var = pattern.variable("C")
    assert corpus_var.in_term_position is True
    assert corpus_var.constraint == TypeName("corpus")

    # The same VarId appears as a term node and inside argument 1's type.
    term_node = pattern.term.args[0].args[0]
    assert isinstance(term_node, PatternVariable)
    index = pattern.term.args[1].inferred_type.indices[0]
    assert isinstance(index, PatternIndex)
    assert index.var == term_node.var

    result = ground_surface(
        pattern, reg, {"C": "pearltrees", "S": "full_dag(pearltrees)"}
    )
    assert result.term == gnd(
        "cross_substrate(principal_tree(pearltrees),full_dag(pearltrees))", reg
    ).term


def test_an_explicit_corpus_must_agree_with_the_one_inferred_from_the_substrate(reg):
    with pytest.raises(GroundingError, match="does not satisfy"):
        ground_surface(
            pat(OWNERSHIP, reg),
            reg,
            {"C": "simplewiki", "S": "full_dag(pearltrees)"},
        )


def test_index_of_matches_what_elaboration_records_for_an_argument(reg):
    """An inferred index and a supplied one are only comparable if built alike."""

    reference = gnd("pearltrees", reg).term
    assert index_of(reference) == ReferenceIndex("pearltrees")
    call = gnd("principal_tree(pearltrees)", reg).term
    assert index_of(call) == TermIndex(call)
    # And that is the shape elaboration itself puts in the result type.
    assert gnd(HOP_DECAY_GROUND, reg).term.inferred_type.indices[0] == TermIndex(call)


# --------------------------------------------------------------------------
# inference of a fresh constraint variable
# --------------------------------------------------------------------------


def test_a_bare_variable_takes_its_constraint_from_the_slot(reg):
    """§3.5: `lineage_op(S)` infers `S::substrate[C]` for a *fresh* C."""

    pattern = pat("lineage_op(S)", reg)
    constraint = pattern.variable("S").constraint
    assert isinstance(constraint, IndexedType)
    assert constraint.name == "substrate"
    index = constraint.indices[0]
    assert isinstance(index, PatternIndex)
    assert index.var.origin == "inferred"

    result = ground_surface(pattern, reg, {"S": "principal_tree(pearltrees)"})
    assert result.term == gnd("lineage_op(principal_tree(pearltrees))", reg).term


def test_an_inferred_constraint_variable_is_not_bindable(reg):
    """It is an artifact of inference, not something the author wrote."""

    pattern = pat("lineage_op(S)", reg)
    inferred = pattern.variable("S").constraint.indices[0].var
    assert inferred not in {v.var for v in pattern.variables}
    with pytest.raises(GroundingError, match="does not belong to this pattern"):
        ground_surface(
            pattern, reg, {"S": "principal_tree(pearltrees)", inferred: "pearltrees"}
        )


def test_two_bare_variables_get_separate_fresh_constraints(reg):
    """A fresh C per *occurrence*, so unrelated patterns cannot appear to share one."""

    first = pat("lineage_op(S)", reg).variable("S").constraint.indices[0].var
    second = pat("lineage_op(S)", reg).variable("S").constraint.indices[0].var
    assert first != second


def test_the_signature_placeholder_never_becomes_a_user_variable(reg):
    """Registry `TypeVar` and pattern variables are different namespaces.

    The fixture's `lineage_op` declares `substrate[C]` and the pattern below
    writes its own `C`.  If the two were conflated, the author's `C` would
    silently pick up whatever the signature meant by it.
    """

    pattern = pat("lineage_op(S::substrate[C])", reg)
    index = pattern.variable("S").constraint.indices[0]
    assert isinstance(index, PatternIndex)
    assert not isinstance(index, TypeVar)
    assert index.var.origin == "named"
    assert index.var == pattern.variable("C").var
    # And no signature placeholder survives anywhere in the pattern.
    assert _typevars_in(pattern.term) == []


# --------------------------------------------------------------------------
# type conflicts
# --------------------------------------------------------------------------


def test_a_judge_cannot_satisfy_a_substrate_slot(reg):
    with pytest.raises(ElaborationError, match="expected substrate"):
        pat("lineage_op(J::judge)", reg)


def test_mismatched_ownership_variables_fail_before_any_binding(reg):
    """The signature says both arguments share one corpus; the pattern does not."""

    with pytest.raises(ElaborationError, match="expected substrate"):
        pat("cross_substrate(S::substrate[C], T::substrate[OtherC])", reg)


def test_a_top_level_unannotated_variable_is_underconstrained(reg):
    with pytest.raises(ElaborationError, match="underconstrained"):
        pat("X", reg)


def test_a_bare_list_literal_is_still_rejected_before_its_elements(reg):
    """A list with no declared type fails as a list, not as a variable.

    Worth pinning: the element diagnostic would be the misleading one, since the
    variable is underconstrained only as a consequence of the list being so.
    """

    with pytest.raises(ElaborationError, match="declared list type"):
        pat("[W]", reg)


def test_a_variable_may_not_index_a_list_element_type(reg):
    with pytest.raises(ElaborationError, match="must be a type"):
        pat("list_probe(weights=W::list[C])", reg)


# --------------------------------------------------------------------------
# binding-set errors
# --------------------------------------------------------------------------


def test_missing_bindings_are_named(reg):
    with pytest.raises(GroundingError, match="missing binding"):
        ground_surface(pat(CROSS, reg), reg, {"S": "principal_tree(pearltrees)"})


def test_an_unknown_binding_name_is_rejected(reg):
    with pytest.raises(GroundingError, match="unknown binding 'Nope'"):
        ground_surface(
            pat(CROSS, reg),
            reg,
            {
                "S": "principal_tree(pearltrees)",
                "T": "full_dag(pearltrees)",
                "Nope": "pearltrees",
            },
        )


def test_a_binding_for_a_variable_the_pattern_does_not_use_is_rejected(reg):
    """An extra binding is an error, not a no-op.

    Silently ignoring it would let a caller believe a constraint was applied
    when the pattern never mentioned the variable.
    """

    with pytest.raises(GroundingError, match="unknown binding 'D'"):
        ground_surface(
            pat(CROSS, reg),
            reg,
            {
                "S": "principal_tree(pearltrees)",
                "T": "full_dag(pearltrees)",
                "D": "0.85",
            },
        )


def test_binding_a_variable_to_a_pattern_is_rejected(reg):
    pattern = pat(CROSS, reg)
    inner = pat("cross_substrate(S, S)", reg)
    with pytest.raises(GroundingError, match="is a PatternAST"):
        ground(
            pattern,
            {
                "S": inner,
                "T": gnd("full_dag(pearltrees)", reg),
            },
        )


def test_binding_a_variable_to_a_nonground_term_is_rejected(reg):
    """A term carrying a variable is not a value, whatever its type says."""

    donor = pat("cross_substrate(S, S)", reg)
    with pytest.raises(GroundingError, match="not ground"):
        ground(
            pat(CROSS, reg),
            {"S": donor.term.args[0], "T": gnd("full_dag(pearltrees)", reg)},
        )


def test_ground_refuses_raw_surface_strings(reg):
    """Parse failures must not be able to masquerade as binding failures."""

    with pytest.raises(GroundingError, match="ground_surface"):
        ground(pat(CROSS, reg), {"S": "principal_tree(pearltrees)"})


def test_ground_refuses_a_non_term_binding_value(reg):
    with pytest.raises(GroundingError, match="not a typed term"):
        ground(pat(CROSS, reg), {"S": 42})


def test_a_binding_key_must_be_a_name_or_a_handle(reg):
    with pytest.raises(GroundingError, match="binding key"):
        ground(pat(CROSS, reg), {7: gnd("full_dag(pearltrees)", reg)})


def test_binding_one_variable_by_both_name_and_handle_is_rejected(reg):
    pattern = pat(CROSS, reg)
    handle = pattern.variable("S").var
    with pytest.raises(GroundingError, match="bound twice"):
        ground_surface(
            pattern,
            reg,
            {
                "S": "principal_tree(pearltrees)",
                handle: "full_dag(pearltrees)",
                "T": "full_dag(pearltrees)",
            },
        )


def test_ground_requires_a_pattern(reg):
    with pytest.raises(GroundingError, match="takes a PatternAST"):
        ground(gnd("pearltrees", reg), {})


def test_a_malformed_binding_string_raises_a_parse_error_not_a_grounding_error(reg):
    """Three stages, three error types — the seam the brief asks for."""

    with pytest.raises(ParseError):
        ground_surface(pat(CROSS, reg), reg, {"S": "principal_tree(", "T": "x"})
    with pytest.raises(ElaborationError):
        # Well-formed text, ill-typed: principal_tree views a corpus, not a judge.
        ground_surface(
            pat(CROSS, reg),
            reg,
            {"S": "principal_tree(haiku)", "T": "full_dag(pearltrees)"},
        )
    with pytest.raises(GroundingError):
        ground_surface(pat(CROSS, reg), reg, {"S": "principal_tree(pearltrees)"})


# --------------------------------------------------------------------------
# groundness proof
# --------------------------------------------------------------------------


def _typevars_in(term: TypedTerm) -> list[str]:
    found: list[str] = []

    def walk_type(declared):
        if isinstance(declared, TypeVar):
            found.append(declared.name)
        elif isinstance(declared, TermIndex):
            walk_term(declared.term)
        elif isinstance(declared, IndexedType):
            for index in declared.indices:
                walk_type(index)
        elif isinstance(declared, ListType):
            walk_type(declared.element)

    def walk_term(node):
        walk_type(node.inferred_type)
        if isinstance(node, Call):
            for arg in node.args:
                walk_term(arg)
            for _name, value in node.fields:
                walk_term(value)
        elif isinstance(node, ListTerm):
            for item in node.items:
                walk_term(item)

    walk_term(term)
    return found


def _variables_in(term: TypedTerm) -> list[object]:
    found: list[object] = []

    def walk_type(declared):
        if isinstance(declared, PatternIndex):
            found.append(declared.var)
        elif isinstance(declared, TermIndex):
            walk_term(declared.term)
        elif isinstance(declared, IndexedType):
            for index in declared.indices:
                walk_type(index)
        elif isinstance(declared, ListType):
            walk_type(declared.element)

    def walk_term(node):
        if isinstance(node, PatternVariable):
            found.append(node.var)
        walk_type(node.inferred_type)
        if isinstance(node, Call):
            for arg in node.args:
                walk_term(arg)
            for _name, value in node.fields:
                walk_term(value)
        elif isinstance(node, ListTerm):
            for item in node.items:
                walk_term(item)

    walk_term(term)
    return found


def test_nothing_variable_survives_anywhere_in_a_ground_result(reg):
    """Calls, fields, list items, inferred types, and nested indices."""

    pattern = pat(
        "list_probe(weights=[W1::real, W2::real])", reg
    )
    listy = ground_surface(pattern, reg, {"W1": "0.4", "W2": "0.6"}).term
    assert _variables_in(listy) == []
    assert _typevars_in(listy) == []
    assert isinstance(listy.fields[0][1], ListTerm)
    assert [i.value for i in listy.fields[0][1].items] == [
        to_real(NumberLexeme("0.4")), to_real(NumberLexeme("0.6"))
    ]

    nested = ground_surface(pat(HOP_DECAY_PATTERN, reg), reg, HOP_DECAY_BINDINGS).term
    assert _variables_in(nested) == []
    assert _typevars_in(nested) == []
    # The result type carries a TermIndex; the check must have looked inside it.
    assert isinstance(nested.inferred_type.indices[0], TermIndex)
    assert is_ground(nested)


def test_the_proof_looks_inside_an_expression_valued_index(reg):
    """A variable hiding in a `TermIndex` must not pass as ground.

    Constructed by hand: elaboration never builds this shape, because a
    variable argument becomes a `PatternIndex` rather than a `TermIndex`.  That
    is the point — the proof has to be sound against shapes the current
    elaborator does not produce, since a later milestone may.
    """

    donor = pat("cross_substrate(S, S)", reg)
    hidden = donor.term.args[0]
    smuggled = Call(
        IndexedType("target_factory", (TermIndex(hidden), TypeName("hop_decay"))),
        "hop_decay_targets",
        (),
        (),
    )
    assert is_ground(smuggled) is False
    with pytest.raises(GroundingError, match="unbound variable"):
        make_ground(smuggled)


def test_the_proof_rejects_a_surviving_signature_placeholder(reg):
    smuggled = Reference(IndexedType("substrate", (TypeVar("C"),)), "pearltrees")
    with pytest.raises(GroundingError, match="unresolved signature type variable"):
        make_ground(smuggled)


def test_a_pattern_is_not_a_ground_ast(reg):
    pattern = pat(CROSS, reg)
    assert is_ground(pattern.term) is False
    with pytest.raises(GroundingError, match="not ground"):
        make_ground(pattern.term)


# --------------------------------------------------------------------------
# immutability
# --------------------------------------------------------------------------


def test_grounding_leaves_the_pattern_untouched(reg):
    pattern = pat(HOP_DECAY_PATTERN, reg)
    before = copy.deepcopy(pattern)
    ground_surface(pattern, reg, HOP_DECAY_BINDINGS)
    assert pattern == before
    assert _variables_in(pattern.term)  # still a pattern, not consumed


def test_one_pattern_grounds_more_than_once_with_different_bindings(reg):
    pattern = pat(CROSS, reg)
    first = ground_surface(
        pattern, reg, {"S": "principal_tree(pearltrees)", "T": "full_dag(pearltrees)"}
    )
    second = ground_surface(
        pattern, reg, {"S": "principal_tree(simplewiki)", "T": "full_dag(simplewiki)"}
    )
    assert first.term != second.term
    assert first.term == gnd(
        "cross_substrate(principal_tree(pearltrees),full_dag(pearltrees))", reg
    ).term


# --------------------------------------------------------------------------
# alpha equivalence
# --------------------------------------------------------------------------


def test_renaming_every_variable_preserves_alpha_equivalence(reg):
    assert alpha_equivalent(
        pat("lineage_op(S::substrate[C])", reg),
        pat("lineage_op(X::substrate[Y])", reg),
    )


def test_different_repeated_variable_structure_is_not_alpha_equivalent(reg):
    assert not alpha_equivalent(
        pat("cross_substrate(S, S)", reg),
        pat("cross_substrate(S::substrate[C], T::substrate[C])", reg),
    )


def test_anonymous_and_named_variables_are_not_interchangeable(reg):
    """`_` twice and `S,T` twice both have two variables, but differ in kind."""

    assert not alpha_equivalent(
        pat(ANON, reg),
        pat("cross_substrate(S::substrate[C], T::substrate[C])", reg),
    )


def test_alpha_equivalence_is_reflexive_and_symmetric(reg):
    left = pat(HOP_DECAY_PATTERN, reg)
    right = pat(HOP_DECAY_PATTERN.replace("S::", "Sub::").replace("D::", "Dec::"), reg)
    assert alpha_equivalent(left, left)
    assert alpha_equivalent(left, right)
    assert alpha_equivalent(right, left)


def test_alpha_equivalence_still_compares_the_ground_parts(reg):
    assert not alpha_equivalent(
        pat("cross_substrate(S::substrate[C], T::substrate[C])", reg),
        pat(OWNERSHIP, reg),
    )


def test_a_pattern_and_its_renaming_are_not_equal_as_data(reg):
    """Alpha-equivalence is needed precisely because `==` is stricter."""

    left = pat("lineage_op(S::substrate[C])", reg)
    right = pat("lineage_op(X::substrate[Y])", reg)
    assert left.term != right.term
    assert alpha_equivalent(left, right)


def test_alpha_equivalent_requires_two_patterns(reg):
    with pytest.raises(TypeError):
        alpha_equivalent(pat(CROSS, reg), gnd("pearltrees", reg))


# --------------------------------------------------------------------------
# state boundary and non-goals
# --------------------------------------------------------------------------


def test_the_compatibility_api_stays_ground_only(reg):
    """`elaborate()` must never silently accept a free variable."""

    for text in ["lineage_op(S)", "cross_substrate(S, S)", "lineage_op(S::substrate[C])"]:
        with pytest.raises(ElaborationError, match="ground expression"):
            elaborate(parse_functional(text, reg), reg)


#: A cross-section of the ground surface: references, annotations, every numeric
#: type, defaults, lists, punctuated names, ownership indices, and the coarse and
#: precise lineage forms of §4.1.
GROUND_SURFACE = [
    "pearltrees",
    "pearltrees::corpus",
    "haiku::judge",
    "lineage_at",
    "unbounded_depth",
    "bounded_depth(4)",
    "principal_tree(pearltrees)",
    "principal_tree(pearltrees)::substrate[pearltrees]",
    "full_dag(simplewiki)",
    "margin(t=1)",
    "margin(t=0.03)",
    "margin(t=1,epsilon=-0.0)",
    "default_probe()",
    "list_probe(weights=[0.4,0.6])",
    "list_probe(weights=[])",
    "judge_view(gpt-5.5-low)",
    "judge_view(haiku.fast)",
    "order_probe(s=principal_tree(pearltrees))",
    "cross_substrate(principal_tree(pearltrees),full_dag(pearltrees))",
    "lineage_op(principal_tree(pearltrees))",
    "lineage_op(principal_tree(pearltrees),estimand='hop_decay')",
    HOP_DECAY_GROUND,
]


@pytest.mark.parametrize("text", GROUND_SURFACE)
def test_elaborate_ground_wraps_exactly_what_elaborate_returns(reg, text):
    source = parse_functional(text, reg)
    assert elaborate_ground(source, reg).term == elaborate(source, reg)


@pytest.mark.parametrize("text", GROUND_SURFACE)
def test_the_pattern_path_agrees_with_the_ground_path(reg, text):
    """Adding variables must not have moved the variable-free surface.

    Both entry points share one ``_elaborate``, so this is the test that would
    catch the scope parameter changing a decision it has no business changing —
    an expectation pushed down differently, or a default resolved in another
    order.
    """

    source = parse_functional(text, reg)
    direct = elaborate(source, reg)
    pattern = elaborate_pattern(source, reg)
    assert pattern.variables == ()
    assert pattern.term == direct
    assert debug_repr(pattern.term) == debug_repr(direct)
    # A variable-free pattern grounds under the empty binding set.
    assert ground(pattern, {}).term == direct
    assert ground(pattern, None).term == direct


def test_a_ground_ast_may_still_be_semantically_coarse(reg):
    """Grounding is not interpretation (§4.1).

    `lineage_op(S)` grounded is ground *and* still underconstrained about which
    lineage estimand it names, and nothing here resolves that.
    """

    result = ground_surface(
        pat("lineage_op(S)", reg), reg, {"S": "principal_tree(pearltrees)"}
    )
    assert isinstance(result, GroundAST)
    assert result.term.fields == ()          # no estimand, no impl, no defaults
    assert str(result.term.inferred_type).startswith("abstract_lineage_process[")


def test_the_package_exports_no_identity_or_digest_surface():
    import process_expression_vnext as package

    exported = set(package.__all__)
    forbidden = {
        "digest", "sha256", "canonical_bytes", "pattern_digest", "identity",
        "serialize", "encode", "deploy", "resolve", "interpret", "represent",
        "verify", "verify_factory_receipt", "receipt",
    }
    assert exported & forbidden == set()
    for name in exported:
        assert hasattr(package, name)


def test_no_module_in_the_package_can_compute_a_digest():
    """A digest is not merely unexported — nothing here can produce one.

    Checked at the import level rather than the export level: an unexported
    helper that hashed a typed term would still be a pattern digest, and a
    pattern digest is what this milestone is forbidden to mint.
    """

    package_dir = ROOT / "process_expression_vnext"
    modules = sorted(package_dir.glob("*.py"))
    assert modules, "package has no modules to check"
    for module in modules:
        text = module.read_text(encoding="utf-8")
        for banned in ("hashlib", "sha256", "blake2"):
            assert banned not in text, f"{module.name} reaches for {banned}"


def test_sealed_bundles_are_untouched():
    for name, expected in SEALED.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == expected, f"{name} changed"
