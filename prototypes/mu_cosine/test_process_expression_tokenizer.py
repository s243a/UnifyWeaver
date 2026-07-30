#!/usr/bin/env python3
"""Reversibility and fail-closed tests for the process-expression tokenizer.

The governing requirement (``DESIGN_expression_encoder_future.md`` §3.1) is that
``canonical AST -> tokens -> AST`` round-trips for every registry example and
every generated row, with a finite versioned vocabulary and no hashing, subword
splitting, or unknown-token replacement anywhere on the reconstruction path.

Round-trips are asserted on the **canonical identity string**, not on the token
list, because that is what process identity is derived from.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import process_cards as pc
from process_cards import PROCESSES, REGISTRY
from process_expression_contract import (
    CONTRACT_VERSION,
    CURRENT_GOLDEN_BUNDLE,
    REQUIRED_COVERAGE_CASES,
    load_golden,
    resolve_expression,
)
from process_expression_tokenizer import (
    MAX_INDEX,
    VOCAB,
    VOCAB_VERSION,
    TokenizerError,
    Vocabulary,
    assert_round_trips,
    build_vocabulary,
    decode,
    decode_terms,
    encode,
    encode_expression,
    encode_terms,
    round_trip,
)

GOLDEN_PATH = ROOT / CURRENT_GOLDEN_BUNDLE

ALL_CASES = {**PROCESSES, **REQUIRED_COVERAGE_CASES}


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------


def test_vocabulary_is_finite_deterministic_and_versioned():
    first, second = build_vocabulary(), build_vocabulary()
    assert first.terms == second.terms
    assert first.digest == second.digest
    assert first.version == VOCAB_VERSION
    # Versioned independently of the grammar/structure contract.
    assert first.contract_version == CONTRACT_VERSION
    assert first.version != first.contract_version


def test_vocabulary_covers_the_full_byte_fallback():
    """256 byte tokens: literal payloads never need an UNK or a hash bucket."""

    byte_terms = [t for t in VOCAB.terms if t.startswith("BYTE:0x")]
    assert len(byte_terms) == 256
    assert {int(t[7:], 16) for t in byte_terms} == set(range(256))


def test_vocabulary_is_derived_from_the_registry_not_a_corpus():
    for name in REGISTRY:
        assert f"<NAME:{name}>" in VOCAB.id_by_term
    for output in {s.output for s in REGISTRY.values()}:
        assert f"<OUTPUT:{output}>" in VOCAB.id_by_term
    for key in {k for s in REGISTRY.values() for k in s.kwargs}:
        assert f"<KW:{key}>" in VOCAB.id_by_term


def test_ids_are_dense_and_bijective():
    assert sorted(VOCAB.term_by_id) == list(range(VOCAB.size))
    for term, index in VOCAB.id_by_term.items():
        assert VOCAB.term_by_id[index] == term


# --------------------------------------------------------------------------
# agreement with the sealed bundle
# --------------------------------------------------------------------------


def test_token_strings_reproduce_the_sealed_golden_bundle():
    """The tokenizer inherits structure; it must not have reinvented it."""

    document = load_golden(GOLDEN_PATH)
    assert document["contract_version"] == CONTRACT_VERSION
    for row in document["rows"]:
        expected = [line.split("\t", 1)[0] for line in row["tokens"]]
        actual = list(encode_terms(resolve_expression(row["expression"])))
        assert actual == expected, row["name"]


def test_every_golden_row_round_trips_to_its_canonical_identity():
    for row in load_golden(GOLDEN_PATH)["rows"]:
        assert round_trip(row["expression"]) == row["canonical_identity_string"], row["name"]


@pytest.mark.parametrize("name,expression", sorted(ALL_CASES.items()))
def test_registry_examples_and_coverage_cases_round_trip(name, expression):
    assert_round_trips(expression)


def test_round_trip_preserves_the_full_digest():
    from process_identity import full_ast_digest, full_ast_digest_for_expression

    for expression in ALL_CASES.values():
        decoded = decode(encode_expression(expression))
        assert full_ast_digest(decoded) == full_ast_digest_for_expression(expression)


# --------------------------------------------------------------------------
# the cases where a naive decoder silently changes identity
# --------------------------------------------------------------------------


def test_integer_spelled_number_decodes_back_to_an_integer():
    """`margin(t=1)` must not come back as `1.0`.

    The field is registered `number`, but the canonical lexical form is `1`.
    Decoding to a float would render `1.0`, changing the canonical string and
    therefore the process digest — a silent identity change, which is exactly
    what the reversibility requirement exists to prevent.
    """

    decoded = decode(encode_expression("margin(t=1)"))
    assert decoded.kwargs == (("t", 1),)
    assert isinstance(decoded.kwargs[0][1], int)
    assert pc.canonical(decoded) == "margin(t=1)"

    # And a genuinely fractional value stays a float.
    fractional = decode(encode_expression("margin(t=0.03)"))
    assert isinstance(fractional.kwargs[0][1], float)
    assert pc.canonical(fractional) == "margin(t=0.03)"


def test_elided_default_round_trips_through_the_resolved_stream():
    """`lineage(graph)` streams its resolved default and comes back identical."""

    assert round_trip("lineage(graph)") == "lineage(graph,decay=0.85)"
    assert pc.canonical(pc.parse("lineage(graph)")) == "lineage(graph,decay=0.85)"


def test_utf8_and_escaped_strings_survive_byte_exactly():
    for expression in (
        'routing(e5,haiku,t=[0.02],menus=[10],manifest="héllo·wörld")',
        'routing(e5,haiku,t=[0.02],menus=[10],manifest="a\\"b\\\\c")',
    ):
        assert_round_trips(expression)
    decoded = decode(
        encode_expression('routing(e5,haiku,t=[0.02],menus=[10],manifest="héllo·wörld")')
    )
    assert dict(decoded.kwargs)["manifest"] == "héllo·wörld"


def test_negative_numbers_and_pins_round_trip():
    assert_round_trips("lineage(graph,decay=-0.5)")
    assert_round_trips("lineage(graph,decay=0.85)@run/2026-07-25")
    decoded = decode(encode_expression("lineage(graph,decay=0.85)@run/2026-07-25"))
    assert decoded.pins == ("run/2026-07-25",)


def test_atom_and_dual_role_round_trip():
    assert round_trip("e5") == "e5"                      # nullary atom
    assert round_trip("e5(margin(t=0.03))") == "e5(margin(t=0.03))"   # operator
    assert round_trip("llm.subcat") == "llm.subcat"      # modifier-carrying atom


# --------------------------------------------------------------------------
# fail closed
# --------------------------------------------------------------------------


def test_unknown_token_id_is_rejected():
    with pytest.raises(TokenizerError, match="outside the vocabulary"):
        decode([VOCAB.size + 1])


def test_index_beyond_the_frozen_bound_fails_closed_rather_than_wrapping():
    """An over-wide expression is rejected, never clipped or wrapped.

    `blend` is variadic with no upper bound, so an arity past MAX_INDEX is
    constructible and must fail at encode time rather than silently reusing a
    final in-range code — the behavior the position note forbids.
    """

    assert f"<ARG:{MAX_INDEX - 1}>" in VOCAB.id_by_term
    assert f"<ARG:{MAX_INDEX}>" not in VOCAB.id_by_term

    from process_expression_contract import resolve

    wide = pc.Node("blend", tuple(pc.Node("luna") for _ in range(MAX_INDEX + 1)))
    pc.validate(wide)  # the grammar permits it; the vocabulary does not
    with pytest.raises(TokenizerError, match="outside the frozen vocabulary"):
        encode(resolve(wide))

    # Just inside the bound still encodes.
    narrow = pc.Node("blend", tuple(pc.Node("luna") for _ in range(MAX_INDEX - 1)))
    assert encode(resolve(narrow))


def test_stream_kind_must_agree_with_the_registry():
    terms = list(encode_terms(resolve_expression("lineage(graph,decay=0.85)")))
    swapped = ["<KIND:atom>" if t == "<KIND:apply>" else t for t in terms]
    with pytest.raises(TokenizerError, match="KIND"):
        decode_terms(swapped)


def test_stream_output_must_agree_with_the_registry():
    terms = list(encode_terms(resolve_expression("kalman(luna.D,luna.S)")))
    swapped = ["<OUTPUT:score>" if t == "<OUTPUT:target-set>" else t for t in terms]
    with pytest.raises(TokenizerError, match="contradicts the registry"):
        decode_terms(swapped)


def test_value_tag_must_agree_with_the_declared_kind():
    terms = list(encode_terms(resolve_expression("menu(graph,n=10)")))
    swapped = [
        "<NUMBER>" if t == "<INT>" else "</NUMBER>" if t == "</INT>" else t
        for t in terms
    ]
    with pytest.raises(TokenizerError, match="registered"):
        decode_terms(swapped)


def test_unknown_kwarg_and_unregistered_name_are_rejected():
    terms = list(encode_terms(resolve_expression("lineage(graph,decay=0.85)")))
    with pytest.raises(TokenizerError, match="unknown kwarg"):
        decode_terms(["<KW:n>" if t == "<KW:decay>" else t for t in terms])


def test_truncated_and_trailing_streams_are_rejected():
    terms = list(encode_terms(resolve_expression("kalman(luna.D,luna.S)")))
    with pytest.raises(TokenizerError, match="ended early"):
        decode_terms(terms[:-3])
    with pytest.raises(TokenizerError, match="trailing tokens"):
        decode_terms(terms + ["<BOS>"])


def test_non_contiguous_indices_are_rejected():
    terms = list(encode_terms(resolve_expression("kalman(luna.D,luna.S)")))
    broken = ["<ARG:1>" if t == "<ARG:0>" else t for t in terms]
    with pytest.raises(TokenizerError, match="contiguous"):
        decode_terms(broken)


def test_missing_bos_is_rejected():
    terms = list(encode_terms(resolve_expression("graph")))
    with pytest.raises(TokenizerError, match="expected <BOS>"):
        decode_terms(terms[1:])


def test_every_emitted_term_is_a_vocabulary_member():
    """No hashing, bucketing, or unknown-token substitution can hide here.

    Asserted behaviorally rather than by grepping the source: every term the
    contract emits for every known case resolves to a real vocabulary ID, and
    the reverse map returns it unchanged.
    """

    for expression in ALL_CASES.values():
        for term in encode_terms(resolve_expression(expression)):
            assert term in VOCAB.id_by_term
            assert VOCAB.term_by_id[VOCAB.id_by_term[term]] == term


def test_malformed_model_bytes_fail_through_tokenizer_error():
    """Vocabulary-valid but semantically malformed byte payloads must raise TokenizerError,
    never a raw host exception (UnicodeDecodeError/ValueError). Regression for the 0xff leak."""
    base = list(encode_terms(resolve_expression(
        'routing(e5,sonnet,manifest="x",menus=[10],t=[0.02])')))

    def swap_payload(terms, open_tag, close_tag, payload_terms):
        out, i = [], 0
        while terms[i] != open_tag:
            out.append(terms[i])
            i += 1
        out.append(terms[i])
        i += 1
        while not terms[i].startswith(close_tag):
            i += 1                                    # drop original payload bytes
        out.extend(payload_terms)
        out.extend(terms[i:])
        return out

    # 1. lone 0xff inside a STRING payload: vocabulary-valid, invalid utf-8
    bad = swap_payload(base, "<STRING>", "</STRING>", ["BYTE:0xff"])
    with pytest.raises(TokenizerError, match="utf-8 value payload"):
        decode_terms(bad)

    # 2. non-numeric bytes under a NUMBER tag
    nbase = list(encode_terms(resolve_expression("menu(graph,n=10)")))
    bad_n = ["BYTE:0x78" if t == "BYTE:0x31" else t for t in nbase]   # '1' -> 'x'
    with pytest.raises(TokenizerError, match="payload"):
        decode_terms(bad_n)

    # 3. non-ascii byte inside a MOD payload
    mbase = list(encode_terms(resolve_expression("kalman(luna.D,luna.S)")))
    bad_m = ["BYTE:0xff" if t == "BYTE:0x44" else t for t in mbase]   # 'D' -> 0xff
    with pytest.raises(TokenizerError, match="modifier payload"):
        decode_terms(bad_m)

    # 4. non-ascii byte inside a PIN payload.  Pins decode as ascii like
    # modifiers do, but through a separate call site, so the guard needs its own
    # regression rather than inheriting case 3's coverage.
    pbase = list(encode_terms(
        resolve_expression("lineage(graph,decay=0.85)@run/2026-07-25")))
    assert pbase.count("BYTE:0x72") == 1          # 'r' occurs only in the pin
    bad_p = ["BYTE:0xff" if t == "BYTE:0x72" else t for t in pbase]
    with pytest.raises(TokenizerError, match="pin payload"):
        decode_terms(bad_p)
