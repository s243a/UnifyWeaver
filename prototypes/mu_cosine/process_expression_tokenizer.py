#!/usr/bin/env python3
"""Reversible typed tokenizer for process expressions — step 2, part 1.

``DESIGN_expression_encoder_future.md`` §3.1 requires a stream where

    canonical AST -> tokens -> AST -> canonical bytes

round-trips exactly, with a finite versioned vocabulary and an explicit
256-byte literal fallback.  No hashing, no subword tokenization, and no
unknown-token replacement is permitted anywhere on the reconstruction path.

Two separations keep this honest:

* **Structure is inherited, not reinvented.** The token *strings* come from
  ``process_expression_contract.token_stream`` — the fixture authority sealed in
  ``PROCESS_EXPRESSION_GOLDEN_v2.json``.  This module assigns IDs to those
  strings and defines the inverse.  It therefore cannot drift from the sealed
  bundle by construction, which is the whole reason the bundle was frozen first.
* **The vocabulary is versioned independently of the grammar** (§3.1).
  ``VOCAB_VERSION`` moves when the ID assignment changes;
  ``CONTRACT_VERSION`` moves when the structure does.

What this module is *not*: it is not the encoder, and the exactness it
guarantees is a serialization property.  Per
``DESIGN_process_expression_generator.md`` §5.0 the learned decoder is graded by
tolerance — "close is good" applies to the model, never to this layer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import sys
from typing import Any, Iterator, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from process_cards import REGISTRY, Node, canonical, parse, validate
from process_expression_contract import (
    CONTRACT_VERSION,
    ContractError,
    ResolvedNode,
    resolve,
    token_stream,
)

#: Bumped when the ID assignment changes.  Independent of CONTRACT_VERSION,
#: which tracks the structure those IDs are assigned to.
VOCAB_VERSION = "tok-v1"

#: Bounds the indexed structural tokens.  Deliberately wider than the measured
#: envelope (arity 3, list length 2, one modifier, no pins) so the vocabulary
#: does not need re-versioning for a slightly larger generated corpus.  A value
#: beyond these bounds fails closed rather than wrapping — §8 of the position
#: note forbids clipping or reusing a final in-range code.
MAX_INDEX = 16

_LIST_ELEMENT_TYPE = {"number_list": "number", "int_list": "int"}


class TokenizerError(ValueError):
    """A token stream could not be built, decoded, or round-tripped."""


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------


def _vocabulary_terms() -> list[str]:
    """Every token string the contract can emit, in a frozen order.

    Derived from the pinned registry rather than collected from a corpus: the
    vocabulary is finite and knowable in advance, so nothing here depends on
    which expressions happen to have been generated.
    """

    terms: list[str] = ["<BOS>", "<EOS>"]
    terms += [
        "<NODE>", "</NODE>",
        "<ARGS>", "</ARGS>",
        "<KWARGS>", "</KWARGS>",
        "<MODS>", "</MODS>",
        "<PINS>", "</PINS>",
        "<LIST>", "</LIST>",
        "<INT>", "</INT>",
        "<NUMBER>", "</NUMBER>",
        "<STRING>", "</STRING>",
        "</ARG>", "</KW>", "</ITEM>", "</MOD>", "</PIN>",
    ]
    terms += ["<KIND:atom>", "<KIND:apply>"]
    terms += [f"<NAME:{name}>" for name in sorted(REGISTRY)]
    terms += [f"<OUTPUT:{output}>" for output in sorted({s.output for s in REGISTRY.values()})]
    keys = sorted({key for s in REGISTRY.values() for key in s.kwargs})
    terms += [f"<KW:{key}>" for key in keys]
    for index in range(MAX_INDEX):
        terms += [f"<ARG:{index}>", f"<ITEM:{index}>", f"<MOD:{index}>", f"<PIN:{index}>"]
    # Explicit 256-byte fallback: every literal payload byte has a token.
    terms += [f"BYTE:0x{value:02x}" for value in range(256)]

    if len(set(terms)) != len(terms):
        raise TokenizerError("vocabulary contains a duplicate term")
    return terms


@dataclass(frozen=True)
class Vocabulary:
    version: str
    contract_version: str
    terms: tuple[str, ...]
    id_by_term: Mapping[str, int]
    term_by_id: Mapping[int, str]

    @property
    def size(self) -> int:
        return len(self.terms)

    @property
    def digest(self) -> str:
        payload = "\n".join(
            [self.version, self.contract_version, *self.terms]
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def build_vocabulary() -> Vocabulary:
    terms = tuple(_vocabulary_terms())
    return Vocabulary(
        version=VOCAB_VERSION,
        contract_version=CONTRACT_VERSION,
        terms=terms,
        id_by_term={term: index for index, term in enumerate(terms)},
        term_by_id=dict(enumerate(terms)),
    )


VOCAB = build_vocabulary()


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------


def encode_terms(node: ResolvedNode) -> tuple[str, ...]:
    """The contract's structural stream, unchanged."""

    return tuple(token.token for token in token_stream(node))


def encode(node: ResolvedNode, vocab: Vocabulary = VOCAB) -> tuple[int, ...]:
    ids = []
    for term in encode_terms(node):
        try:
            ids.append(vocab.id_by_term[term])
        except KeyError as exc:
            # An out-of-vocabulary structural token means an index exceeded
            # MAX_INDEX.  Fail closed: no hashing, no bucketing, no UNK.
            raise TokenizerError(f"token outside the frozen vocabulary: {term}") from exc
    return tuple(ids)


def encode_expression(expression: str, vocab: Vocabulary = VOCAB) -> tuple[int, ...]:
    return encode(resolve(parse(expression)), vocab)


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------


class _Cursor:
    def __init__(self, terms: Sequence[str]):
        self.terms = list(terms)
        self.position = 0

    def peek(self) -> str:
        if self.position >= len(self.terms):
            raise TokenizerError("token stream ended early")
        return self.terms[self.position]

    def take(self) -> str:
        term = self.peek()
        self.position += 1
        return term

    def expect(self, term: str) -> str:
        actual = self.take()
        if actual != term:
            raise TokenizerError(f"expected {term}, got {actual}")
        return actual

    def accept(self, term: str) -> bool:
        if self.position < len(self.terms) and self.terms[self.position] == term:
            self.position += 1
            return True
        return False

    @property
    def exhausted(self) -> bool:
        return self.position >= len(self.terms)


def _tagged(term: str, prefix: str) -> str:
    if not term.startswith(prefix) or not term.endswith(">"):
        raise TokenizerError(f"expected a {prefix}…> token, got {term}")
    return term[len(prefix) : -1]


def _take_payload(cursor: _Cursor, closing: str) -> bytes:
    payload = bytearray()
    while not cursor.accept(closing):
        term = cursor.take()
        if not term.startswith("BYTE:0x"):
            raise TokenizerError(f"expected a literal byte, got {term}")
        payload.append(int(term[7:], 16))
    return bytes(payload)


def _value_from(kind: str, payload: bytes) -> Any:
    text = payload.decode("utf-8")
    if kind == "string":
        return text
    if kind == "int":
        return int(text)
    if kind == "number":
        # The declared kind is `number`, but the canonical lexical form decides
        # int-vs-float: `margin(t=1)` renders "1" and must decode back to the
        # int 1, or the canonical string — and therefore the identity — moves.
        return float(text) if any(ch in text for ch in ".eE") else int(text)
    raise TokenizerError(f"unsupported scalar kind: {kind!r}")


def _decode_value(cursor: _Cursor, kind: str) -> Any:
    if cursor.accept("<LIST>"):
        element_kind = _LIST_ELEMENT_TYPE.get(kind)
        if element_kind is None:
            raise TokenizerError(f"list payload for non-list kind {kind!r}")
        items = []
        while not cursor.accept("</LIST>"):
            index = int(_tagged(cursor.take(), "<ITEM:"))
            if index != len(items):
                raise TokenizerError("list item indices must be contiguous")
            items.append(_decode_value(cursor, element_kind))
            cursor.expect("</ITEM>")
        return tuple(items)

    opener = cursor.take()
    mapping = {"<INT>": ("int", "</INT>"), "<NUMBER>": ("number", "</NUMBER>"),
               "<STRING>": ("string", "</STRING>")}
    if opener not in mapping:
        raise TokenizerError(f"expected a value token, got {opener}")
    tagged_kind, closing = mapping[opener]
    if tagged_kind != kind:
        # The stream's own tag must agree with the registry-declared kind.
        raise TokenizerError(
            f"stream declares {tagged_kind!r} for a field registered {kind!r}"
        )
    return _value_from(kind, _take_payload(cursor, closing))


def _decode_node(cursor: _Cursor) -> Node:
    cursor.expect("<NODE>")
    kind = _tagged(cursor.take(), "<KIND:")
    name = _tagged(cursor.take(), "<NAME:")
    output = _tagged(cursor.take(), "<OUTPUT:")
    if name not in REGISTRY:
        raise TokenizerError(f"unregistered name in stream: {name}")
    signature = REGISTRY[name]
    if output != signature.output:
        raise TokenizerError(f"{name} output {output!r} contradicts the registry")
    if kind not in ("atom", "apply"):
        raise TokenizerError(f"unknown KIND: {kind!r}")

    cursor.expect("<ARGS>")
    args = []
    while not cursor.accept("</ARGS>"):
        index = int(_tagged(cursor.take(), "<ARG:"))
        if index != len(args):
            raise TokenizerError("argument indices must be contiguous")
        args.append(_decode_node(cursor))
        cursor.expect("</ARG>")

    cursor.expect("<KWARGS>")
    kwargs = []
    while not cursor.accept("</KWARGS>"):
        key = _tagged(cursor.take(), "<KW:")
        spec = signature.kwargs.get(key)
        if spec is None:
            raise TokenizerError(f"unknown kwarg for {name}: {key}")
        kwargs.append((key, _decode_value(cursor, spec.kind)))
        cursor.expect("</KW>")

    cursor.expect("<MODS>")
    mods = []
    while not cursor.accept("</MODS>"):
        index = int(_tagged(cursor.take(), "<MOD:"))
        if index != len(mods):
            raise TokenizerError("modifier indices must be contiguous")
        mods.append(_take_payload(cursor, "</MOD>").decode("ascii"))

    cursor.expect("<PINS>")
    pins = []
    while not cursor.accept("</PINS>"):
        index = int(_tagged(cursor.take(), "<PIN:"))
        if index != len(pins):
            raise TokenizerError("pin indices must be contiguous")
        pins.append(_take_payload(cursor, "</PIN>").decode("ascii"))

    cursor.expect("</NODE>")

    node = Node(name, tuple(args), tuple(sorted(kwargs)), tuple(mods), tuple(pins))
    # KIND is derived, so it must agree with what the registry implies rather
    # than being taken on the stream's word (§3.1).
    implied = "apply" if (node.args or node.kwargs) or (
        signature.operator and not signature.atom
    ) else "atom"
    if implied != kind:
        raise TokenizerError(f"{name} KIND {kind!r} contradicts the registry")
    return node


def decode_terms(terms: Sequence[str]) -> Node:
    cursor = _Cursor(terms)
    cursor.expect("<BOS>")
    node = _decode_node(cursor)
    cursor.expect("<EOS>")
    if not cursor.exhausted:
        raise TokenizerError("trailing tokens after <EOS>")
    return validate(node)


def decode(ids: Sequence[int], vocab: Vocabulary = VOCAB) -> Node:
    terms = []
    for token_id in ids:
        try:
            terms.append(vocab.term_by_id[token_id])
        except KeyError as exc:
            raise TokenizerError(f"token id outside the vocabulary: {token_id}") from exc
    return decode_terms(terms)


def round_trip(expression: str, vocab: Vocabulary = VOCAB) -> str:
    """Encode then decode, returning the canonical string of the result."""

    decoded = decode(encode_expression(expression, vocab), vocab)
    return canonical(decoded)


def assert_round_trips(expression: str, vocab: Vocabulary = VOCAB) -> None:
    original = canonical(parse(expression))
    recovered = round_trip(expression, vocab)
    if recovered != original:
        raise TokenizerError(
            f"round trip changed the canonical identity: {original!r} -> {recovered!r}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from process_cards import PROCESSES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expression", action="append", default=[])
    parser.add_argument("--all-registered", action="store_true")
    args = parser.parse_args(argv)

    print(f"vocabulary {VOCAB.version} / contract {VOCAB.contract_version}")
    print(f"  size {VOCAB.size}  digest {VOCAB.digest[:32]}…")
    expressions = list(args.expression)
    if args.all_registered or not expressions:
        expressions += list(PROCESSES.values())
    for expression in expressions:
        ids = encode_expression(expression)
        assert_round_trips(expression)
        print(f"  ok  {len(ids):4d} tokens  {canonical(parse(expression))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
