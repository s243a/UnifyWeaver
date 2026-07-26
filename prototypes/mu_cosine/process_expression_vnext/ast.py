#!/usr/bin/env python3
"""Source AST and typed AST for the vNext frontend.

Two distinct trees, deliberately:

* **Source AST** — elaboration-preserving.  Keeps spans, positional order,
  *every* named-field occurrence including duplicates, exact numeric lexemes,
  and unchecked annotations.  Sorting or de-duplicating fields here would hide
  the duplicate-field error the elaborator must raise, so the parser never does
  either.
* **Typed AST** — immutable semantic nodes with structural equality.  Spans and
  successfully-checked annotations are absent, so ``pearltrees`` and
  ``pearltrees::corpus`` are the same term (§3.5).

The source tree is **elaboration-preserving**, not lossless: it keeps
everything elaboration and diagnostics need, but it does not promise exact
source reconstruction.  Grouping parentheses, whitespace and comments
are not retained, and atom/string nodes hold decoded values rather than original
spellings.  The narrower claim is deliberate — a "lossless" guarantee this
implementation cannot satisfy would be worse than an accurate one.

Nothing here is identity-bearing.  ``DESIGN_process_expression_patterns.md``
§2.1 freezes the canonical wire schema ``pe-typed-ast-v1`` separately, and this
milestone deliberately does not implement it: ``debug_repr`` below is labelled
noncanonical for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .numerics import Float64Value, NumberLexeme, RealValue


# --------------------------------------------------------------------------
# source spans
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def __str__(self) -> str:
        return f"[{self.start}:{self.end}]"


# --------------------------------------------------------------------------
# source AST — elaboration-preserving, span-carrying, unchecked
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceNode:
    span: Span


@dataclass(frozen=True)
class SourceName(SourceNode):
    """A bare registered name: becomes ``Reference`` or fails."""

    name: str


@dataclass(frozen=True)
class SourceCall(SourceNode):
    name: str
    args: tuple["SourceNode", ...]
    #: Every occurrence, in source order, duplicates included.
    fields: tuple[tuple[str, "SourceNode", Span], ...]


@dataclass(frozen=True)
class SourceAtom(SourceNode):
    value: str


@dataclass(frozen=True)
class SourceString(SourceNode):
    value: str


@dataclass(frozen=True)
class SourceNumber(SourceNode):
    lexeme: NumberLexeme


@dataclass(frozen=True)
class SourceList(SourceNode):
    items: tuple["SourceNode", ...]


@dataclass(frozen=True)
class SourceAnnotated(SourceNode):
    """``Term :: Type`` before the assertion has been checked."""

    term: "SourceNode"
    asserted: "SourceType"


@dataclass(frozen=True)
class SourceVariable(SourceNode):
    """Recognized so it can be rejected precisely, not misparsed."""

    name: str


# --------------------------------------------------------------------------
# types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceType:
    span: Span


@dataclass(frozen=True)
class SourceTypeName(SourceType):
    name: str


@dataclass(frozen=True)
class SourceIndexedType(SourceType):
    name: str
    indices: tuple[Any, ...]


@dataclass(frozen=True)
class SourceFunctionType(SourceType):
    """Recognized for a precise rejection; not implemented this milestone."""

    argument_types: tuple[SourceType, ...]
    result_type: SourceType


@dataclass(frozen=True)
class SourceReferenceIndex(SourceType):
    """A registered reference occupying an index position, e.g. ``[pearltrees]``.

    Kept distinct from :class:`SourceTypeName` so the elaborator can produce a
    :class:`ReferenceIndex` rather than conflating it with a type.
    """

    name: str


@dataclass(frozen=True)
class Type:
    """Semantic type.  Structural equality; no spans."""


@dataclass(frozen=True)
class TypeName(Type):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class IndexedType(Type):
    name: str
    indices: tuple[Any, ...]

    def __str__(self) -> str:
        inner = ",".join(str(i) for i in self.indices)
        return f"{self.name}[{inner}]"


@dataclass(frozen=True)
class ListType(Type):
    element: Type

    def __str__(self) -> str:
        return f"list[{self.element}]"


@dataclass(frozen=True)
class TypeVar(Type):
    """A registry signature's index placeholder, e.g. the ``C`` in ``substrate[C]``.

    Only ever appears inside a *signature*; a fully elaborated term never
    retains one, because every index is substituted during elaboration.
    """

    name: str

    def __str__(self) -> str:
        return self.name


# --------------------------------------------------------------------------
# value indices — NOT types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueIndex:
    """An index that is a *value*, not a type.

    §3.2 indexes a substrate by the corpus it views — ``substrate[pearltrees]``
    — where ``pearltrees`` is a registered corpus *reference*, not a type named
    ``pearltrees``.  Representing it as :class:`TypeName` would conflate the two
    namespaces and make ``substrate[pearltrees]`` indistinguishable from an
    indexed type whose index happens to share a spelling.
    """


@dataclass(frozen=True)
class ReferenceIndex(ValueIndex):
    """A registered reference used as an index, e.g. ``substrate[pearltrees]``."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TermIndex(ValueIndex):
    """EXPERIMENTAL: a whole expression used as an index.

    Needed for ``lineage_op(S, ...) :: abstract_lineage_process[S]`` where ``S``
    is a substrate *expression* such as ``principal_tree(pearltrees)`` rather
    than a bare reference.

    The wire representation of an expression-valued index is an **unresolved
    specification decision** (see the PR's deferred list).  This node is
    in-memory only and deliberately has no serialization; it exists so the
    representative signature can be typed honestly instead of being faked as a
    reference or silently dropped.
    """

    term: "TypedTerm"

    def __str__(self) -> str:
        return _index_display(self.term)


def _index_display(term: "TypedTerm") -> str:
    if isinstance(term, Reference):
        return term.name
    if isinstance(term, Call):
        inner = ",".join(_index_display(a) for a in term.args)
        return f"{term.name}({inner})"
    return f"<{type(term).__name__.lower()}>"


# --------------------------------------------------------------------------
# typed AST — immutable, structural equality, no spans, no annotations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedTerm:
    inferred_type: Type


@dataclass(frozen=True)
class Reference(TypedTerm):
    name: str


@dataclass(frozen=True)
class Call(TypedTerm):
    name: str
    args: tuple[TypedTerm, ...]
    #: Name-keyed and sorted, so field order is not semantic (§2.1 rule 3).
    fields: tuple[tuple[str, TypedTerm], ...]


@dataclass(frozen=True)
class Atom(TypedTerm):
    value: str


@dataclass(frozen=True)
class String(TypedTerm):
    value: str


@dataclass(frozen=True)
class Int(TypedTerm):
    value: int


@dataclass(frozen=True)
class Real(TypedTerm):
    value: RealValue


@dataclass(frozen=True)
class Float64(TypedTerm):
    value: Float64Value


@dataclass(frozen=True)
class ListTerm(TypedTerm):
    items: tuple[TypedTerm, ...]


def debug_repr(term: TypedTerm) -> dict[str, Any]:
    """NONCANONICAL, NON-IDENTITY-BEARING debug rendering.

    Explicitly *not* ``pe-typed-ast-v1``.  This milestone does not implement the
    canonical wire schema, and nothing produced here may be hashed, stored as
    provenance, or joined to a deployed cache.
    """

    base: dict[str, Any] = {"type": str(term.inferred_type)}
    if isinstance(term, Reference):
        return {**base, "kind": "ref", "name": term.name}
    if isinstance(term, Call):
        return {
            **base,
            "kind": "call",
            "name": term.name,
            "args": [debug_repr(a) for a in term.args],
            "fields": [{"name": n, "value": debug_repr(v)} for n, v in term.fields],
        }
    if isinstance(term, Atom):
        return {**base, "kind": "atom", "value": term.value}
    if isinstance(term, String):
        return {**base, "kind": "string", "value": term.value}
    if isinstance(term, Int):
        return {**base, "kind": "int", "value": str(term.value)}
    if isinstance(term, Real):
        return {**base, "kind": "real", "value": term.value.plain_string()}
    if isinstance(term, Float64):
        return {**base, "kind": "float64", "value": term.value.hex16()}
    if isinstance(term, ListTerm):
        return {**base, "kind": "list", "items": [debug_repr(i) for i in term.items]}
    raise TypeError(f"unknown typed term: {type(term).__name__}")
