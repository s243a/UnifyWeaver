#!/usr/bin/env python3
"""Registry-driven elaboration: source AST -> typed ground term.

Implements the checking half of ``DESIGN_process_expression_patterns.md`` §3.5:

1. types come from the injected registry;
2. literal types are inferred from the *expected* type, not the lexeme's shape;
3. indexed ownership constraints unify (``substrate[C]`` binds ``C``);
4. every ``::`` assertion is checked;
5. conflicts and unconsumed fields are rejected before any semantic output; and
6. normalized inferred types are retained on every node.

Two rules are easy to get backwards and are therefore stated explicitly:

* an annotation **asserts or narrows, never converts** — ``0.85::int`` is an
  error, not a truncation;
* a *successful redundant* annotation leaves no trace, so ``pearltrees`` and
  ``pearltrees::corpus`` are the same typed term.

Nothing here mints identity.  There is no hashing, no serialization to
``pe-typed-ast-v1``, and no notion of a deployed process.
"""
from __future__ import annotations

from typing import Any, Mapping

from .ast import (
    Atom,
    Call,
    ReferenceIndex,
    SourceReferenceIndex,
    TermIndex,
    ValueIndex,
    Float64,
    IndexedType,
    Int,
    ListTerm,
    ListType,
    Real,
    Reference,
    SourceAnnotated,
    SourceAtom,
    SourceCall,
    SourceIndexedType,
    SourceList,
    SourceName,
    SourceNode,
    SourceNumber,
    SourceString,
    SourceType,
    SourceTypeName,
    String,
    Type,
    TypeName,
    TypeVar,
    TypedTerm,
)
from .numerics import NumericError, to_float64, to_int, to_real
from .registry import Entry, Registry, RegistryError


class ElaborationError(ValueError):
    """A term is well-formed text but ill-typed, or violates the registry."""

    def __init__(self, message: str, span=None):
        self.span = span
        super().__init__(message if span is None else f"{message} {span}")


def _substitute(declared: Type, bindings: Mapping[str, Any]) -> Type:
    """Replace signature index variables with their bound values."""

    if isinstance(declared, TypeVar):
        bound = bindings.get(declared.name)
        return bound if bound is not None else declared
    if isinstance(declared, IndexedType):
        return IndexedType(
            declared.name,
            tuple(
                _substitute(i, bindings) if isinstance(i, Type) else i
                for i in declared.indices
            ),
        )
    if isinstance(declared, ValueIndex):
        return declared
    if isinstance(declared, ListType):
        return ListType(_substitute(declared.element, bindings))
    return declared


def _unify(declared: Type, actual: Type, bindings: dict[str, Any]) -> bool:
    """Match an actual type against a signature type, binding index variables.

    A variable already bound must match consistently — this is what stops
    ``cross(principal_tree(pearltrees), principal_tree(simplewiki))``.
    """

    if isinstance(declared, TypeVar):
        existing = bindings.get(declared.name)
        if existing is None:
            bindings[declared.name] = actual
            return True
        return existing == actual
    if isinstance(declared, TypeName):
        return isinstance(actual, TypeName) and declared.name == actual.name
    if isinstance(declared, ListType):
        return isinstance(actual, ListType) and _unify(
            declared.element, actual.element, bindings
        )
    if isinstance(declared, IndexedType):
        if not isinstance(actual, IndexedType) or declared.name != actual.name:
            return False
        if len(declared.indices) != len(actual.indices):
            return False
        return all(
            _unify(d, a, bindings) if isinstance(d, Type) else d == a
            for d, a in zip(declared.indices, actual.indices)
        )
    return False


def _resolve_source_type(node: SourceType, registry: Registry):
    if isinstance(node, SourceReferenceIndex):
        # An index may name a *reference* value.  A callable is not a value, so
        # `list[principal_tree]` and `substrate[principal_tree]` must not pass.
        try:
            entry = registry.get(node.name)
        except RegistryError as exc:
            raise ElaborationError(str(exc), node.span) from exc
        if entry.kind != "reference":
            raise ElaborationError(
                f"{node.name!r} is a callable and cannot be used as a value index",
                node.span,
            )
        return ReferenceIndex(node.name)
    if isinstance(node, SourceTypeName):
        return TypeName(node.name)
    if isinstance(node, SourceIndexedType):
        indices = tuple(_resolve_source_type(i, registry) for i in node.indices)
        if node.name == "list":
            if len(indices) != 1:
                raise ElaborationError("list takes exactly one element type", node.span)
            element = indices[0]
            if not isinstance(element, Type) or isinstance(element, ValueIndex):
                # A value index is not a type; `list[pearltrees]` is ill-formed.
                raise ElaborationError(
                    f"list element type must be a type, not the value {element}",
                    node.span,
                )
            return ListType(element)
        return IndexedType(node.name, indices)
    raise ElaborationError(f"unsupported type form: {type(node).__name__}", node.span)


def _literal(node: SourceNode, expected: Type | None) -> TypedTerm:
    """Type a literal against the *expected* type (§3.6).

    The declared field type wins over the lexeme's surface shape, which is why
    ``margin(t=1)`` is a ``real`` and not an ``int``.
    """

    if isinstance(node, SourceNumber):
        if expected is None:
            raise ElaborationError(
                "a numeric literal needs a declared type; the surface spelling "
                "does not determine whether it is int, real, or float64",
                node.span,
            )
        if not isinstance(expected, TypeName):
            raise ElaborationError(
                f"numeric literal cannot satisfy {expected}", node.span
            )
        try:
            if expected.name == "int":
                return Int(expected, to_int(node.lexeme))
            if expected.name == "real":
                return Real(expected, to_real(node.lexeme))
            if expected.name == "float64":
                return Float64(expected, to_float64(node.lexeme))
        except NumericError as exc:
            raise ElaborationError(str(exc), node.span) from exc
        raise ElaborationError(
            f"numeric literal cannot satisfy {expected}", node.span
        )
    if isinstance(node, SourceAtom):
        return Atom(TypeName("atom"), node.value)
    if isinstance(node, SourceString):
        return String(TypeName("string"), node.value)
    raise ElaborationError(  # pragma: no cover - caller dispatches
        f"not a literal: {type(node).__name__}", node.span
    )


def _elaborate(
    node: SourceNode, registry: Registry, expected: Type | None
) -> TypedTerm:
    if isinstance(node, SourceAnnotated):
        asserted = _resolve_source_type(node.asserted, registry)
        # The assertion narrows what the inner term may be, but never converts.
        inner = _elaborate(node.term, registry, asserted)
        if inner.inferred_type != asserted:
            raise ElaborationError(
                f"annotation {asserted} conflicts with inferred type "
                f"{inner.inferred_type}; '::' asserts or narrows and never converts",
                node.span,
            )
        if expected is not None and not _unify(expected, asserted, {}):
            raise ElaborationError(
                f"expected {expected}, found {asserted}", node.span
            )
        # A successful redundant annotation leaves no semantic trace (§3.5).
        return inner

    if isinstance(node, (SourceNumber, SourceAtom, SourceString)):
        term = _literal(node, expected)
        if expected is not None and term.inferred_type != expected:
            raise ElaborationError(
                f"expected {expected}, found {term.inferred_type}", node.span
            )
        return term

    if isinstance(node, SourceList):
        if expected is None or not isinstance(expected, ListType):
            raise ElaborationError(
                "a list literal needs a declared list type", node.span
            )
        items = tuple(
            _elaborate(item, registry, expected.element) for item in node.items
        )
        for item in items:
            if item.inferred_type != expected.element:
                raise ElaborationError(
                    f"list is not homogeneous: expected {expected.element}, "
                    f"found {item.inferred_type}",
                    node.span,
                )
        return ListTerm(expected, items)

    if isinstance(node, SourceName):
        try:
            entry = registry.get(node.name)
        except RegistryError as exc:
            raise ElaborationError(str(exc), node.span) from exc
        if entry.kind != "reference":
            raise ElaborationError(
                f"{node.name!r} is a callable and must be applied", node.span
            )
        actual = entry.result_type
        if expected is not None and not _unify(expected, actual, {}):
            raise ElaborationError(
                f"expected {expected}, found {actual}", node.span
            )
        return Reference(actual, node.name)

    if isinstance(node, SourceCall):
        return _elaborate_call(node, registry, expected)

    raise ElaborationError(f"unsupported term: {type(node).__name__}", node.span)


def _elaborate_call(
    node: SourceCall, registry: Registry, expected: Type | None
) -> TypedTerm:
    try:
        entry: Entry = registry.get(node.name)
    except RegistryError as exc:
        raise ElaborationError(str(exc), node.span) from exc
    if entry.kind != "call":
        raise ElaborationError(
            f"{node.name!r} is a reference and takes no arguments", node.span
        )

    if len(node.args) != len(entry.arg_types):
        raise ElaborationError(
            f"{node.name} expects {len(entry.arg_types)} positional argument(s), "
            f"got {len(node.args)}",
            node.span,
        )

    bindings: dict[str, Any] = {}
    args: list[TypedTerm] = []
    for index, (arg, declared) in enumerate(zip(node.args, entry.arg_types)):
        want = _substitute(declared, bindings)
        term = _elaborate(arg, registry, want if _is_concrete(want) else None)
        if not _unify(declared, term.inferred_type, bindings):
            raise ElaborationError(
                f"{node.name} argument {index} expects "
                f"{_substitute(declared, bindings)}, found {term.inferred_type}",
                arg.span,
            )
        args.append(term)

    # Value-indexed ownership: substrate[C] is indexed by the corpus it views,
    # so C binds to an argument's *value*, not to its type.
    for var, position in entry.binds:
        argument = args[position]
        if isinstance(argument, Reference):
            index: Any = ReferenceIndex(argument.name)
        else:
            # EXPERIMENTAL: an expression-valued index.  Its wire form is an
            # open specification decision; see the module docstring in ast.py.
            index = TermIndex(argument)
        existing = bindings.get(var)
        if existing is not None and existing != index:
            # A bind must never silently overwrite an ownership constraint that
            # argument unification already established.
            raise ElaborationError(
                f"{node.name} would rebind index {var} from {existing} to {index}; "
                "ownership constraints are established once",
                node.args[position].span,
            )
        bindings[var] = index

    # Duplicates are detected before any normalization, which is why the parser
    # keeps every occurrence in source order.
    seen: dict[str, Any] = {}
    for field_name, value_node, key_span in node.fields:
        if field_name in seen:
            raise ElaborationError(
                f"duplicate named field {field_name!r} for {node.name}", key_span
            )
        seen[field_name] = value_node

    for field_name, key_span in ((n, s) for n, _, s in node.fields):
        if entry.field(field_name) is None:
            raise ElaborationError(
                f"{node.name} does not consume field {field_name!r}", key_span
            )

    fields: dict[str, TypedTerm] = {}

    # Pass 1: every authored field.  Doing these first makes the result
    # independent of the order fields happen to appear in the JSON — otherwise
    # an authored field declared later could not bind an index that a default
    # declared earlier needs.
    for spec in entry.fields:
        if spec.name in seen:
            want = _substitute(spec.type, bindings)
            term = _elaborate(
                seen[spec.name], registry, want if _is_concrete(want) else None
            )
            if not _unify(spec.type, term.inferred_type, bindings):
                raise ElaborationError(
                    f"{node.name} field {spec.name!r} expects "
                    f"{_substitute(spec.type, bindings)}, found {term.inferred_type}",
                    node.span,
                )
            fields[spec.name] = term

    # Pass 2: required-but-absent, then defaults, against fully bound indices.
    for spec in entry.fields:
        if spec.name in seen:
            continue
        if spec.required:
            raise ElaborationError(
                f"{node.name} is missing required field {spec.name!r}", node.span
            )
        elif spec.default is not None:
            want = _substitute(spec.type, bindings)
            if _contains_typevar(want):
                # Rather than implement a general dependency solver, refuse a
                # default whose type is still unresolved.  Accepting it would
                # make the result depend on JSON field order.
                raise ElaborationError(
                    f"{node.name} default for {spec.name!r} has unresolved type "
                    f"{want}; defaults may not depend on later bindings",
                    node.span,
                )
            term = _default_term(spec, registry, want)
            # An inserted default goes through the same unification as an
            # authored value, so an ill-typed default cannot slip in.
            if not _unify(spec.type, term.inferred_type, bindings):
                raise ElaborationError(
                    f"{node.name} default for {spec.name!r} is ill-typed: expected "
                    f"{want}, found {term.inferred_type}",
                    node.span,
                )
            fields[spec.name] = term

    result = _substitute(entry.result_type, bindings)
    if isinstance(result, TypeVar) or _contains_typevar(result):
        raise ElaborationError(
            f"{node.name} result type {result} is underconstrained", node.span
        )
    if expected is not None and not _unify(expected, result, {}):
        raise ElaborationError(f"expected {expected}, found {result}", node.span)

    return Call(result, node.name, tuple(args), tuple(sorted(fields.items())))


def _default_term(spec, registry: Registry, want: Type) -> TypedTerm:
    """Registry defaults are inserted explicitly (§2.1 rule 4).

    A default is surface *text* elaborated through the same path as an authored
    value, so an explicit and an elided default cannot diverge.  The registry
    loader enforces that a default is a JSON string precisely so no numeric
    rounding can happen before this point.
    """

    from .functional_parser import parse_functional

    source = parse_functional(spec.default, registry)
    return _elaborate(source, registry, want)


def _is_concrete(declared: Type) -> bool:
    return not _contains_typevar(declared)


def _contains_typevar(declared: Type) -> bool:
    if isinstance(declared, TypeVar):
        return True
    if isinstance(declared, IndexedType):
        return any(
            _contains_typevar(i) for i in declared.indices if isinstance(i, Type)
        )
    if isinstance(declared, ListType):
        return _contains_typevar(declared.element)
    return False


def _assert_no_type_vars(term: TypedTerm, path: str = "result") -> None:
    """No ``TypeVar`` may survive into a supposedly ground typed term.

    Checked recursively on the way out rather than trusted per-node: a
    signature bug, an unbound index, or a reference declaring a variable would
    otherwise leak an abstract type into a term the caller believes is ground.
    """

    if _contains_typevar(term.inferred_type):
        raise ElaborationError(
            f"{path} retains unresolved type variable(s) in {term.inferred_type}"
        )
    if isinstance(term, Call):
        for index, arg in enumerate(term.args):
            _assert_no_type_vars(arg, f"{path}.{term.name}[{index}]")
        for name, value in term.fields:
            _assert_no_type_vars(value, f"{path}.{term.name}.{name}")
    elif isinstance(term, ListTerm):
        for index, item in enumerate(term.items):
            _assert_no_type_vars(item, f"{path}[{index}]")


def elaborate(source: SourceNode, registry: Registry) -> TypedTerm:
    """Elaborate a source AST into an in-memory typed ground term.

    The result is *not* identity-bearing and is not serialized to any canonical
    schema by this milestone.
    """

    term = _elaborate(source, registry, None)
    _assert_no_type_vars(term)
    return term
