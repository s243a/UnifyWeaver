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

Milestone 2 adds variables to the same code path rather than beside it.  There
is one ``_elaborate``; a ``scope`` of ``None`` means "ground only" and makes any
variable an error, so ground expressions elaborate through exactly the checks
they did before.  Three entry points name the three states:
:func:`elaborate` (compatibility, ground, bare term), :func:`elaborate_ground`
(ground, wrapped) and :func:`elaborate_pattern` (may contain variables).

Nothing here mints identity.  There is no hashing, no serialization to
``pe-typed-ast-v1``, and no notion of a deployed process.
"""
from __future__ import annotations

from typing import Any, Mapping

from .ast import (
    Atom,
    Call,
    PatternIndex,
    PatternVariable,
    ReferenceIndex,
    SourceReferenceIndex,
    SourceVariable,
    SourceVariableIndex,
    TermIndex,
    ValueIndex,
    VarId,
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
    new_var_id,
)
from .numerics import NumericError, to_float64, to_int, to_real
from .patterns import (
    GroundAST,
    GroundingError,
    PatternAST,
    PatternVar,
    ground,
    make_ground,
)
from .registry import Entry, Registry, RegistryError


class ElaborationError(ValueError):
    """A term is well-formed text but ill-typed, or violates the registry."""

    def __init__(self, message: str, span=None):
        self.span = span
        super().__init__(message if span is None else f"{message} {span}")


class _Scope:
    """Variable scope for one pattern elaboration.

    Its whole job is to answer "is this the same variable as that one?", which
    is why the parser cannot do it: two ``S`` tokens are one variable and two
    ``_`` tokens are two, and only something that spans the whole expression
    knows which.
    """

    def __init__(self) -> None:
        self._named: dict[str, VarId] = {}
        self.constraints: dict[VarId, Type | None] = {}
        self.term_positions: set[VarId] = set()
        #: Author-visible variables in first-occurrence order.  Inferred
        #: constraint variables are deliberately excluded: they have no name, no
        #: author intent behind them, and admitting them to the binding table
        #: would let a caller bind an artifact of type inference.
        self.order: list[VarId] = []

    def _register(self, var: VarId) -> VarId:
        if var not in self.constraints:
            self.constraints[var] = None
            self.order.append(var)
        return var

    def named(self, name: str) -> VarId:
        var = self._named.get(name)
        if var is None:
            var = new_var_id("named", name)
            self._named[name] = var
            self._register(var)
        return var

    def anonymous(self) -> VarId:
        return self._register(new_var_id("anonymous", "_"))

    def inferred(self, hint: str) -> VarId:
        # Not registered: an inferred variable is resolved by unification, never
        # by a caller.
        return new_var_id("inferred", hint)

    def variable_for(self, name: str) -> VarId:
        return self.anonymous() if name == "_" else self.named(name)

    def table(self) -> tuple[PatternVar, ...]:
        return tuple(
            PatternVar(
                var=var,
                constraint=self.constraints[var],
                in_term_position=var in self.term_positions,
            )
            for var in self.order
        )


def _freshen(declared: Any, scope: _Scope) -> Any:
    """Replace a signature's own index placeholders with fresh pattern indices.

    §3.5: ``lineage_op(S)`` infers ``S :: substrate[C]`` *for a fresh C*.  The
    registry's ``C`` is a :class:`TypeVar` shared by every use of the signature,
    so reusing it as the variable's constraint would make two unrelated
    expressions appear to share an ownership constraint.
    """

    if isinstance(declared, TypeVar):
        return PatternIndex(scope.inferred(declared.name))
    if isinstance(declared, IndexedType):
        return IndexedType(
            declared.name,
            tuple(
                _freshen(i, scope) if isinstance(i, Type) else i
                for i in declared.indices
            ),
        )
    if isinstance(declared, ListType):
        return ListType(_freshen(declared.element, scope))
    return declared


def _expectation(node: SourceNode, want: Type) -> Type | None:
    """What to hand down as the expected type.

    Ground elaboration keeps its pre-existing rule — an expectation that still
    carries signature variables is dropped, because a literal cannot satisfy one
    and the diagnostic would be confusing.  A *variable* is the exception: the
    slot's declared type is precisely what constrains it, so dropping it there
    would make every ``lineage_op(S)`` underconstrained.
    """

    if _is_concrete(want):
        return want
    if isinstance(node, (SourceVariable, SourceAnnotated)):
        return want
    return None


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


def _resolve_source_type(node: SourceType, registry: Registry, scope: _Scope | None):
    if isinstance(node, SourceVariableIndex):
        if scope is None:
            raise ElaborationError(
                f"index variable {node.name!r} appears in a ground expression; "
                "use elaborate_pattern() for expressions containing variables",
                node.span,
            )
        return PatternIndex(scope.variable_for(node.name))
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
        indices = tuple(_resolve_source_type(i, registry, scope) for i in node.indices)
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
    node: SourceNode, registry: Registry, expected: Type | None, scope: _Scope | None
) -> TypedTerm:
    if isinstance(node, SourceVariable):
        return _elaborate_variable(node, expected, scope)

    if isinstance(node, SourceAnnotated):
        asserted = _resolve_source_type(node.asserted, registry, scope)
        # The assertion narrows what the inner term may be, but never converts.
        inner = _elaborate(node.term, registry, asserted, scope)
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
            _elaborate(item, registry, expected.element, scope) for item in node.items
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
        return _elaborate_call(node, registry, expected, scope)

    raise ElaborationError(f"unsupported term: {type(node).__name__}", node.span)


def _elaborate_variable(
    node: SourceVariable, expected: Type | None, scope: _Scope | None
) -> TypedTerm:
    if scope is None:
        raise ElaborationError(
            f"variable {node.name!r} appears in a ground expression; use "
            "elaborate_pattern() for expressions containing variables",
            node.span,
        )
    if expected is None:
        raise ElaborationError(
            f"variable {node.name!r} is underconstrained: it carries no '::' "
            "annotation and the position it occupies declares no type",
            node.span,
        )
    constraint = _freshen(expected, scope)
    if not isinstance(constraint, Type):  # pragma: no cover - registry forbids it
        raise ElaborationError(
            f"variable {node.name!r} cannot be constrained by the value index "
            f"{constraint}",
            node.span,
        )
    var = scope.variable_for(node.name)
    previous = scope.constraints.get(var)
    if previous is not None and previous != constraint:
        # Two occurrences of one named variable are one variable, so two
        # constraints must agree; silently keeping either would make the pattern
        # mean something the author did not write.
        raise ElaborationError(
            f"variable {node.name!r} is constrained as both {previous} and "
            f"{constraint}",
            node.span,
        )
    scope.constraints[var] = constraint
    scope.term_positions.add(var)
    return PatternVariable(constraint, var)


def _elaborate_call(
    node: SourceCall, registry: Registry, expected: Type | None, scope: _Scope | None
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
        term = _elaborate(arg, registry, _expectation(arg, want), scope)
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
        elif isinstance(argument, PatternVariable):
            # The ownership index *is* the author's variable, so it must stay a
            # pattern index rather than becoming a TermIndex wrapping a variable
            # — otherwise grounding would have to reach inside an index to
            # substitute, and `binds` would produce a shape that direct ground
            # elaboration never produces.
            index = PatternIndex(argument.var)
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
            value_node = seen[spec.name]
            term = _elaborate(
                value_node, registry, _expectation(value_node, want), scope
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
    # Scope is None even inside a pattern: a registry default is fixed text, and
    # a default that introduced a variable would add a binding obligation the
    # author never wrote.
    return _elaborate(source, registry, want, None)


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

    Compatibility convenience, retained from milestone 1 and **ground-only**: a
    variable anywhere in ``source`` is an :class:`ElaborationError`, never a
    silently accepted free variable.  It returns the bare
    :class:`~.ast.TypedTerm`; :func:`elaborate_ground` returns the same term
    wrapped in the :class:`~.patterns.GroundAST` state type, and
    ``elaborate_ground(s, r).term == elaborate(s, r)`` always holds.  Prefer
    :func:`elaborate_ground` in new code, because a bare ``TypedTerm`` does not
    say which state-machine state it is in.

    The result is *not* identity-bearing and is not serialized to any canonical
    schema by this milestone.
    """

    term = _elaborate(source, registry, None, None)
    _assert_no_type_vars(term)
    return term


def elaborate_ground(source: SourceNode, registry: Registry) -> GroundAST:
    """Elaborate a variable-free source AST into a ``GroundAST``.

    The groundness proof runs on the way out, so the state type is earned rather
    than asserted.
    """

    return make_ground(
        elaborate(source, registry),
        what="ground elaboration",
        registry=registry,
    )


def elaborate_pattern(source: SourceNode, registry: Registry) -> PatternAST:
    """Elaborate a source AST that may contain variables into a ``PatternAST``.

    Type checking is the same checking ground elaboration does — signatures,
    ownership indices, annotations, field consumption — with variables treated
    as terms of their constrained type.  A pattern that cannot type-check is
    rejected here rather than at grounding, which is why
    ``cross_substrate(S::substrate[C], T::substrate[OtherC])`` fails before any
    binding is supplied.
    """

    scope = _Scope()
    term = _elaborate(source, registry, None, scope)
    _assert_no_type_vars(term)
    return PatternAST(term=term, variables=scope.table(), registry=registry)


def ground_surface(
    pattern: PatternAST, registry: Registry, bindings: Mapping[Any, str]
) -> GroundAST:
    """Convenience: parse surface-text bindings, then :func:`~.patterns.ground`.

    Kept separate from :func:`~.patterns.ground` on purpose.  Grounding takes
    elaborated values so that a malformed binding *string* raises a parse or
    elaboration error from this function, and a genuinely unsatisfiable binding
    raises a ``GroundingError`` from that one — collapsing the two would make
    "your text is malformed" and "your bindings do not ground this pattern"
    indistinguishable.

    Each binding is elaborated against its variable's constraint, which is what
    lets ``D = "0.85"`` become a ``real`` rather than needing ``0.85::real``.
    """

    from .functional_parser import parse_functional

    if registry is not pattern.registry:
        raise GroundingError(
            f"ground_surface() was given registry {registry.label!r} but the "
            f"pattern was elaborated against {pattern.registry_label!r}; the "
            "bindings would be parsed against names the pattern never saw"
        )
    prepared: dict[Any, TypedTerm] = {}
    for key, text in bindings.items():
        if isinstance(key, VarId):
            matches = [v for v in pattern.variables if v.var == key]
        else:
            matches = [v for v in pattern.named_variables if v.name == key]
        if not matches:
            # Let ground() own the "unknown binding" diagnostic; passing the
            # value through unchanged keeps one message for one failure.
            prepared[key] = text  # type: ignore[assignment]
            continue
        spec = matches[0]
        source = parse_functional(text, registry)
        # A constraint that still mentions a pattern index cannot serve as an
        # *expected* type here: `substrate[C]` would reject the very value that
        # determines C.  Those constraints are checked by ground(), whose
        # unification can bind index variables; only fully determined
        # constraints are pushed down, and those are what a bare numeric literal
        # needs in order to be a real rather than an error.
        expected = spec.constraint
        if expected is not None and _contains_pattern_index(expected):
            expected = None
        prepared[key] = _elaborate(source, registry, expected, None)
    return ground(pattern, prepared)


def _contains_pattern_index(declared: Any) -> bool:
    if isinstance(declared, PatternIndex):
        return True
    if isinstance(declared, IndexedType):
        return any(_contains_pattern_index(i) for i in declared.indices)
    if isinstance(declared, ListType):
        return _contains_pattern_index(declared.element)
    return False
