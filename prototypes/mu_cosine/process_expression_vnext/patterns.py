#!/usr/bin/env python3
"""Pattern state, grounding, and alpha-equivalence for the vNext frontend.

Implements one edge of the state machine in
``DESIGN_process_expression_patterns.md`` §1::

    PatternAST --ground(bindings)--> GroundAST

and nothing beyond it.  ``interpret``, ``represent``, and
``verify_factory_receipt`` are later edges and are deliberately absent, so a
caller cannot mistake a ``GroundAST`` for something deployable.

The two states are real types with checked invariants, not documentation:

``PatternAST``
    may contain :class:`~.ast.PatternVariable` nodes and :class:`~.ast.PatternIndex`
    value indices.

``GroundAST``
    is *proved* on construction to contain neither, and no registry-signature
    :class:`~.ast.TypeVar` either.  The proof is recursive and covers arguments,
    fields, list items, every inferred type, and the interior of a
    :class:`~.ast.TermIndex` — a variable hiding inside an expression-valued
    index is exactly the leak a per-node check would miss.

A ``GroundAST`` may still be semantically coarse.  ``lineage_op(principal_tree(
pearltrees))`` is ground and remains interpretation-underconstrained (§4.1), so
grounding performs no interpretation, no representation selection, and no
factory verification.

Nothing here mints identity.  There is no pattern digest, no canonical pattern
bytes, no alpha-normalized serialization, and no ``pe-typed-ast-v1``.  A digest
would have to commit to the wire form of an expression-valued ``TermIndex``,
which is an open specification decision; :func:`alpha_equivalent` answers the
comparison question structurally and in memory instead.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .ast import (
    Call,
    IndexedType,
    ListTerm,
    ListType,
    PatternIndex,
    PatternVariable,
    Reference,
    ReferenceIndex,
    TermIndex,
    Type,
    TypeName,
    TypeVar,
    TypedTerm,
    ValueIndex,
    VarId,
)
from .registry import RegistryError


class GroundingError(ValueError):
    """A binding set does not ground a pattern.

    Distinct from ``ParseError`` (malformed text) and ``ElaborationError``
    (well-formed text, ill-typed or registry-violating) so a caller can tell
    which of the three stages rejected the input.
    """


# --------------------------------------------------------------------------
# state types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundAST:
    """A typed term proved free of variables and signature type variables.

    The proof runs in ``__post_init__``, so **every** way of obtaining a
    ``GroundAST`` — including calling the exported constructor directly — is
    checked.  A state type whose invariant held only on the package's own happy
    path would be a claim rather than a guarantee, and it is exported.

    ``registry`` records which registry the term was elaborated against, so a
    term cannot be carried into a pattern built from a different one.  It is
    ``None`` only for a value the caller assembled by hand, which has no
    registry to attribute it to.
    """

    term: TypedTerm
    registry: Any = None

    def __post_init__(self) -> None:
        defects = _term_defects(self.term, "term")
        if defects:
            raise GroundingError(_not_ground_message("term", defects))

    def __str__(self) -> str:
        return f"GroundAST({type(self.term).__name__})"


@dataclass(frozen=True)
class PatternVar:
    """One logical variable of a pattern, with what elaboration learned about it."""

    var: VarId
    #: The variable's constraint, or ``None`` for a variable that occurs only in
    #: index position (``substrate[C]`` declares no type for ``C``).
    constraint: Type | None
    #: True when the variable occurs in term position and therefore needs a
    #: binding.  An index-only variable is satisfied by inference instead.
    in_term_position: bool

    @property
    def name(self) -> str:
        return self.var.display

    @property
    def label(self) -> str:
        """Diagnostic label that stays unambiguous for anonymous variables.

        Two ``_`` occurrences share a display name, so a message that used the
        name alone could not say *which* one failed.
        """

        if self.var.origin == "named":
            return repr(self.name)
        return f"_#{self.var.serial}"


@dataclass(frozen=True)
class PatternAST:
    """A typed term that may contain variables, plus its variable table.

    Immutable: :func:`ground` never mutates a pattern, and grounding the same
    pattern twice with different bindings is well defined.
    """

    term: TypedTerm
    variables: tuple[PatternVar, ...]
    #: The registry this pattern was elaborated against.  Held by identity, not
    #: by label: two registries can share a label, and two loads of one file are
    #: not the same registry — nothing guarantees the file did not change
    #: between them, so being told is better than being guessed at.
    registry: Any

    @property
    def registry_label(self) -> str:
        return getattr(self.registry, "label", str(self.registry))

    @property
    def named_variables(self) -> tuple[PatternVar, ...]:
        return tuple(v for v in self.variables if v.var.origin == "named")

    @property
    def anonymous_variables(self) -> tuple[PatternVar, ...]:
        """Handles for ``_`` occurrences.

        Exposed because an anonymous variable has no name to bind by, and a
        pattern whose ``_`` occurrences could not be bound at all would be
        unusable rather than merely anonymous.
        """

        return tuple(v for v in self.variables if v.var.origin == "anonymous")

    @property
    def required_variables(self) -> tuple[PatternVar, ...]:
        return tuple(v for v in self.variables if v.in_term_position)

    def variable(self, name: str) -> PatternVar:
        matches = [v for v in self.named_variables if v.name == name]
        if not matches:
            raise GroundingError(f"pattern has no variable named {name!r}")
        # The scope keys named variables by name, so this cannot be ambiguous.
        return matches[0]


# --------------------------------------------------------------------------
# ground proof
# --------------------------------------------------------------------------


def _type_defects(declared: Any, path: str) -> list[str]:
    """Every reason ``declared`` is not ground, rather than the first."""

    found: list[str] = []
    if isinstance(declared, TypeVar):
        found.append(f"{path}: unresolved signature type variable {declared.name}")
    elif isinstance(declared, PatternIndex):
        found.append(f"{path}: unbound pattern index {declared.var.display}")
    elif isinstance(declared, TermIndex):
        found.extend(_term_defects(declared.term, f"{path}<index>"))
    elif isinstance(declared, IndexedType):
        for position, index in enumerate(declared.indices):
            found.extend(_type_defects(index, f"{path}[{position}]"))
    elif isinstance(declared, ListType):
        found.extend(_type_defects(declared.element, f"{path}.element"))
    return found


def _term_defects(term: TypedTerm, path: str) -> list[str]:
    found: list[str] = []
    if isinstance(term, PatternVariable):
        found.append(f"{path}: unbound variable {term.var.display}")
    found.extend(_type_defects(term.inferred_type, f"{path}:type"))
    if isinstance(term, Call):
        for position, arg in enumerate(term.args):
            found.extend(_term_defects(arg, f"{path}.{term.name}[{position}]"))
        for name, value in term.fields:
            found.extend(_term_defects(value, f"{path}.{term.name}.{name}"))
    elif isinstance(term, ListTerm):
        for position, item in enumerate(term.items):
            found.extend(_term_defects(item, f"{path}[{position}]"))
    return found


def is_ground(term: TypedTerm) -> bool:
    return not _term_defects(term, "term")


def _not_ground_message(what: str, defects: list[str]) -> str:
    return (
        f"{what} is not ground: "
        + "; ".join(defects[:8])
        + ("; …" if len(defects) > 8 else "")
    )


def make_ground(
    term: TypedTerm, *, what: str = "term", registry: Any = None
) -> GroundAST:
    """Prove ``term`` ground and wrap it, naming the caller's context on failure.

    ``GroundAST`` re-runs the same proof, so this is a diagnostic convenience
    rather than the security boundary: the boundary is in ``__post_init__``.
    """

    defects = _term_defects(term, what)
    if defects:
        raise GroundingError(_not_ground_message(what, defects))
    return GroundAST(term, registry)


# --------------------------------------------------------------------------
# binding
# --------------------------------------------------------------------------


def index_of(term: TypedTerm) -> ValueIndex:
    """The value-index form of a term (§3.2).

    A bare registered reference indexes as itself; anything else is an
    expression-valued index.  This is the same rule the elaborator applies when
    a signature's ``binds`` names an argument, which is what makes an inferred
    index and an explicitly supplied one comparable.
    """

    if isinstance(term, Reference):
        return ReferenceIndex(term.name)
    return TermIndex(term)


def _substitute_type(
    declared: Any,
    term_values: Mapping[VarId, TypedTerm],
    index_values: Mapping[VarId, ValueIndex],
) -> Any:
    if isinstance(declared, PatternIndex):
        bound = index_values.get(declared.var)
        return declared if bound is None else bound
    if isinstance(declared, TermIndex):
        # An expression-valued index is a *term*, so it can contain term
        # variables of its own: `hop_decay_targets(principal_tree(C), ...)`
        # indexes its result by `principal_tree(C)`.  Substituting only indices
        # here would leave that C unbound and make a legitimate pattern
        # ungroundable.
        return TermIndex(_substitute_term(declared.term, term_values, index_values))
    if isinstance(declared, IndexedType):
        return IndexedType(
            declared.name,
            tuple(
                _substitute_type(i, term_values, index_values)
                for i in declared.indices
            ),
        )
    if isinstance(declared, ListType):
        return ListType(_substitute_type(declared.element, term_values, index_values))
    return declared


def _substitute_term(
    term: TypedTerm,
    term_values: Mapping[VarId, TypedTerm],
    index_values: Mapping[VarId, ValueIndex],
) -> TypedTerm:
    if isinstance(term, PatternVariable):
        bound = term_values.get(term.var)
        if bound is None:
            return term
        return bound
    inferred = _substitute_type(term.inferred_type, term_values, index_values)
    if isinstance(term, Call):
        return Call(
            inferred,
            term.name,
            tuple(_substitute_term(a, term_values, index_values) for a in term.args),
            tuple(
                (n, _substitute_term(v, term_values, index_values))
                for n, v in term.fields
            ),
        )
    if isinstance(term, ListTerm):
        return ListTerm(
            inferred,
            tuple(_substitute_term(i, term_values, index_values) for i in term.items),
        )
    if inferred == term.inferred_type:
        return term
    return replace(term, inferred_type=inferred)


def _unify_constraint(
    declared: Any, actual: Any, index_values: dict[VarId, ValueIndex]
) -> bool:
    """Match a pattern constraint against a ground type, binding index variables.

    Only :class:`PatternIndex` binds.  A :class:`TypeVar` reaching here means a
    signature placeholder escaped elaboration, so it deliberately does not
    unify — grounding must not paper over that.
    """

    if isinstance(declared, PatternIndex):
        existing = index_values.get(declared.var)
        if existing is None:
            if not isinstance(actual, ValueIndex):
                return False
            index_values[declared.var] = actual
            return True
        return existing == actual
    if isinstance(declared, TypeVar):
        return False
    if isinstance(declared, TypeName):
        return isinstance(actual, TypeName) and declared.name == actual.name
    if isinstance(declared, ListType):
        return isinstance(actual, ListType) and _unify_constraint(
            declared.element, actual.element, index_values
        )
    if isinstance(declared, IndexedType):
        if not isinstance(actual, IndexedType) or declared.name != actual.name:
            return False
        if len(declared.indices) != len(actual.indices):
            return False
        return all(
            _unify_constraint(d, a, index_values)
            for d, a in zip(declared.indices, actual.indices)
        )
    if isinstance(declared, ValueIndex):
        return declared == actual
    return False  # pragma: no cover - every Type subclass is covered above


def _materialize(index: ValueIndex, registry: Any) -> TypedTerm | None:
    """Recover the term a determined index denotes, if it denotes one.

    This is what makes ownership inference symmetric rather than left-to-right:
    in ``cross_substrate(principal_tree(C), S::substrate[C])``, binding ``S``
    determines ``C``, and requiring the caller to restate ``C`` would be exactly
    the "redundant input" the brief rules out.  Returns ``None`` when the index
    determines no term, in which case the variable stays unbound and is reported
    as missing.
    """

    if isinstance(index, TermIndex):
        return index.term
    if isinstance(index, ReferenceIndex):
        try:
            entry = registry.get(index.name)
        except RegistryError:
            return None
        if entry.kind != "reference":
            return None
        # Built exactly as `_elaborate` builds a `SourceName`, so a recovered
        # binding and a supplied one are the same node.
        return Reference(entry.result_type, index.name)
    return None


def _binding_term(value: Any, label: str, pattern: "PatternAST") -> TypedTerm:
    """Accept a typed ground value or a :class:`GroundAST`; reject the rest."""

    if isinstance(value, GroundAST):
        if value.registry is not None and value.registry is not pattern.registry:
            raise GroundingError(
                f"binding for {label} was elaborated against registry "
                f"{getattr(value.registry, 'label', value.registry)!r}, but the "
                f"pattern was elaborated against {pattern.registry_label!r}; a "
                "term only means what its own registry says it means"
            )
        return value.term
    if isinstance(value, PatternAST):
        raise GroundingError(
            f"binding for {label} is a PatternAST; a variable binds to a ground "
            "value, and nesting a pattern would reintroduce the variables "
            "grounding exists to remove"
        )
    if isinstance(value, str):
        raise GroundingError(
            f"binding for {label} is a string; ground() takes elaborated terms so "
            "that a parse error cannot masquerade as a binding error — use "
            "ground_surface() if you want surface text parsed"
        )
    if not isinstance(value, TypedTerm):
        raise GroundingError(
            f"binding for {label} is {type(value).__name__}, not a typed term"
        )
    defects = _term_defects(value, f"binding for {label}")
    if defects:
        raise GroundingError(
            "binding is not ground: " + "; ".join(defects[:4])
        )
    return value


def ground(
    pattern: PatternAST, bindings: Mapping[Any, Any] | None = None
) -> GroundAST:
    """Substitute ``bindings`` into ``pattern`` and prove the result ground.

    Keys are variable names (``"S"``) or opaque :class:`VarId` handles, the
    latter being the only way to bind an anonymous ``_``.  Values are typed
    ground terms or :class:`GroundAST`.

    The pattern is not mutated, the result does not depend on mapping order, and
    a variable that another binding already determines need not be supplied —
    but if it is, it must agree.  Determination runs to a fixpoint in both
    directions: a binding can determine an index, and a determined index can in
    turn supply the term binding for a variable that occurs in both positions.

    A :class:`GroundAST` binding carries the registry it was elaborated against
    and is refused if that is not this pattern's registry.  A bare
    :class:`~.ast.TypedTerm` has no registry to attribute it to, so it is
    accepted on the caller's authority — assembling one by hand is a deliberate
    act, unlike carrying a ``GroundAST`` in from elsewhere.
    """

    if not isinstance(pattern, PatternAST):
        raise GroundingError(
            f"ground() takes a PatternAST, not {type(pattern).__name__}"
        )
    supplied = dict(bindings or {})
    by_id = {v.var: v for v in pattern.variables}
    by_name = {v.name: v for v in pattern.named_variables}
    specs = sorted(pattern.variables, key=lambda s: s.var.serial)

    # 1. Resolve keys.  Every bad key is collected and reported together in a
    #    sorted order, so which of several problems you hear about does not
    #    depend on the order the caller happened to build the mapping in.
    resolved: dict[VarId, TypedTerm] = {}
    unknown: list[str] = []
    doubled: list[str] = []
    for key, value in supplied.items():
        if isinstance(key, VarId):
            spec = by_id.get(key)
            if spec is None:
                unknown.append(
                    f"handle {key.display!r} (serial {key.serial}) does not belong "
                    "to this pattern"
                )
                continue
        elif isinstance(key, str):
            spec = by_name.get(key)
            if spec is None:
                unknown.append(f"unknown binding {key!r}")
                continue
        else:
            unknown.append(
                f"binding key {key!r} is a {type(key).__name__}, not a variable "
                "name or a VarId handle"
            )
            continue
        if spec.var in resolved:
            doubled.append(spec.label)
            continue
        resolved[spec.var] = _binding_term(value, spec.label, pattern)

    if unknown:
        known = ", ".join(sorted(v.name for v in pattern.named_variables))
        suffix = " plus anonymous handles" if pattern.anonymous_variables else ""
        raise GroundingError(
            "; ".join(sorted(unknown))
            + f"; this pattern binds [{known}]{suffix}"
        )
    if doubled:
        raise GroundingError(
            "bound twice, once by name and once by handle: "
            + ", ".join(sorted(set(doubled)))
        )

    # 2. Solve to a fixpoint rather than in one left-to-right sweep.  A constraint
    #    check can determine an index, and a determined index can in turn supply a
    #    term binding whose type determines further indices, so a single pass
    #    would make the outcome depend on the order variables happen to appear in.
    index_values: dict[VarId, ValueIndex] = {}
    for var in sorted(resolved, key=lambda v: v.serial):
        index_values[var] = index_of(resolved[var])

    for _round in range(len(pattern.variables) + 2):
        # Progress is *either* a new term binding or a newly determined index:
        # a constraint check that binds an index has learned something even
        # though it materialized nothing, and stopping there would leave a
        # variable whose value the next round would have recovered.
        before = (len(resolved), len(index_values))
        for spec in specs:
            value = resolved.get(spec.var)
            if value is None:
                if not spec.in_term_position:
                    continue
                index = index_values.get(spec.var)
                recovered = None if index is None else _materialize(
                    index, pattern.registry
                )
                if recovered is None:
                    continue
                resolved[spec.var] = value = recovered
            if spec.constraint is None:
                continue
            if not _unify_constraint(
                spec.constraint, value.inferred_type, index_values
            ):
                raise GroundingError(
                    f"binding for {spec.label} has type {value.inferred_type}, which "
                    "does not satisfy "
                    f"{_substitute_type(spec.constraint, resolved, index_values)}"
                )
        if (len(resolved), len(index_values)) == before:
            break
    else:  # pragma: no cover - each round adds a binding or an index
        raise GroundingError("binding constraints did not reach a fixpoint")

    # 3. Every variable in term position needs a value; an index-only variable is
    #    allowed to be inferred (§3.5's "fresh C").
    missing = [
        spec.label
        for spec in pattern.required_variables
        if spec.var not in resolved
    ]
    if missing:
        raise GroundingError(
            "missing binding(s) for " + ", ".join(sorted(missing))
        )

    result = _substitute_term(pattern.term, resolved, index_values)
    return make_ground(
        result, what="grounded term", registry=pattern.registry
    )


# --------------------------------------------------------------------------
# alpha equivalence
# --------------------------------------------------------------------------


def alpha_equivalent(left: PatternAST, right: PatternAST) -> bool:
    """True when two patterns differ only by a consistent renaming.

    Implemented as a paired walk carrying a bijection, not by normalizing either
    side.  A normal form would be a byte-level commitment this milestone is not
    allowed to make: the canonical encoding of an expression-valued
    ``TermIndex`` is unresolved, so any normalized rendering would either fix
    that decision or quietly drop it.
    """

    if not isinstance(left, PatternAST) or not isinstance(right, PatternAST):
        raise TypeError("alpha_equivalent compares two PatternAST values")
    forward: dict[VarId, VarId] = {}
    backward: dict[VarId, VarId] = {}
    return _alpha_term(left.term, right.term, forward, backward)


def _alpha_var(a: VarId, b: VarId, fwd, bwd) -> bool:
    # Origin is part of the comparison: `_` and `S` are different kinds of
    # binding occurrence, so a pattern that repeats `S` is not a renaming of one
    # that writes `_` twice even when both have two occurrences.
    if a.origin != b.origin:
        return False
    if a in fwd or b in bwd:
        return fwd.get(a) == b and bwd.get(b) == a
    fwd[a] = b
    bwd[b] = a
    return True


def _alpha_type(a: Any, b: Any, fwd, bwd) -> bool:
    if isinstance(a, PatternIndex) or isinstance(b, PatternIndex):
        return (
            isinstance(a, PatternIndex)
            and isinstance(b, PatternIndex)
            and _alpha_var(a.var, b.var, fwd, bwd)
        )
    if isinstance(a, TermIndex) or isinstance(b, TermIndex):
        return (
            isinstance(a, TermIndex)
            and isinstance(b, TermIndex)
            and _alpha_term(a.term, b.term, fwd, bwd)
        )
    if isinstance(a, IndexedType) or isinstance(b, IndexedType):
        return (
            isinstance(a, IndexedType)
            and isinstance(b, IndexedType)
            and a.name == b.name
            and len(a.indices) == len(b.indices)
            and all(
                _alpha_type(x, y, fwd, bwd) for x, y in zip(a.indices, b.indices)
            )
        )
    if isinstance(a, ListType) or isinstance(b, ListType):
        return (
            isinstance(a, ListType)
            and isinstance(b, ListType)
            and _alpha_type(a.element, b.element, fwd, bwd)
        )
    return a == b


def _alpha_term(a: TypedTerm, b: TypedTerm, fwd, bwd) -> bool:
    if type(a) is not type(b):
        return False
    if not _alpha_type(a.inferred_type, b.inferred_type, fwd, bwd):
        return False
    if isinstance(a, PatternVariable):
        return _alpha_var(a.var, b.var, fwd, bwd)
    if isinstance(a, Call):
        return (
            a.name == b.name
            and len(a.args) == len(b.args)
            and len(a.fields) == len(b.fields)
            and all(_alpha_term(x, y, fwd, bwd) for x, y in zip(a.args, b.args))
            and all(
                xn == yn and _alpha_term(xv, yv, fwd, bwd)
                for (xn, xv), (yn, yv) in zip(a.fields, b.fields)
            )
        )
    if isinstance(a, ListTerm):
        return len(a.items) == len(b.items) and all(
            _alpha_term(x, y, fwd, bwd) for x, y in zip(a.items, b.items)
        )
    if isinstance(a, Reference):
        return a.name == b.name
    return a == b
