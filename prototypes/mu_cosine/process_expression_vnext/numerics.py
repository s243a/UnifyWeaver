#!/usr/bin/env python3
"""Exact numeric lexemes and registry-directed normalization.

``DESIGN_process_expression_patterns.md`` §3.6: *the declared field type wins
over the source-language runtime type.*  A numeric literal therefore cannot be
normalized until elaboration knows what the signature expects, which is why
parsing produces a :class:`NumberLexeme` and never a Python ``int`` or ``float``.

Going through ``float`` during parsing would silently round a decimal the
specification requires to survive, and would erase the ``-0`` / ``0``
distinction that ``float64`` fields must preserve.  Both are tested.

Equality for ``real`` is defined on the normalized ``(sign, digits, exponent)``
triple rather than on a rendered string.  The canonical *string* form of a
``real`` is a deferred contract decision (see the PR's deferred-decisions list);
defining equality structurally lets this milestone stay correct without
inventing bytes that identity would later depend on.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
import struct

#: A finite decimal token.  Deliberately excludes ``nan``/``inf`` spellings so
#: they fail as unregistered names rather than becoming non-finite numbers.
NUMBER_RE = re.compile(r"\A-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


class NumericError(ValueError):
    """A numeric lexeme is malformed or cannot satisfy its declared type."""


@dataclass(frozen=True)
class NumberLexeme:
    """The exact source text of a numeric token, unconverted.

    Held as text on purpose: the same lexeme becomes a different value
    depending on whether the registry declares the field ``int``, ``real``, or
    ``float64``.
    """

    text: str

    def __post_init__(self) -> None:
        if not NUMBER_RE.match(self.text):
            raise NumericError(f"malformed numeric literal: {self.text!r}")

    @property
    def decimal(self) -> Decimal:
        # Decimal(str) is exact: context precision applies to operations, not
        # construction, so a value beyond binary64 precision is preserved.
        try:
            return Decimal(self.text)
        except InvalidOperation as exc:  # pragma: no cover - guarded by regex
            raise NumericError(f"malformed numeric literal: {self.text!r}") from exc


def _normalized_triple(value: Decimal) -> tuple[int, tuple[int, ...], int]:
    """Canonical ``(sign, digits, exponent)`` with trailing zeros stripped.

    Done by hand rather than via ``Decimal.normalize()`` because ``normalize``
    is an *operation* and would round to the context precision, defeating the
    exactness this module exists to provide.
    """

    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):  # pragma: no cover - non-finite guarded earlier
        raise NumericError("non-finite decimal")
    digits = list(digits)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    if digits == [0]:
        # Negative decimal zero normalizes to zero (§3.6).
        return (0, (0,), 0)
    return (sign, tuple(digits), exponent)


@dataclass(frozen=True)
class RealValue:
    """An exact decimal ``real``.  Equality is on the normalized triple."""

    triple: tuple[int, tuple[int, ...], int]
    source_text: str

    @property
    def decimal(self) -> Decimal:
        sign, digits, exponent = self.triple
        return Decimal((sign, digits, exponent))

    def plain_string(self) -> str:
        """Non-canonical debug rendering.  See module docstring."""

        return format(self.decimal, "f")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RealValue) and self.triple == other.triple

    def __hash__(self) -> int:
        return hash(self.triple)


@dataclass(frozen=True)
class Float64Value:
    """An IEEE-754 binary64 value, identified by its exact bits.

    ``+0.0`` and ``-0.0`` are distinct here even though they compare equal as
    Python floats, because §3.6 requires the bit patterns to stay distinct.
    """

    bits: int

    @property
    def as_float(self) -> float:
        return struct.unpack(">d", self.bits.to_bytes(8, "big"))[0]

    def hex16(self) -> str:
        return f"{self.bits:016x}"


def to_int(lexeme: NumberLexeme) -> int:
    """``int`` rejects any spelling that is not an exact integer (§3.6)."""

    value = lexeme.decimal
    sign, digits, exponent = _normalized_triple(value)
    if exponent < 0:
        raise NumericError(
            f"{lexeme.text!r} is not an exact integer and cannot satisfy int"
        )
    magnitude = int("".join(str(d) for d in digits)) * (10 ** exponent)
    return -magnitude if sign else magnitude


def to_real(lexeme: NumberLexeme) -> RealValue:
    """``1``, ``1.0`` and ``1e0`` all normalize to the same ``real``."""

    return RealValue(_normalized_triple(lexeme.decimal), lexeme.text)


def to_float64(lexeme: NumberLexeme) -> Float64Value:
    """Round the exact decimal into binary64 under the default IEEE rule."""

    value = float(lexeme.decimal)
    if value != value or value in (float("inf"), float("-inf")):
        raise NumericError(f"{lexeme.text!r} is not finite in float64")
    # Preserve signed zero: float(Decimal("-0.0")) is -0.0, and struct keeps it.
    bits = int.from_bytes(struct.pack(">d", value), "big")
    return Float64Value(bits)
