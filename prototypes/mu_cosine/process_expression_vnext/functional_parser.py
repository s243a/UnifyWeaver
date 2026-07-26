#!/usr/bin/env python3
"""Functional-surface parser producing an elaboration-preserving source AST.

The tree keeps everything elaboration and diagnostics need — spans, positional
order, duplicate named fields, exact numeric lexemes, unchecked annotations —
but it does **not** promise exact source reconstruction: grouping parentheses,
whitespace and comments are dropped, and atom/string nodes hold decoded values
rather than original spellings.  The weaker claim is deliberate; promising
losslessness the implementation cannot deliver would be worse than stating the
guarantee accurately.

Grammar per ``DESIGN_process_expression_patterns.md`` §4, restricted to the
ground slice this milestone supports.  The parser resolves *names* against the
injected registry only for longest-match lexing — it performs no arity,
type, field, or default checking.  All of that is the elaborator's job, and
keeping the seam sharp is what lets a parse error and a type error carry
different diagnostics.

Deliberate non-behaviors:

* named fields are **not** sorted or de-duplicated, so the elaborator can see
  duplicates and reject them;
* numeric text is **not** converted, so the declared type can decide later
  (§3.6);
* annotations are attached unchecked;
* variables, modifiers, pins, function types and sum types are *recognized* and
  rejected with a "not implemented in this milestone" diagnostic rather than
  being silently misparsed.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .ast import (
    SourceAtom,
    SourceCall,
    SourceIndexedType,
    SourceList,
    SourceName,
    SourceNode,
    SourceNumber,
    SourceReferenceIndex,
    SourceString,
    SourceType,
    SourceTypeName,
    SourceVariable,
    SourceAnnotated,
    Span,
)
from .numerics import NUMBER_RE, NumberLexeme, NumericError

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_VARIABLE = re.compile(r"_\Z|[A-Z][A-Za-z0-9_]*\Z")
_NUMBER_TOKEN = re.compile(r"-?[0-9][0-9A-Za-z_.+-]*")
_ATOM = re.compile(r"'([^'\\]*)'")


class ParseError(ValueError):
    """Lexing or parsing failed.  Distinct from registry/elaboration errors."""

    def __init__(self, message: str, position: int | None = None):
        self.position = position
        super().__init__(
            message if position is None else f"{message} (at offset {position})"
        )


class NotImplementedInMilestone(ParseError):
    """A recognized construct deliberately outside this milestone's slice."""


def _respan(node: SourceNode, span: Span) -> SourceNode:
    """Return ``node`` with a new span.  Source nodes are frozen dataclasses."""

    import dataclasses

    return dataclasses.replace(node, span=span)


def _reject_lone_surrogates(value: str, position: int) -> None:
    """A lone UTF-16 surrogate cannot be encoded as UTF-8.

    ``json`` happily decodes ``"\\ud800"`` into an unpaired surrogate, which
    would then blow up at the first encode.  Rejecting it here keeps the
    invariant that every typed ``String`` is encodable.
    """

    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ParseError(
            "string contains an unpaired UTF-16 surrogate and is not valid "
            "Unicode text",
            position,
        ) from exc


class _Parser:
    def __init__(self, text: str, names: tuple[str, ...]):
        self.text = text
        self.i = 0
        # Longest match first, so `gpt-5.5-low` wins over any shorter prefix.
        self.names = tuple(sorted(names, key=len, reverse=True))

    # -- lexing helpers ----------------------------------------------------

    def _ws(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1

    def _at(self, literal: str) -> bool:
        return self.text.startswith(literal, self.i)

    def _expect(self, literal: str) -> None:
        if not self._at(literal):
            got = self.text[self.i : self.i + 12] or "end of input"
            raise ParseError(f"expected {literal!r}, found {got!r}", self.i)
        self.i += len(literal)

    def _match_registered_name(self) -> str | None:
        for name in self.names:
            end = self.i + len(name)
            if not self.text.startswith(name, self.i):
                continue
            # A registered name must not be a prefix of a longer bare word.
            if end < len(self.text) and (
                self.text[end].isalnum() or self.text[end] == "_"
            ):
                continue
            return name
        return None

    # -- entry -------------------------------------------------------------

    def parse(self) -> SourceNode:
        node = self.parse_expr()
        self._ws()
        if self.i != len(self.text):
            raise ParseError(
                f"trailing input: {self.text[self.i : self.i + 16]!r}", self.i
            )
        return node

    def parse_expr(self) -> SourceNode:
        node = self.parse_primary()
        self._ws()
        if self._at("."):
            raise NotImplementedInMilestone(
                "modifiers are recognized but not implemented in this milestone",
                self.i,
            )
        if self._at("@"):
            raise NotImplementedInMilestone(
                "provenance pins are recognized but not implemented in this milestone",
                self.i,
            )
        if self._at("::"):
            start = self.i
            self.i += 2
            asserted = self.parse_type()
            node = SourceAnnotated(Span(node.span.start, self.i), node, asserted)
            self._ws()
            if self._at("."):
                raise NotImplementedInMilestone(
                    "modifiers are recognized but not implemented in this milestone",
                    self.i,
                )
        return node

    def parse_primary(self) -> SourceNode:
        self._ws()
        if self.i >= len(self.text):
            raise ParseError("unexpected end of input", self.i)
        start = self.i
        char = self.text[self.i]

        if char == "(":
            self.i += 1
            inner = self.parse_expr()
            self._ws()
            self._expect(")")
            # The span covers the whole parenthesized expression, including the
            # delimiters, so a diagnostic points at what the author wrote.
            return _respan(inner, Span(start, self.i))

        if char == "[":
            return self.parse_list()

        if char == '"':
            return self.parse_string()

        if char == "'":
            match = _ATOM.match(self.text, self.i)
            if not match:
                raise ParseError("malformed atom literal", self.i)
            self.i = match.end()
            return SourceAtom(Span(start, self.i), match.group(1))

        if char.isdigit() or (
            char == "-" and self.i + 1 < len(self.text) and self.text[self.i + 1].isdigit()
        ):
            return self.parse_number()

        name = self._match_registered_name()
        if name is not None:
            self.i += len(name)
            # Close the name's span *before* skipping trivia, so trailing
            # whitespace never becomes part of the token.
            name_end = self.i
            self._ws()
            if self._at("("):
                return self.parse_call(name, start)
            self.i = name_end
            return SourceName(Span(start, name_end), name)

        word = _IDENT.match(self.text, self.i)
        if word:
            token = word.group(0)
            if _VARIABLE.match(token):
                self.i = word.end()
                raise NotImplementedInMilestone(
                    f"variable {token!r} is recognized but not implemented in this "
                    "milestone",
                    start,
                )
            raise ParseError(
                f"unregistered name {token!r}; a bare lowercase word is never "
                "guessed to be an atom — register it or quote it",
                start,
            )
        raise ParseError(f"unexpected character {char!r}", self.i)

    def parse_call(self, name: str, start: int) -> SourceCall:
        self._expect("(")
        args: list[SourceNode] = []
        fields: list[tuple[str, SourceNode, Span]] = []
        self._ws()
        if self._at(")"):
            self.i += 1
            return SourceCall(Span(start, self.i), name, (), ())
        while True:
            self._ws()
            if self._at(")"):
                raise ParseError("trailing comma in argument list", self.i)
            field_name = self._try_field_name()
            if field_name is not None:
                key, key_span = field_name
                value = self.parse_expr()
                fields.append((key, value, key_span))
            else:
                if fields:
                    raise ParseError(
                        "positional argument after a named field", self.i
                    )
                args.append(self.parse_expr())
            self._ws()
            if self._at(","):
                self.i += 1
                continue
            self._expect(")")
            break
        return SourceCall(Span(start, self.i), name, tuple(args), tuple(fields))

    def _try_field_name(self) -> tuple[str, Span] | None:
        """A field name is `ident =` not followed by `=`."""

        self._ws()
        save = self.i
        word = _IDENT.match(self.text, self.i)
        if not word:
            return None
        after = word.end()
        j = after
        while j < len(self.text) and self.text[j].isspace():
            j += 1
        if j < len(self.text) and self.text[j] == "=" and not self.text.startswith("==", j):
            self.i = j + 1
            return word.group(0), Span(save, after)
        self.i = save
        return None

    def parse_list(self) -> SourceList:
        start = self.i
        self._expect("[")
        items: list[SourceNode] = []
        self._ws()
        if self._at("]"):
            self.i += 1
            return SourceList(Span(start, self.i), ())
        while True:
            self._ws()
            if self._at("]"):
                raise ParseError("trailing comma in list", self.i)
            items.append(self.parse_expr())
            self._ws()
            if self._at(","):
                self.i += 1
                continue
            self._expect("]")
            break
        return SourceList(Span(start, self.i), tuple(items))

    def parse_string(self) -> SourceString:
        start = self.i
        try:
            value, offset = json.JSONDecoder().raw_decode(self.text[self.i :])
        except json.JSONDecodeError as exc:
            raise ParseError(f"malformed JSON string: {exc}", self.i) from exc
        if not isinstance(value, str):  # pragma: no cover - '"' guarantees str
            raise ParseError("expected a string literal", self.i)
        _reject_lone_surrogates(value, start)
        self.i += offset
        return SourceString(Span(start, self.i), value)

    def parse_number(self) -> SourceNumber:
        start = self.i
        match = _NUMBER_TOKEN.match(self.text, self.i)
        if not match:  # pragma: no cover - caller checked the first char
            raise ParseError("expected a number", self.i)
        token = match.group(0)
        # Do not let a number run into an adjacent identifier character.
        if not NUMBER_RE.match(token):
            raise ParseError(
                f"malformed numeric literal {token!r} — finite decimals only, "
                "no NaN, infinity, or malformed exponent",
                start,
            )
        self.i = match.end()
        try:
            lexeme = NumberLexeme(token)
        except NumericError as exc:
            raise ParseError(str(exc), start) from exc
        return SourceNumber(Span(start, self.i), lexeme)

    # -- types -------------------------------------------------------------

    def parse_type(self) -> SourceType:
        self._ws()
        start = self.i
        word = _IDENT.match(self.text, self.i)
        if not word:
            raise ParseError("expected a type name after '::'", self.i)
        name = word.group(0)
        self.i = word.end()
        if name == "function":
            raise NotImplementedInMilestone(
                "function types are recognized but not implemented in this milestone",
                start,
            )
        self._ws()
        if not self._at("["):
            return SourceTypeName(Span(start, self.i), name)
        self.i += 1
        indices: list[Any] = []
        while True:
            self._ws()
            if self._at("]"):
                raise ParseError("empty or trailing index in type", self.i)
            indices.append(self.parse_type_or_index())
            self._ws()
            if self._at(","):
                self.i += 1
                continue
            self._expect("]")
            break
        return SourceIndexedType(Span(start, self.i), name, tuple(indices))

    def parse_type_or_index(self) -> Any:
        """An index position may hold a *value reference* or a type.

        A registered reference is matched with the same longest-match rule used
        for expressions, so a punctuated name such as ``gpt-5.5-low`` lexes as
        one index rather than failing at ``-5.5-low``.
        """

        self._ws()
        start = self.i
        word = _IDENT.match(self.text, self.i)
        if word and _VARIABLE.match(word.group(0)):
            self.i = word.end()
            raise NotImplementedInMilestone(
                f"type variable {word.group(0)!r} is recognized but not implemented "
                "in this milestone",
                start,
            )

        name = self._match_registered_name()
        if name is not None:
            after = self.i + len(name)
            following = self.text[after : after + 1]
            if following != "(":
                self.i = after
                return SourceReferenceIndex(Span(start, self.i), name)
            raise NotImplementedInMilestone(
                "an expression-valued type index is recognized but not implemented "
                "in this milestone; its wire representation is an open "
                "specification decision",
                start,
            )
        if not word:
            raise ParseError("expected a type or index", self.i)
        return self.parse_type()


def parse_functional(text: str, registry) -> SourceNode:
    """Parse functional-surface text into a lossless source AST.

    ``registry`` supplies registered names for longest-match lexing only; no
    signature, arity, or type information is consulted here.
    """

    if not isinstance(text, str) or not text.strip():
        raise ParseError("expression must be a nonempty string")
    return _Parser(text, registry.names()).parse()
