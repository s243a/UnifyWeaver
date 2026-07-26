#!/usr/bin/env python3
"""Injected experimental registry for the vNext frontend milestone.

There is **no approved production vNext registry**.  This loader therefore
accepts only an explicitly-supplied fixture and never falls back to the frozen
v0.3 registry in ``process_cards.py`` — silently borrowing v0.3 signatures would
make the frontend look more validated than it is, and would smuggle the
retired ``source`` catch-all type into a language that deliberately drops it
(§3.2).

Signature types are written in the functional surface's type syntax and parsed
here, so a fixture cannot express a type the language cannot.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .ast import IndexedType, ListType, Type, TypeName, TypeVar


def _contains_type_var(declared: Type) -> bool:
    if isinstance(declared, TypeVar):
        return True
    if isinstance(declared, IndexedType):
        return any(
            _contains_type_var(i) for i in declared.indices if isinstance(i, Type)
        )
    if isinstance(declared, ListType):
        return _contains_type_var(declared.element)
    return False

_TYPE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INDEX_VAR = re.compile(r"[A-Z][A-Za-z0-9_]*\Z")

#: Kinds a fixture entry may declare.
KIND_REFERENCE = "reference"
KIND_CALL = "call"


class RegistryError(ValueError):
    """The registry fixture is malformed, or a name is not registered."""


def parse_type(text: str) -> Type:
    """Parse a signature type such as ``substrate[C]`` or ``list[real]``."""

    parsed, rest = _parse_type(text.strip(), 0)
    if rest != len(text.strip()):
        raise RegistryError(f"trailing input in type: {text!r}")
    return parsed


def _parse_type(text: str, i: int) -> tuple[Type, int]:
    match = _TYPE_NAME.match(text, i)
    if not match:
        raise RegistryError(f"expected a type name in {text!r} at {i}")
    name = match.group(0)
    i = match.end()
    if i < len(text) and text[i] == "[":
        i += 1
        indices: list[Any] = []
        while True:
            if i < len(text) and text[i] == "]":
                raise RegistryError(f"empty index list in {text!r}")
            inner_match = _TYPE_NAME.match(text, i)
            if inner_match and _INDEX_VAR.match(inner_match.group(0)):
                indices.append(TypeVar(inner_match.group(0)))
                i = inner_match.end()
            else:
                inner, i = _parse_type(text, i)
                indices.append(inner)
            if i < len(text) and text[i] == ",":
                i += 1
                continue
            if i >= len(text) or text[i] != "]":
                raise RegistryError(f"unterminated index list in {text!r}")
            i += 1
            break
        if name == "list":
            if len(indices) != 1:
                raise RegistryError("list takes exactly one element type")
            element = indices[0]
            if isinstance(element, TypeVar):
                raise RegistryError("list element type may not be an index variable")
            return ListType(element), i
        return IndexedType(name, tuple(indices)), i
    return TypeName(name), i


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: Type
    required: bool
    default: Any | None  # source text for a literal default, or None


@dataclass(frozen=True)
class Entry:
    name: str
    kind: str
    result_type: Type
    arg_types: tuple[Type, ...] = ()
    fields: tuple[FieldSpec, ...] = ()
    #: Index variables bound from a positional argument's *value*, e.g.
    #: ``principal_tree(pearltrees) :: substrate[pearltrees]`` binds ``C`` to
    #: argument 0.  §3.2 indexes substrates by the corpus they view, so the
    #: index is a value reference rather than a type.
    binds: tuple[tuple[str, int], ...] = ()

    def field(self, name: str) -> FieldSpec | None:
        for spec in self.fields:
            if spec.name == name:
                return spec
        return None


class Registry:
    """A validated, injected registry.  Never falls back to v0.3."""

    def __init__(self, entries: Mapping[str, Entry], *, label: str):
        self._entries = dict(entries)
        self.label = label

    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def get(self, name: str) -> Entry:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise RegistryError(f"unregistered name: {name}") from exc

    def __contains__(self, name: str) -> bool:
        return name in self._entries


TOP_LEVEL_KEYS = frozenset({"label", "status", "production", "note", "entries"})
ENTRY_KEYS = frozenset({"kind", "type", "args", "fields", "binds", "note"})
FIELD_KEYS = frozenset({"type", "required", "default", "note"})

_ENTRY_NAME = re.compile(r"\A[A-Za-z_][A-Za-z0-9_.\-]*\Z")
_FIELD_NAME = re.compile(r"\A[a-z_][a-z0-9_]*\Z")


def _strict_pairs(pairs):
    """Reject duplicate keys at every object level.

    ``json.loads`` silently keeps the last of a duplicated key, which would let
    a fixture declare one thing and mean another.
    """

    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise RegistryError(f"duplicate key in registry fixture: {key!r}")
        seen[key] = value
    return seen


def _reject_unknown(obj: Mapping[str, Any], allowed: frozenset[str], where: str):
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise RegistryError(f"unknown key(s) in {where}: {unknown}")


def load_registry(path: str | Path) -> Registry:
    try:
        document = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs
        )
    except json.JSONDecodeError as exc:
        raise RegistryError(f"malformed registry fixture: {exc}") from exc
    if not isinstance(document, dict):
        raise RegistryError("registry fixture must be a JSON object")
    _reject_unknown(document, TOP_LEVEL_KEYS, "registry fixture")

    status = document.get("status")
    if status != "experimental-test-fixture":
        raise RegistryError(
            "registry fixture must declare status 'experimental-test-fixture'; "
            "this loader refuses anything that looks like a release registry"
        )
    if document.get("production") is not False:
        raise RegistryError("registry fixture must declare production=false")
    if "registry_version" in document:
        raise RegistryError(
            "a test fixture must not claim a registry_version; versions belong to "
            "sealed release registries only"
        )

    raw_entries = document.get("entries")
    if not isinstance(raw_entries, dict) or not raw_entries:
        raise RegistryError("registry fixture has no entries")

    entries: dict[str, Entry] = {}
    for name, spec in raw_entries.items():
        if not _ENTRY_NAME.match(name):
            raise RegistryError(f"malformed entry name: {name!r}")
        if not isinstance(spec, dict):
            raise RegistryError(f"entry {name!r} must be an object")
        _reject_unknown(spec, ENTRY_KEYS, f"entry {name!r}")
        kind = spec.get("kind")
        if kind not in (KIND_REFERENCE, KIND_CALL):
            raise RegistryError(f"entry {name!r} has unsupported kind {kind!r}")
        if "type" not in spec:
            raise RegistryError(f"entry {name!r} has no type")
        if not isinstance(spec["type"], str):
            raise RegistryError(f"entry {name!r} type must be a string")
        result_type = parse_type(spec["type"])
        if kind == KIND_REFERENCE and _contains_type_var(result_type):
            # A reference has no arguments, so nothing can ever bind its
            # variables; accepting it would let a TypeVar escape into a
            # supposedly ground elaboration result.
            raise RegistryError(
                f"reference {name!r} declares unresolved type variable(s) in "
                f"{spec['type']!r}; a reference has no arguments to bind them"
            )

        arg_types: tuple[Type, ...] = ()
        fields: tuple[FieldSpec, ...] = ()
        binds: tuple[tuple[str, int], ...] = ()
        if kind == KIND_CALL:
            raw_args = spec.get("args", [])
            if not isinstance(raw_args, list):
                raise RegistryError(f"entry {name!r} args must be a list")
            for declared in raw_args:
                if not isinstance(declared, str):
                    raise RegistryError(
                        f"entry {name!r} argument types must be strings"
                    )
            arg_types = tuple(parse_type(t) for t in raw_args)
            raw_binds = spec.get("binds", {})
            if not isinstance(raw_binds, dict):
                raise RegistryError(f"entry {name!r} binds must be an object")
            collected_binds = []
            for var, position in raw_binds.items():
                if not _INDEX_VAR.match(var):
                    raise RegistryError(
                        f"entry {name!r} binds a non-variable index {var!r}"
                    )
                if not isinstance(position, int) or isinstance(position, bool):
                    raise RegistryError(
                        f"entry {name!r} binds {var!r} to a non-integer position"
                    )
                if not 0 <= position < len(arg_types):
                    raise RegistryError(
                        f"entry {name!r} binds {var!r} to out-of-range position "
                        f"{position}"
                    )
                collected_binds.append((var, position))
            binds = tuple(sorted(collected_binds))
            raw_fields = spec.get("fields", {})
            if not isinstance(raw_fields, dict):
                raise RegistryError(f"entry {name!r} fields must be an object")
            collected = []
            for field_name, field_spec in raw_fields.items():
                if not _FIELD_NAME.match(field_name):
                    raise RegistryError(
                        f"entry {name!r} has malformed field name {field_name!r}"
                    )
                if not isinstance(field_spec, dict) or "type" not in field_spec:
                    raise RegistryError(
                        f"entry {name!r} field {field_name!r} needs a type"
                    )
                _reject_unknown(
                    field_spec, FIELD_KEYS, f"entry {name!r} field {field_name!r}"
                )
                if not isinstance(field_spec["type"], str):
                    raise RegistryError(
                        f"entry {name!r} field {field_name!r} type must be a string"
                    )
                required = field_spec.get("required", False)
                if not isinstance(required, bool):
                    raise RegistryError(
                        f"entry {name!r} field {field_name!r} required must be a "
                        f"JSON boolean, got {required!r}"
                    )
                default = field_spec.get("default")
                if default is not None and not isinstance(default, str):
                    # Defaults are *surface text*.  Accepting a JSON number and
                    # later calling str() on the resulting Python float would
                    # round the value before elaboration ever sees it.
                    raise RegistryError(
                        f"entry {name!r} field {field_name!r} default must be a "
                        "JSON string of surface text or null; a JSON number would "
                        "round before elaboration"
                    )
                if required and default is not None:
                    raise RegistryError(
                        f"entry {name!r} field {field_name!r} is required and "
                        "cannot also declare a default"
                    )
                collected.append(
                    FieldSpec(
                        name=field_name,
                        type=parse_type(field_spec["type"]),
                        required=required,
                        default=default,
                    )
                )
            fields = tuple(collected)
        elif spec.get("args") or spec.get("fields"):
            raise RegistryError(f"reference {name!r} cannot declare args or fields")

        entries[name] = Entry(
            name=name,
            kind=kind,
            result_type=result_type,
            arg_types=arg_types,
            fields=fields,
            binds=binds,
        )
    return Registry(entries, label=document.get("label", str(path)))
