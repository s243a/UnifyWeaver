#!/usr/bin/env python3
"""Registry v0.4: AST, registry-driven parser, canonicalizer, verbosity cards.

Lossless process identity = SEMANTIC canonical AST string + registry version
(+ factory fingerprint, supplied by callers). v0.4 splits identity per R9
(`DESIGN_registry_v0.4.md` §10.3): ``canonical_semantic`` strips provenance
pins and is the digest preimage for ``ast_sha``, seals, and cache keys;
``canonical_full`` retains pins for provenance and round-trip. Cards (V0-V3)
are lossy renderings for conditioning only. Embedding cache keys bind
(ast_sha, verbosity, RENDERER_VERSION, embedding revision, prefix) — never the
string — plus the pin-bearing digest at V3, the only verbosity that renders
pins.

v0.4 registry changes (rulings R1-R10, `DESIGN_registry_v0.4.md`):
- the `source` output type splits into `substrate` (walkable structure) and
  `judge` (mu-source); corpora register as substrate atoms;
- `product`, `max`, `hop_decay`, `lca_frac` join the function vocabulary (R8);
- numeric literals are legal in positional argument positions, typed by the
  signature's declared arg kind (§10.1 flagged decision 1);
- `mu=` takes a judge expression (R4); `estimand=` / `impl=` are enumerated,
  fail-closed methodology kwargs (R5, R7) — the estimand names the relation,
  never the procedure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re

RENDERER_VERSION = "r3"
#: v0.5 is a purely ADDITIVE bump over v0.4: the `enwiki` substrate atom, the
#: `cowalk` operator, and the walk/weight enumerated kinds. No existing entry
#: changed meaning — but a v0.5 expression like lineage(enwiki,...) would
#: parse under a grown v0.4 and not the original, and two registries sharing
#: a label while disagreeing on grammaticality is exactly the ambiguity
#: versions exist to prevent. Every digest moves (the version is in every
#: preimage); the v0.4->v0.5 migration is 1:1 with no tombstones.
REGISTRY_VERSION = "v0.5"

#: Versions that sealed artifacts may still cite as provenance.  A superseded
#: version is never the digest preimage for NEW identities; it is accepted
#: only when verifying an artifact that records it explicitly.
SUPERSEDED_REGISTRY_VERSIONS = frozenset({"v0.3", "v0.4"})
SUPERSEDED_RENDERER_VERSIONS = frozenset({"r2"})

#: R7: the estimand enumeration is the RELATION vocabulary. Seven primitives
#: plus two derived names typed by the composition rule (`compose_estimands`).
PRIMITIVE_ESTIMANDS = frozenset(
    {"subcategory", "super_category", "element_of", "subtopic",
     "see_also", "assoc", "bridge"}
)
DERIVED_ESTIMANDS = frozenset({"ancestry", "path"})
ESTIMANDS = PRIMITIVE_ESTIMANDS | DERIVED_ESTIMANDS

#: R5: only implementations that actually exist may be registered.
IMPLS = frozenset({"structural", "attention"})

#: v0.5: composed-walk vocabulary for `cowalk`. Each walk value DECLARES its
#: shape at registration — never inferred at query time — because the family
#: classification (symmetric or not) depends on whether the down-sequence is
#: the reverse inverse of the up-sequence (inverse-palindromic), and depth
#: balance alone is not sufficient. Admission to the enum does NOT require
#: invertibility: closeness is a property any walk can carry (an article
#: one-up-one-down from another is near it whether or not `element_of`
#: inverts), so non-palindromic walks may join later with shape
#: "non_palindromic" and no grammar change. The general step-sequence walk
#: grammar is encoder-lane future work; this enum names the walks the
#: generator actually produces.
WALKS = {"sibling": "palindromic", "cousin": "palindromic"}
WALK_SHAPES = frozenset({"palindromic", "non_palindromic"})

#: v0.5: traversal weighting rules for `cowalk`. `idf_node_size` is the
#: hub down-weighting axis (edge information scales inversely with the
#: size of the node it passes through).
WEIGHTS = frozenset({"uniform", "idf_node_size"})

#: Process output types after the R2 split. `source` no longer exists.
OUTPUT_TYPES = frozenset({"substrate", "judge", "score", "target-set", "pick"})

#: Literal value kinds a signature may declare for kwargs or positional args.
VALUE_KINDS = frozenset(
    {"number", "int", "string", "number_list", "int_list", "estimand", "impl",
     "walk", "weight"}
)


@dataclass(frozen=True)
class KwSpec:
    kind: str
    default: object = None
    required: bool = False
    doc: str = ""


@dataclass(frozen=True)
class Signature:
    atom: bool
    min_args: int | None
    max_args: int | None
    arg_types: tuple[str, ...] = ()
    variadic_arg_type: str | None = None
    kwargs: dict[str, KwSpec] = field(default_factory=dict)
    output: str = "judge"
    modifiers: frozenset[str] = frozenset()

    @property
    def operator(self):
        return self.min_args is not None


def _check_declared_type(declared):
    for alternative in declared.split("|"):
        if alternative not in OUTPUT_TYPES and alternative not in VALUE_KINDS \
                and alternative != "process":
            raise ValueError(f"signature declares an unknown type: {declared!r}")


def _sig(
    *,
    atom=False,
    args=None,
    arg_types=(),
    variadic_arg_type=None,
    kwargs=None,
    output="judge",
    modifiers=(),
):
    minimum, maximum = (None, None) if args is None else args
    if args is None and (arg_types or variadic_arg_type is not None):
        raise ValueError("non-operator signature cannot declare positional types")
    if args is not None:
        if len(arg_types) < minimum:
            raise ValueError("signature lacks a type for a required positional argument")
        if maximum is not None and len(arg_types) > maximum:
            raise ValueError("signature has more positional types than arguments")
        if maximum is None and variadic_arg_type is None:
            raise ValueError("unbounded signature requires a variadic positional type")
    if output not in OUTPUT_TYPES:
        raise ValueError(f"signature declares an unknown output type: {output!r}")
    for declared in arg_types:
        _check_declared_type(declared)
    if variadic_arg_type is not None:
        _check_declared_type(variadic_arg_type)
    for spec in (kwargs or {}).values():
        _check_declared_type(spec.kind)
    return Signature(
        atom=atom,
        min_args=minimum,
        max_args=maximum,
        arg_types=tuple(arg_types),
        variadic_arg_type=variadic_arg_type,
        kwargs=dict(kwargs or {}),
        output=output,
        modifiers=frozenset(modifiers),
    )


def _methodology():
    """R5's two-axis methodology slot: what quantity, and which implementation
    of it. Optional everywhere, enumerated, fail-closed on unknown values;
    absence blocks deployment (`require_deployable`) without breaking parsing."""
    return {"estimand": KwSpec("estimand"), "impl": KwSpec("impl")}


#: R3: `decay` and `gamma` are RETENTION factors — the fraction of weight
#: retained per hop, multiplying per hop — not loss rates.
_RETENTION_DOC = "retention factor: fraction of weight retained per hop; multiplies per hop"

# Longest-match lexing over registered names keeps dotted model names distinct
# from modifiers. Signatures are intentionally strict: process identity must
# fail before hashing if a factory expression has unknown or mistyped inputs.
REGISTRY = {
    # -- substrates (R2): walkable corpora, the structures a process walks --
    "pearltrees": _sig(atom=True, output="substrate"),
    "simplemind": _sig(atom=True, output="substrate"),
    "simplewiki": _sig(atom=True, output="substrate"),
    "enwiki": _sig(atom=True, output="substrate"),
    "fs": _sig(atom=True, output="substrate"),
    # -- judges (R2): mu-sources. `graph` is judge-only under v0.4; its former
    #    substrate role is carried by the real corpora above. --
    "graph": _sig(atom=True, output="judge", modifiers=("discrim",)),
    "human": _sig(atom=True, output="judge"),
    "luna": _sig(atom=True, output="judge", modifiers=("D", "S")),
    "sonnet": _sig(atom=True, output="judge", modifiers=("lineage",)),
    "haiku": _sig(atom=True, output="judge"),
    "gpt-5.5-low": _sig(atom=True, output="judge"),
    "gemini": _sig(atom=True, output="judge"),
    "opus": _sig(atom=True, output="judge"),
    "llm": _sig(atom=True, output="judge", modifiers=("element", "subcat")),
    # -- scorers and operators --
    "e5": _sig(
        atom=True,
        args=(1, 1),
        arg_types=("process",),
        kwargs=_methodology(),
        output="score",
    ),
    "routing": _sig(
        args=(2, 2),
        arg_types=("score", "judge"),
        kwargs={
            "t": KwSpec("number_list"),
            "menus": KwSpec("int_list"),
            "manifest": KwSpec("string"),
            **_methodology(),
        },
        output="pick",
    ),
    "pick": _sig(args=(1, 1), arg_types=("target-set",), output="pick"),
    "kalman": _sig(
        args=(2, 2),
        arg_types=("judge", "judge"),
        kwargs=_methodology(),
        output="target-set",
    ),
    "blend": _sig(
        args=(2, None),
        arg_types=("judge", "judge"),
        variadic_arg_type="judge",
        kwargs={"w": KwSpec("number_list"), **_methodology()},
        output="target-set",
    ),
    "lineage": _sig(
        args=(1, 1),
        arg_types=("substrate",),
        kwargs={
            "mu": KwSpec("judge"),
            "decay": KwSpec("number", 0.85, doc=_RETENTION_DOC),
            "depth": KwSpec("int"),
            **_methodology(),
        },
        output="target-set",
    ),
    # v0.5: composed walks (mixed up/down traversals — sibling, cousin).
    # A DIFFERENT operator from `lineage`, whose semantics are a directional
    # up-walk; direction is recorded by the ESTIMAND (R7 derives "path" for
    # mixed compositions), and the walk's depth is implied by its value
    # (sibling=1, cousin=2) — a depth kwarg would let depth and walk
    # disagree, the same defect class the rejected `terminating=` had.
    "cowalk": _sig(
        args=(1, 1),
        arg_types=("substrate",),
        kwargs={
            "walk": KwSpec("walk", required=True),
            "weight": KwSpec("weight", "uniform"),
            "mu": KwSpec("judge"),
            **_methodology(),
        },
        output="target-set",
    ),
    "distill": _sig(
        args=(1, 1), arg_types=("score",), kwargs=_methodology(), output="target-set"
    ),
    "menu": _sig(
        args=(1, 1),
        arg_types=("judge",),
        kwargs={"n": KwSpec("int", required=True)},
        output="target-set",
    ),
    "margin": _sig(
        args=(0, 0),
        kwargs={"t": KwSpec("number", required=True), **_methodology()},
        output="score",
    ),
    # -- R8 additions: the function vocabulary the graph judge composes --
    "product": _sig(
        args=(2, None),
        arg_types=("score", "score"),
        variadic_arg_type="score",
        kwargs=_methodology(),
        output="score",
    ),
    "max": _sig(
        args=(2, None),
        arg_types=("number|score", "score"),
        variadic_arg_type="score",
        kwargs=_methodology(),
        output="score",
    ),
    "hop_decay": _sig(
        args=(1, 1),
        arg_types=("substrate",),
        kwargs={
            "gamma": KwSpec("number", required=True, doc=_RETENTION_DOC),
            **_methodology(),
        },
        output="score",
    ),
    "lca_frac": _sig(
        args=(1, 1), arg_types=("substrate",), kwargs=_methodology(), output="score"
    ),
}
_NAMES = sorted(REGISTRY, key=len, reverse=True)
_MOD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_NUM = re.compile(r"-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_PIN = re.compile(r"[A-Za-z0-9._/-]+")
_KW = re.compile(r"(?P<name>[a-z][a-z0-9_]*)\s*=")


class ParseError(ValueError):
    pass


class CompositionError(ValueError):
    """An estimand chain violates the R7 composition rules."""


class DeploymentError(ValueError):
    """A process is grammatical but not deployable (missing methodology)."""


@dataclass(frozen=True)
class Node:
    name: str
    args: tuple = ()                 # positional child Nodes and/or numeric literals
    kwargs: tuple = ()               # sorted (kw, value) pairs; values canonical
    mods: tuple = ()                 # dotted modifiers, in order
    pins: tuple = ()                 # provenance pins, in order


def _skip_ws(s, i):
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _lex_name(s, i):
    i = _skip_ws(s, i)
    for n in _NAMES:
        if s.startswith(n, i):
            end = i + len(n)
            if end == len(s) or s[end] in " \t\r\n(),.@=[]":
                return n, end
    raise ParseError(f"unregistered name at {i}: {s[i:i+24]!r}")


def _parse_val(s, i):
    i = _skip_ws(s, i)
    if i >= len(s):
        raise ParseError("missing value at end of input")
    if s[i] == "[":
        out, i = [], i + 1
        i = _skip_ws(s, i)
        if i < len(s) and s[i] == "]":
            return tuple(out), i + 1
        while True:
            v, i = _parse_val(s, i)
            out.append(v)
            i = _skip_ws(s, i)
            if i >= len(s):
                raise ParseError("unterminated list")
            if s[i] == "]":
                return tuple(out), i + 1
            if s[i] == ",":
                i += 1
                i = _skip_ws(s, i)
                if i < len(s) and s[i] == "]":
                    raise ParseError("trailing comma in list")
                continue
            raise ParseError(f"expected ',' or ']' at {i}")
    if s[i] == '"':
        try:
            value, consumed = json.JSONDecoder().raw_decode(s[i:])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ParseError(f"invalid string at {i}: {exc}") from exc
        if not isinstance(value, str):
            raise ParseError(f"expected string at {i}")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ParseError(f"string contains an invalid Unicode surrogate at {i}")
        return value, i + consumed
    m = _NUM.match(s, i)
    if m:
        t = m.group(0)
        value = float(t) if ("." in t or "e" in t.lower()) else int(t)
        if isinstance(value, float) and not math.isfinite(value):
            raise ParseError(f"non-finite number at {i}")
        return value, m.end()
    raise ParseError(f"expected number, list, or quoted string at {i}")


def _value_matches(kind, value):
    number = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    )
    if kind == "number":
        return number
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "string":
        return (
            isinstance(value, str)
            and not any(0xD800 <= ord(character) <= 0xDFFF for character in value)
        )
    if kind == "estimand":
        # Fail closed: an unknown estimand is refused, never defaulted (R5/R7).
        return isinstance(value, str) and value in ESTIMANDS
    if kind == "impl":
        return isinstance(value, str) and value in IMPLS
    if kind == "walk":
        # Fail closed like estimand/impl: only registered walk values, each
        # of which declared its shape when it was added.
        return isinstance(value, str) and value in WALKS
    if kind == "weight":
        return isinstance(value, str) and value in WEIGHTS
    if kind == "number_list":
        return (
            isinstance(value, tuple)
            and bool(value)
            and all(_value_matches("number", item) for item in value)
        )
    if kind == "int_list":
        return isinstance(value, tuple) and bool(value) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
    raise AssertionError(f"unknown registry value kind: {kind}")


def _output_matches(expected, actual):
    return expected == "process" or expected == actual


def _arg_matches(declared, child):
    """A positional slot accepts what its declared type says — alternatives
    separated by '|', each either an output type or a literal value kind
    (§10.1 flagged decision 1: declared-type-wins, same as kwargs)."""
    for alternative in declared.split("|"):
        if isinstance(child, Node):
            if (alternative == "process" or alternative in OUTPUT_TYPES) and \
                    _output_matches(alternative, REGISTRY[child.name].output):
                return True
        elif alternative in VALUE_KINDS and _value_matches(alternative, child):
            return True
    return False


def _describe_arg(child):
    if isinstance(child, Node):
        return REGISTRY[child.name].output
    return f"literal {type(child).__name__}"


def _validate_signature(name, called, args, kwargs, mods):
    signature = REGISTRY[name]
    if called:
        if not signature.operator:
            raise ParseError(f"{name} is not an operator")
        if len(args) < signature.min_args or (
            signature.max_args is not None and len(args) > signature.max_args
        ):
            maximum = "unbounded" if signature.max_args is None else signature.max_args
            raise ParseError(
                f"{name} expects {signature.min_args}..{maximum} positional args, got {len(args)}"
            )
        for index, child in enumerate(args):
            if index < len(signature.arg_types):
                expected = signature.arg_types[index]
            else:
                expected = signature.variadic_arg_type
            if expected is None:
                raise AssertionError(f"{name} signature lacks positional type {index}")
            if not _arg_matches(expected, child):
                raise ParseError(
                    f"{name} argument {index + 1} must be {expected}, "
                    f"got {_describe_arg(child)}"
                )
    elif not signature.atom:
        raise ParseError(f"{name} is not an atom")
    seen = set()
    for key, value in kwargs:
        if key in seen:
            raise ParseError(f"duplicate kwarg for {name}: {key}")
        seen.add(key)
        spec = signature.kwargs.get(key)
        if spec is None:
            raise ParseError(f"unknown kwarg for {name}: {key}")
        if spec.kind in OUTPUT_TYPES or spec.kind == "process":
            if not isinstance(value, Node) or not _output_matches(
                spec.kind, REGISTRY[value.name].output
            ):
                raise ParseError(f"{name}.{key} must be a {spec.kind} expression")
        elif not _value_matches(spec.kind, value):
            enumerated = {
                "estimand": ESTIMANDS, "impl": IMPLS,
                "walk": frozenset(WALKS), "weight": WEIGHTS,
            }
            if spec.kind in enumerated:
                raise ParseError(
                    f"{name}.{key} must be one of the registered {spec.kind} "
                    f"values {sorted(enumerated[spec.kind])}; unknown values "
                    f"are refused, not defaulted"
                )
            raise ParseError(f"{name}.{key} must be {spec.kind}")
    missing = [
        key for key, spec in signature.kwargs.items()
        if spec.required and key not in seen
    ]
    if missing:
        raise ParseError(f"missing required kwargs for {name}: {','.join(sorted(missing))}")
    unknown_mods = set(mods) - signature.modifiers
    if unknown_mods:
        raise ParseError(f"unknown modifiers for {name}: {','.join(sorted(unknown_mods))}")
    if len(mods) != len(set(mods)):
        raise ParseError(f"duplicate modifier for {name}")
    if len(mods) > 1:
        raise ParseError(f"{name} accepts at most one modifier")
    values = dict(kwargs)
    if name == "routing":
        has_thresholds = "t" in values
        has_menus = "menus" in values
        if has_thresholds != has_menus:
            raise ParseError("routing.t and routing.menus must be supplied together")
        if has_thresholds and len(values["t"]) != len(values["menus"]):
            raise ParseError("routing.t and routing.menus must have equal lengths")
    if name == "blend" and "w" in values and len(values["w"]) != len(args):
        raise ParseError("blend.w must contain one weight per positional source")


def _parse_expr(s, i):
    i = _skip_ws(s, i)
    name, i = _lex_name(s, i)
    args, kwargs = [], []
    i = _skip_ws(s, i)
    called = i < len(s) and s[i] == "("
    if called:
        i += 1
        i = _skip_ws(s, i)
        while i < len(s) and s[i] != ")":
            kw_match = _KW.match(s, i)
            if kw_match:
                kw = kw_match.group("name")
                j = _skip_ws(s, kw_match.end())
                if j < len(s) and (s[j].isdigit() or s[j] in '-["'):
                    v, i = _parse_val(s, j)
                else:
                    # A kwarg value that is not a literal is a sub-expression
                    # (e.g. mu=haiku); the signature decides whether that is
                    # legal, the grammar only decides what it is.
                    v, i = _parse_expr(s, j)
                kwargs.append((kw, v))
            elif s[i].isdigit() or s[i] == "-":
                # §10.1 flagged decision 1: numeric literals are legal in
                # positional argument positions, typed by the declared kind.
                v, i = _parse_val(s, i)
                args.append(v)
            else:
                child, i = _parse_expr(s, i)
                args.append(child)
            i = _skip_ws(s, i)
            if i >= len(s):
                raise ParseError(f"unterminated call to {name}")
            if s[i] == ",":
                i += 1
                i = _skip_ws(s, i)
                if i < len(s) and s[i] == ")":
                    raise ParseError(f"trailing comma in call to {name}")
            elif s[i] != ")":
                raise ParseError(f"expected ',' or ')' at {i}")
        if i >= len(s):
            raise ParseError(f"unterminated call to {name}")
        i += 1
    i = _skip_ws(s, i)
    mods, pins = [], []
    while i < len(s) and s[i] == ".":
        m = _MOD.match(s, i + 1)
        if not m:
            raise ParseError(f"bad modifier at {i + 1}")
        mods.append(m.group(0))
        i = m.end()
    while i < len(s) and s[i] == "@":
        m = _PIN.match(s, i + 1)
        if not m:
            raise ParseError(f"bad pin at {i + 1}")
        pins.append(m.group(0))
        i = m.end()
    _validate_signature(name, called, args, kwargs, mods)
    # An explicitly written default is the same process as its elided form.
    defaults = {key: spec.default for key, spec in REGISTRY[name].kwargs.items()}
    kwargs = [(key, value) for key, value in kwargs if defaults.get(key) != value]
    return Node(name, tuple(args), tuple(sorted(kwargs, key=lambda kv: kv[0])), tuple(mods), tuple(pins)), i


def parse(text):
    if not isinstance(text, str) or not text.strip():
        raise ParseError("expression must be a nonempty string")
    node, i = _parse_expr(text, 0)
    i = _skip_ws(text, i)
    if i != len(text):
        raise ParseError(f"trailing input at {i}")
    return node


def _validate_literal(owner, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError(f"{owner} has a malformed positional literal")
    if isinstance(value, float) and not math.isfinite(value):
        raise ParseError(f"{owner} has a non-finite positional literal")


def validate(node):
    """Fail closed on a parsed or programmatically constructed process AST."""
    if not isinstance(node, Node):
        raise ParseError("process AST must contain Node instances")
    if not isinstance(node.name, str) or node.name not in REGISTRY:
        raise ParseError(f"unregistered node name: {node.name!r}")
    if not isinstance(node.args, tuple) or not isinstance(node.kwargs, tuple):
        raise ParseError(f"{node.name} args and kwargs must be tuples")
    if not isinstance(node.mods, tuple) or not isinstance(node.pins, tuple):
        raise ParseError(f"{node.name} modifiers and pins must be tuples")
    for child in node.args:
        if isinstance(child, Node):
            validate(child)
        else:
            _validate_literal(node.name, child)
    for item in node.kwargs:
        if not (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
        ):
            raise ParseError(f"{node.name} has a malformed kwarg")
        if isinstance(item[1], Node):
            validate(item[1])
    for modifier in node.mods:
        if not isinstance(modifier, str) or _MOD.fullmatch(modifier) is None:
            raise ParseError(f"{node.name} has a malformed modifier")
    for pin in node.pins:
        if not isinstance(pin, str) or _PIN.fullmatch(pin) is None:
            raise ParseError(f"{node.name} has a malformed provenance pin")
    signature = REGISTRY[node.name]
    called = bool(node.args or node.kwargs) or (signature.operator and not signature.atom)
    _validate_signature(node.name, called, node.args, node.kwargs, node.mods)
    return node


def _render_val(v):
    if isinstance(v, tuple):
        return "[" + ",".join(_render_val(x) for x in v) + "]"
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("cannot render non-finite process value")
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    if isinstance(v, Node):
        raise ValueError("node values render through the canonicalizer, not _render_val")
    return str(v)


def _validate_verbosity(verbosity):
    if isinstance(verbosity, bool) or not isinstance(verbosity, int) or not 0 <= verbosity <= 3:
        raise ValueError("verbosity must be one of 0, 1, 2, or 3")


def render(node, verbosity=3):
    """V1: names+structure, kwargs elided. V2: + non-default kwargs. V3: + pins. V0: ''.
    Lossless canonical = ``canonical_full`` (identity digests use
    ``canonical_semantic``, which strips pins)."""
    validate(node)
    _validate_verbosity(verbosity)
    if verbosity == 0:
        return ""

    def _value(v):
        return render(v, verbosity) if isinstance(v, Node) else _render_val(v)

    kw = ""
    if verbosity >= 2:
        defaults = {
            key: spec.default for key, spec in REGISTRY[node.name].kwargs.items()
        }
        kept = [(k, v) for k, v in node.kwargs if defaults.get(k) != v]
        if kept:
            kw = ("," if node.args else "") + ",".join(f"{k}={_value(v)}" for k, v in kept)
    inner = ",".join(
        render(a, verbosity) if isinstance(a, Node) else _render_val(a)
        for a in node.args
    )
    called = bool(node.args or kw) or (
        REGISTRY[node.name].operator and not REGISTRY[node.name].atom
    )
    s = node.name + (f"({inner}{kw})" if called else "")
    s += "".join("." + m for m in node.mods)
    if verbosity >= 3:
        s += "".join("@" + p for p in node.pins)
    return s


def _canonical(node, keep_pins):
    defaults = {
        key: spec.default for key, spec in REGISTRY[node.name].kwargs.items()
    }
    kws = dict(defaults)
    kws.update(dict(node.kwargs))

    def _value(v):
        return _canonical(v, keep_pins) if isinstance(v, Node) else _render_val(v)

    kw = ",".join(f"{k}={_value(v)}" for k, v in sorted(kws.items()) if v is not None)
    inner = ",".join(_value(a) for a in node.args)
    body = ",".join(x for x in (inner, kw) if x)
    s = node.name + (f"({body})" if body else "")
    s += "".join("." + m for m in node.mods)
    if keep_pins:
        s += "".join("@" + p for p in node.pins)
    return s


def canonical_semantic(node):
    """The SEMANTIC identity string: V3-with-resolved-defaults, pins stripped
    at every node. This is the digest preimage for ``ast_sha``, seals, and
    cache keys (R9 §10.3): two derivations of the same precise AST share
    semantic identity, so a pin can never mint a new process."""
    validate(node)
    return _canonical(node, keep_pins=False)


def canonical_full(node):
    """The provenance string: semantic form plus pins, for round-trip and
    audit. Never a digest preimage for semantic identity."""
    validate(node)
    return _canonical(node, keep_pins=True)


def canonical(node):
    """Back-compat name for the lossless round-trip string (= canonical_full).
    Identity digests use ``canonical_semantic``."""
    return canonical_full(node)


def ast_sha(node):
    return hashlib.sha256(
        (REGISTRY_VERSION + "|" + canonical_semantic(node)).encode()).hexdigest()[:16]


def embedding_cache_key(node, verbosity, e5_revision, prefix="passage"):
    """Cache key for a conditioning-card embedding (stage 5, §5).

    Cards are e5 embeddings of registry-dependent strings, so the key binds
    ``REGISTRY_VERSION`` explicitly — not only transitively through the
    truncated ``ast_sha`` — so a post-bump lookup can never return a vector
    embedded from a string that no longer exists."""
    _validate_verbosity(verbosity)
    parts = [
        REGISTRY_VERSION,
        ast_sha(node),
        str(verbosity),
        RENDERER_VERSION,
        e5_revision,
        prefix,
    ]
    if verbosity >= 3:
        # V3 is the only verbosity that renders pins, and pins are outside
        # ast_sha under v0.4 — bind the pin-bearing canonical so two pin
        # variants of one semantic process cannot share a V3 card embedding.
        parts.append(
            hashlib.sha256(
                (REGISTRY_VERSION + "|" + canonical_full(node)).encode()
            ).hexdigest()[:16]
        )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def compose_estimands(steps):
    """Type a relation chain per R7. Chains are written ITEM-FIRST (the first
    step leaves the item, later steps continue upward/downward through the
    hierarchy). Returns the composite estimand:

    - a single primitive composes to itself;
    - `element_of` may appear only as the item-end step (index 0) and survives
      descent (`element_of ∘ subcategory ⇒ element_of`);
    - `subtopic` and `subcategory` compose interchangeably — the label is
      curational, the graph relation is primary;
    - `bridge` is transparent to typing;
    - a monotone composition types as `ancestry`; any mixed-direction
      composition types as `path`;
    - `assoc` and `see_also` are excluded from chains entirely.
    """
    if not isinstance(steps, (list, tuple)) or not steps:
        raise CompositionError("an estimand chain must be a nonempty sequence")
    for step in steps:
        if step not in PRIMITIVE_ESTIMANDS:
            raise CompositionError(
                f"chain step must be a primitive estimand, got {step!r}"
            )
        if step in ("assoc", "see_also"):
            raise CompositionError(
                f"{step} is non-transitive and excluded from chains (R7)"
            )
    for index, step in enumerate(steps):
        if step == "element_of" and index != 0:
            raise CompositionError(
                "element_of terminates a chain at the item end; membership is "
                "not transitive through membership (R7)"
            )
    typed = [step for step in steps if step != "bridge"]
    if not typed:
        return "bridge"
    directions = {
        "up" if step in ("subcategory", "subtopic", "element_of") else "down"
        for step in typed
    }
    if len(directions) > 1:
        return "path"
    if len(typed) == 1:
        return typed[0]
    if typed[0] == "element_of":
        return "element_of"
    return "ancestry"


def require_deployable(node):
    """Deployment gate (test obligation 4): a process may not deploy without a
    stated `estimand=` on its root. Ordinary registered defaults (`decay=0.85`)
    keep working — only the methodology slot fails closed on absence. `impl=`
    stays optional: it is a selection constraint, not identity (R9)."""
    validate(node)
    signature = REGISTRY[node.name]
    if "estimand" not in signature.kwargs:
        raise DeploymentError(
            f"{node.name} has no estimand slot and cannot state its methodology"
        )
    if dict(node.kwargs).get("estimand") is None:
        raise DeploymentError(
            f"{node.name} does not state estimand=; absence is unresolved, "
            f"not a default (R5)"
        )
    return node


# Registry of CURRENT processes. The eight v0.3 survivors re-spell unchanged
# (their judge arguments re-type registry-side). `lineage-graph` is retired per
# §10.1 — the stage-4 migration manifest owns its history — and two additions
# exercise the new vocabulary.
PROCESSES = {
    "e5-auto": "e5(margin(t=0.03))",
    "haiku-n10": "e5(routing(e5,haiku,t=[0.02],menus=[10]))",
    "sonnet-lin-n10": "e5(routing(e5,sonnet.lineage,t=[0.02],menus=[10]))",
    "sonnet-lin-n20": "e5(routing(e5,sonnet.lineage,t=[0.02,0.03],menus=[10,20]))",
    "kalman-fused": "kalman(luna.D,luna.S)",
    "blend": "blend(luna.D,luna.S)",
    "dir-blend": "blend(graph.discrim,llm.element,llm.subcat)",
    "distill-3tier": "distill(e5(routing(e5,sonnet.lineage,t=[0.02,0.03],menus=[10,20])))",
    # R8: the graph judge, max(floor, gamma^hops * lca_frac), estimand `path` —
    # parameters are the prototype's measured defaults (gamma 0.6, floor 0.02).
    "graph-judge": (
        'max(0.02,product(hop_decay(simplemind,gamma=0.6),'
        'lca_frac(simplemind)),estimand="path")'
    ),
    # R4's own example: lineage over a real substrate with an explicit mu-source.
    "lineage-haiku": 'lineage(pearltrees,mu=haiku,estimand="ancestry")',
}
