#!/usr/bin/env python3
"""P0 of DESIGN_process_expression_*: AST, registry-driven parser, canonicalizer, verbosity cards.

Lossless process identity = canonical AST string + registry version (+ factory fingerprint,
supplied by callers). Cards (V0-V3) are lossy renderings for conditioning only. Embedding cache
keys bind (ast_sha, verbosity, RENDERER_VERSION, embedding revision, prefix) — never the string.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re

RENDERER_VERSION = "r2"
REGISTRY_VERSION = "v0.3"


@dataclass(frozen=True)
class KwSpec:
    kind: str
    default: object = None
    required: bool = False


@dataclass(frozen=True)
class Signature:
    atom: bool
    min_args: int | None
    max_args: int | None
    arg_types: tuple[str, ...] = ()
    variadic_arg_type: str | None = None
    kwargs: dict[str, KwSpec] = field(default_factory=dict)
    output: str = "source"
    modifiers: frozenset[str] = frozenset()

    @property
    def operator(self):
        return self.min_args is not None


def _sig(
    *,
    atom=False,
    args=None,
    arg_types=(),
    variadic_arg_type=None,
    kwargs=None,
    output="source",
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


# Longest-match lexing over registered names keeps dotted model names distinct
# from modifiers. Signatures are intentionally strict: P0 process identity must
# fail before hashing if a factory expression has unknown or mistyped inputs.
REGISTRY = {
    "e5": _sig(atom=True, args=(1, 1), arg_types=("process",), output="score"),
    "graph": _sig(atom=True, output="source", modifiers=("discrim",)),
    "human": _sig(atom=True),
    "luna": _sig(atom=True, modifiers=("D", "S")),
    "sonnet": _sig(atom=True, modifiers=("lineage",)),
    "haiku": _sig(atom=True),
    "gpt-5.5-low": _sig(atom=True),
    "gemini": _sig(atom=True),
    "opus": _sig(atom=True),
    "routing": _sig(
        args=(2, 2),
        arg_types=("score", "source"),
        kwargs={
            "t": KwSpec("number_list"),
            "menus": KwSpec("int_list"),
            "manifest": KwSpec("string"),
        },
        output="pick",
    ),
    "pick": _sig(args=(1, 1), arg_types=("target-set",), output="pick"),
    "kalman": _sig(
        args=(2, 2),
        arg_types=("source", "source"),
        output="target-set",
    ),
    "blend": _sig(
        args=(2, None),
        arg_types=("source", "source"),
        variadic_arg_type="source",
        kwargs={"w": KwSpec("number_list")},
        output="target-set",
    ),
    "lineage": _sig(
        args=(1, 1),
        arg_types=("source",),
        kwargs={
            "decay": KwSpec("number", 0.85),
            "depth": KwSpec("int"),
        },
        output="target-set",
    ),
    "distill": _sig(args=(1, 1), arg_types=("score",), output="target-set"),
    "menu": _sig(
        args=(1, 1),
        arg_types=("source",),
        kwargs={"n": KwSpec("int", required=True)},
        output="target-set",
    ),
    "margin": _sig(
        args=(0, 0),
        kwargs={"t": KwSpec("number", required=True)},
        output="score",
    ),
    "llm": _sig(atom=True, modifiers=("element", "subcat")),
}
_NAMES = sorted(REGISTRY, key=len, reverse=True)
_MOD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_NUM = re.compile(r"-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_PIN = re.compile(r"[A-Za-z0-9._/-]+")
_KW = re.compile(r"(?P<name>[a-z][a-z0-9_]*)\s*=")


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Node:
    name: str
    args: tuple = ()                 # positional child Nodes
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
            actual = REGISTRY[child.name].output
            if not _output_matches(expected, actual):
                raise ParseError(
                    f"{name} argument {index + 1} must be {expected}, got {actual}"
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
        if not _value_matches(spec.kind, value):
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
                v, i = _parse_val(s, kw_match.end())
                kwargs.append((kw, v))
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
    return Node(name, tuple(args), tuple(sorted(kwargs)), tuple(mods), tuple(pins)), i


def parse(text):
    if not isinstance(text, str) or not text.strip():
        raise ParseError("expression must be a nonempty string")
    node, i = _parse_expr(text, 0)
    i = _skip_ws(text, i)
    if i != len(text):
        raise ParseError(f"trailing input at {i}")
    return node


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
        validate(child)
    for item in node.kwargs:
        if not (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
        ):
            raise ParseError(f"{node.name} has a malformed kwarg")
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
    return str(v)


def _validate_verbosity(verbosity):
    if isinstance(verbosity, bool) or not isinstance(verbosity, int) or not 0 <= verbosity <= 3:
        raise ValueError("verbosity must be one of 0, 1, 2, or 3")


def render(node, verbosity=3):
    """V1: names+structure, kwargs elided. V2: + non-default kwargs. V3: + pins. V0: ''. Lossless
    canonical = V3 plus default kwargs made explicit (identity string)."""
    validate(node)
    _validate_verbosity(verbosity)
    if verbosity == 0:
        return ""
    kw = ""
    if verbosity >= 2:
        defaults = {
            key: spec.default for key, spec in REGISTRY[node.name].kwargs.items()
        }
        kept = [(k, v) for k, v in node.kwargs if defaults.get(k) != v]
        if kept:
            kw = ("," if node.args else "") + ",".join(f"{k}={_render_val(v)}" for k, v in kept)
    inner = ",".join(render(a, verbosity) for a in node.args)
    called = bool(node.args or kw) or (
        REGISTRY[node.name].operator and not REGISTRY[node.name].atom
    )
    s = node.name + (f"({inner}{kw})" if called else "")
    s += "".join("." + m for m in node.mods)
    if verbosity >= 3:
        s += "".join("@" + p for p in node.pins)
    return s


def canonical(node):
    """Lossless identity string: V3 rendering with ALL kwargs explicit (defaults resolved)."""
    validate(node)
    defaults = {
        key: spec.default for key, spec in REGISTRY[node.name].kwargs.items()
    }
    kws = dict(defaults)
    kws.update(dict(node.kwargs))
    kw = ",".join(f"{k}={_render_val(v)}" for k, v in sorted(kws.items()) if v is not None)
    inner = ",".join(canonical(a) for a in node.args)
    body = ",".join(x for x in (inner, kw) if x)
    s = node.name + (f"({body})" if body else "")
    return s + "".join("." + m for m in node.mods) + "".join("@" + p for p in node.pins)


def ast_sha(node):
    return hashlib.sha256(
        (REGISTRY_VERSION + "|" + canonical(node)).encode()).hexdigest()[:16]


def embedding_cache_key(node, verbosity, e5_revision, prefix="passage"):
    _validate_verbosity(verbosity)
    return hashlib.sha256("|".join(
        [ast_sha(node), str(verbosity), RENDERER_VERSION, e5_revision, prefix]
    ).encode()).hexdigest()[:16]


# Registry of CURRENT processes (P0 exit requirement)
PROCESSES = {
    "e5-auto": "e5(margin(t=0.03))",
    "haiku-n10": "e5(routing(e5,haiku,t=[0.02],menus=[10]))",
    "sonnet-lin-n10": "e5(routing(e5,sonnet.lineage,t=[0.02],menus=[10]))",
    "sonnet-lin-n20": "e5(routing(e5,sonnet.lineage,t=[0.02,0.03],menus=[10,20]))",
    "kalman-fused": "kalman(luna.D,luna.S)",
    "blend": "blend(luna.D,luna.S)",
    "dir-blend": "blend(graph.discrim,llm.element,llm.subcat)",
    "lineage-graph": "lineage(graph,decay=0.85)",
    "distill-3tier": "distill(e5(routing(e5,sonnet.lineage,t=[0.02,0.03],menus=[10,20])))",
}
