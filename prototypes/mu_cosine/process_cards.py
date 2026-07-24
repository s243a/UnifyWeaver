#!/usr/bin/env python3
"""P0 of DESIGN_process_expression_*: AST, registry-driven parser, canonicalizer, verbosity cards.

Lossless process identity = canonical AST string + registry version (+ factory fingerprint,
supplied by callers). Cards (V0-V3) are lossy renderings for conditioning only. Embedding cache
keys bind (ast_sha, verbosity, RENDERER_VERSION, embedding revision, prefix) — never the string.
"""
from dataclasses import dataclass, field
import hashlib
import re

RENDERER_VERSION = "r1"
REGISTRY_VERSION = "v0.2"

# name -> (is_atom, is_operator, {kw: default}) ; longest-match lexing over these names
REGISTRY = {
    "e5": (True, True, {}),
    "graph": (True, True, {}),
    "human": (True, False, {}),
    "luna": (True, False, {}), "sonnet": (True, False, {}), "haiku": (True, False, {}),
    "gpt-5.5-low": (True, False, {}), "gemini": (True, False, {}), "opus": (True, False, {}),
    "routing": (False, True, {"t": None, "menus": None}),
    "pick": (False, True, {}),
    "kalman": (False, True, {}),
    "blend": (False, True, {"w": None}),
    "lineage": (False, True, {"decay": 0.85, "depth": None}),
    "distill": (False, True, {}),
    "menu": (False, True, {"n": None}),
    "margin": (False, True, {"t": None}),
    "llm": (True, False, {}),
}
_NAMES = sorted(REGISTRY, key=len, reverse=True)
_MOD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_NUM = re.compile(r"-?[0-9]+(\.[0-9]+)?")
_PIN = re.compile(r"[A-Za-z0-9._/-]+")
_KW = re.compile(r"[a-z][a-z0-9_]*=")


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Node:
    name: str
    args: tuple = ()                 # positional child Nodes
    kwargs: tuple = ()               # sorted (kw, value) pairs; values canonical
    mods: tuple = ()                 # dotted modifiers, in order
    pins: tuple = ()                 # provenance pins, in order


def _lex_name(s, i):
    for n in _NAMES:
        if s.startswith(n, i):
            return n, i + len(n)
    raise ParseError(f"unregistered name at {i}: {s[i:i+24]!r}")


def _parse_val(s, i):
    if s[i] == "[":
        out, i = [], i + 1
        while s[i] != "]":
            v, i = _parse_val(s, i)
            out.append(v)
            if s[i] == ",":
                i += 1
        return tuple(out), i + 1
    if s[i] == '"':
        j = s.index('"', i + 1)
        return s[i + 1:j], j + 1
    m = _NUM.match(s, i)
    if m:
        t = m.group(0)
        return (float(t) if "." in t else int(t)), m.end()
    n, i = _lex_name(s, i)
    return n, i


def _parse_expr(s, i):
    name, i = _lex_name(s, i)
    args, kwargs = [], []
    if i < len(s) and s[i] == "(":
        if not REGISTRY[name][1]:
            raise ParseError(f"{name} is not an operator")
        i += 1
        while s[i] != ")":
            if _KW.match(s, i):
                kw = s[i:s.index("=", i)]
                v, i = _parse_val(s, s.index("=", i) + 1)
                kwargs.append((kw, v))
            else:
                child, i = _parse_expr(s, i)
                args.append(child)
            if s[i] == ",":
                i += 1
        i += 1
    elif not REGISTRY[name][0]:
        raise ParseError(f"{name} is not an atom")
    # normalize: an explicitly-written default kwarg is the same process as the elided form
    defaults = REGISTRY[name][2]
    kwargs = [(k, v) for k, v in kwargs if defaults.get(k) != v]
    mods, pins = [], []
    while i < len(s) and s[i] in ".@":
        c, i2 = s[i], i + 1
        m = (_MOD if c == "." else _PIN).match(s, i2)
        if not m:
            raise ParseError(f"bad {'modifier' if c=='.' else 'pin'} at {i2}")
        (mods if c == "." else pins).append(m.group(0))
        i = m.end()
    return Node(name, tuple(args), tuple(sorted(kwargs)), tuple(mods), tuple(pins)), i


def parse(text):
    node, i = _parse_expr(text.replace(" ", ""), 0)
    if i != len(text.replace(" ", "")):
        raise ParseError(f"trailing input at {i}")
    return node


def _render_val(v):
    if isinstance(v, tuple):
        return "[" + ",".join(_render_val(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def render(node, verbosity=3):
    """V1: names+structure, kwargs elided. V2: + non-default kwargs. V3: + pins. V0: ''. Lossless
    canonical = V3 plus default kwargs made explicit (identity string)."""
    if verbosity == 0:
        return ""
    kw = ""
    if verbosity >= 2:
        defaults = REGISTRY[node.name][2]
        kept = [(k, v) for k, v in node.kwargs if defaults.get(k) != v]
        if kept:
            kw = ("," if node.args else "") + ",".join(f"{k}={_render_val(v)}" for k, v in kept)
    inner = ",".join(render(a, verbosity) for a in node.args)
    s = node.name + (f"({inner}{kw})" if (node.args or kw) else "")
    s += "".join("." + m for m in node.mods)
    if verbosity >= 3:
        s += "".join("@" + p for p in node.pins)
    return s


def canonical(node):
    """Lossless identity string: V3 rendering with ALL kwargs explicit (defaults resolved)."""
    defaults = dict(REGISTRY[node.name][2])
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
