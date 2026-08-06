#!/usr/bin/env python3
"""Resolved AST DTO, complete typed role paths, and reference token stream.

Step 1 of ``DESIGN_expression_encoder_future.md`` §11: contract fixtures, with
no change to current P0 behavior.  Nothing here trains, embeds, or conditions
anything — it freezes *what the encoder's input must look like* so that step 2's
real tokenizer is written against bytes that already exist.

Three layers, in the order §3 defines them:

1. **Resolved canonical AST DTO** (§3) — a ``process_cards.Node`` plus the
   pinned registry, with every default and derived signature field resolved.
   ``Node`` itself stores neither resolved defaults nor ``KIND``/``OUTPUT``;
   this transform derives them, and its kwarg resolution is checked against
   ``canonical()`` so the DTO can never disagree with process identity.
2. **Complete typed role paths** (§3.2) — every item carries its whole
   root-to-item path, not a lossy ``(depth, breadth)`` pair.  The root has the
   empty path; ``ROOT`` is a distinguished token, not an edge step.
3. **Reference token stream** (§3.1) — the structural serialization those paths
   decorate.

Scope boundary, stated plainly: this is the *fixture authority*, not the
production tokenizer.  Step 2 owns the versioned vocabulary IDs, the explicit
256-byte literal fallback, and the reversibility guarantee.  What this module
guarantees is that step 2 has a frozen structural target to reproduce.  The
token names below are therefore structural strings, not vocabulary indices.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from process_cards import (
    OUTPUT_TYPES,
    REGISTRY,
    REGISTRY_VERSION,
    VALUE_KINDS,
    Node,
    _render_val,
    canonical_full,
    canonical_semantic,
    parse,
    validate,
)
from process_identity import full_ast_digest

#: Bumped whenever the structural stream or role-path serialization changes.
#: Versioned independently of the grammar, per §3.1.
#:
#: pec-v2: scalar literals carry their *declared registry kind* rather than a
#: type inferred from the Python runtime value.  ``margin(t=1)`` now emits
#: ``<NUMBER>`` instead of ``<INT>``.  No pec-v1 fixture row was affected — that
#: is precisely why pec-v2 adds a row covering an integer-spelled ``number``.
#:
#: pec-v3 (registry v0.4): numeric literals may occupy positional argument
#: slots — they serialize as value tokens inside the ``<ARG:i>`` fence
#: (§10.1 flagged decision 1); kwargs may carry sub-expression values
#: (``mu=haiku``), serialized as a ``<NODE>`` inside the ``<KW:…>`` fence;
#: the enumerated methodology kinds get their own fences (``<ESTIMAND>``,
#: ``<IMPL>``); and golden rows split identity per R9 — the identity string
#: is the pin-free ``canonical_semantic``, with ``canonical_full_string``
#: carried alongside for provenance and round-trip.
#: pec-v4 (registry v0.5): the composed-walk kinds get their own fences
#: (``<WALK>``, ``<WEIGHT>``), same shape as the pec-v3 methodology fences —
#: the declared kind stays on the stream so a decoder holds the value to the
#: registry enumeration. This is a token-stream grammar change, so the
#: contract version moves and bundle v4 is sealed; and per the standing note
#: on PR #4077, the parked `pick` and scientific-notation rows join
#: REQUIRED_COVERAGE_CASES at this change.
CONTRACT_VERSION = "pec-v4"

GOLDEN_SCHEMA = "unifyweaver.process-expression-golden.v2"

#: The bundle a consumer must reproduce.  When a contract change seals a new
#: bundle, this pointer and every document naming it move together — see
#: ``DESIGN_process_expression_generator.md`` §0 for the supersession procedure.
CURRENT_GOLDEN_BUNDLE = "PROCESS_EXPRESSION_GOLDEN_v4.json"

#: Superseded bundles, retained as audit-only provenance.  They are never
#: mutated and never accepted by the current loader; their integrity is pinned
#: separately so corruption is detectable independently of the version check.
SUPERSEDED_GOLDEN_BUNDLES = {
    "PROCESS_EXPRESSION_GOLDEN_v1.json": {
        "contract_version": "pec-v1",
        "sha256": "b053351a2a419ac58b7ab644afe15c60543846ce8b9d5a3d9bcbc332ca24db29",
    },
    "PROCESS_EXPRESSION_GOLDEN_v2.json": {
        "contract_version": "pec-v2",
        "sha256": "85e6421f5a1347fca5937d1243dc01500a9aa5b7221571b4918248e57ece6344",
    },
    "PROCESS_EXPRESSION_GOLDEN_v3.json": {
        "contract_version": "pec-v3",
        "sha256": "90cc484021150aa9916be2f8c4fdb57b66f3a2e7d18dafff7e40c3c566af8ef7",
    },
}

#: Coverage the golden bundle must carry beyond the registered processes.
#: Canonical source: the committed bundle is reproducible from this module
#: alone, rather than from command-line flags someone typed once.
REQUIRED_COVERAGE_CASES = {
    "atom-bare": "graph.discrim",
    "atom-dual-bare": "e5",
    # v0.4 re-spellings: `graph` is judge-only, so substrate slots take corpora.
    "pinned": "lineage(pearltrees,decay=0.85)@run/2026-07-25",
    "utf8-string": 'routing(e5,haiku,t=[0.02],menus=[10],manifest="héllo·wörld")',
    "escaped-string": 'routing(e5,haiku,t=[0.02],menus=[10],manifest="a\\"b\\\\c")',
    "menu-required-int": "menu(graph,n=10)",
    "blend-variadic": "blend(luna.D,luna.S,graph)",
    "margin-number": "margin(t=0.03)",
    "neg-number": "lineage(fs,decay=-0.5)",
    # pec-v2: a `number` field spelled as an integer must keep its declared kind.
    "int-spelled-number": "margin(t=1)",
    "int-spelled-number-list": "routing(e5,haiku,t=[1],menus=[10])",
    # pec-v3 / registry v0.4 coverage: each new grammar feature gets a row.
    "substrate-atom": "fs",
    "numeric-positional-literal": "max(0.02,e5(margin(t=0.03)))",
    "mu-judge-kwarg": "lineage(simplewiki,mu=sonnet.lineage)",
    "estimand-impl": 'lca_frac(simplemind,estimand="path",impl="structural")',
    # pec-v4 / registry v0.5 coverage: each new grammar feature gets a row.
    "enwiki-substrate": 'lineage(enwiki,mu=graph,estimand="ancestry")',
    "cowalk-sibling": 'cowalk(enwiki,walk="sibling",estimand="path")',
    "cowalk-weighted-cousin": (
        'cowalk(simplewiki,walk="cousin",weight="idf_node_size",'
        'mu=haiku,estimand="path")'
    ),
    # Parked on PR #4077 for "the next contract change", which is this one:
    "pick-root": "pick(menu(graph,n=10))",
    "scientific-notation-number": "margin(t=1e-05)",
}

KIND_ATOM = "atom"
KIND_APPLY = "apply"

ROLE_ARG = "ARG"
ROLE_KWARG = "KWARG"
ROLE_LIST_ITEM = "LIST_ITEM"
ROLE_MOD = "MOD"
ROLE_PIN = "PIN"
ROLE_LITERAL_BYTE = "LITERAL_BYTE"

#: The element type carried by each registered list-valued kwarg kind.
_LIST_ELEMENT_TYPE = {"number_list": "number", "int_list": "int"}


class ContractError(ValueError):
    """A resolved DTO, role path, or token stream failed closed."""


# --------------------------------------------------------------------------
# role paths
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleStep:
    """One typed edge of a root-to-item path (§3.2)."""

    role: str
    index: int | None = None
    key: str | None = None
    type_name: str | None = None

    def serialize(self) -> str:
        if self.role == ROLE_ARG:
            return f"ARG({self.index},{self.type_name})"
        if self.role == ROLE_KWARG:
            return f"KWARG({self.key},{self.type_name})"
        if self.role == ROLE_LIST_ITEM:
            return f"LIST_ITEM({self.index},{self.type_name})"
        if self.role in (ROLE_MOD, ROLE_PIN, ROLE_LITERAL_BYTE):
            return f"{self.role}({self.index})"
        raise ContractError(f"unknown role: {self.role!r}")


RolePath = tuple[RoleStep, ...]


def serialize_path(path: Sequence[RoleStep]) -> str:
    """The authoritative serialized path.  The root serializes to ``ROOT``."""

    if not path:
        return "ROOT"
    return "/".join(step.serialize() for step in path)


def local_role(path: Sequence[RoleStep]) -> str:
    """The lossy ``(depth, local_role)`` feature of §1 — never an identity."""

    return "ROOT" if not path else path[-1].serialize()


def depth_breadth(path: Sequence[RoleStep]) -> tuple[int, int | None]:
    """The deliberately insufficient coordinate pair, for the aliasing tests."""

    if not path:
        return (0, None)
    return (len(path), path[-1].index)


# --------------------------------------------------------------------------
# resolved canonical AST DTO
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedKwarg:
    key: str
    value_type: str
    value: Any

    def as_record(self) -> dict[str, Any]:
        if isinstance(self.value, ResolvedNode):
            # pec-v3: a kwarg may carry a sub-expression (mu=haiku); it is a
            # node record, never a lexical string, so nothing re-parses it.
            return {
                "key": self.key,
                "value_type": self.value_type,
                "node": self.value.as_record(),
            }
        return {
            "key": self.key,
            "value_type": self.value_type,
            "lexical": _lexical(self.value),
        }


@dataclass(frozen=True)
class ResolvedLiteral:
    """pec-v3: a numeric literal occupying a positional argument slot."""

    value_type: str
    value: Any

    def as_record(self) -> dict[str, Any]:
        return {
            "literal": True,
            "value_type": self.value_type,
            "lexical": _lexical(self.value),
        }


@dataclass(frozen=True)
class ResolvedNode:
    """A node with every registry default and derived field resolved (§3)."""

    name: str
    kind: str
    output: str
    args: tuple["ResolvedNode | ResolvedLiteral", ...]
    kwargs: tuple[ResolvedKwarg, ...]
    mods: tuple[str, ...]
    pins: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "output": self.output,
            "args": [child.as_record() for child in self.args],
            "kwargs": [kw.as_record() for kw in self.kwargs],
            "mods": list(self.mods),
            "pins": list(self.pins),
        }


def _lexical(value: Any) -> str:
    """The canonical lexical form, reusing the canonicalizer's own renderer.

    Reusing ``_render_val`` rather than reimplementing it is deliberate: an
    independent number formatter here could drift from the identity string and
    silently create a second serialization convention (§2).
    """

    return _render_val(value)


_SCALAR_KINDS = ("number", "int", "string", "estimand", "impl",
                 "walk", "weight")


def _value_type(value: Any, declared_kind: str | None) -> str:
    """Resolve a literal's type, preferring the *declared registry kind*.

    §3.1 of the encoder handoff requires derived tokens to agree with the pinned
    registry during both encoding and decoding.  Inferring a scalar's type from
    its Python runtime value violates that: ``margin(t=1)`` is valid, and
    ``margin.t`` is registered ``number``, but the runtime value is an ``int``.
    Inferring would emit ``<INT>`` for a ``number`` field, which a stream-driven
    grader would then hold to exact reconstruction instead of the tolerance
    rule in the generator specification §5.0.1.

    Runtime inference remains only as a fallback for values with no declared
    kind, and it never overrides the registry.
    """

    if isinstance(value, tuple):
        if declared_kind in _LIST_ELEMENT_TYPE:
            return declared_kind
        raise ContractError(f"list value has no registered list kind: {declared_kind!r}")
    if isinstance(value, bool):
        raise ContractError("booleans are not a registered process value type")
    if declared_kind in _SCALAR_KINDS:
        return declared_kind
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    raise ContractError(f"unsupported literal shape: {type(value).__name__}")


def resolve(node: Node) -> ResolvedNode:
    """Derive KIND/OUTPUT and resolve every registry default.

    The resolved kwarg set is exactly the set ``canonical()`` renders, so the
    DTO and the identity string cannot diverge.
    """

    validate(node)
    signature = REGISTRY[node.name]

    defaults = {key: spec.default for key, spec in signature.kwargs.items()}
    merged = dict(defaults)
    merged.update(dict(node.kwargs))
    # The canonicalizer drops resolved kwargs that are still ``None``; the DTO
    # serializes the same set so identity and input stay in agreement.
    resolved_kwargs = []
    for key, value in sorted(merged.items()):
        if value is None:
            continue
        declared = signature.kwargs[key].kind if key in signature.kwargs else None
        if isinstance(value, Node):
            # pec-v3: a node-valued kwarg (mu=haiku) resolves recursively and
            # carries its declared output type, per the registry.
            if declared not in OUTPUT_TYPES and declared != "process":
                raise ContractError(
                    f"{node.name}.{key} carries a node but is not declared "
                    f"an output type: {declared!r}"
                )
            resolved_kwargs.append(
                ResolvedKwarg(key=key, value_type=declared, value=resolve(value))
            )
        else:
            resolved_kwargs.append(
                ResolvedKwarg(
                    key=key, value_type=_value_type(value, declared), value=value
                )
            )
    resolved_kwargs = tuple(resolved_kwargs)

    called = bool(node.args or resolved_kwargs) or (
        signature.operator and not signature.atom
    )
    kind = KIND_APPLY if called else KIND_ATOM
    # Atom/operator dual roles are distinguished by KIND and validated against
    # the signature (§3.1) — a derived token never overrides the registry.
    if kind == KIND_ATOM and not signature.atom:
        raise ContractError(f"{node.name} is not registered as an atom")
    if kind == KIND_APPLY and not signature.operator:
        raise ContractError(f"{node.name} is not registered as an operator")

    resolved_args = []
    for index, child in enumerate(node.args):
        if isinstance(child, Node):
            resolved_args.append(resolve(child))
        else:
            resolved_args.append(
                ResolvedLiteral(
                    value_type=_literal_arg_kind(signature, index, child), value=child
                )
            )

    resolved = ResolvedNode(
        name=node.name,
        kind=kind,
        output=signature.output,
        args=tuple(resolved_args),
        kwargs=resolved_kwargs,
        mods=tuple(node.mods),
        pins=tuple(node.pins),
    )
    return resolved


def resolve_expression(expression: str) -> ResolvedNode:
    return resolve(parse(expression))


def _expected_arg_type(signature, index: int) -> str:
    if index < len(signature.arg_types):
        return signature.arg_types[index]
    if signature.variadic_arg_type is not None:
        return signature.variadic_arg_type
    raise ContractError(f"positional argument {index} has no registered type")


def _literal_arg_kind(signature, index: int, value: Any) -> str:
    """The declared value kind a positional literal resolves under.

    A declared type may be a union (``number|score``); the literal takes the
    single value-kind alternative — declared-type-wins, never runtime
    inference alone (§3.1).
    """

    declared = _expected_arg_type(signature, index)
    kinds = [alt for alt in declared.split("|") if alt in VALUE_KINDS]
    if len(kinds) != 1:
        raise ContractError(
            f"positional literal at {index} needs exactly one declared value "
            f"kind, got {declared!r}"
        )
    return _value_type(value, kinds[0])


# --------------------------------------------------------------------------
# reference token stream
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """A structural token plus its complete typed role path."""

    token: str
    path: RolePath

    def as_record(self) -> dict[str, Any]:
        return {"token": self.token, "path": serialize_path(self.path)}

    def as_line(self) -> str:
        """One reviewable line per token: ``<TOKEN>\\tROLE/PATH``."""

        return f"{self.token}\t{serialize_path(self.path)}"


def _literal_tokens(payload: bytes, path: RolePath) -> Iterator[Token]:
    """Literal payloads expand to bytes under ``LITERAL_BYTE(index)`` roles."""

    for index, byte in enumerate(payload):
        step = RoleStep(ROLE_LITERAL_BYTE, index=index)
        yield Token(f"BYTE:0x{byte:02x}", path + (step,))


def _value_tokens(value: Any, value_type: str, path: RolePath) -> Iterator[Token]:
    if isinstance(value, tuple):
        element_type = _LIST_ELEMENT_TYPE[value_type]
        yield Token("<LIST>", path)
        for index, item in enumerate(value):
            step = RoleStep(ROLE_LIST_ITEM, index=index, type_name=element_type)
            yield Token(f"<ITEM:{index}>", path)
            yield from _value_tokens(item, element_type, path + (step,))
            yield Token("</ITEM>", path)
        yield Token("</LIST>", path)
        return

    if value_type == "int":
        open_tag, close_tag = "<INT>", "</INT>"
        payload = _lexical(value).encode("utf-8")
    elif value_type == "number":
        open_tag, close_tag = "<NUMBER>", "</NUMBER>"
        payload = _lexical(value).encode("utf-8")
    elif value_type == "string":
        open_tag, close_tag = "<STRING>", "</STRING>"
        # Exact UTF-8 bytes of the string itself, not its JSON quoting.
        payload = value.encode("utf-8")
    elif value_type in ("estimand", "impl", "walk", "weight"):
        # pec-v3 (estimand/impl) and pec-v4 (walk/weight): enumerated kinds
        # keep their declared kind on the stream, so a decoder can hold them
        # to the registry enumeration.
        open_tag = f"<{value_type.upper()}>"
        close_tag = f"</{value_type.upper()}>"
        payload = value.encode("utf-8")
    else:
        raise ContractError(f"unsupported scalar value type: {value_type!r}")

    yield Token(open_tag, path)
    yield from _literal_tokens(payload, path)
    yield Token(close_tag, path)


def _node_tokens(node: ResolvedNode, path: RolePath) -> Iterator[Token]:
    signature = REGISTRY[node.name]
    yield Token("<NODE>", path)
    yield Token(f"<KIND:{node.kind}>", path)
    yield Token(f"<NAME:{node.name}>", path)
    yield Token(f"<OUTPUT:{node.output}>", path)

    yield Token("<ARGS>", path)
    for index, child in enumerate(node.args):
        step = RoleStep(
            ROLE_ARG, index=index, type_name=_expected_arg_type(signature, index)
        )
        yield Token(f"<ARG:{index}>", path)
        if isinstance(child, ResolvedLiteral):
            # pec-v3: a positional literal is value tokens inside the ARG
            # fence — no <NODE> wrapper, because there is no node.
            yield from _value_tokens(child.value, child.value_type, path + (step,))
        else:
            yield from _node_tokens(child, path + (step,))
        yield Token("</ARG>", path)
    yield Token("</ARGS>", path)

    yield Token("<KWARGS>", path)
    for kwarg in node.kwargs:
        step = RoleStep(ROLE_KWARG, key=kwarg.key, type_name=kwarg.value_type)
        yield Token(f"<KW:{kwarg.key}>", path)
        if isinstance(kwarg.value, ResolvedNode):
            # pec-v3: a node-valued kwarg (mu=haiku) is a whole node inside
            # the KW fence.
            yield from _node_tokens(kwarg.value, path + (step,))
        else:
            yield from _value_tokens(kwarg.value, kwarg.value_type, path + (step,))
        yield Token("</KW>", path)
    yield Token("</KWARGS>", path)

    yield Token("<MODS>", path)
    for index, modifier in enumerate(node.mods):
        step = RoleStep(ROLE_MOD, index=index)
        yield Token(f"<MOD:{index}>", path)
        yield from _literal_tokens(modifier.encode("ascii"), path + (step,))
        yield Token("</MOD>", path)
    yield Token("</MODS>", path)

    yield Token("<PINS>", path)
    for index, pin in enumerate(node.pins):
        step = RoleStep(ROLE_PIN, index=index)
        yield Token(f"<PIN:{index}>", path)
        yield from _literal_tokens(pin.encode("ascii"), path + (step,))
        yield Token("</PIN>", path)
    yield Token("</PINS>", path)

    yield Token("</NODE>", path)


def token_stream(node: ResolvedNode) -> tuple[Token, ...]:
    """The reference structural stream of §3.1, with complete role paths."""

    if not isinstance(node, ResolvedNode):
        raise ContractError("token_stream requires a resolved DTO, not a raw Node")
    tokens = [Token("<BOS>", ())]
    tokens.extend(_node_tokens(node, ()))
    tokens.append(Token("<EOS>", ()))
    return tuple(tokens)


def role_paths(tokens: Sequence[Token]) -> tuple[str, ...]:
    """Every distinct serialized role path in the stream, in first-seen order."""

    seen: dict[str, None] = {}
    for token in tokens:
        seen.setdefault(serialize_path(token.path), None)
    return tuple(seen)


# --------------------------------------------------------------------------
# golden vectors
# --------------------------------------------------------------------------


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


def golden_row(name: str, expression: str) -> dict[str, Any]:
    node = parse(expression)
    resolved = resolve(node)
    tokens = token_stream(resolved)
    return {
        "name": name,
        "expression": expression,
        # R9 split: identity is the pin-free semantic form; the full form is
        # provenance and the round-trip target. They differ only on pinned rows
        # — which is exactly what test obligation 10 checks.
        "canonical_identity_string": canonical_semantic(node),
        "canonical_full_string": canonical_full(node),
        "full_process_digest": full_ast_digest(node),
        "resolved_ast": resolved.as_record(),
        "token_count": len(tokens),
        "tokens": [token.as_line() for token in tokens],
        "role_paths": list(role_paths(tokens)),
    }


def build_golden(cases: Mapping[str, str]) -> dict[str, Any]:
    rows = [golden_row(name, expression) for name, expression in sorted(cases.items())]
    document = {
        "schema": GOLDEN_SCHEMA,
        "registry_version": REGISTRY_VERSION,
        "contract_version": CONTRACT_VERSION,
        "note": (
            "Frozen structural fixtures for the step-2 tokenizer. Token names are "
            "structural strings, not vocabulary IDs; step 2 assigns versioned IDs "
            "and the explicit 256-byte literal fallback while reproducing this "
            "structure and these role paths exactly."
        ),
        "rows": rows,
    }
    document["golden_sha256"] = hashlib.sha256(
        _canonical_json_bytes({k: v for k, v in document.items()})
    ).hexdigest()
    return document


def verify_golden(document: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every frozen row from the expression alone."""

    if not isinstance(document, Mapping) or document.get("schema") != GOLDEN_SCHEMA:
        raise ContractError("unsupported golden fixture schema")
    if document.get("registry_version") != REGISTRY_VERSION:
        raise ContractError(
            "golden fixtures were frozen under a different registry version; "
            "regenerating them is a contract change, not a refresh"
        )
    if document.get("contract_version") != CONTRACT_VERSION:
        raise ContractError("golden fixtures were frozen under a different contract version")
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ContractError("golden fixtures contain no rows")

    core = {key: value for key, value in document.items() if key != "golden_sha256"}
    if hashlib.sha256(_canonical_json_bytes(core)).hexdigest() != document.get(
        "golden_sha256"
    ):
        raise ContractError("golden_sha256 does not bind the fixture content")

    for row in rows:
        rebuilt = golden_row(row["name"], row["expression"])
        if rebuilt != row:
            raise ContractError(f"golden row drifted: {row.get('name')!r}")
    return dict(document)


def load_golden(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "rb") as handle:
        document = json.loads(handle.read().decode("utf-8"))
    return verify_golden(document)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from process_cards import PROCESSES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="golden fixture path to write")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME=EXPRESSION",
        help="additional case beyond the canonical set; may not override a name",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing bundle (a sealed bundle must never be rewritten)",
    )
    args = parser.parse_args(argv)

    # The committed bundle is reproducible from this module alone.
    cases = dict(PROCESSES)
    for name, expression in REQUIRED_COVERAGE_CASES.items():
        if name in cases:
            raise SystemExit(f"coverage case collides with a registered process: {name}")
        cases[name] = expression
    for item in args.extra:
        name, _, expression = item.partition("=")
        if not name or not expression:
            raise SystemExit("--extra expects NAME=EXPRESSION")
        if name in cases:
            # Silent replacement would let a bundle drift from its canonical
            # source while still looking self-consistent.
            raise SystemExit(f"--extra may not override an existing case: {name}")
        cases[name] = expression

    target = Path(args.out)
    if target.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite a sealed bundle: {target}. A contract change "
            "seals a NEW bundle; it never rewrites an existing one."
        )

    document = build_golden(cases)
    verify_golden(document)
    with open(target, "wb") as handle:
        handle.write(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True).encode(
                "utf-8"
            )
            + b"\n"
        )
    print(f"{len(document['rows'])} golden rows -> {args.out}")
    print(f"golden_sha256 {document['golden_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
