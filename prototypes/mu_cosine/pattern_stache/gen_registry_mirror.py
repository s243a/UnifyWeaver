#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (@s243a)
#
# gen_registry_mirror.py - generate the closed-constraint Prolog mirror
# from the sealed v0.4 registry surface (ruling 5(b) of
# DESIGN_prolog_elaborator.md, decided in PR #4093).
#
# Reads REGISTRY from the SEALED process_cards.py (the v0.4 grammar
# authority — NOT process_expression_vnext, which stays untouched per
# the standing rule) and emits pe_registry_mirror.pl: the one generated
# mirror that pe_emit, pe_where, and pe_elaborate consume, replacing
# the hand-maintained copy that lived in pe_emit.pl (the consumer-3
# liberty, retired by this generator).
#
# The sha256 of process_cards.py's bytes is embedded at generation and
# re-checked when the mirror LOADS, failing closed on mismatch — so a
# drifted mirror refuses to load rather than silently serving stale
# registry facts (PROJECT_PHILOSOPHY §4: the decision is in the
# artifact).
#
# Usage (from prototypes/mu_cosine/pattern_stache/):
#   python3 gen_registry_mirror.py          # writes pe_registry_mirror.pl
#
# Deterministic: facts are emitted in sorted order, so regeneration
# without a registry change is a no-op diff.

import hashlib
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
MU_COSINE = HERE.parent
sys.path.insert(0, str(MU_COSINE))

import process_cards  # the sealed v0.4 registry surface

SOURCE = MU_COSINE / "process_cards.py"
OUT = HERE / "pe_registry_mirror.pl"

ATOM_RE = re.compile(r"[a-z][a-zA-Z0-9_]*$")


def q(name):
    """Quote a Prolog atom when needed (hyphens, uppercase, dots)."""
    s = str(name)
    if ATOM_RE.fullmatch(s):
        return s
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def pl_value(v):
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, int):
        return str(v)
    return q(v)


def main():
    source_bytes = SOURCE.read_bytes()
    sha = hashlib.sha256(source_bytes).hexdigest()
    reg = process_cards.REGISTRY
    version = process_cards.REGISTRY_VERSION

    atoms, operators, variadics, outputs, modifiers = [], [], [], [], []
    kwspecs, required = [], []

    for name in sorted(reg):
        sig = reg[name]
        outputs.append((name, sig.output))
        if sig.atom:
            atoms.append(name)
        if sig.operator:
            operators.append(name)
            if sig.max_args is None:
                variadics.append(name)
        for mod in sorted(sig.modifiers):
            modifiers.append((name, mod))
        for kw in sorted(sig.kwargs):
            spec = sig.kwargs[kw]
            default = "none" if spec.default is None else f"default({pl_value(spec.default)})"
            kwspecs.append((name, kw, spec.kind, default))
            if spec.required:
                required.append((name, kw))

    lines = []
    a = lines.append
    a(":- encoding(utf8).")
    a("% SPDX-License-Identifier: MIT OR Apache-2.0")
    a("%")
    a("% pe_registry_mirror.pl - GENERATED closed-constraint mirror of the")
    a(f"% sealed {version} registry surface (process_cards.py REGISTRY).")
    a("%")
    a("% DO NOT EDIT: regenerate with gen_registry_mirror.py (ruling 5(b),")
    a("% DESIGN_prolog_elaborator.md / PR #4093).  The embedded sha256 of")
    a("% process_cards.py is re-checked at load; a mismatch means the sealed")
    a("% registry changed since generation, and the mirror REFUSES TO LOAD")
    a("% until regenerated (fail closed - no stale registry facts).")
    a("")
    a(":- module(pe_registry_mirror, [")
    a("    pe_registry_version/1,")
    a("    pe_atom/1,")
    a("    pe_operator/1,")
    a("    pe_variadic/1,")
    a("    pe_output/2,")
    a("    pe_modifier/2,")
    a("    pe_kwspec/4,")
    a("    pe_required/2")
    a("]).")
    a("")
    a(":- use_module(library(sha)).")
    a("")
    a(f"pe_registry_version('{version}').")
    a("")
    a(f"pe_mirror_source_sha256('{sha}').")
    a("")
    a("% pe_atom(Name): forms that may appear bare.")
    for n in atoms:
        a(f"pe_atom({q(n)}).")
    a("")
    a("% pe_operator(Name): anything that may be applied to arguments.")
    for n in operators:
        a(f"pe_operator({q(n)}).")
    a("")
    a("% pe_variadic(Name): open-ended positional arity.")
    for n in variadics:
        a(f"pe_variadic({q(n)}).")
    a("")
    a("% pe_output(Name, OutputType).")
    for n, o in outputs:
        a(f"pe_output({q(n)}, {q(o)}).")
    a("")
    a("% pe_modifier(Name, Modifier).")
    for n, m in modifiers:
        a(f"pe_modifier({q(n)}, {q(m)}).")
    a("")
    a("% pe_kwspec(Op, Kw, Kind, Default).  Kind is the registry's declared")
    a("% kind verbatim (value kinds: number/int/string/number_list/int_list/")
    a("% estimand/impl; an output-type kind such as judge marks a node-valued")
    a("% kwarg).  Default is none or default(Value).")
    for n, k, kind, d in kwspecs:
        a(f"pe_kwspec({q(n)}, {q(k)}, {q(kind)}, {d}).")
    a("")
    a("% pe_required(Op, Kw): kwargs the registry marks required.")
    for n, k in required:
        a(f"pe_required({q(n)}, {q(k)}).")
    a("")
    a("% Load-time drift check: fail closed if the sealed source moved.")
    a("pe_mirror_verify :-")
    a("    module_property(pe_registry_mirror, file(Here)),")
    a("    file_directory_name(Here, Dir),")
    a("    atomic_list_concat([Dir, '/../process_cards.py'], Src),")
    a("    (   exists_file(Src)")
    a("    ->  true")
    a("    ;   throw(error(pe_registry_mirror(source_missing(Src)), _))")
    a("    ),")
    a("    read_file_to_string(Src, S, [encoding(octet)]),")
    a("    sha_hash(S, H, [algorithm(sha256), encoding(octet)]),")
    a("    hash_atom(H, Actual),")
    a("    pe_mirror_source_sha256(Expected),")
    a("    (   Actual == Expected")
    a("    ->  true")
    a("    ;   throw(error(pe_registry_mirror(source_drift(expected(Expected), actual(Actual))), _))")
    a("    ).")
    a("")
    a(":- initialization(pe_mirror_verify).")
    a("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(atoms)} atoms, {len(operators)} operators, "
          f"{len(kwspecs)} kwspecs, source sha256 {sha[:16]}...")


if __name__ == "__main__":
    main()
