// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// resolver.mjs -- EDGE of the JS-WAM compiled uw-resolve P0.
//
// WHAT IS IN HERE, exhaustively: conversion between JSON catalogs/requests
// and WAM terms, plus driving Runtime.run / lowered_dispatch. There is NO
// resolver logic — no candidate order, no constraint arithmetic, no
// layer walk. Those live in js/generated_program.js (compiler output from
// examples/pkg_resolver/resolver.pl).

import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const generated = require(join(dirname(fileURLToPath(import.meta.url)), "js", "generated_program.js"));
const { Runtime, V } = generated;
const program = generated.program || generated.M && generated.M.program;

function internAtom(name) {
  return V.Atom(Runtime.intern(program.intern_table, String(name)));
}

function functorId(name) {
  return Runtime.intern(program.intern_table, String(name));
}

function functorName(fid) {
  return Runtime.functor_name(program.intern_table, fid);
}

function isCons(term) {
  if (!term || term.tag !== "struct" || (term.args || []).length !== 2) return false;
  const n = functorName(term.fid);
  return n === "[|]" || n === "." || n === "./2" || n === "[|]/2";
}

function jsListToTerm(arr, mapItem) {
  let cur = internAtom("[]");
  const fid = functorId("[|]");
  for (let i = arr.length - 1; i >= 0; i--) {
    cur = V.Struct(fid, [mapItem(arr[i]), cur]);
  }
  return cur;
}

function vTerm(triple) {
  return V.Struct(functorId("v"), [
    V.Int(triple[0] | 0),
    V.Int(triple[1] | 0),
    V.Int(triple[2] | 0)
  ]);
}

function constraintTerm(c) {
  if (c === "any" || c == null) return internAtom("any");
  if (typeof c === "object" && c.op === "eq") return V.Struct(functorId("eq"), [vTerm(c.v)]);
  if (typeof c === "object" && c.op === "gte") return V.Struct(functorId("gte"), [vTerm(c.v)]);
  if (typeof c === "object" && c.op === "lt") return V.Struct(functorId("lt"), [vTerm(c.v)]);
  if (typeof c === "object" && c.op === "range") {
    return V.Struct(functorId("range"), [vTerm(c.lo), vTerm(c.hi)]);
  }
  throw new Error("resolver shim: unknown constraint " + JSON.stringify(c));
}

function pairTerm(name, ver) {
  return V.Struct(functorId("-"), [internAtom(name), vTerm(ver)]);
}

function pkgTerm(row) {
  return V.Struct(functorId("package"), [internAtom(row[0]), vTerm(row[1])]);
}

function depTerm(row) {
  return V.Struct(functorId("depends"), [
    internAtom(row[0]), vTerm(row[1]), internAtom(row[2]), constraintTerm(row[3])
  ]);
}

function confTerm(row) {
  return V.Struct(functorId("conflicts"), [
    internAtom(row[0]), vTerm(row[1]), internAtom(row[2])
  ]);
}

function requestTerm(req) {
  if (req && typeof req === "object" && req.req) {
    return V.Struct(functorId("req"), [internAtom(req.req), constraintTerm(req.constraint)]);
  }
  return internAtom(req);
}

export function catalogToTerm(cat) {
  const c = cat || {};
  return V.Struct(functorId("catalog"), [
    jsListToTerm(c.packages || [], pkgTerm),
    jsListToTerm(c.depends || [], depTerm),
    jsListToTerm(c.conflicts || [], confTerm),
    jsListToTerm(c.base || [], (p) => pairTerm(p[0], p[1])),
    jsListToTerm(c.installed || [], (p) => pairTerm(p[0], p[1])),
    jsListToTerm(c.requested || [], internAtom)
  ]);
}

function termToJs(state, term0) {
  const term = Runtime.deref(state, term0);
  if (!term || typeof term !== "object") return term;
  if (term.tag === "int" || term.tag === "float") return term.val;
  if (term.tag === "string") return String(term.val);
  if (term.tag === "atom") {
    const n = Runtime.string_of(program.intern_table, term.id);
    if (n === "[]") return [];
    if (n === "any") return "any";
    if (n === "true") return true;
    if (n === "false") return false;
    return n;
  }
  if (term.tag === "unbound") return null;
  if (isCons(term)) {
    const out = [];
    let cur = term;
    const nilId = Runtime.intern(program.intern_table, "[]");
    while (true) {
      cur = Runtime.deref(state, cur);
      if (cur.tag === "atom" && cur.id === nilId) return out;
      if (!isCons(cur)) throw new Error("resolver shim: expected a list");
      out.push(termToJs(state, cur.args[0]));
      cur = cur.args[1];
    }
  }
  if (term.tag === "struct") {
    const n = functorName(term.fid).replace(/\/\d+$/, "");
    const args = (term.args || []).map((a) => termToJs(state, a));
    if (n === "v" && args.length === 3) return args;
    if (n === "-" && args.length === 2) return [args[0], args[1]];
    if (n === "blocked" && args.length === 3) {
      // blocked(Name, needs(C), base_has(V))
      const needs = args[1] && args[1][0] === "needs" ? args[1][1] : args[1];
      const bh = args[2] && args[2][0] === "base_has" ? args[2][1] : args[2];
      return { name: args[0], needs: needs, base_has: bh };
    }
    if (n === "needs" || n === "base_has" || n === "eq" || n === "gte" || n === "lt") {
      return [n, args[0]];
    }
    if (n === "range") return { op: "range", lo: args[0], hi: args[1] };
    return [n, ...args];
  }
  throw new Error("resolver shim: unhandled term " + JSON.stringify(term));
}

function normalizeConstraint(c) {
  if (c === "any") return "any";
  if (Array.isArray(c) && (c[0] === "gte" || c[0] === "eq" || c[0] === "lt")) {
    return { op: c[0], v: c[1] };
  }
  if (c && typeof c === "object" && c.op) return c;
  if (Array.isArray(c) && c[0] === "range") return { op: "range", lo: c[1], hi: c[2] };
  return c;
}

function normalizeBlocked(b) {
  if (b && typeof b === "object" && b.name) {
    // Key order matches SWI json_write_dict (alpha) so corpus stringify compares.
    return {
      base_has: b.base_has,
      name: b.name,
      needs: normalizeConstraint(b.needs)
    };
  }
  return b;
}

function runPred(predArity, argTerms) {
  const state = Runtime.new_state();
  state.program = program;
  const slash = String(predArity).lastIndexOf("/");
  const arity = slash >= 0 ? Number(predArity.slice(slash + 1)) : argTerms.length;
  const saved = [];
  // Hold the result cells themselves: lowered T4 list-walks (map_requests/2)
  // overwrite A-registers with the recursive tail, but unification still
  // binds this unbound. Deref after run — same trick as cliArgs.mjs.
  for (let i = 0; i < arity; i++) {
    const t = i < argTerms.length && argTerms[i] !== undefined
      ? argTerms[i]
      : Runtime.new_var(state);
    Runtime.put_reg(state, i + 1, t);
    saved.push(t);
  }
  const lowered = program.lowered_dispatch && program.lowered_dispatch[predArity];
  const startPc = program.labels && program.labels[predArity];
  let ok;
  if (typeof lowered === "function") {
    ok = lowered(program, state) === true;
  } else if (startPc !== undefined) {
    state.pc = startPc;
    ok = Runtime.run(program, state) === true;
  } else {
    throw new Error("unknown predicate: " + predArity);
  }
  return { ok: ok === true, state, saved };
}

function readSaved(state, saved, n) {
  return termToJs(state, saved[n - 1]);
}

export function resolve(catalog, requests) {
  const cat = catalogToTerm(catalog);
  const reqs = jsListToTerm(requests || [], requestTerm);
  const { ok, state, saved } = runPred("resolve/3", [cat, reqs, undefined]);
  if (!ok) return { fail: true };
  return { ok: readSaved(state, saved, 3) };
}

export function resolveLayered(catalog, requests) {
  const cat = catalogToTerm(catalog);
  const reqs = jsListToTerm(requests || [], requestTerm);
  const { ok, state, saved } = runPred("resolve_layered/3", [cat, reqs, undefined]);
  if (!ok) return { fail: true };
  return { ok: readSaved(state, saved, 3) };
}

export function explainBlocked(catalog, request) {
  const cat = catalogToTerm(catalog);
  const req = requestTerm(request);
  const { ok, state, saved } = runPred("explain_blocked_list/3", [cat, req, undefined]);
  if (!ok) return { fail: true };
  const list = readSaved(state, saved, 3) || [];
  return { ok: list.map(normalizeBlocked) };
}

export function layerClosure(catalog, request) {
  const cat = catalogToTerm(catalog);
  const req = requestTerm(request);
  const { ok, state, saved } = runPred("layer_closure/3", [cat, req, undefined]);
  if (!ok) return { fail: true };
  return { ok: readSaved(state, saved, 3) };
}

export function removalOrphans(catalog, pkg) {
  const cat = catalogToTerm(catalog);
  const { ok, state, saved } = runPred("removal_orphans/3", [cat, internAtom(pkg), undefined]);
  if (!ok) return { fail: true };
  return { ok: readSaved(state, saved, 3) };
}

export function runCase(row) {
  const cat = row.catalog;
  const q = row.query;
  const args = row.args;
  if (q === "resolve") return resolve(cat, args);
  if (q === "resolve_layered") return resolveLayered(cat, args);
  if (q === "explain_blocked") return explainBlocked(cat, args);
  if (q === "layer_closure") return layerClosure(cat, args);
  if (q === "removal_orphans") return removalOrphans(cat, args);
  throw new Error("unknown query " + q);
}
