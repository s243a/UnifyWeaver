// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// resolver_store.mjs -- EDGE of the JS-WAM compiled uw-resolve P2 store adapter.
//
// WHAT IS IN HERE, exhaustively: conversion between JSON env/requests
// and WAM terms, plus driving Runtime.run / lowered_dispatch. There is NO
// resolver logic. Catalog facts come from D43 indexed stores compiled in.
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

function vTerm(ver) {
  return verTerm(ver);
}

function verTerm(ver) {
  if (ver && typeof ver === "object" && !Array.isArray(ver) && ver.deb) {
    const d = ver.deb;
    return V.Struct(functorId("deb"), [
      V.Int(d[0] | 0),
      segsTerm(d[1] || []),
      segsTerm(d[2] || [])
    ]);
  }
  return V.Struct(functorId("v"), [
    V.Int(ver[0] | 0),
    V.Int(ver[1] | 0),
    V.Int(ver[2] | 0)
  ]);
}

function segsTerm(segs) {
  return jsListToTerm(segs, (seg) => {
    const order = String(seg[0] || "");
    const codes = [];
    for (let i = 0; i < order.length; i++) codes.push(order.charCodeAt(i));
    return V.Struct(functorId("s"), [jsListToTerm(codes, (c) => V.Int(c)), V.Int(seg[1] | 0)]);
  });
}

function constraintTerm(c) {
  if (c === "any" || c == null) return internAtom("any");
  if (typeof c === "object" && (c.op === "eq" || c.op === "gte" || c.op === "lt" ||
      c.op === "lte" || c.op === "gt")) {
    return V.Struct(functorId(c.op), [verTerm(c.v)]);
  }
  if (typeof c === "object" && c.op === "range") {
    return V.Struct(functorId("range"), [verTerm(c.lo), verTerm(c.hi)]);
  }
  throw new Error("resolver shim: unknown constraint " + JSON.stringify(c));
}

function pairTerm(name, ver) {
  return V.Struct(functorId("-"), [internAtom(name), vTerm(ver)]);
}

function holdTerm(row) {
  if (!row) return internAtom("[]");
  if (row.length >= 3) {
    return V.Struct(functorId("base"), [pairTerm(row[0], row[1]), internAtom(row[2])]);
  }
  return pairTerm(row[0], row[1]);
}

function layerTerm(row) {
  const name = (row && row.name) || row[0];
  const pkgs = (row && row.packages) || row[1] || [];
  return V.Struct(functorId("layer"), [internAtom(name), jsListToTerm(pkgs, holdTerm)]);
}

function aliasTerm(row) {
  return V.Struct(functorId("alias"), [internAtom(row[0]), internAtom(row[1])]);
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

export function envToTerm(env0) {
  const e = env0 || {};
  const catId = e.catalog_id || e.catalogId || "default";
  return V.Struct(functorId("env"), [
    internAtom(catId),
    jsListToTerm(e.base || [], holdTerm),
    jsListToTerm(e.installed || [], (p) => pairTerm(p[0], p[1])),
    jsListToTerm(e.requested || [], internAtom),
    jsListToTerm(e.layers || [], layerTerm),
    jsListToTerm(e.excluded || [], internAtom),
    jsListToTerm(e.aliases || [], aliasTerm)
  ]);
}

function envOf(row) {
  if (row.env) {
    const e = { ...row.env };
    if (row.catalog_id && !e.catalog_id) e.catalog_id = row.catalog_id;
    return e;
  }
  const c = row.catalog || {};
  return {
    catalog_id: row.catalog_id || "default",
    base: c.base || [],
    installed: c.installed || [],
    requested: c.requested || [],
    layers: c.layers || [],
    excluded: c.excluded || [],
    aliases: c.aliases || []
  };
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
    if (n === "s" && args.length === 2) {
      const order = Array.isArray(args[0])
        ? String.fromCharCode.apply(null, args[0])
        : "";
      return [order, args[1]];
    }
    if (n === "deb" && args.length === 3) return { deb: args };
    if (n === "-" && args.length === 2) return [args[0], args[1]];
    if (n === "blocked" && args.length === 3) {
      const needs = args[1] && args[1][0] === "needs" ? args[1][1] : args[1];
      const third = args[2];
      if (third && third[0] === "providers") {
        return { name: args[0], needs: needs, providers: third[1] };
      }
      const bh = third && third[0] === "base_has" ? third[1] : third;
      return { name: args[0], needs: needs, base_has: bh };
    }
    if (n === "blocked" && args.length === 1 && args[0] && args[0][0] === "alternatives") {
      return { alternatives: args[0][1] };
    }
    if (n === "alt" && args.length === 2) return { dep: args[0], reason: args[1] };
    if (n === "safe" && args.length === 1) {
      const cost = Array.isArray(args[0]) && args[0][0] === "cost" ? args[0][1] : args[0];
      return { cost: cost, verdict: "safe" };
    }
    if (n === "coordinated" && args.length === 1) {
      return { set: args[0], verdict: "coordinated" };
    }
    if (n === "unsafe" && args.length === 1) {
      return { reason: args[0], verdict: "unsafe" };
    }
    if (n === "audit" && args.length === 2) {
      return normalizeAuditTerm(args[0], args[1]);
    }
    if (n === "ok" && args.length === 1) return { __ok_set: args[0] };
    if (n === "needs" || n === "base_has" || n === "eq" || n === "gte" || n === "lt"
        || n === "lte" || n === "gt"
        || n === "cost" || n === "held" || n === "suggest" || n === "providers"
        || n === "alternatives") {
      return [n, args[0]];
    }
    if (n === "range") return { op: "range", lo: args[0], hi: args[1] };
    return [n, ...args];
  }
  throw new Error("resolver shim: unhandled term " + JSON.stringify(term));
}

function normalizeConstraint(c) {
  if (c === "any") return "any";
  if (Array.isArray(c) && (c[0] === "gte" || c[0] === "eq" || c[0] === "lt"
      || c[0] === "lte" || c[0] === "gt")) {
    return { op: c[0], v: c[1] };
  }
  if (c && typeof c === "object" && c.op) return c;
  if (Array.isArray(c) && c[0] === "range") return { op: "range", lo: c[1], hi: c[2] };
  return c;
}

function normalizeAuditTerm(name, payload) {
  if (payload === "over_frozen") return { kind: "over_frozen", name: name };
  if (Array.isArray(payload) && payload[0] === "suggest") {
    return { kind: "suggest", name: name, reason: payload[1] };
  }
  if (Array.isArray(payload) && payload[0] === "held") {
    return { kind: "held", name: name, reason: payload[1] };
  }
  if (payload && typeof payload === "object" && payload.kind) return payload;
  return { kind: "held", name: name, reason: payload };
}

function normalizeVerdict(v) {
  if (v === "no_candidate") return { verdict: "no_candidate" };
  if (v && typeof v === "object" && v.verdict) return v;
  return v;
}

function normalizeUpgrade(r) {
  if (r === "no_candidate") return { fail: true };
  if (r && typeof r === "object" && r.__ok_set) return { ok: r.__ok_set };
  if (Array.isArray(r)) return { ok: r };
  if (r && typeof r === "object" && r.name && r.base_has !== undefined) {
    return { ok: { blocked: normalizeBlocked(r) } };
  }
  return r;
}

function normalizeBlocked(b) {
  if (b && typeof b === "object" && b.alternatives) {
    return { alternatives: b.alternatives };
  }
  if (b && typeof b === "object" && b.providers) {
    return {
      name: b.name,
      needs: normalizeConstraint(b.needs),
      providers: (b.providers || []).map(normalizeBlocked)
    };
  }
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

export function resolve(env, requests) {
  const e = envToTerm(env);
  const reqs = jsListToTerm(requests || [], requestTerm);
  const { ok, state, saved } = runPred("resolve_store/3", [e, reqs, undefined]);
  if (!ok) return { fail: true };
  return { ok: readSaved(state, saved, 3) };
}

export function resolveLayered(env, requests) {
  const e = envToTerm(env);
  const reqs = jsListToTerm(requests || [], requestTerm);
  const { ok, state, saved } = runPred("resolve_layered_store/3", [e, reqs, undefined]);
  if (!ok) return { fail: true };
  return { ok: readSaved(state, saved, 3) };
}

export function explainBlocked(env, request) {
  const e = envToTerm(env);
  const req = requestTerm(request);
  const { ok, state, saved } = runPred("explain_blocked_list_store/3", [e, req, undefined]);
  if (!ok) return { fail: true };
  const list = readSaved(state, saved, 3) || [];
  return { ok: list.map(normalizeBlocked) };
}

export function layerClosure(env, request) {
  const e = envToTerm(env);
  const req = requestTerm(request);
  const { ok, state, saved } = runPred("layer_closure_store/3", [e, req, undefined]);
  if (!ok) return { fail: true };
  return { ok: readSaved(state, saved, 3) };
}

export function removalOrphans(env, pkg) {
  const e = envToTerm(env);
  const { ok, state, saved } = runPred("removal_orphans_store/3", [e, internAtom(pkg), undefined]);
  if (!ok) return { fail: true };
  return { ok: readSaved(state, saved, 3) };
}

export function safeUpgrade(env, pkg, ver) {
  const e = envToTerm(env);
  const { ok, state, saved } = runPred("safe_upgrade_store/4", [e, internAtom(pkg), vTerm(ver), undefined]);
  if (!ok) return { fail: true };
  return { ok: normalizeVerdict(readSaved(state, saved, 4)) };
}

export function upgradeSet(env, pkg, ver) {
  const e = envToTerm(env);
  const { ok, state, saved } = runPred("upgrade_set_result_store/4", [e, internAtom(pkg), vTerm(ver), undefined]);
  if (!ok) return { fail: true };
  return normalizeUpgrade(readSaved(state, saved, 4));
}

export function freezeAudit(env) {
  const e = envToTerm(env);
  const { ok, state, saved } = runPred("freeze_audit_store/2", [e, undefined]);
  if (!ok) return { fail: true };
  return { ok: readSaved(state, saved, 2) || [] };
}

export function dependents(env, pkg) {
  const e = envToTerm(env);
  const { ok, state, saved } = runPred("dependents_store/3", [e, internAtom(pkg), undefined]);
  if (!ok) return { fail: true };
  return { ok: readSaved(state, saved, 3) };
}

export function dependentsInstalled(env, pkg) {
  const e = envToTerm(env);
  const { ok, state, saved } = runPred("dependents_installed_store/3", [e, internAtom(pkg), undefined]);
  if (!ok) return { fail: true };
  return { ok: readSaved(state, saved, 3) };
}

function pkgVerArgs(args) {
  return { pkg: args[0], ver: args[1] };
}

export function runCase(row) {
  const env = envOf(row);
  const q = row.query;
  const args = row.args;
  if (q === "resolve") return resolve(env, args);
  if (q === "resolve_layered") return resolveLayered(env, args);
  if (q === "explain_blocked") return explainBlocked(env, args);
  if (q === "layer_closure") return layerClosure(env, args);
  if (q === "removal_orphans") return removalOrphans(env, args);
  if (q === "safe_upgrade") {
    const a = pkgVerArgs(args);
    return safeUpgrade(env, a.pkg, a.ver);
  }
  if (q === "upgrade_set") {
    const a = pkgVerArgs(args);
    return upgradeSet(env, a.pkg, a.ver);
  }
  if (q === "freeze_audit") return freezeAudit(env);
  if (q === "dependents") return dependents(env, args);
  if (q === "dependents_installed") return dependentsInstalled(env, args);
  throw new Error("unknown query " + q);
}
