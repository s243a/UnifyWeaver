// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// cliArgs.mjs -- the EDGE of the JS-WAM compiled parser.
//
// It exports the same two names peerhailer's `src/cliArgs.js` does --
// `parseArgs(argv, registry?)` and `class CliError` -- so the oracle's own test
// corpus and the differential runner import this file with nothing changed but
// the import specifier.
//
// WHAT IS IN HERE, exhaustively: conversion between the JS WAM term
// representation and plain JavaScript values, at the module boundary. There is
// NO parse logic. Every decision about what an argv line means -- the two flag
// regexes, the strict/lenient split, the schema lookup, the arity check, the
// exact wording of every error message -- lives in the compiled WAM module
// (js/generated_program.js), which is compiler output from
// examples/cli_args/cli_args.pl.
//
// The compiled wrappers return only true/false and allocate a fresh WAM state,
// so the shim drives Runtime.run itself and reads A2 (or A3) after success.
//
// Term representation (JS WAM):
//   atom     { tag: "atom", id }
//   string   { tag: "string", val }
//   int      { tag: "int", val }
//   cons     { tag: "struct", fid: [|], args: [H, T] }
//   compound { tag: "struct", fid, args }

import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const generated = require(join(dirname(fileURLToPath(import.meta.url)), "js", "generated_program.js"));
const { Runtime, V } = generated;
const program = generated.program;

/** A user-facing parse error; mirrors the oracle's class exactly. */
export class CliError extends Error {}

function internAtom(name) {
  return V.Atom(Runtime.intern(program.intern_table, name));
}

function functorId(name) {
  return Runtime.intern(program.intern_table, name);
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

function termToJsList(state, term) {
  const items = [];
  let cur = Runtime.deref(state, term);
  const nilId = Runtime.intern(program.intern_table, "[]");
  while (true) {
    cur = Runtime.deref(state, cur);
    if (typeof cur !== "object" || cur === null) {
      throw new Error("cliArgs shim: expected a list");
    }
    if (cur.tag === "atom" && cur.id === nilId) return items;
    if (!isCons(cur)) throw new Error("cliArgs shim: expected a list cons");
    items.push(cur.args[0]);
    cur = cur.args[1];
  }
}

function scalarToJs(state, term) {
  const t = Runtime.deref(state, term);
  if (!t || typeof t !== "object") return t;
  if (t.tag === "string") return String(t.val);
  if (t.tag === "int" || t.tag === "float") return t.val;
  if (t.tag === "atom") {
    const name = Runtime.string_of(program.intern_table, t.id);
    if (name === "true") return true;
    if (name === "false") return false;
    return name;
  }
  throw new Error(`cliArgs shim: unexpected scalar ${JSON.stringify(t)}`);
}

function positionalToJs(state, term) {
  return termToJsList(state, term).map((item) => scalarToJs(state, item));
}

/**
 * A Prolog list of `Key-Value` pairs becomes a plain object, in pair order.
 *
 * A plain `{}` with `obj[k] = v` is deliberate rather than a Map or a
 * null-prototype object: the oracle builds its `flags` exactly this way, so JS
 * object semantics -- key-insertion order, and the inherited `__proto__` setter
 * that silently drops a primitive assignment -- are reproduced rather than
 * re-implemented. cli_args.pl models the same semantics on the Prolog side
 * (flags_set/4), so the two agree before this function is ever reached.
 */
function pairsToObject(state, pairsTerm) {
  const out = {};
  const dash = functorId("-");
  for (const p of termToJsList(state, pairsTerm)) {
    const t = Runtime.deref(state, p);
    if (!t || t.tag !== "struct" || t.fid !== dash || (t.args || []).length !== 2) {
      throw new Error("cliArgs shim: expected a Key-Value pair");
    }
    const k = scalarToJs(state, t.args[0]);
    out[k] = scalarToJs(state, t.args[1]);
  }
  return out;
}

function pairTerm(k, v) {
  return V.Struct(functorId("-"), [k, v]);
}

function toEntryTerm(entry) {
  if (entry && entry.actions) {
    const actions = jsListToTerm(Object.keys(entry.actions), (a) =>
      pairTerm(V.String(a), toEntryTerm(entry.actions[a]))
    );
    return V.Struct(functorId("group"), [actions]);
  }
  const optionsObj = entry && entry.options ? entry.options : {};
  const positionals = entry && entry.positionals ? entry.positionals : [];
  const options = jsListToTerm(Object.keys(optionsObj), (k) =>
    pairTerm(V.String(k), internAtom(String(optionsObj[k])))
  );
  const pos = jsListToTerm(positionals, (p) => V.String(p));
  return V.Struct(functorId("schema"), [options, pos]);
}

function toRegistryTerm(registry) {
  return jsListToTerm(Object.keys(registry), (name) =>
    pairTerm(V.String(name), toEntryTerm(registry[name]))
  );
}

function runParse(argvTerm, registryTerm) {
  const state = Runtime.new_state();
  state.program = program;
  Runtime.put_reg(state, 1, argvTerm);
  // Hold the result cell itself: parse_args/2 tail-calls parse_args/3, so A2
  // is overwritten. Unification still binds this unbound; deref it after run.
  const resultVar = Runtime.new_var(state);
  let label;
  if (registryTerm === undefined) {
    Runtime.put_reg(state, 2, resultVar);
    label = "parse_args/2";
  } else {
    Runtime.put_reg(state, 2, registryTerm);
    Runtime.put_reg(state, 3, resultVar);
    label = "parse_args/3";
  }
  const startPc = program.labels[label];
  if (startPc === undefined) {
    throw new Error(`cliArgs shim: missing label ${label}`);
  }
  state.pc = startPc;
  const ok = Runtime.run(program, state) === true;
  if (!ok) {
    throw new Error("cliArgs shim: parse_args failed in the WAM");
  }
  return { state, result: Runtime.deref(state, resultVar) };
}

/**
 * parseArgs(argv, registry?) -> { positional, flags }
 * Throws CliError with the oracle's exact message on a usage error.
 */
export function parseArgs(argv, registry) {
  const argvTerm = jsListToTerm(Array.from(argv, String), (s) => V.String(s));
  const registryTerm = registry === undefined ? undefined : toRegistryTerm(registry);
  const { state, result } = runParse(argvTerm, registryTerm);
  if (!result || result.tag !== "struct") {
    throw new Error(`cliArgs shim: parse_args answered neither ok/2 nor error/1 (${String(result)})`);
  }
  const fname = functorName(result.fid);
  if (fname === "ok" && (result.args || []).length === 2) {
    return {
      positional: positionalToJs(state, result.args[0]),
      flags: pairsToObject(state, result.args[1])
    };
  }
  if (fname === "error" && (result.args || []).length === 1) {
    throw new CliError(String(scalarToJs(state, result.args[0])));
  }
  throw new Error(
    `cliArgs shim: parse_args answered neither ok/2 nor error/1 (${fname})`
  );
}
