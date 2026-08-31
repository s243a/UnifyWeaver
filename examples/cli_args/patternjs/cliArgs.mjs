// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// cliArgs.mjs -- the EDGE of the transpiled parser.
//
// It exports the same two names peerhailer's `src/cliArgs.js` does --
// `parseArgs(argv, registry?)` and `class CliError` -- so the oracle's own test
// corpus and the differential runner import this file with nothing changed but
// the import specifier.
//
// WHAT IS IN HERE, exhaustively: conversion between UnifyWeaver's term
// representation and plain JavaScript values, at the module boundary. There is
// NO parse logic. Every decision about what an argv line means -- the two flag
// regexes, the strict/lenient split, the schema lookup, the arity check, the
// exact wording of every error message -- lives in cliArgs.generated.mjs, which
// is compiler output from examples/cli_args/cli_args.pl.
//
// The term representation (typescript_target's, see G-A3-12/G-A3-13):
//
//   Prolog atom / string    a JS string
//   true / false            a JS boolean
//   list                    a JS array
//   compound f(A1..An)      { $: "f", args: [...] }
//
// so `ok(Positional, Flags)` arrives as
// `{ $: "ok", args: [ [...], [ {$:"-",args:[K,V]}, ... ] ] }`.

import { parse_args_2, parse_args_3 } from "./cliArgs.generated.mjs";

/** A user-facing parse error; mirrors the oracle's class exactly. */
export class CliError extends Error {}

/** Is `t` the compound term `f(...)`? */
const isTerm = (t, f) =>
  t !== null && typeof t === "object" && !Array.isArray(t) && t.$ === f;

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
function pairsToObject(pairs) {
  const out = {};
  for (const p of pairs) {
    if (!isTerm(p, "-") || p.args.length !== 2) {
      throw new Error(`cliArgs shim: expected a Key-Value pair, got ${JSON.stringify(p)}`);
    }
    out[p.args[0]] = p.args[1];
  }
  return out;
}

/**
 * A JS registry (the oracle's `COMMANDS` shape) becomes the Prolog registry
 * term cli_args.pl expects. Only reached when a caller passes one; with no
 * registry the compiled `parse_args/2` supplies cli_args.pl's own
 * `default_registry/1`, which is the transpiled copy of the oracle's COMMANDS.
 */
const pair = (k, v) => ({ $: "-", args: [k, v] });

function toRegistryTerm(registry) {
  return Object.keys(registry).map((name) => pair(name, toEntryTerm(registry[name])));
}

function toEntryTerm(entry) {
  if (entry && entry.actions) {
    return {
      $: "group",
      args: [Object.keys(entry.actions).map((a) => pair(a, toEntryTerm(entry.actions[a])))],
    };
  }
  const options = entry && entry.options ? entry.options : {};
  const positionals = entry && entry.positionals ? entry.positionals : [];
  return {
    $: "schema",
    args: [Object.keys(options).map((k) => pair(k, options[k])), positionals.slice()],
  };
}

/**
 * parseArgs(argv, registry?) -> { positional, flags }
 * Throws CliError with the oracle's exact message on a usage error.
 */
export function parseArgs(argv, registry) {
  const tokens = Array.from(argv, String);
  const result =
    registry === undefined
      ? parse_args_2(tokens)
      : parse_args_3(tokens, toRegistryTerm(registry));

  if (isTerm(result, "ok") && result.args.length === 2) {
    return { positional: result.args[0], flags: pairsToObject(result.args[1]) };
  }
  if (isTerm(result, "error") && result.args.length === 1) {
    throw new CliError(result.args[0]);
  }
  // parse_args/2 is semidet in the compiled calling convention, so it CAN answer
  // the failure sentinel. cli_args.pl documents that it never fails; if it ever
  // does, that is a compiler or reference-implementation bug and must be loud.
  throw new Error(
    `cliArgs shim: parse_args answered neither ok/2 nor error/1 (${String(result)})`
  );
}
