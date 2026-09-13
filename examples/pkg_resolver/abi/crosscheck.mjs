#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// crosscheck.mjs -- compare the two provider tiers for one soname:
//   readelf(lib.so)  ->  symprov rows ["so|sym@node", ["at", rel, binding]]
//   .symbols file    ->  symprov rows ["so|sym@node", ["since", minver, rel, binding]]
//
// (1) EXACT IDENTITY: the sets of sym@node must be equal. This is the check
//     the model actually relies on (a requirement matches a provider by the
//     exact triple), and on Ubuntu 22.04 libc6 it is 3006/3006.
// (2) PER-NAME COMPARISON: for every symbol NAME the SET of nodes must agree
//     on both sides. This is the corrected form of the legacy per-name figure
//     (the original script compared the LAST curated row of a symbol against
//     the EARLIEST ELF node, so every two-node symbol -- e.g.
//     pthread_setname_np@GLIBC_2.12 + @GLIBC_2.34 -- "disagreed": 91.7%).
//     Node names are opaque labels: nothing is parsed out of them and nothing
//     is ordered (Sol P3 removed the local dotted-version comparator); the
//     fixture fixtures/crosscheck/ keeps the regression pinned.
// Both figures must be 100% AND every denominator must be nonzero: two empty
// inputs (e.g. /dev/null vs /dev/null) FAIL instead of passing on NaN (Sol P2c).
//
// usage: crosscheck.mjs <symbols.symprov.jsonl> <elf.symprov.jsonl> <soname>

import { readFileSync } from "node:fs";

const [symFile, elfFile, so] = process.argv.slice(2);
if (!symFile || !elfFile || !so) { console.error("usage: crosscheck.mjs <symbols.symprov.jsonl> <elf.symprov.jsonl> <soname>"); process.exit(2); }

function loadRows(file, tag) {
  const rows = [];                         // {sym, node}
  for (const l of readFileSync(file, "utf8").split("\n")) {
    if (!l) continue;
    const [k, v] = JSON.parse(l);
    if (!k.startsWith(so + "|") || v[0] !== tag) continue;
    const ident = k.slice(so.length + 1);
    const at = ident.lastIndexOf("@");
    rows.push({ sym: ident.slice(0, at), node: ident.slice(at + 1) });
  }
  return rows;
}

function fail(msg) { console.error(`  FAIL: ${msg}`); process.exit(1); }

const S = loadRows(symFile, "since"), E = loadRows(elfFile, "at");
const sKeys = new Set(S.map((r) => `${r.sym}@${r.node}`));
const eKeys = new Set(E.map((r) => `${r.sym}@${r.node}`));
let both = 0; const onlyE = [], onlyS = [];
for (const k of eKeys) (sKeys.has(k) ? both++ : onlyE.push(k));
for (const k of sKeys) if (!eKeys.has(k)) onlyS.push(k);
console.log(`  exact sym@node identity: .symbols=${sKeys.size} readelf=${eKeys.size} shared=${both} only-readelf=${onlyE.length} only-.symbols=${onlyS.length}`);
if (onlyE.length) console.log(`    only-readelf sample: ${onlyE.slice(0, 5).join(", ")}`);
if (onlyS.length) console.log(`    only-.symbols sample: ${onlyS.slice(0, 5).join(", ")}`);
if (sKeys.size === 0 || eKeys.size === 0) fail(`empty symbol set for ${so} (.symbols=${sKeys.size} readelf=${eKeys.size}); nothing to compare`);
const denom = Math.max(sKeys.size, eKeys.size);
const identityPct = (100 * both) / denom;
console.log(`  identity agreement: ${both}/${denom} (${identityPct.toFixed(1)}%)`);

// Per-name node-set agreement (no ordering, no parsing of node names).
function nodeSets(rows) {
  const m = new Map();
  for (const r of rows) { if (!m.has(r.sym)) m.set(r.sym, new Set()); m.get(r.sym).add(r.node); }
  return m;
}
const sN = nodeSets(S), eN = nodeSets(E);
const names = new Set([...sN.keys(), ...eN.keys()]);
let agree = 0; const dis = [];
for (const sym of names) {
  const a = sN.get(sym) || new Set(), b = eN.get(sym) || new Set();
  const same = a.size === b.size && [...a].every((n) => b.has(n));
  if (same) agree++;
  else if (dis.length < 5) dis.push(`${sym} (.symbols={${[...a].join(",")}} readelf={${[...b].join(",")}})`);
}
if (names.size === 0) fail("no symbol names to compare");
const pct = (100 * agree) / names.size;
console.log(`  per-name node-set agreement: ${agree}/${names.size} (${pct.toFixed(1)}%)`);
if (dis.length) console.log(`    disagreements: ${dis.join(", ")}`);
if (!(identityPct === 100)) fail("exact identity sets differ");
if (!(pct === 100)) fail("per-name node sets differ");
console.log("  PASS: readelf and .symbols agree on every exact sym@node identity");
