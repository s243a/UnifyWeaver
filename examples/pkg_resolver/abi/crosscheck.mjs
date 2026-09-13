#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// crosscheck.mjs -- compare the two provider tiers for one soname:
//   readelf(lib.so)  ->  symprov rows ["so|sym@node", ["at", rel]]
//   .symbols file    ->  symprov rows ["so|sym@node", ["since", minver]]
//
// (1) EXACT IDENTITY: the sets of sym@node must be equal. This is the check
//     the model actually relies on (a requirement matches a provider by the
//     exact triple), and on Ubuntu 22.04 libc6 it is 3006/3006.
// (2) LEGACY PER-NAME COMPARISON, CORRECTED: the original script compared the
//     LAST curated row of a symbol against the EARLIEST ELF node of that
//     symbol, so every symbol with two nodes (e.g. pthread_setname_np@GLIBC_2.12
//     + @GLIBC_2.34 after the 2.34 libpthread merge) "disagreed" (91.7%). With
//     the same earliest-row aggregation on both sides the figure is 100%; the
//     "glibc 2.34 merge divergence" explanation was an artifact. Note this
//     comparison mixes axes (node label number vs package version) and is kept
//     only as the corrected regression figure.
//
// usage: crosscheck.mjs <symbols.symprov.jsonl> <elf.symprov.jsonl> <soname>

import { readFileSync } from "node:fs";

const [symFile, elfFile, so] = process.argv.slice(2);

function loadRows(file, tag) {
  const rows = [];                         // {sym, node, v}
  for (const l of readFileSync(file, "utf8").split("\n")) {
    if (!l) continue;
    const [k, v] = JSON.parse(l);
    if (!k.startsWith(so + "|") || v[0] !== tag) continue;
    const ident = k.slice(so.length + 1);
    const at = ident.lastIndexOf("@");
    rows.push({ sym: ident.slice(0, at), node: ident.slice(at + 1), v: v[1] });
  }
  return rows;
}

const S = loadRows(symFile, "since"), E = loadRows(elfFile, "at");
const sKeys = new Set(S.map((r) => `${r.sym}@${r.node}`));
const eKeys = new Set(E.map((r) => `${r.sym}@${r.node}`));
let both = 0; const onlyE = [], onlyS = [];
for (const k of eKeys) (sKeys.has(k) ? both++ : onlyE.push(k));
for (const k of sKeys) if (!eKeys.has(k)) onlyS.push(k);
console.log(`  exact sym@node identity: .symbols=${sKeys.size} readelf=${eKeys.size} shared=${both} only-readelf=${onlyE.length} only-.symbols=${onlyS.length}`);
if (onlyE.length) console.log(`    only-readelf sample: ${onlyE.slice(0, 5).join(", ")}`);
if (onlyS.length) console.log(`    only-.symbols sample: ${onlyS.slice(0, 5).join(", ")}`);
const identityPct = (100 * both) / Math.max(sKeys.size, eKeys.size);
console.log(`  identity agreement: ${both}/${Math.max(sKeys.size, eKeys.size)} (${identityPct.toFixed(1)}%)`);

// Legacy per-name figure, corrected: EARLIEST row on BOTH sides.
function nodeNum(node) { const m = node.match(/_([0-9][0-9.]*)$/); return m ? m[1] : null; }
function cmpDotted(a, b) {
  const pa = a.split("."), pb = b.split(".");
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) { const x = +(pa[i] || 0), y = +(pb[i] || 0); if (x !== y) return x - y; }
  return 0;
}
function earliest(rows, pick) {
  const m = new Map();
  for (const r of rows) { const n = pick(r); if (n === null) continue; if (!m.has(r.sym) || cmpDotted(n, m.get(r.sym)) < 0) m.set(r.sym, n); }
  return m;
}
const sE = earliest(S, (r) => nodeNum(r.node));   // earliest numeric node per name (curated rows)
const eE = earliest(E, (r) => nodeNum(r.node));   // earliest numeric node per name (readelf rows)
let shared = 0, agree = 0; const dis = [];
for (const [sym, n] of eE) { if (!sE.has(sym)) continue; shared++; if (sE.get(sym) === n) agree++; else if (dis.length < 5) dis.push(`${sym} (.symbols=${sE.get(sym)} readelf=${n})`); }
const pct = (100 * agree) / shared;
console.log(`  legacy per-name earliest-row comparison (corrected aggregation): ${agree}/${shared} (${pct.toFixed(1)}%)`);
if (dis.length) console.log(`    disagreements: ${dis.join(", ")}`);
if (identityPct < 100) { console.error("  FAIL: exact identity sets differ"); process.exit(1); }
if (pct < 100) { console.error("  FAIL: corrected per-name comparison below 100%"); process.exit(1); }
console.log("  PASS: readelf and .symbols agree on every exact sym@node identity");
