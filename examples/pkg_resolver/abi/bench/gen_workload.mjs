// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// gen_workload.mjs -- generate query-key workloads for the ABI store crossover
// benchmark from the ACTUAL store keys (examples/pkg_resolver/abi/.out/bench/
// symprov.p2.jsonl, lines = ["soname|sym@node", "scalar"]).
//
// Two workloads (one key per line, in randomized order):
//   *.skewed.keys   -- a small HOT set (libc.so.6 / libstdc++.so.6 / libm.so.6
//                      keys) is hit HOT_FRACTION of the time, the rest is a
//                      uniform tail over all keys. Models real ABI resolution,
//                      which re-touches a few libraries constantly.
//   *.uniform.keys  -- uniform-random over all keys (the contrast workload).
// Both mix in MISS_FRACTION well-formed keys that are NOT in the store (a real
// soname band, a synthetic symbol) so the read path does real work and returns
// nothing.
//
// Usage:
//   node gen_workload.mjs <symprov.p2.jsonl> <out-prefix> [N] [HOT_FRAC] [MISS_FRAC] [SEED]
// Deterministic (seeded LCG) for reproducibility. Writes <out-prefix>.skewed.keys,
// <out-prefix>.uniform.keys and <out-prefix>.manifest.json.

import fs from 'node:fs';
import readline from 'node:readline';

const [,, jsonl, outPrefix,
       nArg = '50000', hotArg = '0.80', missArg = '0.15', seedArg = '1234567'] = process.argv;
if (!jsonl || !outPrefix) {
  console.error('usage: node gen_workload.mjs <symprov.p2.jsonl> <out-prefix> [N] [HOT_FRAC] [MISS_FRAC] [SEED]');
  process.exit(2);
}
const N = parseInt(nArg, 10);
const HOT_FRACTION = parseFloat(hotArg);
const MISS_FRACTION = parseFloat(missArg);
let state = (parseInt(seedArg, 10) >>> 0) || 1;
// Numerical Recipes LCG -> [0,1)
function rnd() { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296; }
function pick(arr) { return arr[Math.floor(rnd() * arr.length)]; }

const HOT_SONAMES = new Set(['libc.so.6', 'libstdc++.so.6', 'libm.so.6']);
const MISS_SYM = '__uwbenchmiss__@__UWBENCH__';

const keys = [];
const hotKeys = [];
const rl = readline.createInterface({ input: fs.createReadStream(jsonl) });
for await (const line of rl) {
  const t = line.trim();
  if (!t) continue;
  let a;
  try { a = JSON.parse(t); } catch { continue; }
  const k = a[0];
  if (typeof k !== 'string') continue;
  keys.push(k);
  const so = k.slice(0, k.indexOf('|'));
  if (HOT_SONAMES.has(so)) hotKeys.push(k);
}
if (keys.length === 0) { console.error('no keys read'); process.exit(1); }

// Build a distinct set of well-formed miss keys: real soname band + synthetic sym.
const sonames = [...new Set(keys.map(k => k.slice(0, k.indexOf('|'))))];
function missKey() { return `${pick(sonames)}|${MISS_SYM}`; }

function genSkewed() {
  const out = [];
  let nMiss = 0, nHot = 0, nTail = 0;
  for (let i = 0; i < N; i++) {
    if (rnd() < MISS_FRACTION) { out.push(missKey()); nMiss++; continue; }
    if (rnd() < HOT_FRACTION && hotKeys.length) { out.push(pick(hotKeys)); nHot++; }
    else { out.push(pick(keys)); nTail++; }
  }
  return { out, nMiss, nHot, nTail };
}
function genUniform() {
  const out = [];
  let nMiss = 0, nHit = 0;
  for (let i = 0; i < N; i++) {
    if (rnd() < MISS_FRACTION) { out.push(missKey()); nMiss++; }
    else { out.push(pick(keys)); nHit++; }
  }
  return { out, nMiss, nHit };
}

const sk = genSkewed();
const un = genUniform();
fs.writeFileSync(`${outPrefix}.skewed.keys`, sk.out.join('\n') + '\n');
fs.writeFileSync(`${outPrefix}.uniform.keys`, un.out.join('\n') + '\n');

const distinctSkewed = new Set(sk.out).size;
const distinctUniform = new Set(un.out).size;
const manifest = {
  source: jsonl,
  total_store_keys: keys.length,
  distinct_sonames: sonames.length,
  hot_sonames: [...HOT_SONAMES],
  hot_key_count: hotKeys.length,
  hot_key_fraction_of_store: +(hotKeys.length / keys.length).toFixed(4),
  N, HOT_FRACTION, MISS_FRACTION, seed: parseInt(seedArg, 10),
  skewed: { queries: N, miss: sk.nMiss, hot: sk.nHot, tail: sk.nTail, distinct_keys: distinctSkewed },
  uniform: { queries: N, miss: un.nMiss, hit: un.nHit, distinct_keys: distinctUniform },
  note: 'miss keys use a real soname band + synthetic symbol so the seek path does real work and returns 0 rows',
};
fs.writeFileSync(`${outPrefix}.manifest.json`, JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify(manifest, null, 2));
