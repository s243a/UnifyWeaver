#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// gen_snapshots.mjs -- seeded correlated MULTI-SNAPSHOT generator for the
// memory-efficient dedup store. Snapshot 0 is a base catalog (same seeded DAG
// rule as gen_scale_catalog.mjs); each later snapshot applies a small, seeded
// churn (bump/add/remove) so MOST packages are unchanged between snapshots --
// the property that makes cross-snapshot dedup pay off.
//
// The package POOL is APPEND-ONLY: a bumped/removed (Name,Ver) row is never
// deleted, only membership shifts, so every snapshot that selected an old
// version still resolves. Output:
//   pool.rich.jsonl              -- union of all package/depends rows ever
//                                   minted, catalog=<PoolId> (feeds rich_to_p2
//                                   unchanged: it keys by row.catalog)
//   membership/snap-<t>.jsonl    -- {kind:"snapshot_member",snap,name,ver} rows
//
//   node gen_snapshots.mjs --out=DIR [--base=5000 --n=100 --churn-bump=0.02
//        --churn-add=0.005 --churn-remove=0.002 --seed=0xc0ffee01]
//   node gen_snapshots.mjs --out=DIR --materialize=<t>   # naive flat rich.jsonl
//        for one snapshot (catalog=snap-<t>) -- the O(N×pkg) baseline, for the
//        dedup-ratio comparison and single-snapshot compat testing.

import { mkdirSync, writeFileSync } from "node:fs";

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const pick = (rng, n) => Math.floor(rng() * n);

const args = {};
for (const a of process.argv.slice(2)) {
  const m = a.match(/^--([^=]+)=(.*)$/);
  if (m) args[m[1]] = m[2];
}
const OUT = args.out;
if (!OUT) { console.error("usage: gen_snapshots.mjs --out=DIR [flags]"); process.exit(2); }
const BASE = parseInt(args.base || "5000", 10);
const N = parseInt(args.n || "100", 10);
const P_BUMP = parseFloat(args["churn-bump"] || "0.02");
const P_ADD = parseFloat(args["churn-add"] || "0.005");
const P_REMOVE = parseFloat(args["churn-remove"] || "0.002");
const SEED = args.seed ? Number(args.seed) : 0xc0ffee01;
const POOL_ID = "s5k-snap";

// --- pool state (append-only) -------------------------------------------------
// names[]: index -> name (index = DAG rank; deps only reference lower indices).
// A "version key" is "a.b.c". poolSeen guards duplicate (name,ver) minting.
const names = [];
const nameIndex = new Map();          // name -> index
const poolPkg = [];                   // {name, ver}
const poolDep = [];                   // {name, ver, dep, constraint}
const poolSeen = new Set();           // "name@a.b.c"
const depsByNameVer = new Map();      // "name@a.b.c" -> [depRow,...]  (for materialize)
const vkey = (v) => v[0] + "." + v[1] + "." + v[2];

function mintName(name) {
  if (!nameIndex.has(name)) { nameIndex.set(name, names.length); names.push(name); }
  return nameIndex.get(name);
}
// Mint (name,ver) with a seeded DAG dep list drawn from names of LOWER index
// (acyclic). Idempotent per (name,ver). Returns nothing; fills the pool.
function mintVersion(rng, name, ver) {
  const idx = mintName(name);
  const k = name + "@" + vkey(ver);
  if (poolSeen.has(k)) return;
  poolSeen.add(k);
  poolPkg.push({ name, ver });
  const rows = [];
  const nDep = idx === 0 ? 0 : 2 + pick(rng, 3); // 2-4
  const used = new Set();
  for (let d = 0; d < nDep; d++) {
    const j = pick(rng, idx);                    // lower index only -> acyclic
    if (used.has(j)) continue;
    used.add(j);
    const dep = names[j];
    let constraint = "any";
    if (rng() < 0.25) constraint = { op: "gte", v: [0, 0, 0] };
    const row = { name, ver, dep, constraint };
    poolDep.push(row);
    rows.push(row);
  }
  depsByNameVer.set(k, rows);
}

// --- snapshot 0: base catalog -------------------------------------------------
const rng0 = mulberry32(SEED);
const membership = [];                 // membership[t]: Map name -> ver
const m0 = new Map();
for (let i = 0; i < BASE; i++) {
  const name = "p" + i;
  const ver = [0, 0, 0];
  mintVersion(rng0, name, ver);
  m0.set(name, ver);
}
membership.push(m0);

// --- snapshots 1..N-1: seeded churn -------------------------------------------
let addCounter = BASE;                  // new names get indices >= BASE
let bumpTotal = 0, addTotal = 0, removeTotal = 0;
for (let t = 1; t < N; t++) {
  const rng = mulberry32(SEED + t);     // independently reproducible per snapshot
  const prev = membership[t - 1];
  const m = new Map(prev);              // inherit; churn mutates this copy
  const live = Array.from(m.keys());

  // bump: mint a new version for p% of live names (append-only pool row).
  const nBump = Math.round(P_BUMP * BASE);
  for (let b = 0; b < nBump && live.length; b++) {
    const name = live[pick(rng, live.length)];
    const cur = m.get(name);
    const nver = [cur[0] + 1, 0, 0];
    mintVersion(rng, name, nver);
    m.set(name, nver);
    bumpTotal++;
  }
  // add: brand-new names, deps drawn from already-existing names (acyclic).
  const nAdd = Math.round(P_ADD * BASE);
  for (let a = 0; a < nAdd; a++) {
    const name = "p" + (addCounter++);
    const ver = [0, 0, 0];
    mintVersion(rng, name, ver);
    m.set(name, ver);
    addTotal++;
  }
  // remove: drop r% of current members from THIS snapshot only (pool untouched).
  const nRemove = Math.round(P_REMOVE * BASE);
  const cur = Array.from(m.keys());
  for (let r = 0; r < nRemove && cur.length; r++) {
    const name = cur[pick(rng, cur.length)];
    m.delete(name);
    removeTotal++;
  }
  membership.push(m);
}

// --- materialize one snapshot to naive flat rich (the O(N×pkg) baseline) ------
if (args.materialize != null) {
  const t = parseInt(args.materialize, 10);
  if (!(t >= 0 && t < N)) { console.error("materialize: snap out of range"); process.exit(2); }
  const cat = "snap-" + t;
  const rich = [];
  for (const [name, ver] of membership[t]) {
    rich.push({ kind: "package", catalog: cat, name, ver });
    for (const row of (depsByNameVer.get(name + "@" + vkey(ver)) || [])) {
      rich.push({ kind: "depends", catalog: cat, name, ver, dep: row.dep, constraint: row.constraint });
    }
  }
  mkdirSync(OUT, { recursive: true });
  writeFileSync(OUT + "/rich.jsonl", rich.map((r) => JSON.stringify(r)).join("\n") + "\n");
  console.log("gen_snapshots: materialized snap " + t + " -> " + OUT +
    "/rich.jsonl  rows=" + rich.length + " members=" + membership[t].size);
  process.exit(0);
}

// --- write the deduped pool + membership --------------------------------------
mkdirSync(OUT + "/membership", { recursive: true });
const pool = [];
for (const p of poolPkg) pool.push({ kind: "package", catalog: POOL_ID, name: p.name, ver: p.ver });
for (const d of poolDep) pool.push({ kind: "depends", catalog: POOL_ID, name: d.name, ver: d.ver, dep: d.dep, constraint: d.constraint });
writeFileSync(OUT + "/pool.rich.jsonl", pool.map((r) => JSON.stringify(r)).join("\n") + "\n");

let memberRows = 0;
for (let t = 0; t < N; t++) {
  const rows = [];
  for (const [name, ver] of membership[t]) rows.push({ kind: "snapshot_member", snap: t, name, ver });
  memberRows += rows.length;
  writeFileSync(OUT + "/membership/snap-" + t + ".jsonl", rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
}

// --- git-tree membership: base commit + per-snapshot deltas -------------------
// The COMPACT canonical form (the "git tree"): snapshot 0 is a full `set`
// commit; each later snapshot records only what CHANGED vs the previous --
// op:"set" (a name's new-or-bumped current version) or op:"del" (a dropped
// name). Materialize snapshot t's membership by folding deltas 0..t
// (materialize_membership.mjs). Because most packages don't change between
// snapshots, this is a small fraction of the full per-snapshot lists above.
mkdirSync(OUT + "/membership-git", { recursive: true });
const vk = (v) => JSON.stringify(v);
let deltaRows = 0;
for (let t = 0; t < N; t++) {
  const rows = [];
  if (t === 0) {
    for (const [name, ver] of membership[0]) rows.push({ snap: 0, op: "set", name, ver });
  } else {
    const prev = membership[t - 1], cur = membership[t];
    for (const [name, ver] of cur) {
      const pv = prev.get(name);
      if (!pv || vk(pv) !== vk(ver)) rows.push({ snap: t, op: "set", name, ver });
    }
    for (const [name] of prev) if (!cur.has(name)) rows.push({ snap: t, op: "del", name });
  }
  deltaRows += rows.length;
  writeFileSync(OUT + "/membership-git/delta-" + t + ".jsonl",
    rows.map((r) => JSON.stringify(r)).join("\n") + (rows.length ? "\n" : ""));
}

const uniqueVersions = poolPkg.length;
const naiveVersions = memberRows; // = sum of members over snapshots = naive (Name,Ver) rows
console.log("gen_snapshots: pool_id=" + POOL_ID + " base=" + BASE + " N=" + N +
  " churn(bump/add/remove)=" + P_BUMP + "/" + P_ADD + "/" + P_REMOVE);
console.log("  pool: unique_versions=" + uniqueVersions + " dep_rows=" + poolDep.length);
console.log("  membership rows (naive versions) =" + naiveVersions +
  "  dedup ratio (versions) = " + (naiveVersions / uniqueVersions).toFixed(2) + "x");
console.log("  git-tree membership rows (deltas) =" + deltaRows +
  "  vs full per-snapshot =" + memberRows +
  "  membership compression = " + (memberRows / Math.max(1, deltaRows)).toFixed(2) + "x");
console.log("  churn totals: bump=" + bumpTotal + " add=" + addTotal + " remove=" + removeTotal);
