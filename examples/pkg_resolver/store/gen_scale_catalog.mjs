#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// gen_scale_catalog.mjs -- seeded 5k-package catalog as rich JSONL + env cases.
//
//   node gen_scale_catalog.mjs <out-dir>
// Writes rich.jsonl, then caller runs rich_to_p2 + build_stores.
// Writes cases.jsonl (500 store-backed differential cases) and probe.json
// (the bound resolve_layered request used for the bytes-read proof).

import { mkdirSync, writeFileSync } from "node:fs";
import { packKey } from "./pack.mjs";

const SEED = 0xc0ffee01;
const N_PKGS = 5000;
const N_CASES = 500;
const CAT = "s5k";

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick(rng, n) {
  return Math.floor(rng() * n);
}

const rng = mulberry32(SEED);
const outDir = process.argv[2];
if (!outDir) {
  console.error("usage: gen_scale_catalog.mjs <out-dir>");
  process.exit(2);
}
mkdirSync(outDir, { recursive: true });

const names = Array.from({ length: N_PKGS }, (_, i) => "p" + i);
const rich = [];
let nDeps = 0;

for (let i = 0; i < N_PKGS; i++) {
  const name = names[i];
  const nVer = 1 + pick(rng, 2); // 1–2 versions
  const vers = [];
  for (let k = 0; k < nVer; k++) vers.push([k, 0, 0]);
  for (const v of vers) {
    rich.push({ kind: "package", catalog: CAT, name, ver: v });
  }
  const nDep = i === 0 ? 0 : 2 + pick(rng, 3); // 2–4, DAG onto earlier names
  const used = new Set();
  for (let d = 0; d < nDep && i > 0; d++) {
    const j = pick(rng, i);
    if (used.has(j)) continue;
    used.add(j);
    const ver = vers[0];
    const dep = names[j];
    const kind = rng();
    let constraint = "any";
    if (kind < 0.25) constraint = { op: "gte", v: [0, 0, 0] };
    rich.push({
      kind: "depends",
      catalog: CAT,
      name,
      ver,
      dep,
      constraint
    });
    nDeps += 1;
  }
}

if (nDeps < 15000) {
  // pad with extra edges on later packages
  for (let i = 3; i < N_PKGS && nDeps < 15000; i++) {
    const j = (i * 7 + 3) % i;
    rich.push({
      kind: "depends",
      catalog: CAT,
      name: names[i],
      ver: [0, 0, 0],
      dep: names[j],
      constraint: "any"
    });
    nDeps += 1;
  }
}

const richPath = outDir + "/rich.jsonl";
writeFileSync(richPath, rich.map((r) => JSON.stringify(r)).join("\n") + "\n");

// Probe: a mid-graph package whose closure is small-ish (depends on earlier).
const probeName = "p30";
const probe = {
  catalog_id: CAT,
  query: "resolve_layered",
  args: [probeName],
  env: {
    catalog_id: CAT,
    base: [["p0", [0, 0, 0]]],
    installed: [],
    requested: [],
    layers: [],
    excluded: [],
    aliases: []
  }
};
writeFileSync(outDir + "/probe.json", JSON.stringify(probe, null, 2));

const cases = [];
for (let i = 0; i < N_CASES; i++) {
  const qn = ["resolve", "resolve_layered", "explain_blocked", "dependents",
    "freeze_audit", "dependents_installed"][i % 6];
  // Closure queries stay in the first 25 packages so a request touches a
  // slice of the 5k store. dependents* may target any package (pure seek).
  const small = names[pick(rng, 25)];
  const any = names[pick(rng, N_PKGS)];
  const nBase = pick(rng, 3);
  const base = [];
  for (let b = 0; b < nBase; b++) {
    const bn = names[pick(rng, 25)];
    base.push([bn, [0, 0, 0]]);
  }
  let args;
  if (qn === "resolve" || qn === "resolve_layered") args = [small];
  else if (qn === "explain_blocked") args = small;
  else if (qn === "freeze_audit") args = [];
  else args = any;
  cases.push({
    id: "s" + i,
    catalog_id: CAT,
    query: qn,
    args,
    env: {
      catalog_id: CAT,
      base,
      installed: base.slice(0, 2),
      requested: base.length ? [base[0][0]] : [],
      layers: [],
      excluded: [],
      aliases: []
    }
  });
}
writeFileSync(outDir + "/cases.jsonl", cases.map((c) => JSON.stringify(c)).join("\n") + "\n");

console.log("gen_scale_catalog: packages=" + N_PKGS + " dep_edges=" + nDeps +
  " rich_rows=" + rich.length + " cases=" + N_CASES + " cat=" + CAT);
console.log("  key example " + packKey(CAT, "p0"));
