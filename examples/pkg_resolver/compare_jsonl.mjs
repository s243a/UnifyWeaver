#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// compare_jsonl.mjs -- compare SWI oracle vs wamjs results for pkg_resolver.
//
//   node compare_jsonl.mjs <cases.jsonl> <swi.jsonl> <wamjs.jsonl>

import { readFileSync } from "node:fs";

const [, , casesPath, swiPath, wamPath] = process.argv;
if (!casesPath || !swiPath || !wamPath) {
  console.error("usage: compare_jsonl.mjs <cases.jsonl> <swi.jsonl> <wamjs.jsonl>");
  process.exit(2);
}

const readLines = (p) => readFileSync(p, "utf8").split("\n").filter((l) => l !== "");
const cases = readLines(casesPath);
const swi = readLines(swiPath);
const wam = readLines(wamPath);

if (swi.length !== cases.length || wam.length !== cases.length) {
  console.error(
    `FATAL: line-count mismatch — cases=${cases.length} swi=${swi.length} wamjs=${wam.length}`
  );
  process.exit(2);
}

function stableStringify(x) {
  if (x === null || typeof x !== "object") return JSON.stringify(x);
  if (Array.isArray(x)) return "[" + x.map(stableStringify).join(",") + "]";
  const keys = Object.keys(x).sort();
  return "{" + keys.map((k) => JSON.stringify(k) + ":" + stableStringify(x[k])).join(",") + "}";
}

function canon(obj) {
  if (obj && typeof obj === "object" && obj.id !== undefined) {
    const { id: _id, ...rest } = obj;
    return stableStringify(rest);
  }
  return stableStringify(obj);
}

let divergences = 0;
let crashes = 0;
const samples = [];

for (let i = 0; i < cases.length; i += 1) {
  const s = JSON.parse(swi[i]);
  const w = JSON.parse(wam[i]);
  if (s.crash !== undefined || w.crash !== undefined) {
    crashes += 1;
    if (samples.length < 8) samples.push({ i, id: JSON.parse(cases[i]).id, swi: s, wamjs: w });
    continue;
  }
  if (canon(s) !== canon(w)) {
    divergences += 1;
    if (samples.length < 8) samples.push({ i, id: JSON.parse(cases[i]).id, swi: s, wamjs: w });
  }
}

console.log("pkg_resolver differential");
console.log("  cases:        " + cases.length);
console.log("  divergences:  " + divergences);
console.log("  crashes:      " + crashes);
if (samples.length > 0) {
  console.log("  samples:");
  for (const s of samples) {
    console.log("    " + s.id + " swi=" + JSON.stringify(s.swi) + " wamjs=" + JSON.stringify(s.wamjs));
  }
}
if (divergences !== 0 || crashes !== 0) process.exit(1);
console.log("  result:       0 divergences");
