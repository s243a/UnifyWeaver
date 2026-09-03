#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// test_cache_equiv.mjs -- L1 cache is semantics-invisible.
// Same results, same order, cache on or off, in one process.
//
//   node test_cache_equiv.mjs <wam-project-dir> <cases.jsonl>
//
// Requires the compiled wamjs_store project (resolver_store.mjs + js/).

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join, resolve } from "node:path";
import { createRequire } from "node:module";

const wamDir = resolve(process.argv[2]);
const casesPath = resolve(process.argv[3]);
if (!wamDir || !process.argv[3]) {
  process.stderr.write("usage: node test_cache_equiv.mjs <wam-project-dir> <cases.jsonl>\n");
  process.exit(2);
}

function stableStringify(x) {
  if (x === null || typeof x !== "object") return JSON.stringify(x);
  if (Array.isArray(x)) return "[" + x.map(stableStringify).join(",") + "]";
  const keys = Object.keys(x).sort();
  return "{" + keys.map((k) => JSON.stringify(k) + ":" + stableStringify(x[k])).join(",") + "}";
}

const shimUrl = pathToFileURL(join(wamDir, "resolver_store.mjs")).href;
const { runCase } = await import(shimUrl);
const require = createRequire(join(wamDir, "resolver_store.mjs"));
const generated = require(join(wamDir, "js", "generated_program.js"));
const Runtime = generated.Runtime;

const lines = readFileSync(casesPath, "utf8").split("\n").filter((l) => l !== "");
const cases = lines.map((l) => JSON.parse(l));

function runAll() {
  return cases.map((row) => {
    try {
      return runCase(row);
    } catch (err) {
      return { crash: String((err && err.stack) || err) };
    }
  });
}

Runtime.configure_fact_cache(false);
const off = runAll();
const offStats = Runtime.fact_cache_stats();

Runtime.configure_fact_cache(true, 4096);
const on = runAll();
const onStats = Runtime.fact_cache_stats();
const replay = runAll();
const replayStats = Runtime.fact_cache_stats();

let divergences = 0;
for (let i = 0; i < cases.length; i++) {
  const a = stableStringify(off[i]);
  const b = stableStringify(on[i]);
  const d = stableStringify(replay[i]);
  if (a !== b || a !== d) {
    divergences += 1;
    process.stderr.write("CACHE_DIVERGE " + String(cases[i].id) + "\n");
    process.stderr.write("  off    " + a + "\n");
    process.stderr.write("  on     " + b + "\n");
  }
}

if (offStats.hits !== 0) {
  process.stderr.write("cache_equiv: cache-off must not record hits, got " + offStats.hits + "\n");
  process.exit(1);
}
if (onStats.misses < 1) {
  process.stderr.write("cache_equiv: cache-on first pass should miss, got misses=" + onStats.misses + "\n");
  process.exit(1);
}
if (replayStats.hits <= onStats.hits || replayStats.hits < 1) {
  process.stderr.write(
    "cache_equiv: replay should add hits (on_hits=" +
    onStats.hits + " replay_hits=" + replayStats.hits + ")\n"
  );
  process.exit(1);
}
if (divergences !== 0) {
  process.stderr.write("cache_equiv: " + divergences + " divergences / " + cases.length + "\n");
  process.exit(1);
}

process.stdout.write(
  "cache_equiv ok cases=" + cases.length +
  " off_hits=" + offStats.hits +
  " on_hits=" + onStats.hits +
  " on_misses=" + onStats.misses +
  " replay_hits=" + replayStats.hits + "\n"
);
