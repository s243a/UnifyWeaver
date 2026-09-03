#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
//   node run_probe.mjs <wam-project-dir> <probe.json> [--repeat N]
// Must be launched with cwd or imports resolving generated_program.js
// next to resolver_store.mjs inside the wam project dir.
//
// UW_PROBE_REPEAT overrides --repeat. Same process: L1 cache (if on)
// warms on the first query.

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join, resolve } from "node:path";
import { createRequire } from "node:module";

const argv = process.argv.slice(2);
const wamDir = resolve(argv[0]);
const probePath = resolve(argv[1]);
let repeat = Number(process.env.UW_PROBE_REPEAT || "1");
for (let i = 2; i < argv.length; i++) {
  if (argv[i] === "--repeat" && argv[i + 1]) {
    repeat = Number(argv[i + 1]);
    i += 1;
  }
}
if (!Number.isFinite(repeat) || repeat < 1) repeat = 1;

const shimUrl = pathToFileURL(join(wamDir, "resolver_store.mjs")).href;
const { resolveLayered } = await import(shimUrl);
const require = createRequire(join(wamDir, "resolver_store.mjs"));
const generated = require(join(wamDir, "js", "generated_program.js"));
const Runtime = generated.Runtime;
const probe = JSON.parse(readFileSync(probePath, "utf8"));
const t0 = process.hrtime.bigint();
let got;
for (let i = 0; i < repeat; i++) {
  got = resolveLayered(probe.env, probe.args);
}
const t1 = process.hrtime.bigint();
process.stdout.write("wamjs_result " + JSON.stringify(got) + "\n");
process.stdout.write("wamjs_wall_ms " + (Number(t1 - t0) / 1e6).toFixed(3) + "\n");
process.stdout.write("wamjs_repeat " + String(repeat) + "\n");
if (Runtime && typeof Runtime.fact_cache_stats === "function") {
  const s = Runtime.fact_cache_stats();
  process.stdout.write(
    "wamjs_cache on=" + s.on +
    " hits=" + s.hits +
    " misses=" + s.misses +
    " size=" + s.size +
    " cap=" + s.cap + "\n"
  );
}
