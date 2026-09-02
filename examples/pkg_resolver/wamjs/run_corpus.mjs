#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// run_corpus.mjs -- compare dump_corpus JSONL (SWI expected) to the JS-WAM shim.

import { readFileSync, writeFileSync } from "node:fs";
import { runCase } from "./resolver.mjs";

const src = process.argv[2];
const dest = process.argv[3];
if (!src || !dest) {
  console.error("usage: node run_corpus.mjs <swi.jsonl> <wamjs.jsonl>");
  process.exit(2);
}

function stableStringify(x) {
  if (x === null || typeof x !== "object") return JSON.stringify(x);
  if (Array.isArray(x)) return "[" + x.map(stableStringify).join(",") + "]";
  const keys = Object.keys(x).sort();
  return "{" + keys.map((k) => JSON.stringify(k) + ":" + stableStringify(x[k])).join(",") + "}";
}

const lines = readFileSync(src, "utf8").split("\n").filter((l) => l !== "");
const out = [];
let divergences = 0;
for (const line of lines) {
  const row = JSON.parse(line);
  let got;
  try {
    got = runCase(row);
  } catch (err) {
    got = { crash: String((err && err.stack) || err) };
  }
  out.push(JSON.stringify({ id: row.id, got }));
  const exp = row.expected;
  if (stableStringify(got) !== stableStringify(exp)) {
    divergences += 1;
    console.error("DIVERGE", row.id);
    console.error("  expected", JSON.stringify(exp));
    console.error("  got     ", JSON.stringify(got));
  } else {
    console.log("ok", row.id);
  }
}
writeFileSync(dest, out.join("\n") + "\n");
if (divergences !== 0) {
  console.error("corpus-under-node: " + divergences + " divergences / " + lines.length);
  process.exit(1);
}
console.log("corpus-under-node: " + lines.length + "/" + lines.length + " matched SWI");
