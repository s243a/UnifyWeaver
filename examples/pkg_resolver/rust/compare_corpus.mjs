#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// compare_corpus.mjs -- compare the corpus JSONL (SWI expected, from
// dump_corpus.pl) against the Rust-WAM shim's results line by line.
//
//   node compare_corpus.mjs <swi.jsonl> <rust.jsonl>

import { readFileSync } from "node:fs";

const [, , swiPath, rustPath] = process.argv;
if (!swiPath || !rustPath) {
  console.error("usage: compare_corpus.mjs <swi.jsonl> <rust.jsonl>");
  process.exit(2);
}

const readLines = (p) => readFileSync(p, "utf8").split("\n").filter((l) => l !== "");
const swi = readLines(swiPath);
const rust = readLines(rustPath);

if (swi.length !== rust.length) {
  console.error(`FATAL: line-count mismatch — swi=${swi.length} rust=${rust.length}`);
  process.exit(2);
}

function stableStringify(x) {
  if (x === null || typeof x !== "object") return JSON.stringify(x);
  if (Array.isArray(x)) return "[" + x.map(stableStringify).join(",") + "]";
  const keys = Object.keys(x).sort();
  return "{" + keys.map((k) => JSON.stringify(k) + ":" + stableStringify(x[k])).join(",") + "}";
}

let divergences = 0;
for (let i = 0; i < swi.length; i += 1) {
  const row = JSON.parse(swi[i]);
  const got = JSON.parse(rust[i]);
  const { id: _id, ...gotRest } = got;
  const exp = row.expected;
  if (stableStringify(gotRest) !== stableStringify(exp)) {
    divergences += 1;
    console.error("DIVERGE", row.id, "(" + row.query + ")");
    console.error("  expected", JSON.stringify(exp));
    console.error("  got     ", JSON.stringify(gotRest));
  } else {
    console.log("ok", row.id);
  }
}

if (divergences !== 0) {
  console.error(`corpus-under-rust: ${divergences} divergences / ${swi.length}`);
  process.exit(1);
}
console.log(`corpus-under-rust: ${swi.length}/${swi.length} matched SWI`);
