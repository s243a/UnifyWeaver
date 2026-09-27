// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// compare_jsonl.mjs -- the semantic comparer for the differential harness.
//
//   node compare_jsonl.mjs <cases.txt> <oracle.jsonl> <prolog.jsonl>
//
// Compares the two runners' results line-for-line by CONTENT, not by text:
// `positional` is an ordered array, `flags` is an unordered object (key order
// inside "flags" is irrelevant -- JS object key order and the Prolog list order
// need not agree), booleans stay distinct from the strings "true"/"false".
//
// Reports two counts separately:
//   divergences         - class mismatch (ok vs error) or differing ok contents
//   message mismatches  - both sides errored, but with different messages
// Exit status is non-zero if either count is non-zero.

import { readFileSync } from "node:fs";

const [, , casesPath, oraclePath, prologPath] = process.argv;
if (!casesPath || !oraclePath || !prologPath) {
  console.error("usage: compare_jsonl.mjs <cases.txt> <oracle.jsonl> <prolog.jsonl>");
  process.exit(2);
}

const readLines = (p) => readFileSync(p, "utf8").split("\n").filter((l) => l !== "");
const tokenize = (line) => line.split(" ").filter(Boolean);

const cases = readFileSync(casesPath, "utf8")
  .split("\n")
  .filter((l) => tokenize(l).length > 0);
const oracle = readLines(oraclePath);
const prolog = readLines(prologPath);

if (oracle.length !== cases.length || prolog.length !== cases.length) {
  console.error(
    `FATAL: line-count mismatch — cases=${cases.length} oracle=${oracle.length} prolog=${prolog.length}`,
  );
  process.exit(2);
}

const sameFlags = (a, b) => {
  const ka = Object.keys(a).sort();
  const kb = Object.keys(b).sort();
  if (ka.length !== kb.length) return false;
  for (let i = 0; i < ka.length; i += 1) {
    if (ka[i] !== kb[i]) return false;
    const va = a[ka[i]];
    const vb = b[kb[i]];
    // strict: true !== "true", false !== "false"
    if (typeof va !== typeof vb || va !== vb) return false;
  }
  return true;
};

const sameArray = (a, b) =>
  a.length === b.length && a.every((x, i) => x === b[i]);

let divergences = 0;
let messageMismatches = 0;
let crashes = 0;
const samples = [];

for (let i = 0; i < cases.length; i += 1) {
  const o = JSON.parse(oracle[i]);
  const p = JSON.parse(prolog[i]);

  if (o.crash !== undefined || p.crash !== undefined) {
    crashes += 1;
    if (samples.length < 10) samples.push({ line: cases[i], oracle: o, prolog: p, why: "crash" });
    continue;
  }

  const oIsOk = o.ok !== undefined;
  const pIsOk = p.ok !== undefined;

  if (oIsOk !== pIsOk) {
    divergences += 1;
    if (samples.length < 10) samples.push({ line: cases[i], oracle: o, prolog: p, why: "class" });
    continue;
  }

  if (oIsOk) {
    if (!sameArray(o.ok.positional, p.ok.positional) || !sameFlags(o.ok.flags, p.ok.flags)) {
      divergences += 1;
      if (samples.length < 10) samples.push({ line: cases[i], oracle: o, prolog: p, why: "content" });
    }
    continue;
  }

  if (o.error !== p.error) {
    messageMismatches += 1;
    if (samples.length < 10) samples.push({ line: cases[i], oracle: o, prolog: p, why: "message" });
  }
}

const okCount = oracle.filter((l) => l.startsWith('{"ok"')).length;

console.log(`sample size:          ${cases.length} argv-lines`);
console.log(`  oracle ok results:  ${okCount}`);
console.log(`  oracle errors:      ${cases.length - okCount}`);
console.log(`divergences:          ${divergences}`);
console.log(`message mismatches:   ${messageMismatches}`);
if (crashes) console.log(`crashes:              ${crashes}`);

if (samples.length) {
  console.log("\nfirst differing cases:");
  for (const s of samples) {
    console.log(`  [${s.why}] ${JSON.stringify(s.line)}`);
    console.log(`      oracle: ${JSON.stringify(s.oracle)}`);
    console.log(`      prolog: ${JSON.stringify(s.prolog)}`);
  }
}

process.exit(divergences === 0 && messageMismatches === 0 && crashes === 0 ? 0 : 1);
