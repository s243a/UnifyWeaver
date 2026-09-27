// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// compare_corpus.mjs -- the per-contract-point reporter for the corpus gate.
//
//   node compare_corpus.mjs <map.tsv> <oracle.jsonl> <cljs.jsonl>
//
// map.tsv is `extract_corpus.mjs --map` output: `<test index>\t<test name>\t<argv line>`,
// one row per argv-line, in the same order as the two .jsonl files.
//
// A contract point passes when EVERY argv-line it exercises agrees between the
// oracle and the transpiled parser -- same class (ok vs error), same positional
// array, same flags (key order irrelevant, booleans strict, so `true` never
// matches the string "true"), and, for an error, the SAME MESSAGE. The message
// check is the one the import-swap corpus gets for free from its `assert.throws`
// regexes, so it is not optional here.

import { readFileSync } from "node:fs";

const [, , mapPath, oraclePath, cljsPath] = process.argv;
if (!mapPath || !oraclePath || !cljsPath) {
  console.error("usage: compare_corpus.mjs <map.tsv> <oracle.jsonl> <cljs.jsonl>");
  process.exit(2);
}

const readLines = (p) => readFileSync(p, "utf8").split("\n").filter((l) => l !== "");
const rows = readLines(mapPath).map((l) => {
  const [idx, name, ...rest] = l.split("\t");
  return { idx: Number(idx), name, line: rest.join("\t") };
});
const oracle = readLines(oraclePath);
const cljs = readLines(cljsPath);

if (oracle.length !== rows.length || cljs.length !== rows.length) {
  console.error(
    `FATAL: line-count mismatch — cases=${rows.length} oracle=${oracle.length} cljs=${cljs.length}`,
  );
  process.exit(2);
}

const sameFlags = (a, b) => {
  const ka = Object.keys(a).sort();
  const kb = Object.keys(b).sort();
  if (ka.length !== kb.length) return false;
  for (let i = 0; i < ka.length; i += 1) {
    if (ka[i] !== kb[i]) return false;
    if (typeof a[ka[i]] !== typeof b[kb[i]] || a[ka[i]] !== b[kb[i]]) return false;
  }
  return true;
};
const sameArray = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

const tests = new Map();
let lineFailures = 0;

for (let i = 0; i < rows.length; i += 1) {
  const r = rows[i];
  if (!tests.has(r.idx)) tests.set(r.idx, { name: r.name, ok: true, why: [] });
  const t = tests.get(r.idx);

  const o = JSON.parse(oracle[i]);
  const c = JSON.parse(cljs[i]);
  let why = null;

  if (o.crash !== undefined || c.crash !== undefined) why = "crash";
  else if ((o.ok !== undefined) !== (c.ok !== undefined)) why = "class";
  else if (o.ok !== undefined) {
    if (!sameArray(o.ok.positional, c.ok.positional)) why = "positional";
    else if (!sameFlags(o.ok.flags, c.ok.flags)) why = "flags";
  } else if (o.error !== c.error) why = "message";

  if (why) {
    lineFailures += 1;
    t.ok = false;
    t.why.push({ why, line: r.line, oracle: o, cljs: c });
  }
}

const passed = [...tests.values()].filter((t) => t.ok).length;
console.log(`contract points (test() blocks): ${passed} / ${tests.size}`);
console.log(`argv-lines compared:             ${rows.length - lineFailures} / ${rows.length}`);
console.log(`  (error MESSAGES compared exactly, not just the error class)`);

if (lineFailures) {
  console.log("\nfailing contract points:");
  for (const [idx, t] of tests) {
    if (t.ok) continue;
    console.log(`  [${idx}] ${t.name}`);
    for (const f of t.why) {
      console.log(`      [${f.why}] ${JSON.stringify(f.line)}`);
      console.log(`         oracle: ${JSON.stringify(f.oracle)}`);
      console.log(`         cljs:   ${JSON.stringify(f.cljs)}`);
    }
  }
}

process.exit(lineFailures === 0 ? 0 : 1);
