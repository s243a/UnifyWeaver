// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// extract_corpus.mjs -- pull the argv-lines the 17-test contract corpus
// exercises straight out of oracle/cliArgs.test.mjs, and print one per line.
//
//   node examples/cli_args/cljs/extract_corpus.mjs > corpus.txt
//
// WHY THIS EXISTS. The patternjs lane could run the corpus by changing ONE line
// of the vendored test file -- its `import`. That is impossible for nbb: the
// corpus is an ESM module driven by `node --test`, and the transpiled parser is
// a ClojureScript namespace in a different runtime. There is no import to swap.
//
// The honest equivalent is to drive the SAME argv-lines through both parsers and
// compare the results, error messages included -- which is what
// run_corpus_cljs.sh does. This script is the part that has to be right for that
// to mean anything: the lines must come from the corpus SOURCE, not from a
// hand-copied list that could drift away from it.
//
// The 17 `test(...)` blocks exercise 25 distinct argv-lines (several assert two
// or three spellings of the same contract point), in three source spellings:
//
//   of("<line>")                        the common case
//   for (const line of ["a", "b"])      test 1 & 2
//   parseArgs([...])                    test 9, whose PEM is a const
//
// All three are read below. The count is asserted so a corpus edit that this
// extractor cannot see fails loudly instead of silently shrinking the gate.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = join(HERE, "..", "oracle", "cliArgs.test.mjs");
const src = readFileSync(SRC, "utf8");

// `const pem = "...";` -- the one identifier the corpus passes to parseArgs.
const consts = {};
for (const m of src.matchAll(/^\s*const\s+([A-Za-z_$][\w$]*)\s*=\s*"([^"]*)"\s*;/gm)) {
  consts[m[1]] = m[2];
}

// Split the file into its `test("<name>", ... )` blocks so every argv-line can
// be attributed to the contract point that exercises it. That attribution is
// what makes an N/17 claim mean anything.
const blocks = [];
const testRe = /^test\(\s*"((?:[^"\\]|\\.)*)"/gm;
let m;
const starts = [];
while ((m = testRe.exec(src)) !== null) starts.push({ name: m[1], at: m.index });
for (let i = 0; i < starts.length; i += 1) {
  const end = i + 1 < starts.length ? starts[i + 1].at : src.length;
  blocks.push({ name: starts[i].name, body: src.slice(starts[i].at, end) });
}

function linesOf(body) {
  const found = [];
  // 1. of("<line>")
  for (const x of body.matchAll(/\bof\(\s*"([^"]*)"\s*\)/g)) found.push(x[1]);
  // 2. for (const line of [ "...", "..." ])
  for (const x of body.matchAll(/for\s*\(\s*const\s+line\s+of\s*\[([^\]]*)\]/g)) {
    for (const s of x[1].matchAll(/"([^"]*)"/g)) found.push(s[1]);
  }
  // 3. parseArgs([ ... ]) -- string literals and resolvable const identifiers.
  for (const x of body.matchAll(/\bparseArgs\(\s*\[([^\]]*)\]\s*\)/g)) {
    const toks = [];
    let ok = true;
    for (const raw of x[1].split(",")) {
      const t = raw.trim();
      if (t === "") continue;
      const q = t.match(/^"([^"]*)"$/);
      if (q) { toks.push(q[1]); continue; }
      if (Object.prototype.hasOwnProperty.call(consts, t)) { toks.push(consts[t]); continue; }
      ok = false;
      break;
    }
    if (ok && toks.length) found.push(toks.join(" "));
  }
  const seen = new Set();
  const out = [];
  for (const l of found) {
    const norm = l.split(" ").filter(Boolean).join(" ");
    if (!norm || seen.has(norm)) continue;
    seen.add(norm);
    out.push(norm);
  }
  return out;
}

const perTest = blocks.map((b) => ({ name: b.name, lines: linesOf(b.body) }));
const total = perTest.reduce((n, t) => n + t.lines.length, 0);

const EXPECTED_TESTS = 17;
const EXPECTED_LINES = 25;
if (perTest.length !== EXPECTED_TESTS) {
  console.error(`extract_corpus: expected ${EXPECTED_TESTS} test() blocks, found ${perTest.length}`);
  process.exit(2);
}
if (total !== EXPECTED_LINES) {
  console.error(`extract_corpus: expected ${EXPECTED_LINES} argv-lines, found ${total}`);
  process.exit(2);
}
for (const t of perTest) {
  if (t.lines.length === 0) {
    console.error(`extract_corpus: test ${JSON.stringify(t.name)} yielded no argv-line`);
    process.exit(2);
  }
}

// Default output: one argv-line per line, in corpus order -- the case file the
// two runners read. `--map` instead prints `<test index>\t<line>` so the corpus
// runner can report the result per contract point.
if (process.argv.includes("--map")) {
  const rows = [];
  perTest.forEach((t, i) => t.lines.forEach((l) => rows.push(`${i + 1}\t${t.name}\t${l}`)));
  process.stdout.write(rows.join("\n") + "\n");
} else {
  const rows = [];
  perTest.forEach((t) => t.lines.forEach((l) => rows.push(l)));
  process.stdout.write(rows.join("\n") + "\n");
}
