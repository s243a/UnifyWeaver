// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// diff_runner_patternjs.mjs -- the TRANSPILED side of the differential harness.
//
// Byte-for-byte the same protocol as examples/cli_args/diff_runner.mjs (the
// oracle side) and diff_runner.pl (the SWI side): read argv-lines on stdin, one
// case per line, tokens separated by spaces, skip lines with no tokens, and
// print one JSON object per line --
//
//   {"ok":{"positional":[...],"flags":{...}}}
//   {"error":"<CliError message>"}
//   {"crash":"<unexpected error>"}          // never expected; a hard failure
//
// The only difference from diff_runner.mjs is the import: `parseArgs` comes from
// the transpiled parser's edge shim instead of from the vendored oracle.
//
//   node examples/cli_args/patternjs/diff_runner_patternjs.mjs < lines.txt > patternjs.jsonl

import { parseArgs, CliError } from "./cliArgs.mjs";

const tokenize = (line) => line.split(" ").filter(Boolean);

function runLine(line) {
  try {
    const { positional, flags } = parseArgs(tokenize(line));
    return { ok: { positional, flags } };
  } catch (err) {
    if (err instanceof CliError) return { error: err.message };
    return { crash: String((err && err.message) || err) };
  }
}

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const input = Buffer.concat(chunks).toString("utf8");

const out = [];
for (const line of input.split("\n")) {
  if (tokenize(line).length === 0) continue;
  out.push(JSON.stringify(runLine(line)));
}
process.stdout.write(out.length ? out.join("\n") + "\n" : "");
