// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// diff_runner.mjs -- the ORACLE side of the differential harness.
//
// Protocol: reads argv-lines on stdin (one line per case, tokens separated by
// spaces; blank lines are skipped), runs peerhailer's `parseArgs` from the
// vendored oracle copy, and prints one JSON object per line:
//
//   {"ok":{"positional":[...],"flags":{...}}}
//   {"error":"<CliError message>"}
//   {"crash":"<unexpected error>"}          // never expected; a hard failure
//
// The Prolog side (diff_runner.pl) prints the identical shape.
//
//   node examples/cli_args/diff_runner.mjs < lines.txt > oracle.jsonl

import { parseArgs, CliError } from "./oracle/cliArgs.js";

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
  // Skip only lines that carry no tokens at all -- the exact rule diff_runner.pl
  // applies, so the two runners stay line-for-line aligned.
  if (tokenize(line).length === 0) continue;
  out.push(JSON.stringify(runLine(line)));
}
process.stdout.write(out.length ? out.join("\n") + "\n" : "");
