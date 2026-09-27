#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// diff_runner_wamjs.mjs -- JS-WAM side of the pkg_resolver differential.
// Reads the same JSONL as diff_runner.pl; writes one result object per line.
//
//   node examples/pkg_resolver/wamjs/diff_runner_wamjs.mjs < cases.jsonl

import { runCase } from "./resolver.mjs";

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const input = Buffer.concat(chunks).toString("utf8");

for (const line of input.split("\n")) {
  if (line === "") continue;
  const row = JSON.parse(line);
  let got;
  try {
    got = runCase(row);
  } catch (err) {
    got = { crash: String((err && err.stack) || err) };
  }
  process.stdout.write(JSON.stringify({ id: row.id, ...got }) + "\n");
}
