#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// build_membership_p2.mjs -- compile one snapshot membership JSONL
// (gen_snapshots.mjs output: {kind:"snapshot_member",snap,name,ver}) into the
// per-snapshot store_pkg P/2 JSONL, keyed SnapId|Name -> Ver. This is the ONLY
// per-snapshot table in the dedup store (the deps/conflicts/provides/revdeps
// live once in the shared pool, keyed PoolId|Name).
//
//   node build_membership_p2.mjs <membership/snap-<t>.jsonl> <out-dir> <SnapId>
// Writes <out-dir>/pkg.jsonl.

import { createReadStream, mkdirSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { packKey, packVer } from "./pack.mjs";

const src = process.argv[2];
const outDir = process.argv[3];
const snapId = process.argv[4];
if (!src || !outDir || !snapId) {
  console.error("usage: build_membership_p2.mjs <membership.jsonl> <out-dir> <SnapId>");
  process.exit(2);
}
mkdirSync(outDir, { recursive: true });

const pkg = [];
const rl = createInterface({ input: createReadStream(src), crlfDelay: Infinity });
for await (const line of rl) {
  if (!line) continue;
  const row = JSON.parse(line);
  if (row.kind !== "snapshot_member") continue;
  pkg.push(JSON.stringify([packKey(snapId, row.name), packVer(row.ver)]));
}
writeFileSync(outDir + "/pkg.jsonl", pkg.join("\n") + (pkg.length ? "\n" : ""));
process.stdout.write("build_membership_p2: snap=" + snapId + " members=" + pkg.length + " -> " + outDir + "/pkg.jsonl\n");
