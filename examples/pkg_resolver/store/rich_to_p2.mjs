#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// rich_to_p2.mjs -- compile a rich JSONL catalog dump into P/2 JSONL
// files (the D43 indexer input). Reverse-deps are precomputed here so
// dependents/upgrade_set is a seek, not a scan. Provides are keyed by
// virtual name (CatId|Virtual).
//
//   node examples/pkg_resolver/store/rich_to_p2.mjs <rich.jsonl> <out-dir>

import { createReadStream } from "node:fs";
import { mkdirSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { packKey, packVer, packDep, packConflict, packRev, packProvide } from "./pack.mjs";

const src = process.argv[2];
const outDir = process.argv[3];
if (!src || !outDir) {
  console.error("usage: rich_to_p2.mjs <rich.jsonl> <out-dir>");
  process.exit(2);
}

mkdirSync(outDir, { recursive: true });
const pkg = [];
const dep = [];
const conflict = [];
const rev = [];
const provide = [];

function pair(k, v) {
  return JSON.stringify([k, v]);
}

function altName(a) {
  return a.dep || a.name;
}

const rl = createInterface({ input: createReadStream(src), crlfDelay: Infinity });
for await (const line of rl) {
  if (!line) continue;
  const row = JSON.parse(line);
  const cat = row.catalog || "default";
  const kind = row.kind;
  if (kind === "package") {
    pkg.push(pair(packKey(cat, row.name), packVer(row.ver)));
  } else if (kind === "depends") {
    dep.push(pair(packKey(cat, row.name), packDep(row.ver, row.dep, row.constraint)));
    if (row.dep && typeof row.dep === "object" && row.dep.alternatives) {
      for (const a of row.dep.alternatives) {
        rev.push(pair(packKey(cat, altName(a)), packRev(row.name, row.ver, a.constraint)));
      }
    } else {
      rev.push(pair(packKey(cat, row.dep), packRev(row.name, row.ver, row.constraint)));
    }
  } else if (kind === "conflicts") {
    conflict.push(pair(packKey(cat, row.name), packConflict(row.ver, row.other)));
  } else if (kind === "provides") {
    provide.push(pair(
      packKey(cat, row.virtual),
      packProvide(row.name, row.ver, row.virtual_ver)
    ));
  }
}

writeFileSync(outDir + "/pkg.jsonl", pkg.join("\n") + (pkg.length ? "\n" : ""));
writeFileSync(outDir + "/dep.jsonl", dep.join("\n") + (dep.length ? "\n" : ""));
writeFileSync(outDir + "/conflict.jsonl", conflict.join("\n") + (conflict.length ? "\n" : ""));
writeFileSync(outDir + "/revdep.jsonl", rev.join("\n") + (rev.length ? "\n" : ""));
writeFileSync(outDir + "/provide.jsonl", provide.join("\n") + (provide.length ? "\n" : ""));
process.stdout.write(
  "rich_to_p2: pkg=" + pkg.length + " dep=" + dep.length +
  " conflict=" + conflict.length + " revdep=" + rev.length +
  " provide=" + provide.length + " -> " + outDir + "\n"
);
