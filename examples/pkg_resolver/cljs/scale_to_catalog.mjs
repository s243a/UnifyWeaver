#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// scale_to_catalog.mjs -- turn the 5k scale generator's rich.jsonl into the
// ONE catalog JSON document the ClojureScript resolver edge consumes (the same
// shape cli/generated/catalogs/*.json use).
//
// The base hold is [p0 0.0.0] and installed/requested are empty, matching
// store/scale_demo.pl's load_rich_catalog/2 exactly, so the SWI leg of B3 and
// the ClojureScript leg answer the same question.
//
//   node scale_to_catalog.mjs <scale-dir>   # writes <scale-dir>/catalog.json

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const dir = process.argv[2];
if (!dir) {
  console.error("usage: scale_to_catalog.mjs <scale-dir>");
  process.exit(2);
}

const packages = [];
const depends = [];
const provides = [];

for (const line of readFileSync(join(dir, "rich.jsonl"), "utf8").split("\n")) {
  if (line === "") continue;
  const r = JSON.parse(line);
  if (r.kind === "package") packages.push([r.name, r.ver]);
  else if (r.kind === "depends") depends.push([r.name, r.ver, r.dep, r.constraint]);
  else if (r.kind === "provides") {
    const row = [r.name, r.ver, r.virtual];
    if (r.virtual_ver != null) row.push(r.virtual_ver);
    provides.push(row);
  }
}

const catalog = {
  packages,
  depends,
  conflicts: [],
  base: [["p0", [0, 0, 0]]],
  installed: [],
  requested: [],
  provides
};

const out = join(dir, "catalog.json");
writeFileSync(out, JSON.stringify(catalog));
console.error(
  `scale_to_catalog: ${packages.length} packages, ${depends.length} depends -> ${out}`
);
