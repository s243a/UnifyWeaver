#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// scale_to_case.mjs -- turn the 5k-package scale catalog (rich.jsonl +
// probe.json from examples/pkg_resolver/store/gen_scale_catalog.mjs) into ONE
// case in the JSON form the Rust shim reads. Same catalog and same bound
// request scale_demo.pl uses for its SWI term-leg reference, so the B3
// numbers are comparable.
//
//   node scale_to_case.mjs <scale-dir> [maxPackages] > case.json
//
// The optional package cap keeps the first N package names (p0..pN-1) and the
// dependency edges between them, so the same probe can be run at several
// catalog sizes to plot the backend's scaling.

import { readFileSync } from "node:fs";

const dir = process.argv[2];
const cap = process.argv[3] ? Number(process.argv[3]) : Infinity;
if (!dir) {
  console.error("usage: scale_to_case.mjs <scale-dir> [maxPackages]");
  process.exit(2);
}

const keep = (name) => {
  if (cap === Infinity) return true;
  const m = /^p(\d+)$/.exec(name);
  return m ? Number(m[1]) < cap : true;
};

const probe = JSON.parse(readFileSync(dir + "/probe.json", "utf8"));
const packages = [];
const depends = [];
const conflicts = [];

for (const line of readFileSync(dir + "/rich.jsonl", "utf8").split("\n")) {
  if (line === "") continue;
  const row = JSON.parse(line);
  if (row.kind === "package") {
    if (keep(row.name)) packages.push([row.name, row.ver]);
  } else if (row.kind === "depends") {
    if (keep(row.name) && keep(row.dep)) {
      depends.push([row.name, row.ver, row.dep, row.constraint ?? "any"]);
    }
  } else if (row.kind === "conflicts") {
    if (keep(row.name) && keep(row.other)) {
      conflicts.push([row.name, row.ver, row.other]);
    }
  }
}

const env = probe.env || {};
const args = Array.isArray(probe.args) ? probe.args : [probe.args];

process.stdout.write(
  JSON.stringify({
    id: "scale5k",
    catalog: {
      packages,
      depends,
      conflicts,
      base: env.base || [],
      installed: env.installed || [],
      requested: env.requested || [],
      layers: env.layers || [],
      excluded: env.excluded || [],
      aliases: env.aliases || []
    },
    query: probe.query || "resolve_layered",
    args
  }) + "\n"
);
