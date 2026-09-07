#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// build_sample.mjs — parse sample_packages → rich JSONL + P/2 + Prolog catalog.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { parsePackagesText, parseDebVersion } from "./parse_packages.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const CAT = "debian_slice";
const src = process.argv[2] || join(HERE, "sample_packages");
const outDir = process.argv[3] || HERE;

const text = readFileSync(src, "utf8");
const { stanzas, rows } = parsePackagesText(text, CAT);

mkdirSync(join(outDir, "stores"), { recursive: true });
writeFileSync(join(outDir, "sample.jsonl"), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");

const root = join(HERE, "../../..");
const richToP2 = join(HERE, "../store/rich_to_p2.mjs");
const r = spawnSync(process.execPath, [richToP2, join(outDir, "sample.jsonl"), join(outDir, "stores")], {
  cwd: root,
  encoding: "utf8"
});
if (r.status !== 0) {
  process.stderr.write(r.stderr || r.stdout || "rich_to_p2 failed\n");
  process.exit(r.status || 1);
}
process.stdout.write(r.stdout);

const packages = [];
const depends = [];
const conflicts = [];
const provides = [];
const essential = [];
for (const row of rows) {
  if (row.kind === "package") packages.push([row.name, row.ver]);
  else if (row.kind === "depends") depends.push([row.name, row.ver, row.dep, row.constraint]);
  else if (row.kind === "conflicts") conflicts.push([row.name, row.ver, row.other]);
  else if (row.kind === "provides") {
    if (row.virtual_ver) provides.push([row.name, row.ver, row.virtual, row.virtual_ver]);
    else provides.push([row.name, row.ver, row.virtual]);
  } else if (row.kind === "essential") essential.push([row.name, row.ver]);
}

const catalog = {
  packages, depends, conflicts, provides,
  base: [], installed: [], requested: [],
  layers: [], excluded: [], aliases: [],
  essential
};
writeFileSync(join(outDir, "sample_catalog.json"), JSON.stringify(catalog, null, 2));

function emitVer(v) {
  if (v && v.deb) {
    const [e, up, rev] = v.deb;
    return `deb(${e},[${(up || []).map(emitSeg).join(",")}],[${(rev || []).map(emitSeg).join(",")}])`;
  }
  return `v(${v[0]},${v[1]},${v[2]})`;
}
function emitSeg(seg) {
  const order = String(seg[0] || "");
  const codes = [...order].map((c) => c.charCodeAt(0)).join(",");
  return `s([${codes}],${seg[1] | 0})`;
}
function emitAtom(n) {
  return "'" + String(n).replace(/'/g, "''") + "'";
}
function emitC(c) {
  if (c === "any" || c == null) return "any";
  if (c.op === "range") return `range(${emitVer(c.lo)},${emitVer(c.hi)})`;
  return `${c.op}(${emitVer(c.v)})`;
}
function emitDepNeed(d) {
  if (d && typeof d === "object" && d.alternatives) {
    const alts = d.alternatives.map((a) => `dep(${emitAtom(a.dep)},${emitC(a.constraint)})`);
    return `alternatives([${alts.join(",")}])`;
  }
  return emitAtom(d);
}

const pkgT = packages.map(([n, v]) => `package(${emitAtom(n)},${emitVer(v)})`);
const depT = depends.map(([n, v, d, c]) =>
  `depends(${emitAtom(n)},${emitVer(v)},${emitDepNeed(d)},${emitC(c)})`);
const confT = conflicts.map(([n, v, o]) =>
  `conflicts(${emitAtom(n)},${emitVer(v)},${emitAtom(o)})`);
const prT = provides.map((row) => {
  if (row.length >= 4) {
    return `provides(${emitAtom(row[0])},${emitVer(row[1])},${emitAtom(row[2])},${emitVer(row[3])})`;
  }
  return `provides(${emitAtom(row[0])},${emitVer(row[1])},${emitAtom(row[2])})`;
});

const oldLibc = parseDebVersion("2.31-13+deb11u11");
const essHolds = essential.map(([n, v]) => {
  if (n === "libc6") return `base(${emitAtom(n)}-${emitVer(oldLibc)},blanket)`;
  return `base(${emitAtom(n)}-${emitVer(v)},footprint)`;
});
if (!essential.some(([n]) => n === "libc6")) {
  const libc = packages.find(([n]) => n === "libc6");
  if (libc) essHolds.push(`base('libc6'-${emitVer(oldLibc)},blanket)`);
}

const pl = `:- encoding(utf8).
% Generated from sample_packages — do not edit by hand.
:- module(sample_catalog, [sample_catalog/1, sample_layered_catalog/1, sample_stats/1]).

sample_stats(_{stanzas: ${stanzas.length}, packages: ${packages.length},
               depends: ${depends.length}, provides: ${provides.length},
               essential: ${essential.length}}).

sample_catalog(catalog(
    [${pkgT.join(",\n     ")}],
    [${depT.join(",\n     ")}],
    [${confT.join(",\n     ")}],
    [],
    [],
    [],
    [],
    [],
    [],
    [${prT.join(",\n     ")}])).

sample_layered_catalog(catalog(
    [${pkgT.join(",\n     ")}],
    [${depT.join(",\n     ")}],
    [${confT.join(",\n     ")}],
    [${essHolds.join(",\n     ")}],
    [],
    [],
    [],
    [],
    [],
    [${prT.join(",\n     ")}])).
`;
writeFileSync(join(outDir, "sample_catalog.pl"), pl);

writeFileSync(join(outDir, "PROVENANCE.txt"), [
  "Debian Packages slice for uw-resolve P3.",
  "Source: https://deb.debian.org/debian/dists/bookworm/main/binary-amd64/Packages.gz",
  "Snapshot Last-Modified: Sat, 11 Jul 2026 (deb.debian.org).",
  "Stanzas: " + stanzas.length,
  "Parser: examples/pkg_resolver/debian/parse_packages.mjs",
  "Notes: Pre-Depends treated as Depends; Breaks treated as Conflicts;",
  "       Essential:yes mapped to base-candidate (kind=essential / footprint holds).",
  ""
].join("\n"));

process.stdout.write(
  "build_sample: stanzas=" + stanzas.length +
  " packages=" + packages.length +
  " depends=" + depends.length +
  " provides=" + provides.length +
  " essential=" + essential.length + "\n"
);
