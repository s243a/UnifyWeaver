#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// slice_packages.mjs — pick a 300–1000 stanza slice from a Packages file
// (Essential + libc-ish + awk providers + enough neighbors).

import { readFileSync, writeFileSync } from "node:fs";
import { parseStanzas } from "./parse_packages.mjs";

const src = process.argv[2];
const dest = process.argv[3];
const target = Number(process.argv[4] || 500);
if (!src || !dest) {
  process.stderr.write("usage: slice_packages.mjs <Packages> <out> [target=500]\n");
  process.exit(2);
}

const MUST = new Set([
  "libc6", "libc-bin", "libgcc-s1", "gcc-12-base", "mawk", "gawk",
  "dash", "bash", "coreutils", "dpkg", "perl-base", "tar", "gzip",
  "sed", "grep", "hostname", "util-linux", "debianutils", "base-files",
  "base-passwd", "bash-static", "awk"
]);

const text = readFileSync(src, "utf8");
const stanzas = parseStanzas(text);
const byName = new Map();
for (const st of stanzas) {
  if (!byName.has(st.Package)) byName.set(st.Package, st);
}

const picked = new Map();
function take(st) {
  if (!st || picked.has(st.Package)) return;
  picked.set(st.Package, st);
}

for (const st of stanzas) {
  if (String(st.Essential || "").toLowerCase() === "yes") take(st);
  if (MUST.has(st.Package)) take(st);
}

for (const st of stanzas) {
  if (picked.size >= target) break;
  const deps = String(st.Depends || "") + " " + String(st.Provides || "");
  if (/\blibc6\b/.test(deps) || /\bawk\b/.test(deps) || /\|/.test(st.Depends || "")) {
    take(st);
  }
}

for (const st of stanzas) {
  if (picked.size >= target) break;
  take(st);
}

const ordered = [];
const seen = new Set();
for (const st of stanzas) {
  if (picked.has(st.Package) && !seen.has(st.Package)) {
    ordered.push(st);
    seen.add(st.Package);
  }
}

function stanzaText(st) {
  const keys = Object.keys(st);
  return keys.map((k) => {
    const lines = String(st[k]).split("\n");
    return k + ": " + lines[0] + lines.slice(1).map((l) => "\n " + l).join("");
  }).join("\n") + "\n";
}

writeFileSync(dest, ordered.map(stanzaText).join("\n"));
process.stdout.write(
  "slice_packages: in=" + stanzas.length +
  " out=" + ordered.length +
  " essential=" + ordered.filter((s) => String(s.Essential || "").toLowerCase() === "yes").length +
  " -> " + dest + "\n"
);
