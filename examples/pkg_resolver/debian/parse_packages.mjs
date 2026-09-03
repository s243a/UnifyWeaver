#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// parse_packages.mjs — Debian Packages control-stanza parser (ingestion edge).
//
//   node parse_packages.mjs <Packages> [--catalog ID] [--jsonl out] [--limit N]
//
// Covered fields:
//   Package, Version, Depends, Pre-Depends (treated as Depends; noted),
//   Provides (unversioned + versioned), Conflicts, Breaks (treated as
//   Conflicts; noted), Essential (base-candidate marker).
// Unknown fields are skipped loudly-once on stderr.
//
// Debian relation map: >= → gte, <= → lte, >> → gt, << → lt, = → eq.

import { readFileSync, writeFileSync } from "node:fs";

export const REL_MAP = {
  ">=": "gte",
  "<=": "lte",
  ">>": "gt",
  "<<": "lt",
  "=": "eq"
};

const KNOWN = new Set([
  "Package", "Version", "Depends", "Pre-Depends", "Provides",
  "Conflicts", "Breaks", "Essential", "Status", "Priority", "Section",
  "Installed-Size", "Maintainer", "Architecture", "Source", "Filename",
  "Size", "MD5sum", "SHA256", "SHA1", "Description", "Homepage",
  "Multi-Arch", "Recommends", "Suggests", "Enhances", "Replaces",
  "Built-Using", "Tag", "Task", "Important"
]);

const skipped = new Set();

export function parseDebVersion(str) {
  const s = String(str).trim();
  let epoch = 0;
  let rest = s;
  const ci = s.indexOf(":");
  if (ci > 0 && /^\d+$/.test(s.slice(0, ci))) {
    epoch = Number(s.slice(0, ci));
    rest = s.slice(ci + 1);
  }
  const hi = rest.lastIndexOf("-");
  let up;
  let rev;
  if (hi >= 0) {
    up = rest.slice(0, hi);
    rev = rest.slice(hi + 1);
  } else {
    up = rest;
    rev = "";
  }
  return { deb: [epoch, segment(up), segment(rev)] };
}

function segment(s) {
  const segs = [];
  let i = 0;
  while (i < s.length) {
    let order = "";
    while (i < s.length && (s[i] < "0" || s[i] > "9")) {
      order += s[i];
      i += 1;
    }
    let digits = "";
    while (i < s.length && s[i] >= "0" && s[i] <= "9") {
      digits += s[i];
      i += 1;
    }
    segs.push([order, digits === "" ? 0 : Number(digits)]);
  }
  return segs;
}

function splitTop(s, sep) {
  const out = [];
  let cur = "";
  let depth = 0;
  for (let i = 0; i < s.length; i += 1) {
    const ch = s[i];
    if (ch === "(") depth += 1;
    else if (ch === ")") depth -= 1;
    if (ch === sep && depth === 0) {
      if (cur.trim()) out.push(cur.trim());
      cur = "";
    } else {
      cur += ch;
    }
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

export function parseRelationAtom(raw) {
  const s = raw.trim().replace(/:[a-z0-9]+$/i, ""); // drop :any / :amd64
  const m = s.match(/^(\S+?)(?:\s*\(\s*(>=|<=|>>|<<|=)\s*([^)]+?)\s*\))?(?:\s*\[[^\]]*\])?$/);
  if (!m) return null;
  const name = m[1].replace(/:any$/, "");
  if (!m[2]) return { name, constraint: "any" };
  const op = REL_MAP[m[2]];
  if (!op) return { name, constraint: "any" };
  return { name, constraint: { op, v: parseDebVersion(m[3]) } };
}

export function parseDependsField(s) {
  if (!s || !String(s).trim()) return [];
  const groups = splitTop(String(s), ",");
  const out = [];
  for (const g of groups) {
    const alts = splitTop(g, "|").map(parseRelationAtom).filter(Boolean);
    if (alts.length === 0) continue;
    if (alts.length === 1) out.push(alts[0]);
    else {
      out.push({
        alternatives: alts.map((a) => ({ dep: a.name, constraint: a.constraint }))
      });
    }
  }
  return out;
}

export function parseProvidesField(s) {
  if (!s || !String(s).trim()) return [];
  return splitTop(String(s), ",").map(parseRelationAtom).filter(Boolean);
}

export function parseStanzas(text) {
  const stanzas = [];
  let cur = {};
  let last = null;
  const lines = String(text).split(/\r?\n/);
  for (const line of lines) {
    if (line === "") {
      if (cur.Package) stanzas.push(cur);
      cur = {};
      last = null;
      continue;
    }
    if (line[0] === " " || line[0] === "\t") {
      if (last) cur[last] = (cur[last] || "") + "\n" + line.slice(1);
      continue;
    }
    const colon = line.indexOf(":");
    if (colon < 0) continue;
    const field = line.slice(0, colon);
    const value = line.slice(colon + 1).trim();
    if (!KNOWN.has(field) && !skipped.has(field)) {
      skipped.add(field);
      process.stderr.write("parse_packages: skipping unknown field " + field + "\n");
    }
    cur[field] = value;
    last = field;
  }
  if (cur.Package) stanzas.push(cur);
  return stanzas;
}

export function stanzaToRows(st, catalog) {
  const rows = [];
  const name = st.Package;
  if (!name || !st.Version) return rows;
  const ver = parseDebVersion(st.Version);
  rows.push({ kind: "package", catalog, name, ver, version_raw: st.Version });
  const notes = [];
  const depFields = [];
  if (st.Depends) depFields.push(st.Depends);
  if (st["Pre-Depends"]) {
    depFields.push(st["Pre-Depends"]);
    notes.push("pre-depends-as-depends");
  }
  for (const field of depFields) {
    for (const d of parseDependsField(field)) {
      if (d.alternatives) {
        rows.push({
          kind: "depends", catalog, name, ver,
          dep: { alternatives: d.alternatives },
          constraint: "any"
        });
      } else {
        rows.push({
          kind: "depends", catalog, name, ver,
          dep: d.name, constraint: d.constraint
        });
      }
    }
  }
  if (st.Conflicts) {
    for (const c of parseProvidesField(st.Conflicts)) {
      rows.push({ kind: "conflicts", catalog, name, ver, other: c.name });
    }
  }
  if (st.Breaks) {
    notes.push("breaks-as-conflicts");
    for (const c of parseProvidesField(st.Breaks)) {
      rows.push({ kind: "conflicts", catalog, name, ver, other: c.name });
    }
  }
  if (st.Provides) {
    for (const p of parseProvidesField(st.Provides)) {
      const row = { kind: "provides", catalog, name, ver, virtual: p.name };
      if (p.constraint && p.constraint !== "any" && p.constraint.v) {
        row.virtual_ver = p.constraint.v;
      }
      rows.push(row);
    }
  }
  if (String(st.Essential || "").toLowerCase() === "yes") {
    rows.push({
      kind: "essential",
      catalog,
      name,
      ver,
      marker: "base-candidate"
    });
  }
  if (notes.length) {
    rows[0].notes = notes;
  }
  return rows;
}

export function parsePackagesText(text, catalog) {
  const stanzas = parseStanzas(text);
  const rows = [];
  for (const st of stanzas) rows.push(...stanzaToRows(st, catalog));
  return { stanzas, rows, skipped: [...skipped] };
}

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args[0] === "-h" || args[0] === "--help") {
    process.stderr.write("usage: parse_packages.mjs <Packages> [--catalog ID] [--jsonl out] [--limit N]\n");
    process.exit(args.length ? 0 : 2);
  }
  const src = args[0];
  let catalog = "debian";
  let jsonl = null;
  let limit = Infinity;
  for (let i = 1; i < args.length; i += 1) {
    if (args[i] === "--catalog") catalog = args[++i];
    else if (args[i] === "--jsonl") jsonl = args[++i];
    else if (args[i] === "--limit") limit = Number(args[++i]);
  }
  const text = src === "-" ? readFileSync(0, "utf8") : readFileSync(src, "utf8");
  const { stanzas, rows } = parsePackagesText(text, catalog);
  const kept = stanzas.slice(0, Number.isFinite(limit) ? limit : stanzas.length);
  const outRows = [];
  for (const st of kept) outRows.push(...stanzaToRows(st, catalog));
  const body = outRows.map((r) => JSON.stringify(r)).join("\n") + (outRows.length ? "\n" : "");
  if (jsonl) writeFileSync(jsonl, body);
  else process.stdout.write(body);
  process.stderr.write(
    "parse_packages: stanzas=" + stanzas.length +
    " emitted=" + kept.length +
    " rows=" + outRows.length + "\n"
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
