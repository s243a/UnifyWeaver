#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// gen_snap_cases.mjs -- emit a differential cases.jsonl for ONE snapshot's
// membership, used to prove interval-mode resolves == per-snapshot store_pkg
// replay resolves. Member names + versions are read from a replayed snapshot's
// pkg.jsonl (keys "<snapid>|Name" -> VerPacked); the SAME cases file is fed to
// both the store_pkg-path runner and the interval-mode runner (the env is
// identical, so any answer difference is a store/interval divergence).
//
//   node gen_snap_cases.mjs <rep-dir>/pkg.jsonl <snapid> > cases.jsonl
//
// Emits ~80 cases spanning resolve / resolve_layered / dependents /
// dependents_installed / explain_blocked / freeze_audit over that snapshot's
// members, with a fixed env (base + installed drawn from the members) so the
// base/installed/freeze_audit paths are all exercised.

import { readFileSync } from "node:fs";

const PKG = process.argv[2];
const SNAPID = process.argv[3];
if (!PKG || !SNAPID) {
  console.error("usage: gen_snap_cases.mjs <rep-dir>/pkg.jsonl <snapid>");
  process.exit(2);
}

// Parse "<snapid>|Name" -> Ver (JSON) from the replayed membership table.
const prefix = SNAPID + "|";
function unpackVer(s) {
  if (s.startsWith("d:")) {
    // d:E:up:rev  (deb) -> {deb:[E,[[order,n],...],[[order,n],...]]}
    const [, e, up, rev] = s.split(":");
    const segs = (part) =>
      part === "" ? [] : part.split(";").map((seg) => {
        const [order, n] = seg.split("|");
        return [order, parseInt(n, 10)];
      });
    return { deb: [parseInt(e, 10), segs(up), segs(rev)] };
  }
  const [a, b, c] = s.split(".").map((x) => parseInt(x, 10));
  return [a, b, c];
}

const members = [];
for (const line of readFileSync(PKG, "utf8").split("\n")) {
  if (!line.trim()) continue;
  const [key, verPacked] = JSON.parse(line);
  if (!key.startsWith(prefix)) continue;
  members.push({ name: key.slice(prefix.length), ver: unpackVer(verPacked) });
}
members.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));

// A deterministic subset for base / installed holds (a slice of the members).
const nBase = Math.min(8, members.length);
const nInst = Math.min(8, members.length);
const base = members.slice(0, nBase).map((m, i) =>
  [m.name, m.ver, i % 2 === 0 ? "blanket" : "footprint"]);
const installed = members.slice(nBase, nBase + nInst).map((m) => [m.name, m.ver]);

const env = {
  catalog_id: SNAPID,
  base,
  installed,
  requested: [],
  layers: [],
  excluded: [],
  aliases: [],
};

// Choose ~16 member names spread across the corpus for the per-name queries.
const N = Math.min(16, members.length);
const step = Math.max(1, Math.floor(members.length / N));
const picks = [];
for (let i = 0; i < members.length && picks.length < N; i += step) picks.push(members[i].name);

const cases = [];
const emit = (id, query, args) =>
  cases.push({ id, query, args, catalog_id: SNAPID, env });

// resolve / resolve_layered take an args LIST; dependents / dependents_installed
// / explain_blocked take a bare arg (matches store_diff_runner_snap.pl dispatch).
for (const name of picks) {
  emit("resolve_" + name, "resolve", [name]);
  emit("layered_" + name, "resolve_layered", [name]);
  emit("dependents_" + name, "dependents", name);
  emit("depinst_" + name, "dependents_installed", name);
  emit("blocked_" + name, "explain_blocked", name);
}
// A few freeze_audit cases (args ignored; exercises env.base holds).
for (let i = 0; i < 4; i++) emit("audit_" + i, "freeze_audit", []);

process.stdout.write(cases.map((c) => JSON.stringify(c)).join("\n") + "\n");
console.error(
  "gen_snap_cases: snapid=" + SNAPID + " members=" + members.length +
  " picks=" + picks.length + " cases=" + cases.length);
