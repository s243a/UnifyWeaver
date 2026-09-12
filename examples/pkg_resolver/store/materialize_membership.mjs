#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// materialize_membership.mjs -- materialize a per-snapshot store_pkg table (or
// the (Name,Ver,from,to) validity-interval table) FROM the compact git-tree
// membership deltas (gen_snapshots.mjs' membership-git/delta-<t>.jsonl). The
// git tree is the source of truth; this is the derived, rebuildable query index.
//
//   # Replay deltas 0..t -> one snapshot's store_pkg P/2 (SnapId|Name -> Ver):
//   node materialize_membership.mjs --git=DIR/membership-git --snapshot=t \
//        --snapid=snap<t> --out=OUTDIR      # writes OUTDIR/pkg.jsonl
//
//   # Fold ALL deltas -> validity intervals (one row per version tenure):
//   node materialize_membership.mjs --git=DIR/membership-git --intervals \
//        --out=OUTDIR                        # writes OUTDIR/intervals.jsonl
//
// Prints the materialization time (ms) and row count on stderr.

import { readFileSync, mkdirSync, writeFileSync, existsSync } from "node:fs";
import { packKey, packVer } from "./pack.mjs";

const args = {};
for (const a of process.argv.slice(2)) {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/);
  if (m) args[m[1]] = m[2] === undefined ? true : m[2];
}
const GIT = args.git;
const OUT = args.out;
if (!GIT || !OUT) { console.error("usage: materialize_membership.mjs --git=DIR --out=DIR (--snapshot=t --snapid=ID | --intervals)"); process.exit(2); }
mkdirSync(OUT, { recursive: true });

function readDelta(t) {
  const p = GIT + "/delta-" + t + ".jsonl";
  if (!existsSync(p)) return [];
  return readFileSync(p, "utf8").split("\n").filter(Boolean).map(JSON.parse);
}
const vk = (v) => JSON.stringify(v);

if (args.intervals) {
  // Fold every delta once; emit a (Name, Ver, from, to) row per version tenure.
  const t0 = process.hrtime.bigint();
  let t = 0;
  const open = new Map();            // name -> { ver, from }
  const out = [];
  const close = (name, to) => { const o = open.get(name); if (o) { out.push([name, o.ver, o.from, to]); open.delete(name); } };
  for (;;) {
    const rows = readDelta(t);
    if (rows.length === 0 && t > 0 && !existsSync(GIT + "/delta-" + t + ".jsonl")) break;
    for (const r of rows) {
      if (r.op === "set") {
        const o = open.get(r.name);
        if (o && vk(o.ver) !== vk(r.ver)) close(r.name, t - 1);      // bump: close old tenure
        if (!open.has(r.name)) open.set(r.name, { ver: r.ver, from: t });
      } else if (r.op === "del") {
        close(r.name, t - 1);
      }
    }
    t++;
    if (!existsSync(GIT + "/delta-" + t + ".jsonl")) break;
  }
  const last = t - 1;
  for (const [name] of open) close(name, last);                       // to = last snapshot
  writeFileSync(OUT + "/intervals.jsonl", out.map((r) => JSON.stringify(r)).join("\n") + (out.length ? "\n" : ""));
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  console.error("materialize(intervals): snapshots=" + t + " tenures=" + out.length + " time=" + ms.toFixed(1) + "ms -> " + OUT + "/intervals.jsonl");
} else {
  // Replay deltas 0..T into the current membership, emit store_pkg P/2.
  const T = parseInt(args.snapshot, 10);
  const snapId = args.snapid || ("snap" + T);
  if (!(T >= 0)) { console.error("--snapshot=<t> required (>=0)"); process.exit(2); }
  const t0 = process.hrtime.bigint();
  const cur = new Map();             // name -> ver
  for (let t = 0; t <= T; t++) {
    for (const r of readDelta(t)) {
      if (r.op === "set") cur.set(r.name, r.ver);
      else if (r.op === "del") cur.delete(r.name);
    }
  }
  const pkg = [];
  for (const [name, ver] of cur) pkg.push(JSON.stringify([packKey(snapId, name), packVer(ver)]));
  writeFileSync(OUT + "/pkg.jsonl", pkg.join("\n") + (pkg.length ? "\n" : ""));
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  console.error("materialize(snapshot " + T + "): members=" + pkg.length + " time=" + ms.toFixed(1) + "ms -> " + OUT + "/pkg.jsonl (SnapId=" + snapId + ")");
}
