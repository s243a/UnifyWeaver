#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// resolve_auto.mjs -- size-gated auto-router for the Rust WAM pkg resolver.
//
// The Rust target ships TWO resolve lanes that give byte-identical selections
// (D70/D73): the term lane (examples/pkg_resolver/rust, loads the whole catalog
// as WAM `Value` terms) and the store lane (examples/pkg_resolver/rust_store,
// serves package/dep/conflict/revdep/provides from the D43 indexed seek store
// and reads only the few KB a query touches). The store lane resolves the 5k
// catalog in ~40-50 ms vs ~400-535 ms for the term lane (~10x; D89/D91), but at
// small catalog sizes the two are within ~1.3x and the term lane needs no
// pre-built store. This driver picks the faster lane automatically from the
// catalog size, so large-catalog resolution gets the store win by DEFAULT while
// small catalogs stay on the simpler term path.
//
// WHAT IS IN HERE, exhaustively:
//   1. an ORDERED size->backend LADDER (a small table of {minPackages, backend})
//      with the backend as an enum/extension point, so a future LMDB tier slots
//      in as a third rung (term -> indexed-store -> lmdb) with no routing rework;
//   2. per-case routing: pick the highest rung whose threshold the catalog meets
//      AND whose backend is available, else demote to the next lower available
//      rung (never a failed/wrong resolve -- fall back to term if no store);
//   3. batched dispatch: cases are grouped by chosen backend and each backend
//      binary is run ONCE over its group, then results are merged back in input
//      order. Both binaries accept the SAME term-format case line (the store
//      shim's env_term falls back to the catalog's env fields), so routing does
//      not transform the payload -- it only chooses which binary reads it.
//
// There is NO resolver logic here (not one candidate ordering, constraint
// comparison, layer walk or selection rule) and NO catalog transform -- the
// case line handed to the term binary is byte-for-byte the line handed to the
// store binary. The selection is produced entirely by the compiled resolver.pl
// / resolver_store.pl behind whichever binary this driver spawns.
//
// Usage:
//   node resolve_auto.mjs [options] < cases.jsonl > results.jsonl
//
//   --threshold N        packages at/above which the indexed store is used
//                        (default 500; env UW_RESOLVE_STORE_THRESHOLD)
//   --backend B          force a lane: auto | term | indexed
//                        (default auto; env UW_RESOLVE_BACKEND)
//   --store-dir DIR      the pre-built store dir (must hold pkg.data) whose
//                        catalog the store binary was compiled against
//                        (env UW_RESOLVE_STORE_DIR)
//   --catalog-id ID      the store's catalog id; store keys are `CatId|Name`, so
//                        a case routed to the store must name the catalog whose
//                        facts were baked in. Discovered from <store-dir>/
//                        probe.json when omitted (env UW_RESOLVE_CATALOG_ID).
//                        The term lane ignores catalog_id (its catalog travels
//                        in the case), so this is pure routing metadata and does
//                        not change any selection.
//   --term-bin PATH      term binary (default rust/.../release/uw_resolve;
//                        env UW_RESOLVE_TERM_BIN)
//   --store-bin PATH     store binary (default rust_store/.../uw_resolve_store;
//                        env UW_RESOLVE_STORE_BIN)
//   --explain            print the routing decision for each case to stderr
//
// Exit: 0 on success; 2 on a forced backend that is unavailable.

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve as pathResolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

const DEFAULT_TERM_BIN = join(
  HERE,
  "rust",
  "uw_resolve_wam",
  "target",
  "release",
  "uw_resolve"
);
const DEFAULT_STORE_BIN = join(
  HERE,
  "rust_store",
  "uw_resolve_wam_store",
  "target",
  "release",
  "uw_resolve_store"
);

// ---------------------------------------------------------------------------
// Backend registry -- the extension point. Each backend knows its binary and
// how to decide whether it can serve a request right now. A future `lmdb`
// backend adds ONE entry here (bin + available()) and ONE ladder rung below;
// nothing in the routing loop changes.
// ---------------------------------------------------------------------------

function makeBackends(opts) {
  return {
    // The always-available baseline: the catalog travels in the case line.
    term: {
      name: "term",
      bin: opts.termBin,
      available() {
        return existsSync(this.bin);
      },
      unavailableReason() {
        return `term binary missing: ${this.bin} (run rust/build.sh)`;
      }
    },

    // The D43 indexed seek store. Needs the store binary AND a pre-built store
    // dir (pkg.data present) whose catalog the binary was compiled against.
    indexed: {
      name: "indexed",
      bin: opts.storeBin,
      available() {
        if (!existsSync(this.bin)) return false;
        // A store dir is how we assert a store was actually built; the binary
        // bakes the absolute store path at compile time, so its presence plus
        // pkg.data is our availability signal.
        if (opts.storeDir) return existsSync(join(opts.storeDir, "pkg.data"));
        // No dir given: trust the compiled-in store iff the binary exists.
        return true;
      },
      unavailableReason() {
        if (!existsSync(this.bin))
          return `store binary missing: ${this.bin} (run rust_store/build.sh)`;
        return `store dir has no pkg.data: ${opts.storeDir} (run store/build_stores.sh)`;
      }
    }

    // Future rung, deferred this round (LMDB fails loud on the Rust lane today):
    // lmdb: {
    //   name: "lmdb",
    //   bin: opts.storeBin,           // same binary, UW_STORE_BACKEND=lmdb build
    //   available() { return existsSync(join(opts.storeDir, "lmdb", "pkg")); },
    //   unavailableReason() { return "no lmdb store built"; }
    // }
  };
}

// ---------------------------------------------------------------------------
// The ladder -- ordered by minPackages ascending. The router walks it from the
// TOP rung down and takes the first rung whose threshold is met and whose
// backend is available. Add the lmdb rung here (above `indexed`) when it lands.
// ---------------------------------------------------------------------------

function makeLadder(threshold) {
  return [
    { minPackages: 0, backend: "term" },
    { minPackages: threshold, backend: "indexed" }
    // { minPackages: lmdbThreshold, backend: "lmdb" }   // future third rung
  ].sort((a, b) => a.minPackages - b.minPackages);
}

/** Choose a backend name for a catalog of `size` packages.
 *  Walks the ladder top-down: the highest rung whose threshold is met and whose
 *  backend is available wins; otherwise demote to the next lower available rung.
 *  `term` (rung 0) is always available when its binary exists, so this never
 *  returns a wrong/failed route as long as the term binary is present. */
function chooseBackend(size, ladder, backends) {
  for (let i = ladder.length - 1; i >= 0; i--) {
    const rung = ladder[i];
    if (size < rung.minPackages) continue;
    const b = backends[rung.backend];
    if (b && b.available()) return { name: rung.backend, demotedFrom: null };
    // rung matched by size but backend unavailable -> keep walking down,
    // remembering what we skipped so --explain can report the demotion.
    for (let j = i - 1; j >= 0; j--) {
      const lower = ladder[j];
      if (size < lower.minPackages) continue;
      const lb = backends[lower.backend];
      if (lb && lb.available())
        return { name: lower.backend, demotedFrom: rung.backend };
    }
  }
  return { name: "term", demotedFrom: null };
}

// ---------------------------------------------------------------------------
// arg / env parsing
// ---------------------------------------------------------------------------

function parseOpts(argv) {
  const o = {
    threshold: numEnv("UW_RESOLVE_STORE_THRESHOLD", 500),
    backend: process.env.UW_RESOLVE_BACKEND || "auto",
    storeDir: process.env.UW_RESOLVE_STORE_DIR || "",
    catalogId: process.env.UW_RESOLVE_CATALOG_ID || "",
    termBin: process.env.UW_RESOLVE_TERM_BIN || DEFAULT_TERM_BIN,
    storeBin: process.env.UW_RESOLVE_STORE_BIN || DEFAULT_STORE_BIN,
    explain: false
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === "--threshold") o.threshold = Number(next());
    else if (a === "--backend") o.backend = next();
    else if (a === "--store-dir") o.storeDir = pathResolve(next());
    else if (a === "--catalog-id") o.catalogId = next();
    else if (a === "--term-bin") o.termBin = pathResolve(next());
    else if (a === "--store-bin") o.storeBin = pathResolve(next());
    else if (a === "--explain") o.explain = true;
    else if (a === "--help" || a === "-h") o.help = true;
    else throw new Error(`resolve_auto: unknown option ${a}`);
  }
  if (!["auto", "term", "indexed", "lmdb"].includes(o.backend))
    throw new Error(`resolve_auto: bad --backend ${o.backend} (auto|term|indexed)`);
  if (!Number.isFinite(o.threshold) || o.threshold < 0)
    throw new Error(`resolve_auto: bad --threshold ${o.threshold}`);
  return o;
}

function numEnv(name, dflt) {
  const v = process.env[name];
  if (v === undefined || v === "") return dflt;
  const n = Number(v);
  return Number.isFinite(n) ? n : dflt;
}

/** The store keys facts as `CatId|Name`, so a case routed to a store backend
 *  must name the catalog whose facts were baked in. Resolve it from (in order):
 *  an explicit --catalog-id / env, then <store-dir>/probe.json's catalog_id.
 *  Returns "" when unknown (then a case's own catalog_id, if any, is kept). */
function resolveCatalogId(opts) {
  if (opts.catalogId) return opts.catalogId;
  if (opts.storeDir) {
    const probe = join(opts.storeDir, "probe.json");
    if (existsSync(probe)) {
      try {
        const p = JSON.parse(readFileSync(probe, "utf8"));
        const id = p.catalog_id || (p.env && p.env.catalog_id);
        if (id) return id;
      } catch {
        /* fall through */
      }
    }
  }
  return "";
}

/** Ensure a store-routed case names the baked catalog. The term lane never reads
 *  catalog_id, so this only affects which store records a store lookup hits; a
 *  case that already carries a catalog_id (top-level or in env) is left as-is. */
function withCatalogId(line, catalogId) {
  if (!catalogId) return line;
  let obj;
  try {
    obj = JSON.parse(line);
  } catch {
    return line;
  }
  if (obj.catalog_id !== undefined) return line;
  if (obj.env && obj.env.catalog_id !== undefined) return line;
  obj.catalog_id = catalogId;
  return JSON.stringify(obj);
}

function catalogSize(caseObj) {
  const cat = caseObj && caseObj.catalog;
  if (cat && Array.isArray(cat.packages)) return cat.packages.length;
  // A store-format case (env only, no catalog) carries no size signal; treat as
  // 0 so it takes rung 0 unless a backend is forced.
  return 0;
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

function runBackend(backend, lines) {
  const res = spawnSync(backend.bin, [], {
    input: lines.join("\n") + (lines.length ? "\n" : ""),
    maxBuffer: 1 << 30,
    encoding: "utf8"
  });
  if (res.status !== 0) {
    throw new Error(
      `resolve_auto: ${backend.name} binary (${backend.bin}) exited ${res.status}\n${res.stderr || ""}`
    );
  }
  return res.stdout.split("\n").filter((l) => l.trim() !== "");
}

function main() {
  const opts = parseOpts(process.argv.slice(2));
  if (opts.help) {
    process.stdout.write(usage());
    return;
  }
  const backends = makeBackends(opts);
  const ladder = makeLadder(opts.threshold);

  // If a backend is forced, honour it (but refuse a forced backend that is
  // unavailable rather than silently swapping -- callers who force `indexed`
  // want the store, not a quiet term fallback).
  const forced = opts.backend === "auto" ? null : opts.backend;
  if (forced) {
    const b = backends[forced];
    if (!b) throw new Error(`resolve_auto: no such backend ${forced}`);
    if (!b.available()) {
      process.stderr.write(`resolve_auto: forced backend ${forced} unavailable: ${b.unavailableReason()}\n`);
      process.exit(2);
    }
  }

  const catalogId = resolveCatalogId(opts);

  const input = readFileSync(0, "utf8");
  const lines = input.split("\n").filter((l) => l.trim() !== "");

  // Assign each case (by input index) to a backend.
  const assignment = new Array(lines.length); // backend name per line
  const payload = new Array(lines.length); // (possibly catalog_id-stamped) line
  const groups = {}; // backend name -> [indices]
  for (let i = 0; i < lines.length; i++) {
    let obj;
    try {
      obj = JSON.parse(lines[i]);
    } catch {
      obj = null;
    }
    let name, demotedFrom = null;
    if (forced) {
      name = forced;
    } else {
      const size = catalogSize(obj);
      const pick = chooseBackend(size, ladder, backends);
      name = pick.name;
      demotedFrom = pick.demotedFrom;
      if (opts.explain) {
        const id = obj && obj.id !== undefined ? JSON.stringify(obj.id) : `#${i}`;
        const note = demotedFrom ? ` (demoted from ${demotedFrom}: ${backends[demotedFrom].unavailableReason()})` : "";
        process.stderr.write(`resolve_auto: case ${id} size=${size} threshold=${opts.threshold} -> ${name}${note}\n`);
      }
    }
    assignment[i] = name;
    // Term forwards the case verbatim; a store backend needs the baked catalog's
    // id so its `CatId|Name` seeks land on the right records.
    payload[i] = name === "term" ? lines[i] : withCatalogId(lines[i], catalogId);
    (groups[name] = groups[name] || []).push(i);
  }

  // Run each backend once over its group; scatter results back by index.
  const out = new Array(lines.length);
  for (const name of Object.keys(groups)) {
    const idxs = groups[name];
    const backend = backends[name];
    const results = runBackend(backend, idxs.map((i) => payload[i]));
    if (results.length !== idxs.length) {
      throw new Error(
        `resolve_auto: ${name} returned ${results.length} lines for ${idxs.length} cases`
      );
    }
    for (let k = 0; k < idxs.length; k++) out[idxs[k]] = results[k];
  }

  process.stdout.write(out.length ? out.join("\n") + "\n" : "");
}

function usage() {
  return [
    "usage: resolve_auto.mjs [options] < cases.jsonl > results.jsonl",
    "",
    "  --threshold N     packages at/above which the indexed store is used (default 500)",
    "  --backend B       auto | term | indexed  (default auto)",
    "  --store-dir DIR   pre-built store dir (must hold pkg.data)",
    "  --term-bin PATH   term binary path",
    "  --store-bin PATH  store binary path",
    "  --explain         print routing decisions to stderr",
    ""
  ].join("\n");
}

main();
