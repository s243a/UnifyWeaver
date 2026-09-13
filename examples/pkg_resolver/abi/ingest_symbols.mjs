#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// ingest_symbols.mjs -- ingest symbol-level ABI metadata into the interval
// store as P/2 JSONL rows, mirroring the pkg_resolver store shape one level
// down (symbol tenures instead of package tenures). Three ingest tiers:
//
//   symbols-file <path>   parse a Debian/Ubuntu `.symbols` control member ->
//                         symprov interval rows. ZERO binary download; the
//                         maintainer already computed the [sym, intro] table.
//   elf <lib.so>          readelf fallback for a `.so` with no `.symbols`:
//                         exported versioned symbols -> symprov rows
//                         (intro = the symbol's own version tag).
//   requires <binary>     readelf: the binary's referenced versioned symbols
//                         -> symreq rows (attributed to their soname via the
//                         verneed/.gnu.version_r table) + NEEDED sonames.
//   symbols-dir <dir>     batch: ingest every *.symbols under <dir>.
//
// Store rows are P/2 pairs `[key, value]` (the D43 indexer / load_p2_jsonl
// shape). Intervals are `intro#inf` within a soname (to="inf").
//
//   symprov.jsonl : ["<soname>|<sym>", "<intro>#inf"]
//   symreq.jsonl  : ["<binary>|<sym>", "<soname>#<ver>"]
//   needed.jsonl  : ["<binary>", "<soname>"]
//
// Usage:
//   node ingest_symbols.mjs symbols-file <path>  [--out DIR] [--stdout]
//   node ingest_symbols.mjs elf         <lib.so> [--out DIR] [--stdout]
//   node ingest_symbols.mjs requires    <binary> [--out DIR] [--stdout]
//   node ingest_symbols.mjs symbols-dir <dir>    [--out DIR]

import { execFileSync } from "node:child_process";
import { readFileSync, appendFileSync, writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";

function readelf(args, file) {
  try {
    return execFileSync("readelf", [...args, file], { encoding: "utf8", maxBuffer: 1 << 26 });
  } catch {
    return "";
  }
}

function pair(k, v) {
  return JSON.stringify([k, v]);
}

// A versioned symbol's numeric intro version, extracted from a version tag such
// as GLIBC_2.34 / LIBSELINUX_1.0 -> "2.34" / "1.0". Non-numeric namespaces
// (e.g. GLIBC_PRIVATE, @Base) return null so they never set a bound.
function verNum(ver) {
  const m = String(ver).match(/_([0-9][0-9.]*)$/);
  return m ? m[1] : null;
}

// ---------------------------------------------------------------------------
// Tier 1: parse a Debian/Ubuntu `.symbols` control member.
// ---------------------------------------------------------------------------
// Format (per soname block):
//   <soname> <package> #MINVER#            <- header (no leading space)
//   | libc6 (>> 2.35), libc6 (<< 2.36)     <- alt-dep template (skip)
//   * Build-Depends-Package: libc6-dev     <- meta field (skip)
//    <symbol>@<version> <minimum-version> [<id>]   <- symbol (leading space)
// A symbol line may carry leading `(tag=value|...)` selectors which we strip.
// The `minimum-version` field IS the symbol's introduced version (our intro).
function parseSymbolsFile(path) {
  const text = readFileSync(path, "utf8");
  let soname = null;
  const rows = [];               // {soname, sym, intro}
  const sonames = new Set();
  let symCount = 0, skipped = 0;
  for (const raw of text.split("\n")) {
    if (!raw) continue;
    // Header / meta lines are NOT indented.
    if (!/^[ \t]/.test(raw)) {
      const c = raw[0];
      if (c === "|" || c === "*" || c === "#") continue;   // alt-dep / meta / comment
      soname = raw.trim().split(/\s+/)[0];
      if (soname) sonames.add(soname);
      continue;
    }
    if (!soname) continue;
    let line = raw.trim();
    if (!line || line[0] === "|" || line[0] === "*" || line[0] === "#") continue;
    // Strip a leading (tag=value|...) selector group, e.g. "(optional)sym@Base".
    if (line[0] === "(") {
      const close = line.indexOf(")");
      if (close >= 0) line = line.slice(close + 1).trimStart();
    }
    // "<sym>@<version> <minimum-version> [id]" -- symbol names have no '@'.
    const at = line.indexOf("@");
    if (at < 0) { skipped++; continue; }
    const sym = line.slice(0, at);
    const rest = line.slice(at + 1).split(/\s+/);
    const version = rest[0];            // the symver tag, e.g. GLIBC_2.34 or Base
    const minver = rest[1];             // the introduced-version field
    if (!sym || minver === undefined) { skipped++; continue; }
    // intro = the curated minimum-version field. "0" (private) stays 0.0.0.
    const intro = /^[0-9]/.test(minver) ? minver : (verNum(version) || "0");
    rows.push({ soname, sym, intro });
    symCount++;
  }
  return { rows, sonames: [...sonames], symCount, skipped };
}

// ---------------------------------------------------------------------------
// readelf helpers (Tiers 2 & 3).
// ---------------------------------------------------------------------------
// Defined versioned syms = provides; UND versioned syms = requires.
function dynsyms(file) {
  const out = readelf(["-W", "--dyn-syms"], file);
  const defined = [], undef = [];
  for (const raw of out.split("\n")) {
    const t = raw.trim().split(/\s+/);
    if (t.length < 8 || !/^\d+:$/.test(t[0])) continue;
    const ndx = t[6], name = t[7];
    if (!name.includes("@")) continue;
    const [sym, ver] = name.replace("@@", "@").split("@");
    if (!sym || !ver) continue;
    (ndx === "UND" ? undef : defined).push([sym, ver]);
  }
  return { defined, undef };
}

function dynamic(file) {
  const out = readelf(["-d"], file);
  let soname = null;
  const needed = [];
  for (const raw of out.split("\n")) {
    let m;
    if ((m = raw.match(/\(SONAME\)\s+Library soname: \[([^\]]+)\]/))) soname = m[1];
    else if ((m = raw.match(/\(NEEDED\)\s+Shared library: \[([^\]]+)\]/))) needed.push(m[1]);
  }
  return { soname, needed };
}

// Map each version tag (GLIBC_2.34) to the soname (File) that provides it,
// from the `.gnu.version_r` (verneed) table.
function verneedMap(file) {
  const out = readelf(["-V"], file);
  const map = new Map();          // versionName -> soname
  let inNeeds = false, curFile = null;
  for (const raw of out.split("\n")) {
    if (/Version needs section/.test(raw)) { inNeeds = true; continue; }
    if (inNeeds && /Version (definition|symbols) section/.test(raw)) inNeeds = false;
    if (!inNeeds) continue;
    let m;
    if ((m = raw.match(/File:\s+(\S+)/))) curFile = m[1];
    if ((m = raw.match(/Name:\s+(\S+)/)) && curFile) map.set(m[1], curFile);
  }
  return map;
}

// ---------------------------------------------------------------------------
// Output sink.
// ---------------------------------------------------------------------------
function makeSink(outDir, toStdout) {
  const buffers = { symprov: [], symreq: [], needed: [] };
  return {
    add(store, k, v) { buffers[store].push(pair(k, v)); },
    flush(append) {
      if (outDir) mkdirSync(outDir, { recursive: true });
      for (const store of Object.keys(buffers)) {
        const lines = buffers[store];
        if (!lines.length) continue;
        if (outDir) {
          const f = join(outDir, store + ".jsonl");
          const body = lines.join("\n") + "\n";
          if (append) appendFileSync(f, body);
          else writeFileSync(f, body);
        }
        if (toStdout) for (const l of lines) process.stdout.write(l + "\n");
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Commands.
// ---------------------------------------------------------------------------
function parseArgs(rest) {
  const opts = { out: null, stdout: false, append: false, positional: [] };
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    if (a === "--out") opts.out = rest[++i];
    else if (a === "--stdout") opts.stdout = true;
    else if (a === "--append") opts.append = true;
    else opts.positional.push(a);
  }
  return opts;
}

const [cmd, ...rest] = process.argv.slice(2);
const opts = parseArgs(rest);
// symbols-dir manages its own append semantics; other modes default to
// overwrite unless --append. If neither --out nor --stdout, echo to stdout.
if (!opts.out && !opts.stdout) opts.stdout = true;

if (cmd === "symbols-file") {
  const path = opts.positional[0];
  const { rows, sonames, symCount, skipped } = parseSymbolsFile(path);
  const sink = makeSink(opts.out, opts.stdout);
  for (const { soname, sym, intro } of rows) sink.add("symprov", soname + "|" + sym, intro + "#inf");
  sink.flush(opts.append);
  process.stderr.write(
    `symbols-file ${path}: symprov=${symCount} sonames=${sonames.length} [${sonames.slice(0, 6).join(", ")}${sonames.length > 6 ? ", ..." : ""}] skipped=${skipped}\n`
  );
} else if (cmd === "elf") {
  const path = opts.positional[0];
  const { defined } = dynsyms(path);
  const { soname } = dynamic(path);
  const so = soname || path;
  // intro(sym) = earliest version tag seen on the exported symbol.
  const intro = new Map();
  for (const [sym, ver] of defined) {
    const n = verNum(ver);
    if (!n) continue;
    if (!intro.has(sym) || cmpVer(n, intro.get(sym)) < 0) intro.set(sym, n);
  }
  const sink = makeSink(opts.out, opts.stdout);
  for (const [sym, v] of intro) sink.add("symprov", so + "|" + sym, v + "#inf");
  sink.flush(opts.append);
  process.stderr.write(`elf ${path}: symprov=${intro.size} soname=${so}\n`);
} else if (cmd === "requires") {
  const path = opts.positional[0];
  const { undef } = dynsyms(path);
  const { needed } = dynamic(path);
  const vmap = verneedMap(path);
  const sink = makeSink(opts.out, opts.stdout);
  let n = 0;
  for (const [sym, ver] of undef) {
    const num = verNum(ver);
    if (!num) continue;                        // ignore GLIBC_PRIVATE etc.
    const so = vmap.get(ver) || "?";           // soname from verneed
    sink.add("symreq", path + "|" + sym, so + "#" + num);
    n++;
  }
  for (const so of needed) sink.add("needed", path, so);
  sink.flush(opts.append);
  process.stderr.write(`requires ${path}: symreq=${n} needed=[${needed.join(", ")}]\n`);
} else if (cmd === "symbols-dir") {
  const dir = opts.positional[0];
  const files = readdirSync(dir).filter((f) => f.endsWith(".symbols")).map((f) => join(dir, f));
  if (opts.out) { mkdirSync(opts.out, { recursive: true }); writeFileSync(join(opts.out, "symprov.jsonl"), ""); }
  let total = 0, nfiles = 0;
  for (const f of files) {
    let parsed;
    try { parsed = parseSymbolsFile(f); } catch { continue; }
    const sink = makeSink(opts.out, false);
    for (const { soname, sym, intro } of parsed.rows) sink.add("symprov", soname + "|" + sym, intro + "#inf");
    sink.flush(true);            // append across all files
    total += parsed.symCount;
    nfiles++;
  }
  process.stderr.write(`symbols-dir ${dir}: files=${nfiles} symprov=${total}\n`);
} else {
  process.stderr.write("usage: ingest_symbols.mjs symbols-file|elf|requires|symbols-dir <path> [--out DIR] [--stdout] [--append]\n");
  process.exit(2);
}

function cmpVer(a, b) {
  const pa = a.split("."), pb = b.split(".");
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const x = +(pa[i] || 0), y = +(pb[i] || 0);
    if (x !== y) return x - y;
  }
  return 0;
}
