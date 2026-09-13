#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// ingest_symbols.mjs -- ingest symbol-level ABI evidence into the store as P/2
// JSONL rows (`[key, value]`, the load_p2_jsonl shape).
//
// DESIGN (post-review redesign; see REVIEW_NOTES.md):
//
//   * Identity is the exact (soname, symbol, version-node) triple. A version
//     node (`GLIBC_2.34`, `LIBSELINUX_1.0`, `COMMON_1`, `PUBLIC`, `Base`) is an
//     opaque ELF label matched by string equality -- never parsed into a number,
//     never collapsed to the bare symbol name. `Base` is dpkg's spelling of "no
//     version" (an unversioned export); ELF unversioned exports are recorded
//     under that same spelling so the two provider tiers agree.
//   * The Debian PACKAGE-version axis (the `.symbols` minimum-version field,
//     the release candidates) is a separate axis, carried verbatim as the deb
//     version string and parsed/ordered on the Prolog side by the frozen
//     resolver's deb/3 machinery (debian/deb_parse.pl + resolver:version_lt/2).
//   * Every obligation is preserved: versioned requirements with non-numeric
//     nodes, unversioned requirements, weak references (flagged, not dropped).
//   * Requirements are attributed to their soname through the per-symbol
//     version INDEX (.gnu.version -> .gnu.version_r), not through the version
//     NAME, so two libraries sharing a node name (COMMON_1) never collide.
//   * Evidence completeness is explicit: every successful ingest emits an
//     `evidence` row; a missing / unreadable ELF is a loud failure (exit 3)
//     that -- when --out is given -- records a failure evidence row instead
//     of an empty "success". A `.symbols` file is ingested atomically: one
//     unsupported row, one unknown tag, one block whose evidence release is
//     unknown, or a bad --release rejects the WHOLE file (also in batch
//     mode), nothing partial is written, and the run exits 3 (Sol P2).
//   * Default-version binding is retained (Sol P1b): every provider row says
//     whether an UNVERSIONED reference binds to it. From readelf that is the
//     `.gnu.version` hidden bit (`@` = hidden = "nondefault", `@@` =
//     "default"); the loader additionally binds legacy unversioned references
//     to the OLDEST version node (verdef index 2) even when hidden, so index 2
//     is recorded as "default" too (verified with the loader, fixture D10c).
//     A `.symbols` file carries no `@@` information: its rows are "unproven"
//     unless cross-checked against the ELF with --elf, and the resolver then
//     answers unknown -- never compatible -- for an unversioned reference.
//     `Base` (unversioned export) always binds.
//
// Store rows (all JSON arrays `[key, value]`):
//   symprov.jsonl  ["<soname>|<sym>@<node>", ["since", "<debver>", "<evidence-release>", <binding>]]  (.symbols)
//                  ["<soname>|<sym>@<node>", ["at", "<evidence-release>", <binding>]]                (readelf)
//                  <binding> = "default" | "nondefault" | "unproven"
//   symreq.jsonl   ["<binary>|<sym>@<node>", ["<soname>", "GLOBAL"|"WEAK"]]
//                  ["<binary>|<sym>",        ["", "GLOBAL"|"WEAK"]]   (unversioned)
//   needed.jsonl   ["<binary>", "<soname>"]
//   evidence.jsonl ["provides|<soname>", ["symbols"|"elf", "<release-id>", "complete", "<source>"]]
//                  ["requires|<binary>", ["readelf", "complete"|"missing_file"|"readelf_failed"|"inconsistent", "<detail>"]]
//   releases.jsonl ["<soname>", "<debver>"]                            (candidate axis)
//   replaces.jsonl ["<new-soname>", "<old-soname>"]                    (declared soname succession)
//
// Usage:
//   node ingest_symbols.mjs symbols-file <path>  [--release V] [--arch A] [--elf lib.so] [--out DIR] [--append] [--stdout]
//   node ingest_symbols.mjs symbols-dir  <dir>   [--release V] [--arch A] --out DIR
//   node ingest_symbols.mjs elf          <lib.so> [--release V] [--out DIR] [--append] [--stdout]
//   node ingest_symbols.mjs requires     <binary> [--out DIR] [--append] [--stdout]
//   node ingest_symbols.mjs releases     <soname> <debver>... [--out DIR] [--append] [--stdout]
//   node ingest_symbols.mjs replaces     <new-soname> <old-soname>... [--out DIR] [--append] [--stdout]
//
// Exit codes: 0 ok; 2 usage; 3 evidence failure (missing file, readelf failure,
// unsupported/unknown .symbols template construct, unknown or invalid evidence
// release, (optional) row without ELF cross-check, ELF/.symbols disagreement).

import { execFileSync } from "node:child_process";
import { readFileSync, existsSync, appendFileSync, writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";

const EXIT_EVIDENCE = 3;

function die(msg, code = EXIT_EVIDENCE) {
  process.stderr.write(`ingest_symbols: ${msg}\n`);
  process.exit(code);
}

function pair(k, v) {
  return JSON.stringify([k, v]);
}

function run(cmd, args) {
  return execFileSync(cmd, args, { encoding: "utf8", maxBuffer: 1 << 26, stdio: ["ignore", "pipe", "ignore"] });
}

// readelf: returns null (never "") on failure so callers cannot mistake a
// failed read for an empty section.
function readelf(args, file) {
  if (!existsSync(file)) return null;
  try { return run("readelf", [...args, file]); } catch { return null; }
}

// ---------------------------------------------------------------------------
// Debian package-version helpers (ingestion edge only; ordering is Prolog's).
// ---------------------------------------------------------------------------
// The evidence release of a provider = the package version the evidence was
// taken from. For an installed .symbols file that is the installed package
// version (dpkg-query); for an ELF file it is the owning package's version
// (dpkg -S). Callers may override with --release.
function dpkgVersion(pkg) {
  try { return run("dpkg-query", ["-W", "-f", "${Version}", pkg]).trim() || null; } catch { return null; }
}

function dpkgOwner(path) {
  try {
    const out = run("dpkg", ["-S", path]).trim();
    const m = out.match(/^([^:\s]+(?::[^:\s]+)?):\s/);
    return m ? m[1] : null;
  } catch { return null; }
}

// A syntactically valid Debian version: [epoch:]upstream[-revision], upstream
// starts with a digit (Policy 5.6.12). We only validate; parsing is Prolog's.
const DEB_VERSION_RE = /^(?:\d+:)?\d[A-Za-z0-9.+~:-]*$/;

// ---------------------------------------------------------------------------
// Tier: parse a Debian/Ubuntu `.symbols` file (binary control member form).
// ---------------------------------------------------------------------------
// Binary form (dpkg-gensymbols output, /var/lib/dpkg/info/*.symbols):
//   <soname> <package> #MINVER#           header (no leading whitespace)
//   | libc6 (>> 2.35), libc6 (<< 2.36)    alternative dependency template
//   * Build-Depends-Package: libc6-dev    meta field
//    <symbol>@<node> <minimum-version> [<dep-id>]   symbol row (indented)
//
// SOURCE-TEMPLATE syntax (debian/*.symbols in source packages) differs and is
// only partially supportable without the binary at hand. Tags are WHITELISTED:
// the ones below are processed, EVERY other tag rejects the file (Sol P2).
//   (optional)          dpkg lets an optional symbol stay in the template
//                       after it disappeared from the binary, so the row is
//                       NOT an export fact by itself (Sol P1c). It is kept only
//                       when --elf <lib> is given and the ELF exports that exact
//                       sym@node; an optional row absent from the ELF is
//                       dropped (reported on stderr); without --elf the file
//                       is rejected.
//   (arch=..)/(arch-bits=..)/(arch-endian=..)
//                       processed when --arch is given (row kept iff it
//                       selects the arch; arch-bits/endian derived from it);
//                       rejected otherwise
//   (ignore-blacklist)  processed (ignored; does not affect identity)
//   (symver), (regex), (c++...), any unknown tag
//                       rejected: symver/regex/c++ need the binary to expand;
//                       an unknown tag has unknown semantics
//   #include "file"     rejected (template include)
//   #PACKAGE#           accepted in the header; the package name is then
//                       unknown so --release becomes mandatory
// Minimum-version semantics: a curated LOWER BOUND on the package version a
// dependent needs (Debian policy lets maintainers raise it after a compatible
// behaviour change), NOT a ground-truth introduction date. Stored verbatim.

const KNOWN_ARCHES_64 = new Set(["amd64", "arm64", "ppc64el", "s390x", "riscv64", "ia64", "mips64el", "sparc64", "ppc64", "alpha", "loong64"]);
const BIG_ENDIAN = new Set(["s390x", "ppc64", "sparc64", "hppa", "m68k", "mips", "powerpc"]);

function archSelects(spec, arch) {
  // spec: "amd64 !i386 any-arm linux-any" -- dpkg-architecture style; we
  // support exact names, `!` negation, and the `any`/`linux-any` wildcards.
  const terms = spec.trim().split(/\s+/);
  let selected = null;
  for (let t of terms) {
    let neg = false;
    if (t.startsWith("!")) { neg = true; t = t.slice(1); }
    const hit = t === "any" || t === "linux-any" || t === arch || t === `linux-${arch}` ||
      (t.startsWith("any-") && arch.endsWith(t.slice(4)));
    if (neg) { if (hit) return false; if (selected === null) selected = true; }
    else if (hit) selected = true;
    else if (selected === null) selected = false;
  }
  return selected === null ? true : selected;
}

function parseTags(line) {
  // Leading `(tag|tag=value|...)` group(s). Returns {tags: Map, rest}.
  const tags = new Map();
  let rest = line;
  while (rest[0] === "(") {
    const close = rest.indexOf(")");
    if (close < 0) break;
    for (const t of rest.slice(1, close).split("|")) {
      const eq = t.indexOf("=");
      if (eq < 0) tags.set(t.trim(), true);
      else tags.set(t.slice(0, eq).trim(), t.slice(eq + 1).trim());
    }
    rest = rest.slice(close + 1);
  }
  return { tags, rest };
}

// Whitelist (Sol P2): every tag not listed here rejects the file.
const SUPPORTED_TAGS = new Set(["optional", "arch", "arch-bits", "arch-endian", "ignore-blacklist"]);

function parseSymbolsFile(path, { arch = null } = {}) {
  if (!existsSync(path)) die(`symbols file not found: ${path}`);
  const text = readFileSync(path, "utf8");
  const blocks = [];                // {soname, package, rows: [{sym, node, minver, optional}]}
  const errors = [];                // unsupported template constructs (line numbers)
  let cur = null;
  let lineNo = 0;
  for (const raw of text.split("\n")) {
    lineNo++;
    if (!raw.trim()) continue;
    if (!/^[ \t]/.test(raw)) {
      const c = raw[0];
      if (c === "|" || c === "*") continue;                       // alt-dep template / meta field
      if (c === "#") {
        if (/^#include\b/.test(raw)) errors.push(`${lineNo}: template #include`);
        continue;                                                 // comment
      }
      const [soname, pkg] = raw.trim().split(/\s+/);
      if (!soname) { errors.push(`${lineNo}: malformed header`); continue; }
      cur = { soname, package: pkg && pkg !== "#PACKAGE#" ? pkg : null, rows: [] };
      blocks.push(cur);
      continue;
    }
    if (!cur) { errors.push(`${lineNo}: symbol row before any soname header`); continue; }
    let line = raw.trim();
    if (line[0] === "|" || line[0] === "*" || line[0] === "#") continue;
    const { tags, rest } = parseTags(line);
    line = rest.trimStart();
    // Whitelist: reject every tag whose semantics we do not implement.
    const unknown = [...tags.keys()].filter((t) => !SUPPORTED_TAGS.has(t));
    if (unknown.length) { errors.push(`${lineNo}: unsupported template tag (${unknown.join("|")}): ${raw.trim()}`); continue; }
    if (line[0] === '"') { errors.push(`${lineNo}: quoted (pattern) symbol needs the binary to expand: ${raw.trim()}`); continue; }
    // Architecture selectors.
    let archOk = true;
    for (const key of ["arch", "arch-bits", "arch-endian"]) {
      if (!tags.has(key)) continue;
      if (!arch) { errors.push(`${lineNo}: (${key}=...) selector but no --arch given: ${raw.trim()}`); archOk = null; break; }
      const v = String(tags.get(key));
      if (key === "arch") archOk = archOk && archSelects(v, arch);
      else if (key === "arch-bits") archOk = archOk && (v === (KNOWN_ARCHES_64.has(arch) ? "64" : "32"));
      else archOk = archOk && (v === (BIG_ENDIAN.has(arch) ? "big" : "little"));
    }
    if (archOk === null) continue;
    if (!archOk) continue;                                         // row does not apply to this arch
    // "<sym>@<node> <minver> [<dep-id>]" -- split at the LAST '@' (symbol
    // names never contain '@'; node names never do either).
    const parts = line.split(/\s+/);
    const ident = parts[0], minver = parts[1];
    const at = ident.lastIndexOf("@");
    if (at <= 0 || minver === undefined) { errors.push(`${lineNo}: malformed symbol row: ${raw.trim()}`); continue; }
    const sym = ident.slice(0, at), node = ident.slice(at + 1);
    if (!node) { errors.push(`${lineNo}: empty version node: ${raw.trim()}`); continue; }
    if (!DEB_VERSION_RE.test(minver)) { errors.push(`${lineNo}: minimum-version is not a Debian version: ${raw.trim()}`); continue; }
    cur.rows.push({ sym, node, minver, optional: tags.has("optional"), lineNo });
  }
  return { blocks, errors };
}

// ---------------------------------------------------------------------------
// readelf: version-index-aware symbol tables (Tiers 2 & 3).
// ---------------------------------------------------------------------------
// readelf -W --dyn-syms rows:  "  7: 0000000000000000  0 FUNC GLOBAL DEFAULT UND __libc_start_main@GLIBC_2.34 (5)"
// readelf -W -V:
//   Version symbols section '.gnu.version' ...
//     000:   0 (*local*)  2 (GLIBC_2.3)  3 (GLIBC_2.2.5)  5h (GLIBC_2.34)   <- versym per .dynsym index; 'h' = hidden
//   Version definition section '.gnu.version_d' ...
//     0x0000: Rev: 1  Flags: base  Index: 1  Cnt: 1  Name: libc.so.6
//     0x001c: Rev: 1  Flags: none  Index: 2  Cnt: 1  Name: GLIBC_2.2.5
//   Version needs section '.gnu.version_r' ...
//     0x0000: Version: 1  File: libc.so.6  Cnt: 9
//     0x0010:   Name: GLIBC_2.28  Flags: none  Version: 11
function elfTables(file) {
  const symOut = readelf(["-W", "--dyn-syms"], file);
  const verOut = readelf(["-W", "-V"], file);
  const dynOut = readelf(["-W", "-d"], file);
  if (symOut === null || verOut === null || dynOut === null) return null;

  const syms = [];                        // {idx, bind, ndx, name, verName (from name@VER), defaultFromName (@@)}
  for (const raw of symOut.split("\n")) {
    const m = raw.match(/^\s*(\d+):\s+\S+\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)?(?:\s+\((\d+)\))?\s*$/);
    if (!m) continue;
    const [, idx, , bind, , ndx, name0] = m;
    if (!name0) continue;
    let name = name0, verName = null, defaultFromName = null;
    const dd = name0.indexOf("@@"), d = name0.indexOf("@");
    if (dd >= 0) { name = name0.slice(0, dd); verName = name0.slice(dd + 2); defaultFromName = true; }
    else if (d > 0) { name = name0.slice(0, d); verName = name0.slice(d + 1); defaultFromName = false; }
    syms.push({ idx: +idx, bind, ndx, name, verName, defaultFromName });
  }

  // .gnu.version: index -> {ver, hidden}
  const versym = new Map();
  const verdef = new Map();               // verIdx -> name  (definitions; Index 1 = base/unversioned)
  const verneed = new Map();              // verIdx -> {file, name, weak}
  let section = null, curFile = null;
  for (const raw of verOut.split("\n")) {
    if (/^Version symbols section/.test(raw)) { section = "sym"; continue; }
    if (/^Version definition section/.test(raw)) { section = "def"; continue; }
    if (/^Version needs section/.test(raw)) { section = "need"; continue; }
    if (section === "sym") {
      const m = raw.match(/^\s*([0-9a-f]+):\s+(.*)$/);
      if (!m) continue;
      const base = parseInt(m[1], 16);
      const re = /([0-9a-f]+)(h?)\s*\(([^)]*)\)/g;    // "23 (GLIBC_2.34)" or hidden "10h(GLIBC_2.12)"
      let e, i = 0;
      while ((e = re.exec(m[2]))) { versym.set(base + i, { ver: parseInt(e[1], 16), hidden: e[2] === "h" }); i++; }
    } else if (section === "def") {
      const m = raw.match(/Index:\s+(\d+)\s+Cnt:\s+\d+\s+Name:\s+(\S+)/);
      if (m) verdef.set(+m[1], m[2]);
    } else if (section === "need") {
      let m;
      if ((m = raw.match(/Version:\s+\d+\s+File:\s+(\S+)\s+Cnt:/))) { curFile = m[1]; continue; }
      if ((m = raw.match(/Name:\s+(\S+)\s+Flags:\s+([^\s]+(?:\s+[^\s]+)*?)\s+Version:\s+(\d+)/)) && curFile) {
        verneed.set(+m[3], { file: curFile, name: m[1], weak: /WEAK/.test(m[2]) });
      }
    }
  }

  let soname = null;
  const needed = [];
  for (const raw of dynOut.split("\n")) {
    let m;
    if ((m = raw.match(/\(SONAME\)\s+Library soname: \[([^\]]+)\]/))) soname = m[1];
    else if ((m = raw.match(/\(NEEDED\)\s+Shared library: \[([^\]]+)\]/))) needed.push(m[1]);
  }
  return { syms, versym, verdef, verneed, soname, needed, hasVersioning: versym.size > 0 };
}

// Provides: defined dynamic symbols with their exact version node and their
// default-version binding (Sol P1b):
//   binding = "default"    an unversioned reference binds to it: the `@@`
//                          default (versym hidden bit clear), or the oldest
//                          version node (verdef index 2), which the loader
//                          accepts for legacy unversioned references even
//                          when hidden (glibc dl-lookup: index < 3 is taken
//                          before the hidden test; fixture D10c proves it)
//           = "nondefault" hidden (`@`) at verdef index >= 3: an unversioned
//                          reference does NOT bind to it (fixture D10)
//   `Base` is always "default".
function elfProvides(t) {
  const rows = [];                        // {sym, node, binding}
  const problems = [];
  for (const s of t.syms) {
    if (s.ndx === "UND" || s.ndx === "Ndx") continue;
    if (s.bind !== "GLOBAL" && s.bind !== "WEAK") continue;   // LOCAL never exported
    let node, binding = "default";
    if (!t.hasVersioning) node = "Base";
    else {
      const v = t.versym.get(s.idx);
      if (!v) { problems.push(`no versym entry for dynsym ${s.idx} (${s.name})`); continue; }
      if (v.ver === 0) continue;                                // *local*
      if (v.ver === 1) node = "Base";                           // *global* = unversioned
      else {
        node = t.verdef.get(v.ver);
        if (!node) { problems.push(`dynsym ${s.idx} (${s.name}) versym ${v.ver} has no verdef entry`); continue; }
        if (s.verName && s.verName !== node) problems.push(`dynsym ${s.idx}: name says @${s.verName} but verdef index says ${node}`);
        if (s.defaultFromName !== null && s.defaultFromName === v.hidden)
          problems.push(`dynsym ${s.idx} (${s.name}@${node}): name says ${s.defaultFromName ? "@@ default" : "@ hidden"} but versym hidden bit says ${v.hidden ? "hidden" : "default"}`);
        binding = (!v.hidden || v.ver === 2) ? "default" : "nondefault";
      }
    }
    rows.push({ sym: s.name, node, binding });
  }
  return { rows, problems };
}

// Exact export map of an ELF: "sym@node" -> binding. Used to cross-check a
// `.symbols` file against the binary it describes (Sol P1c).
function elfExportMap(path, what) {
  const t = elfTables(path);
  if (!t) die(`${what}: --elf ${path}: cannot read (missing file or readelf failure)`);
  const { rows, problems } = elfProvides(t);
  if (problems.length) die(`${what}: --elf ${path}: inconsistent version tables:\n  ${problems.slice(0, 10).join("\n  ")}`);
  const map = new Map();
  for (const { sym, node, binding } of rows) {
    const k = `${sym}@${node}`;
    if (!map.has(k) || binding === "default") map.set(k, binding);
  }
  return { soname: t.soname || path, map };
}

// Requires: undefined dynamic symbols, each attributed to (file, node) via its
// version INDEX. Never keyed by node name.
function elfRequires(t) {
  const rows = [];                        // {sym, node|null, soname|"", bind}
  const problems = [];
  for (const s of t.syms) {
    if (s.ndx !== "UND" || !s.name) continue;
    if (s.bind !== "GLOBAL" && s.bind !== "WEAK") continue;
    let bind = s.bind;
    if (!t.hasVersioning) { rows.push({ sym: s.name, node: null, soname: "", bind }); continue; }
    const v = t.versym.get(s.idx);
    if (!v) { problems.push(`no versym entry for dynsym ${s.idx} (${s.name})`); continue; }
    if (v.ver <= 1) { rows.push({ sym: s.name, node: null, soname: "", bind }); continue; }   // unversioned reference
    const need = t.verneed.get(v.ver);
    if (!need) { problems.push(`dynsym ${s.idx} (${s.name}) versym ${v.ver} has no verneed entry`); continue; }
    if (s.verName && s.verName !== need.name) problems.push(`dynsym ${s.idx}: name says @${s.verName} but verneed index ${v.ver} says ${need.name}`);
    if (!t.needed.includes(need.file)) problems.push(`verneed file ${need.file} (for ${s.name}@${need.name}) is not in DT_NEEDED`);
    if (need.weak) bind = "WEAK";
    rows.push({ sym: s.name, node: need.name, soname: need.file, bind });
  }
  return { rows, problems };
}

// ---------------------------------------------------------------------------
// Output sink.
// ---------------------------------------------------------------------------
const STORES = ["symprov", "symreq", "needed", "evidence", "releases", "replaces"];

function makeSink(outDir, toStdout) {
  const buffers = Object.fromEntries(STORES.map((s) => [s, []]));
  return {
    add(store, k, v) { buffers[store].push(pair(k, v)); },
    flush(append) {
      if (outDir) mkdirSync(outDir, { recursive: true });
      for (const store of STORES) {
        const lines = buffers[store];
        if (!lines.length) continue;
        if (outDir) {
          const f = join(outDir, store + ".jsonl");
          const body = lines.join("\n") + "\n";
          if (append) appendFileSync(f, body); else writeFileSync(f, body);
        }
        if (toStdout) for (const l of lines) process.stdout.write(l + "\n");
      }
    },
  };
}

function parseArgs(rest) {
  const opts = { out: null, stdout: false, append: false, release: null, arch: null, elf: null, positional: [] };
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    if (a === "--out") opts.out = rest[++i];
    else if (a === "--stdout") opts.stdout = true;
    else if (a === "--append") opts.append = true;
    else if (a === "--release") opts.release = rest[++i];
    else if (a === "--arch") opts.arch = rest[++i];
    else if (a === "--elf") opts.elf = rest[++i];
    else opts.positional.push(a);
  }
  if (!opts.out && !opts.stdout) opts.stdout = true;
  return opts;
}

// One Debian-version gate for every evidence release, whichever command
// supplied it (Sol P2): --release, dpkg-query, or the `releases` axis.
function validRelease(rel, what) {
  if (!rel) die(`${what}: evidence release unknown (not owned by an installed package); pass --release <debver>`);
  if (!DEB_VERSION_RE.test(rel)) die(`${what}: release '${rel}' is not a Debian version`);
  return rel;
}

function requireRelease(opts, guess, what) {
  return validRelease(opts.release || guess, what);
}

// ---------------------------------------------------------------------------
// Commands.
// ---------------------------------------------------------------------------
// A `.symbols` file is ingested ATOMICALLY: any error in any block returns
// null (nothing is added to the sink), in single-file and batch mode alike.
function cmdSymbolsFile(opts, path, sink, elf = null) {
  const { blocks, errors } = parseSymbolsFile(path, { arch: opts.arch });
  if (!blocks.length && !errors.length) errors.push("no soname blocks");
  const rows = [];                                   // deferred until the whole file is clean
  const sonames = [];
  let dropped = 0;
  for (const b of blocks) {
    const guess = b.package ? dpkgVersion(b.package) : null;
    const rel = opts.release || guess;
    if (!rel) { errors.push(`${b.soname}: evidence release unknown (package ${b.package || "#PACKAGE#"} not installed); pass --release`); continue; }
    if (!DEB_VERSION_RE.test(rel)) { errors.push(`${b.soname}: release '${rel}' is not a Debian version`); continue; }
    const xcheck = elf && elf.soname === b.soname ? elf.map : null;
    const seen = new Set();
    for (const { sym, node, minver, optional, lineNo } of b.rows) {
      const k = `${sym}@${node}`;
      if (xcheck) {
        const binding = xcheck.get(k);
        if (binding === undefined) {
          if (optional) { dropped++; process.stderr.write(`ingest_symbols: ${path}:${lineNo}: (optional) ${k} not exported by ${opts.elf}; row dropped\n`); continue; }
          errors.push(`${lineNo}: ${k} is in the .symbols file but not exported by --elf ${opts.elf}`);
          continue;
        }
        seen.add(k);
        rows.push([`${b.soname}|${k}`, ["since", minver, rel, binding]]);
      } else {
        if (optional) { errors.push(`${lineNo}: (optional) ${k} needs an ELF cross-check (--elf <lib>) to count as an export`); continue; }
        rows.push([`${b.soname}|${k}`, ["since", minver, rel, node === "Base" ? "default" : "unproven"]]);
      }
    }
    if (xcheck) for (const k of xcheck.keys()) if (!seen.has(k)) errors.push(`${b.soname}: ${k} is exported by --elf ${opts.elf} but absent from the .symbols file (evidence would not be complete)`);
    rows.push([`provides|${b.soname}`, ["symbols", rel, "complete", path], "evidence"]);
    sonames.push(b.soname);
  }
  if (errors.length) {
    process.stderr.write(`ingest_symbols: ${path}: ${errors.length} problem(s) -- rejecting the whole file, nothing written:\n`);
    for (const e of errors.slice(0, 20)) process.stderr.write(`  ${e}\n`);
    if (errors.length > 20) process.stderr.write(`  ... ${errors.length - 20} more\n`);
    return null;
  }
  let n = 0;
  for (const [k, v, store] of rows) { sink.add(store || "symprov", k, v); if (!store) n++; }
  return { n, sonames, dropped };
}

const [cmd, ...rest] = process.argv.slice(2);
const opts = parseArgs(rest);
if (opts.release !== null) validRelease(opts.release, cmd);

if (cmd === "symbols-file") {
  const path = opts.positional[0];
  if (!path) die("symbols-file: missing path", 2);
  const elf = opts.elf ? elfExportMap(opts.elf, `symbols-file ${path}`) : null;
  const sink = makeSink(opts.out, opts.stdout);
  const r = cmdSymbolsFile(opts, path, sink, elf);
  if (!r) process.exit(EXIT_EVIDENCE);
  sink.flush(opts.append);
  process.stderr.write(`symbols-file ${path}: symprov=${r.n} sonames=${r.sonames.length} [${r.sonames.slice(0, 6).join(", ")}${r.sonames.length > 6 ? ", ..." : ""}]${elf ? ` cross-checked=${elf.soname} optional-dropped=${r.dropped}` : ""}\n`);
} else if (cmd === "symbols-dir") {
  const dir = opts.positional[0];
  if (!dir || !opts.out) die("symbols-dir: needs <dir> and --out DIR", 2);
  if (opts.elf) die("symbols-dir: --elf applies to a single file; use symbols-file", 2);
  const files = readdirSync(dir).filter((f) => f.endsWith(".symbols")).map((f) => join(dir, f));
  mkdirSync(opts.out, { recursive: true });
  for (const s of ["symprov", "evidence"]) writeFileSync(join(opts.out, s + ".jsonl"), "");
  let total = 0, ok = 0, rejected = 0;
  for (const f of files) {
    const sink = makeSink(opts.out, false);
    const r = cmdSymbolsFile(opts, f, sink);
    if (!r) { rejected++; continue; }                // atomic: no block of a rejected file is written
    sink.flush(true);
    total += r.n; ok++;
  }
  process.stderr.write(`symbols-dir ${dir}: files=${ok} rejected=${rejected} symprov=${total}\n`);
  if (rejected) process.exit(EXIT_EVIDENCE);
} else if (cmd === "elf") {
  const path = opts.positional[0];
  if (!path) die("elf: missing path", 2);
  const t = elfTables(path);
  if (!t) die(`elf: cannot read ${path} (missing file or readelf failure); no evidence emitted`);
  const so = t.soname || path;
  const owner = dpkgOwner(path);
  const rel = requireRelease(opts, owner ? dpkgVersion(owner) : null, `elf ${path}`);
  const { rows, problems } = elfProvides(t);
  if (problems.length) die(`elf ${path}: inconsistent version tables:\n  ${problems.slice(0, 10).join("\n  ")}`);
  const sink = makeSink(opts.out, opts.stdout);
  const seen = new Map();                      // one identity per sym@node; "default" wins if both appear
  for (const { sym, node, binding } of rows) {
    const k = `${so}|${sym}@${node}`;
    if (!seen.has(k) || binding === "default") seen.set(k, binding);
  }
  let nd = 0;
  for (const [k, binding] of seen) { sink.add("symprov", k, ["at", rel, binding]); if (binding === "nondefault") nd++; }
  sink.add("evidence", `provides|${so}`, ["elf", rel, "complete", path]);
  sink.flush(opts.append);
  process.stderr.write(`elf ${path}: soname=${so} release=${rel} symprov=${seen.size} (nondefault=${nd})\n`);
} else if (cmd === "requires") {
  const path = opts.positional[0];
  if (!path) die("requires: missing path", 2);
  const t = elfTables(path);
  const sink = makeSink(opts.out, opts.stdout);
  if (!t) {
    const status = existsSync(path) ? "readelf_failed" : "missing_file";
    sink.add("evidence", `requires|${path}`, ["readelf", status, path]);
    sink.flush(opts.append);
    die(`requires ${path}: ${status}; recorded INCOMPLETE evidence, no requirement rows`);
  }
  const { rows, problems } = elfRequires(t);
  if (problems.length) {
    sink.add("evidence", `requires|${path}`, ["readelf", "inconsistent", problems[0]]);
    sink.flush(opts.append);
    die(`requires ${path}: inconsistent version tables:\n  ${problems.slice(0, 10).join("\n  ")}`);
  }
  let nv = 0, nu = 0;
  for (const { sym, node, soname, bind } of rows) {
    if (node === null) { sink.add("symreq", `${path}|${sym}`, ["", bind]); nu++; }
    else { sink.add("symreq", `${path}|${sym}@${node}`, [soname, bind]); nv++; }
  }
  for (const so of t.needed) sink.add("needed", path, so);
  sink.add("evidence", `requires|${path}`, ["readelf", "complete", path]);
  sink.flush(opts.append);
  process.stderr.write(`requires ${path}: symreq=${nv} versioned + ${nu} unversioned; needed=[${t.needed.join(", ")}]\n`);
} else if (cmd === "releases") {
  const [so, ...vers] = opts.positional;
  if (!so || !vers.length) die("releases: needs <soname> <debver>...", 2);
  for (const v of vers) validRelease(v, `releases ${so}`);
  const sink = makeSink(opts.out, opts.stdout);
  for (const v of vers) sink.add("releases", so, v);
  sink.flush(opts.append);
  process.stderr.write(`releases ${so}: ${vers.length} candidate(s)\n`);
} else if (cmd === "replaces") {
  // Declared soname succession (Sol P2d): offering <new> to a binary whose
  // DT_NEEDED names <old> is a soname_mismatch only under this relation; any
  // other name that is not NEEDED is simply not_needed (no stem heuristic).
  const [so, ...olds] = opts.positional;
  if (!so || !olds.length) die("replaces: needs <new-soname> <old-soname>...", 2);
  const sink = makeSink(opts.out, opts.stdout);
  for (const o of olds) sink.add("replaces", so, o);
  sink.flush(opts.append);
  process.stderr.write(`replaces ${so}: ${olds.join(", ")}\n`);
} else {
  process.stderr.write("usage: ingest_symbols.mjs symbols-file|symbols-dir|elf|requires|releases|replaces <args> [--release V] [--arch A] [--elf lib.so] [--out DIR] [--append] [--stdout]\n");
  process.exit(2);
}
