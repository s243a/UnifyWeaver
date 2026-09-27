#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// pkg.mjs -- the `pkg` CLI. Two Prolog specs, composed at the edges.
//
//   argv --> transpiled argparser (examples/cli_args/cli_args.pl, driven with
//            the registry rendered from cli/pkg_schema.pl)
//        --> dispatch
//        --> transpiled resolver  (examples/pkg_resolver/resolver.pl, via
//            ../wamjs/resolver.mjs)
//        --> output
//
// WHAT IS IN HERE, exhaustively:
//   1. conversion   -- version strings <-> [M,I,P] triples, catalog file ->
//                      JSON data, resolver answers -> the documented JSON doc;
//   2. dispatch     -- which command calls which resolver query;
//   3. formatting   -- the human tables and the JSON serialisation;
//   4. exit codes   -- a table keyed by the doc's `status`.
//
// There is NO parse logic (not one flag rule, not one arity check, not one
// error message -- every usage error you will ever see from `pkg` is a string
// produced by compiled cli_args) and NO resolve logic (not one candidate
// ordering, constraint comparison, layer walk, freeze rule or upgrade closure
// -- those are all compiled resolver.pl).
//
// The single deliberate exception is `pkg deps`, which is a *projection* over
// the catalog's own `depends` rows rather than a resolver query: it filters and
// sorts rows, it does not interpret them. That is called out in the README, and
// it is the only command that reads the catalog directly.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { parseArgs, CliError } from "../../cli_args/wamjs/cliArgs.mjs";
import * as R from "../wamjs/resolver.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

/** The command grammar, rendered from cli/pkg_schema.pl by cli/derive.pl. */
export const REGISTRY = JSON.parse(
  readFileSync(join(HERE, "generated", "pkg_registry.json"), "utf8")
);

/** Spellings that name the same query. `layer` is Pkg's `sfs-combine` word. */
const COMMAND_ALIASES = { layer: "install-plan" };

/** status -> process exit code. 0 success, 1 query-false/blocked, 2 usage. */
export const EXIT_FOR_STATUS = {
  ok: 0,
  clear: 0,
  blocked: 1,
  fail: 1,
  "not-frozen": 1
};

// ---------------------------------------------------------------------------
// 1. conversion
// ---------------------------------------------------------------------------

function verStr(v) {
  return `${v[0]}.${v[1]}.${v[2]}`;
}

function parseVer(s) {
  const m = /^(\d+)\.(\d+)\.(\d+)$/.exec(String(s));
  if (!m) throw new CliError(`bad version: ${s} (expected MAJOR.MINOR.PATCH)`);
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

function pairDoc(p) {
  return { name: p[0], version: verStr(p[1]) };
}

/** The catalog/resolver constraint forms, as one tagged JSON object. */
function constraintDoc(c) {
  if (c === "any" || c == null) return { op: "any" };
  if (c.op === "range") return { op: "range", lo: verStr(c.lo), hi: verStr(c.hi) };
  if (c.op === "eq" || c.op === "gte" || c.op === "lt") {
    return { op: c.op, version: verStr(c.v) };
  }
  throw new Error(`pkg: unhandled constraint ${JSON.stringify(c)}`);
}

function loadCatalog(path) {
  let text;
  try {
    text = readFileSync(path, "utf8");
  } catch (e) {
    throw new CliError(`cannot read catalog ${path}: ${e.code || e.message}`);
  }
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new CliError(`catalog ${path} is not valid JSON: ${e.message}`);
  }
}

// ---------------------------------------------------------------------------
// 2. dispatch -- command -> resolver query
// ---------------------------------------------------------------------------

const DISPATCH = {
  "resolve": (cat, names) => {
    const r = R.resolve(cat, names);
    if (r.fail) {
      return { command: "resolve", status: "fail", requests: names, reason: "no_solution" };
    }
    return { command: "resolve", status: "ok", requests: names, selection: r.ok.map(pairDoc) };
  },

  "install-plan": (cat, names) => {
    const lay = R.resolveLayered(cat, names);
    if (lay.fail) {
      return {
        command: "install-plan", status: "fail", requests: names,
        reason: "no_solution", manifests: []
      };
    }
    const manifests = [];
    for (const n of names) {
      const lc = R.layerClosure(cat, n);
      if (lc.fail) {
        return {
          command: "install-plan", status: "fail", requests: names,
          reason: `no_manifest_for:${n}`, manifests: []
        };
      }
      manifests.push({ request: n, order: lc.ok.map(pairDoc) });
    }
    return {
      command: "install-plan", status: "ok", requests: names,
      selection: lay.ok.map(pairDoc), manifests
    };
  },

  "why-blocked": (cat, [name]) => {
    const b = R.explainBlocked(cat, name);
    if (b.fail) {
      return { command: "why-blocked", status: "fail", request: name, reason: "query_failed", blocked: [] };
    }
    const blocked = b.ok.map((x) => ({
      name: x.name,
      needs: constraintDoc(x.needs),
      base_has: verStr(x.base_has)
    }));
    return {
      command: "why-blocked",
      status: blocked.length ? "blocked" : "clear",
      request: name,
      blocked
    };
  },

  // The one catalog projection: `depends` rows for this name, no interpretation.
  "deps": (cat, [name]) => {
    const rows = (cat.depends || [])
      .filter((r) => r[0] === name)
      .sort((a, b) => cmpVer(a[1], b[1]) || cmpStr(a[2], b[2]))
      .map((r) => ({ version: verStr(r[1]), dep: r[2], constraint: constraintDoc(r[3]) }));
    return { command: "deps", status: "ok", package: name, depends: rows };
  },

  "what-needs": (cat, [name], flags) => {
    const only = flags.installed === true;
    const d = only ? R.dependentsInstalled(cat, name) : R.dependents(cat, name);
    if (d.fail) {
      return {
        command: "what-needs", status: "fail", package: name,
        installed_only: only, reason: "query_failed", dependents: []
      };
    }
    return {
      command: "what-needs", status: "ok", package: name,
      installed_only: only, dependents: d.ok.map(pairDoc)
    };
  },

  "orphans": (cat, [name]) => {
    const o = R.removalOrphans(cat, name);
    if (o.fail) {
      return { command: "orphans", status: "fail", package: name, reason: "query_failed", orphans: [] };
    }
    return { command: "orphans", status: "ok", package: name, orphans: o.ok.map(pairDoc) };
  },

  "why-frozen": (cat, [name]) => {
    const a = R.freezeAudit(cat);
    if (a.fail) {
      return { command: "why-frozen", status: "fail", package: name, reason: "query_failed", kind: null };
    }
    const hit = a.ok.find((row) => row.name === name);
    if (!hit) {
      return { command: "why-frozen", status: "not-frozen", package: name, kind: null, reason: null };
    }
    return {
      command: "why-frozen", status: "ok", package: name,
      kind: hit.kind, reason: hit.reason === undefined ? null : hit.reason
    };
  },

  "audit": (cat) => {
    const a = R.freezeAudit(cat);
    if (a.fail) return { command: "audit", status: "fail", reason: "query_failed", audit: [] };
    return {
      command: "audit", status: "ok",
      audit: a.ok.map((row) => ({
        name: row.name,
        kind: row.kind,
        reason: row.reason === undefined ? null : row.reason
      }))
    };
  },

  "safe-upgrade": (cat, [name, version]) => {
    const ver = parseVer(version);
    const s = R.safeUpgrade(cat, name, ver);
    const base = { command: "safe-upgrade", package: name, version: verStr(ver) };
    if (s.fail) return { ...base, status: "fail", reason: "query_failed", result: null };
    const v = s.ok;
    if (v.verdict === "safe") {
      return { ...base, status: "ok", result: { verdict: "safe", cost: v.cost } };
    }
    if (v.verdict === "coordinated") {
      return { ...base, status: "ok", result: { verdict: "coordinated", set: v.set.map(pairDoc) } };
    }
    if (v.verdict === "unsafe") {
      return { ...base, status: "fail", result: { verdict: "unsafe", reason: v.reason } };
    }
    if (v.verdict === "no_candidate") {
      return { ...base, status: "fail", result: { verdict: "no_candidate" } };
    }
    throw new Error(`pkg: unhandled verdict ${JSON.stringify(v)}`);
  }
};

function cmpVer(a, b) {
  return (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]);
}
function cmpStr(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

// ---------------------------------------------------------------------------
// 3. formatting
// ---------------------------------------------------------------------------

function table(rows, indent = "  ") {
  if (rows.length === 0) return "";
  const widths = [];
  for (const r of rows) {
    r.forEach((c, i) => {
      widths[i] = Math.max(widths[i] || 0, String(c).length);
    });
  }
  return rows
    .map((r) =>
      (indent + r.map((c, i) => (i === r.length - 1 ? String(c) : String(c).padEnd(widths[i]))).join("  "))
        .replace(/\s+$/, "")
    )
    .join("\n");
}

export function constraintText(c) {
  if (c.op === "any") return "*";
  if (c.op === "range") return `>=${c.lo},<${c.hi}`;
  return { eq: "==", gte: ">=", lt: "<" }[c.op] + c.version;
}

function plural(n, one, many) {
  return `${n} ${n === 1 ? one : many}`;
}

/** doc -> the human-readable form. A pure function of the JSON document. */
export function renderHuman(doc) {
  const L = [];
  switch (doc.command) {
    case "resolve":
      if (doc.status === "fail") {
        L.push(`no solution for: ${doc.requests.join(" ")}`);
        L.push("(the catalog has no candidate set that satisfies every request)");
        break;
      }
      L.push(`selection for ${doc.requests.join(" ")} — ${plural(doc.selection.length, "package", "packages")}`);
      L.push(table(doc.selection.map((p) => [p.name, p.version])));
      break;

    case "install-plan":
      if (doc.status === "fail") {
        L.push(`no layered plan for: ${doc.requests.join(" ")}`);
        L.push(`(run \`pkg why-blocked ${doc.requests[0]}\` for the ceiling)`);
        break;
      }
      L.push(`plan for ${doc.requests.join(" ")} — ${plural(doc.selection.length, "package", "packages")} (frozen base untouched)`);
      if (doc.selection.length === 0) {
        L.push("  nothing to install: every request is already satisfied by a loaded layer");
      } else {
        L.push(table(doc.selection.map((p) => [p.name, p.version])));
      }
      for (const m of doc.manifests) {
        L.push(`install order for ${m.request}:`);
        if (m.order.length === 0) L.push("  (empty)");
        else L.push(table(m.order.map((p, i) => [`${i + 1}.`, p.name, p.version])));
      }
      break;

    case "why-blocked":
      if (doc.status === "clear") {
        L.push(`${doc.request} is not blocked by the frozen base`);
        break;
      }
      L.push(`${doc.request} is blocked by the frozen base — ${plural(doc.blocked.length, "ceiling", "ceilings")}`);
      L.push(table(doc.blocked.map((b) => [b.name, `needs ${constraintText(b.needs)}`, `base has ${b.base_has}`])));
      break;

    case "deps":
      L.push(`direct deps of ${doc.package} — ${plural(doc.depends.length, "row", "rows")} (catalog projection)`);
      if (doc.depends.length) {
        L.push(table(doc.depends.map((d) => [d.version, d.dep, constraintText(d.constraint)])));
      }
      break;

    case "what-needs":
      L.push(
        `${plural(doc.dependents.length, "package", "packages")} directly need ${doc.package}` +
        (doc.installed_only ? " (installed or loaded-layer only)" : " (all catalog versions)")
      );
      if (doc.dependents.length) {
        L.push(table(doc.dependents.map((p) => [p.name, p.version])));
      }
      break;

    case "orphans":
      if (doc.orphans.length === 0) {
        L.push(`removing ${doc.package} would orphan nothing`);
        break;
      }
      L.push(`removing ${doc.package} would orphan ${plural(doc.orphans.length, "package", "packages")}`);
      L.push(table(doc.orphans.map((p) => [p.name, p.version])));
      break;

    case "why-frozen":
      if (doc.status === "not-frozen") {
        L.push(`${doc.package} is not held in the frozen base`);
        break;
      }
      if (doc.kind === "held") {
        L.push(`${doc.package} is held in the frozen base: ${doc.reason}`);
      } else if (doc.kind === "suggest") {
        L.push(`${doc.package} is held in the frozen base: blanket`);
        L.push(`  a base package pins it — suggest ${doc.reason}`);
      } else {
        L.push(`${doc.package} is held in the frozen base: blanket`);
        L.push("  nothing in the base pins it — over-frozen");
      }
      break;

    case "audit": {
      L.push(`freeze audit — ${plural(doc.audit.length, "hold", "holds")}`);
      L.push(table(doc.audit.map((a) => [a.name, a.kind, a.reason === null ? "" : a.reason])));
      const over = doc.audit.filter((a) => a.kind === "over_frozen").length;
      const sug = doc.audit.filter((a) => a.kind === "suggest").length;
      L.push(`${over} over-frozen, ${sug} with a suggested reason`);
      break;
    }

    case "safe-upgrade": {
      const head = `${doc.package} -> ${doc.version}`;
      const r = doc.result;
      if (!r) {
        L.push(`${head}: query failed`);
        break;
      }
      if (r.verdict === "safe") L.push(`${head}: safe (cost: ${r.cost})`);
      else if (r.verdict === "unsafe") L.push(`${head}: unsafe (${r.reason})`);
      else if (r.verdict === "no_candidate") {
        L.push(`${head}: no candidate`);
        L.push("  (no such catalog version, or the package is not a frozen-base hold)");
      } else {
        L.push(`${head}: coordinated — ${plural(r.set.length, "package", "packages")} must move together`);
        L.push(table(r.set.map((p) => [p.name, p.version])));
      }
      break;
    }

    default:
      throw new Error(`pkg: no renderer for ${doc.command}`);
  }
  return L.filter((s) => s !== "").join("\n") + "\n";
}

// ---------------------------------------------------------------------------
// 4. the run
// ---------------------------------------------------------------------------

const USAGE = [
  "usage: pkg <command> [args] [--catalog <file>] [--json]",
  "",
  "  resolve <name>...          classic closure (may move the frozen base)",
  "  install-plan <name>...     layered closure + install order  (alias: layer)",
  "  why-blocked <name>         the frozen-base ceilings that stop a layered plan",
  "  deps <name>                direct catalog deps of a package",
  "  what-needs <name>          reverse deps          [--installed]",
  "  orphans <name>             what removing it would orphan",
  "  why-frozen <name>          one hold's freeze reason",
  "  audit                      every hold's freeze reason",
  "  safe-upgrade <name> <ver>  verdict + the coordinated set",
  "",
  "  --catalog <file>  catalog JSON (or set PKG_CATALOG)",
  "  --json            machine-readable output",
  "",
  "exit: 0 success · 1 query false / blocked · 2 usage error"
].join("\n");

export function run(argv) {
  // cli_args' own leading-global scan is for ITS two globals (--state/--name);
  // an option before the command would silently drop `pkg` onto the legacy
  // lenient parser, where nothing is checked. Refuse instead.
  if (argv.length > 0 && argv[0].startsWith("--") && argv[0] !== "--") {
    throw new CliError(`the command must come first (got ${argv[0]})`);
  }

  // ---- the ONLY parse: compiled cli_args/parse_args/3 with our registry ----
  const { positional, flags } = parseArgs(argv, REGISTRY);

  const spelled = positional[0];
  if (spelled === undefined) throw new CliError("no command given");
  const command = COMMAND_ALIASES[spelled] || spelled;
  const handler = DISPATCH[command];
  if (!handler) throw new CliError(`unknown command: ${spelled}`);

  const catalogPath = flags.catalog || process.env.PKG_CATALOG;
  if (!catalogPath) throw new CliError("no catalog: pass --catalog <file> or set PKG_CATALOG");
  const catalog = loadCatalog(catalogPath);

  const doc = handler(catalog, positional.slice(1), flags);
  return { doc, json: flags.json === true };
}

function main(argv) {
  let out;
  try {
    out = run(argv);
  } catch (e) {
    if (e instanceof CliError) {
      process.stderr.write(`pkg: ${e.message}\n${USAGE}\n`);
      process.exitCode = 2;
      return;
    }
    throw e;
  }
  const text = out.json ? JSON.stringify(out.doc, null, 2) + "\n" : renderHuman(out.doc);
  process.stdout.write(text);
  const code = EXIT_FOR_STATUS[out.doc.status];
  process.exitCode = code === undefined ? 1 : code;
}

const invokedDirectly =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invokedDirectly) main(process.argv.slice(2));
