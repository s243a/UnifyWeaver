#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// gen_catalogs.mjs -- seeded random catalogs for the uw-resolve P0.5
// differential. mulberry32, same construction as examples/cli_args/gen_cases.mjs.
//
// Each line is one JSON object:
//   { "id", "catalog", "query", "args" }
//
//   node examples/pkg_resolver/gen_catalogs.mjs > cases.jsonl

const SEED = 0xa5b6c7d8;
const CASES = 2400;

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick(rng, n) {
  return Math.floor(rng() * n);
}

function pickOne(rng, arr) {
  return arr[pick(rng, arr.length)];
}

function constraintFor(rng, versions) {
  const kind = rng();
  if (kind < 0.45) return "any";
  const v = pickOne(rng, versions);
  if (kind < 0.6) return { op: "eq", v };
  if (kind < 0.75) return { op: "gte", v };
  if (kind < 0.88) {
    const hi = [...v];
    hi[0] = v[0] + 1;
    return { op: "lt", v: hi };
  }
  const lo = versions[0];
  const last = versions[versions.length - 1];
  const hi = [last[0] + 1, 0, 0];
  return { op: "range", lo, hi };
}

function genCatalog(rng) {
  const bucket = rng();
  let nPkgs;
  let maxVer;
  if (bucket < 0.7) {
    nPkgs = 10 + pick(rng, 11); // 10–20
    maxVer = 1 + pick(rng, 2);  // 1–2
  } else if (bucket < 0.9) {
    nPkgs = 20 + pick(rng, 21); // 20–40
    maxVer = 1 + pick(rng, 3);  // 1–3
  } else {
    nPkgs = 40 + pick(rng, 21); // 40–60
    maxVer = 1;                 // linear, no version search
  }

  const names = Array.from({ length: nPkgs }, (_, i) => "p" + i);
  const versions = {};
  const packages = [];
  for (const name of names) {
    const nVer = 1 + pick(rng, maxVer);
    const vers = [];
    for (let k = 0; k < nVer; k += 1) {
      vers.push([k, pick(rng, 3), 0]);
    }
    vers.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    versions[name] = vers;
    for (const v of vers) packages.push([name, v]);
  }

  const depends = [];
  for (let i = 1; i < nPkgs; i += 1) {
    const nDeps = pick(rng, 4); // 0–3, DAG: only earlier names
    const used = new Set();
    for (let d = 0; d < nDeps; d += 1) {
      const j = pick(rng, i);
      if (used.has(j)) continue;
      used.add(j);
      const name = names[i];
      const dep = names[j];
      const ver = pickOne(rng, versions[name]);
      depends.push([name, ver, dep, constraintFor(rng, versions[dep])]);
    }
  }

  const conflicts = [];
  if (nPkgs >= 2 && rng() < 0.35) {
    const a = pick(rng, nPkgs);
    let b = pick(rng, nPkgs);
    if (b === a) b = (b + 1) % nPkgs;
    const va = pickOne(rng, versions[names[a]]);
    conflicts.push([names[a], va, names[b]]);
  }

  const REASONS = ["layer_shadow", "abi_anchor", "modified", "footprint", "blanket"];
  const base = [];
  const installed = [];
  const requested = [];
  const layerPkgs = [];
  const excluded = [];
  const aliases = [];
  for (let i = 0; i < nPkgs; i += 1) {
    const name = names[i];
    const ver = versions[name][0];
    if (rng() < 0.2) {
      if (rng() < 0.55) base.push([name, ver, pickOne(rng, REASONS)]);
      else base.push([name, ver]);
    } else if (rng() < 0.08) {
      layerPkgs.push([name, ver]);
    }
    if (rng() < 0.35) {
      installed.push([name, ver]);
      if (rng() < 0.5) requested.push(name);
    }
    if (rng() < 0.04) excluded.push(name);
    if (rng() < 0.05) aliases.push(["alias_" + name, name]);
  }

  const layers = [];
  if (layerPkgs.length > 0) layers.push({ name: "devx", packages: layerPkgs });

  return {
    packages,
    depends,
    conflicts,
    base,
    installed,
    requested,
    layers,
    excluded,
    aliases
  };
}

function requestFor(rng, cat) {
  if (cat.packages.length === 0) return "p0";
  const name = pickOne(rng, cat.packages)[0];
  if (rng() < 0.25) {
    const vers = cat.packages.filter((p) => p[0] === name).map((p) => p[1]);
    return { req: name, constraint: constraintFor(rng, vers) };
  }
  return name;
}

function requestsFor(rng, cat) {
  const n = 1 + pick(rng, 3);
  const out = [];
  const seen = new Set();
  for (let i = 0; i < n; i += 1) {
    const r = requestFor(rng, cat);
    const key = typeof r === "string" ? r : r.req;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(r);
  }
  if (out.length === 0) out.push(requestFor(rng, cat));
  return out;
}

function removalPkg(rng, cat) {
  if (cat.installed.length > 0 && rng() < 0.8) {
    return pickOne(rng, cat.installed)[0];
  }
  if (cat.packages.length > 0) return pickOne(rng, cat.packages)[0];
  return "ghost";
}

function safeUpgradeArgs(rng, cat) {
  if (cat.base.length > 0) {
    const b = pickOne(rng, cat.base);
    const name = b[0];
    const vers = cat.packages.filter((p) => p[0] === name).map((p) => p[1]);
    const ver = vers.length > 0 ? pickOne(rng, vers) : b[1];
    return [name, ver];
  }
  if (cat.packages.length > 0) {
    const p = pickOne(rng, cat.packages);
    return [p[0], p[1]];
  }
  return ["ghost", [1, 0, 0]];
}

function dependentsPkg(rng, cat) {
  if (cat.packages.length > 0) return pickOne(rng, cat.packages)[0];
  return "ghost";
}

function aliasOrName(rng, cat, name) {
  const hits = (cat.aliases || []).filter((a) => a[1] === name);
  if (hits.length > 0 && rng() < 0.5) return hits[0][0];
  return name;
}

const QUERIES = [
  "resolve", "resolve_layered", "explain_blocked", "layer_closure", "removal_orphans",
  "safe_upgrade", "upgrade_set", "freeze_audit", "dependents", "dependents_installed"
];

const rng = mulberry32(SEED);
for (let i = 0; i < CASES; i += 1) {
  const catalog = genCatalog(rng);
  const query = QUERIES[i % QUERIES.length];
  let args;
  if (query === "resolve" || query === "resolve_layered") {
    args = requestsFor(rng, catalog).map((r) => {
      if (typeof r === "string") return aliasOrName(rng, catalog, r);
      return { ...r, req: aliasOrName(rng, catalog, r.req) };
    });
  } else if (query === "removal_orphans") args = removalPkg(rng, catalog);
  else if (query === "safe_upgrade" || query === "upgrade_set") args = safeUpgradeArgs(rng, catalog);
  else if (query === "freeze_audit") args = [];
  else if (query === "dependents" || query === "dependents_installed") args = dependentsPkg(rng, catalog);
  else args = requestFor(rng, catalog);
  process.stdout.write(JSON.stringify({ id: "g" + i, catalog, query, args }) + "\n");
}
