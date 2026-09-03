#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// pack.mjs -- P/2 packing matching resolver_store.pl (keep in lockstep).
// v/3 packing is byte-identical to P2. deb/3 uses d:Epoch:UpSegs:RevSegs
// where each seg is order|num joined by `;`.

function isDeb(v) {
  return v && typeof v === "object" && !Array.isArray(v) && v.deb;
}

export function packVer(v) {
  if (isDeb(v)) {
    const [e, up, rev] = v.deb;
    return "d:" + (e | 0) + ":" + packSegs(up || []) + ":" + packSegs(rev || []);
  }
  return `${v[0]}.${v[1]}.${v[2]}`;
}

function packSegs(segs) {
  return segs.map((seg) => {
    const order = String(seg[0] || "");
    const n = seg[1] | 0;
    return order + "|" + n;
  }).join(";");
}

export function packConstraint(c) {
  if (c === "any" || c == null) return "any";
  if (c.op === "eq") return "eq:" + packVer(c.v);
  if (c.op === "gte") return "gte:" + packVer(c.v);
  if (c.op === "lt") return "lt:" + packVer(c.v);
  if (c.op === "lte") return "lte:" + packVer(c.v);
  if (c.op === "gt") return "gt:" + packVer(c.v);
  if (c.op === "range") {
    const lo = packVer(c.lo);
    const hi = packVer(c.hi);
    if (lo.startsWith("d:") || hi.startsWith("d:")) return "range@" + lo + "@" + hi;
    return "range:" + lo + ":" + hi;
  }
  throw new Error("unknown constraint " + JSON.stringify(c));
}

export function packKey(catId, name) {
  return String(catId) + "|" + String(name);
}

export function packDep(ver, dep, c) {
  if (dep && typeof dep === "object" && dep.alternatives) {
    const cells = dep.alternatives.map((a) => {
      const n = a.dep || a.name;
      return n + "=" + packConstraint(a.constraint);
    }).join("+");
    return packVer(ver) + "#@alts:" + cells + "#" + packConstraint(c);
  }
  return packVer(ver) + "#" + dep + "#" + packConstraint(c);
}

export function packConflict(ver, other) {
  return packVer(ver) + "#" + other;
}

export function packRev(name, ver, c) {
  return name + "#" + packVer(ver) + "#" + packConstraint(c);
}

export function packProvide(name, ver, virtualVer) {
  const vv = virtualVer == null || virtualVer === "-" ? "-" : packVer(virtualVer);
  return name + "#" + packVer(ver) + "#" + vv;
}
