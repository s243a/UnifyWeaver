#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// pack.mjs -- P/2 packing matching resolver_store.pl (keep in lockstep).

export function packVer(v) {
  return `${v[0]}.${v[1]}.${v[2]}`;
}

export function packConstraint(c) {
  if (c === "any" || c == null) return "any";
  if (c.op === "eq") return "eq:" + packVer(c.v);
  if (c.op === "gte") return "gte:" + packVer(c.v);
  if (c.op === "lt") return "lt:" + packVer(c.v);
  if (c.op === "range") return "range:" + packVer(c.lo) + ":" + packVer(c.hi);
  throw new Error("unknown constraint " + JSON.stringify(c));
}

export function packKey(catId, name) {
  return String(catId) + "|" + String(name);
}

export function packDep(ver, dep, c) {
  return packVer(ver) + "#" + dep + "#" + packConstraint(c);
}

export function packConflict(ver, other) {
  return packVer(ver) + "#" + other;
}

export function packRev(name, ver, c) {
  return name + "#" + packVer(ver) + "#" + packConstraint(c);
}
