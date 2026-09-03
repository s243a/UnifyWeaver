#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// uw_fact_lmdb — load the same TSV/CSV/JSONL into an LMDB environment (backend A).
//
//   node scripts/js_wam/uw_fact_lmdb.js build <tsv|csv|jsonl> <lmdb-dir>
//   node scripts/js_wam/uw_fact_lmdb.js build-all <jsonl-dir> <lmdb-parent-dir>
//
// build-all writes <lmdb-parent-dir>/{pkg,dep,conflict,revdep,provide}/ from the
// matching JSONL files in <jsonl-dir> (resolver store layout).
//
// Requires the `lmdb` npm package (opt-in; not a repo dependency):
//   npm install lmdb
// Missing package is a loud error; never falls back to indexed(...).
//
// Key scheme (see also docs/WAM_JAVASCRIPT_STATUS.md):
//   seq (unbound enum, source order): 0x00 || uint64be(seq) → payload
//   A1  (bound lookup):               0x01 || uint16be(key_len) || encodeIndexKey(A1) || uint64be(seq) → payload
// encodeIndexKey preserves D34 tags: 0x49 int, 0x46 float, 0x53 string, 0x41 atom.

"use strict";

const fs = require("fs");
const path = require("path");
const codec = require(path.join(__dirname, "uw_fact_codec.js"));

const RESOLVER_STORES = ["pkg", "dep", "conflict", "revdep", "provide"];

function usage() {
  process.stderr.write(
    "usage: node scripts/js_wam/uw_fact_lmdb.js build <tsv|csv|jsonl> <lmdb-dir>\n" +
    "       node scripts/js_wam/uw_fact_lmdb.js build-all <jsonl-dir> <lmdb-parent-dir>\n"
  );
  process.exit(2);
}

function loadLmdb(storePath) {
  try {
    const { createRequire } = require("module");
    const req = createRequire(__filename);
    return req("lmdb");
  } catch (err) {
    throw new Error(codec.lmdbMissingError("build", storePath));
  }
}

function putAll(store, list) {
  if (typeof store.clearSync === "function") store.clearSync();
  else if (typeof store.clear === "function") store.clear();
  for (let i = 0; i < list.length; i++) {
    const payload = codec.recordPayload(list[i]);
    const keyBytes = codec.encodeIndexKey(list[i].a1);
    const seq = codec.seqKey(i);
    const a1k = codec.a1RangeKey(keyBytes, i);
    if (typeof store.putSync === "function") {
      store.putSync(seq, payload);
      store.putSync(a1k, payload);
    } else {
      store.put(seq, payload);
      store.put(a1k, payload);
    }
  }
  return list.length;
}

function commit(store, list) {
  if (typeof store.transactionSync === "function") {
    return store.transactionSync(function () { return putAll(store, list); });
  }
  if (typeof store.transaction === "function") {
    return store.transaction(function () { return putAll(store, list); });
  }
  return putAll(store, list);
}

function openEnv(lmdb, storeDir) {
  const open = lmdb.open || (lmdb.default && lmdb.default.open);
  if (typeof open !== "function") {
    throw new Error("uw_fact_lmdb: unexpected lmdb package API (no open())");
  }
  fs.mkdirSync(storeDir, { recursive: true });
  return open({
    path: storeDir,
    encoding: "binary",
    keyEncoding: "binary",
    // lmdb-js defaults noSubdir when the path has an extension (e.g. edges.lmdb).
    // GP-LMDB A is always an environment *directory*.
    noSubdir: false
  });
}

function buildOne(lmdb, input, storeDir) {
  const rows = codec.readFlatFactRows(input);
  const db = openEnv(lmdb, storeDir);
  return Promise.resolve(commit(db, rows)).then(function (n) {
    if (db.flushed && typeof db.flushed.then === "function") {
      return db.flushed.then(function () { return n; });
    }
    return n;
  }).then(function (n) {
    if (typeof db.close === "function") db.close();
    return n;
  });
}

const argv = process.argv.slice(2);
if (argv[0] !== "build" && argv[0] !== "build-all") usage();

let lmdb;
try {
  lmdb = loadLmdb(argv[2] || argv[1] || ".");
} catch (err) {
  process.stderr.write(String(err.message || err) + "\n");
  process.exit(1);
}

if (argv[0] === "build") {
  if (argv.length < 3) usage();
  const input = argv[1];
  const storeDir = path.resolve(argv[2]);
  Promise.resolve(buildOne(lmdb, input, storeDir)).then(function (n) {
    process.stdout.write("uw_fact_lmdb: wrote " + n + " records -> " + storeDir + "\n");
  }).catch(function (err) {
    process.stderr.write(String((err && err.stack) || err) + "\n");
    process.exit(1);
  });
} else {
  if (argv.length < 3) usage();
  const jsonlDir = path.resolve(argv[1]);
  const parent = path.resolve(argv[2]);
  fs.mkdirSync(parent, { recursive: true });
  (async function () {
    for (let i = 0; i < RESOLVER_STORES.length; i++) {
      const name = RESOLVER_STORES[i];
      const src = path.join(jsonlDir, name + ".jsonl");
      if (!fs.existsSync(src)) {
        fs.writeFileSync(src, "");
      }
      const dest = path.join(parent, name);
      const n = await buildOne(lmdb, src, dest);
      process.stdout.write("uw_fact_lmdb: wrote " + n + " records -> " + dest + "\n");
    }
  })().catch(function (err) {
    process.stderr.write(String((err && err.stack) || err) + "\n");
    process.exit(1);
  });
}
