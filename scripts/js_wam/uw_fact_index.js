#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// uw_fact_index — build a dependency-free indexed fact store (backend B).
//
//   node scripts/js_wam/uw_fact_index.js build <tsv|csv|jsonl> <store-prefix>
//
// Writes <store-prefix>.data (length-prefixed records, source order) and
// <store-prefix>.idx (sorted first-arg key table). This is LMDB-style
// (persistent + indexed + seek-based), not LMDB.

"use strict";

const path = require("path");
const codec = require(path.join(__dirname, "uw_fact_codec.js"));

function usage() {
  process.stderr.write(
    "usage: node scripts/js_wam/uw_fact_index.js build <tsv|csv|jsonl> <store-prefix>\n"
  );
  process.exit(2);
}

const argv = process.argv.slice(2);
if (argv[0] !== "build" || argv.length < 3) usage();
const input = argv[1];
const store = argv[2];
const result = codec.writeIndexedStore(input, store);
process.stdout.write(
  "uw_fact_index: wrote " + result.rows + " records / " +
  result.keys + " keys -> " + result.dataPath + " + " + result.idxPath + "\n"
);
