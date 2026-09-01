#!/usr/bin/env node
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// Shared on-disk encoding for JS WAM persistent fact stores.
// Used by uw_fact_index.js (backend B) and uw_fact_lmdb.js (backend A).
// Key tags preserve D34 atom / string / number distinctions.

"use strict";

const fs = require("fs");
const path = require("path");

const DATA_MAGIC = Buffer.from("UWFI");
const IDX_MAGIC = Buffer.from("UWIX");
const VERSION = 1;
const DATA_HEADER = 16;
const IDX_HEADER = 24;
const IDX_ENTRY = 16;

const TAG_ATOM = 0x41;   // A
const TAG_STRING = 0x53; // S
const TAG_INT = 0x49;    // I
const TAG_FLOAT = 0x46;  // F

function trim(s) {
  return String(s).replace(/^\s+|\s+$/g, "");
}

function isJsonlPath(p) {
  return /\.jsonl$/i.test(p) || /\.ndjson$/i.test(p);
}

function classifyCellText(text) {
  const t = trim(text);
  if (/^-?\d+$/.test(t)) {
    const n = Number(t);
    if (Number.isSafeInteger(n)) return { tag: "int", val: n, text: t };
  }
  if (/^-?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?$/.test(t) ||
      /^-?\d+[eE][+-]?\d+$/.test(t)) {
    return { tag: "float", val: Number(t), text: t };
  }
  if (t.length >= 2 && t.charAt(0) === '"' && t.charAt(t.length - 1) === '"') {
    const inner = t.slice(1, -1).replace(/\\"/g, '"').replace(/\\\\/g, "\\");
    return { tag: "string", val: inner, text: t };
  }
  if (t.length >= 2 && t.charAt(0) === "'" && t.charAt(t.length - 1) === "'") {
    const inner = t.slice(1, -1).replace(/\\'/g, "'").replace(/\\\\/g, "\\");
    return { tag: "atom", val: inner, text: t };
  }
  return { tag: "atom", val: t, text: t };
}

function jsonValueToCell(val) {
  if (typeof val === "number") {
    if (Number.isInteger(val)) {
      return { tag: "int", val: val, text: String(val) };
    }
    return { tag: "float", val: val, text: String(val) };
  }
  if (typeof val === "boolean") {
    const t = val ? "true" : "false";
    return { tag: "atom", val: t, text: t };
  }
  if (val === null) {
    return { tag: "atom", val: "[]", text: "[]" };
  }
  if (typeof val === "string") {
    return classifyCellText(val);
  }
  return null;
}

function parseDelimitedLine(line) {
  const sep = line.indexOf("\t") >= 0 ? "\t" : ",";
  const parts = line.split(sep);
  if (parts.length < 2) return null;
  const a1 = classifyCellText(parts[0]);
  const a2 = classifyCellText(parts[1]);
  return { a1: a1, a2: a2 };
}

function parseJsonlLine(line) {
  let obj;
  try { obj = JSON.parse(line); } catch (e) { return null; }
  if (Array.isArray(obj) && obj.length >= 2) {
    const a1 = jsonValueToCell(obj[0]);
    const a2 = jsonValueToCell(obj[1]);
    if (!a1 || !a2) return null;
    return { a1: a1, a2: a2 };
  }
  if (obj && typeof obj === "object") {
    if (Array.isArray(obj.args) && obj.args.length >= 2) {
      const a1 = jsonValueToCell(obj.args[0]);
      const a2 = jsonValueToCell(obj.args[1]);
      if (!a1 || !a2) return null;
      return { a1: a1, a2: a2 };
    }
    if (obj.a1 !== undefined && obj.a2 !== undefined) {
      const a1 = jsonValueToCell(obj.a1);
      const a2 = jsonValueToCell(obj.a2);
      if (!a1 || !a2) return null;
      return { a1: a1, a2: a2 };
    }
  }
  return null;
}

function firstDataLineLooksJson(lines) {
  for (let i = 0; i < lines.length; i++) {
    const cleaned = trim(lines[i]);
    if (cleaned === "" || cleaned.charAt(0) === "#") continue;
    const c = cleaned.charAt(0);
    return c === "{" || c === "[";
  }
  return false;
}

function readFlatFactRows(inputPath) {
  const text = fs.readFileSync(inputPath, "utf8");
  const lines = String(text).split(/\r?\n/);
  const jsonl = isJsonlPath(inputPath) || firstDataLineLooksJson(lines);
  const rows = [];
  for (let i = 0; i < lines.length; i++) {
    const cleaned = trim(lines[i]);
    if (cleaned === "" || cleaned.charAt(0) === "#") continue;
    const pair = jsonl ? parseJsonlLine(cleaned) : parseDelimitedLine(cleaned);
    if (!pair) continue;
    rows.push(pair);
  }
  return rows;
}

function encodeIndexKey(cell) {
  if (cell.tag === "int") {
    const b = Buffer.alloc(9);
    b[0] = TAG_INT;
    b.writeBigInt64BE(BigInt(cell.val), 1);
    return b;
  }
  if (cell.tag === "float") {
    const b = Buffer.alloc(9);
    b[0] = TAG_FLOAT;
    b.writeDoubleBE(cell.val, 1);
    return b;
  }
  if (cell.tag === "string") {
    return Buffer.concat([Buffer.from([TAG_STRING]), Buffer.from(String(cell.val), "utf8")]);
  }
  return Buffer.concat([Buffer.from([TAG_ATOM]), Buffer.from(String(cell.val), "utf8")]);
}

function recordPayload(row) {
  const a1 = Buffer.from(row.a1.text, "utf8");
  const a2 = Buffer.from(row.a2.text, "utf8");
  const buf = Buffer.alloc(4 + a1.length + a2.length);
  buf.writeUInt16LE(a1.length, 0);
  buf.writeUInt16LE(a2.length, 2);
  a1.copy(buf, 4);
  a2.copy(buf, 4 + a1.length);
  return buf;
}

function writeIndexedStore(inputPath, storePrefix) {
  const rows = readFlatFactRows(inputPath);
  const dir = path.dirname(path.resolve(storePrefix));
  fs.mkdirSync(dir, { recursive: true });
  const dataPath = storePrefix + ".data";
  const idxPath = storePrefix + ".idx";

  const dataChunks = [Buffer.alloc(DATA_HEADER)];
  DATA_MAGIC.copy(dataChunks[0], 0);
  dataChunks[0].writeUInt8(VERSION, 4);
  dataChunks[0].writeUInt32LE(rows.length, 8);

  const byKey = new Map();
  let offset = DATA_HEADER;
  for (let i = 0; i < rows.length; i++) {
    const payload = recordPayload(rows[i]);
    const rec = Buffer.alloc(4 + payload.length);
    rec.writeUInt32LE(payload.length, 0);
    payload.copy(rec, 4);
    dataChunks.push(rec);
    const key = encodeIndexKey(rows[i].a1);
    const keyHex = key.toString("hex");
    if (!byKey.has(keyHex)) byKey.set(keyHex, { key: key, offsets: [] });
    byKey.get(keyHex).offsets.push(offset);
    offset += rec.length;
  }
  fs.writeFileSync(dataPath, Buffer.concat(dataChunks));

  const entries = Array.from(byKey.values());
  entries.sort(function (a, b) { return Buffer.compare(a.key, b.key); });

  const keyBlobParts = [];
  const hitsParts = [];
  const table = Buffer.alloc(IDX_ENTRY * entries.length);
  let keyRel = 0;
  let hitsRel = 0;
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    const base = i * IDX_ENTRY;
    table.writeUInt32LE(keyRel, base);
    table.writeUInt16LE(e.key.length, base + 4);
    table.writeUInt16LE(e.offsets.length, base + 6);
    table.writeUInt32LE(hitsRel, base + 8);
    keyBlobParts.push(e.key);
    keyRel += e.key.length;
    const hits = Buffer.alloc(4 * e.offsets.length);
    for (let j = 0; j < e.offsets.length; j++) hits.writeUInt32LE(e.offsets[j], j * 4);
    hitsParts.push(hits);
    hitsRel += hits.length;
  }
  const keyBlob = Buffer.concat(keyBlobParts);
  const hitsBlob = Buffer.concat(hitsParts);
  const header = Buffer.alloc(IDX_HEADER);
  IDX_MAGIC.copy(header, 0);
  header.writeUInt8(VERSION, 4);
  header.writeUInt32LE(entries.length, 8);
  header.writeUInt32LE(IDX_HEADER + table.length, 12);
  header.writeUInt32LE(IDX_HEADER + table.length + keyBlob.length, 16);
  header.writeUInt32LE(rows.length, 20);
  fs.writeFileSync(idxPath, Buffer.concat([header, table, keyBlob, hitsBlob]));
  return { rows: rows.length, keys: entries.length, dataPath: dataPath, idxPath: idxPath };
}

function lmdbMissingError(predKey, storePath) {
  return (
    "JS WAM fact source " + JSON.stringify(String(predKey || "")) +
    " is declared as lmdb(" + JSON.stringify(String(storePath || "")) +
    ") but the 'lmdb' npm package is not installed. " +
    "Install it in this environment with: npm install lmdb " +
    "This backend is opt-in; default builds do not require it. " +
    "The indexed(...) store is a different format and is not used as a fallback."
  );
}

function seqKey(seq) {
  const b = Buffer.alloc(9);
  b[0] = 0x00;
  b.writeBigUInt64BE(BigInt(seq), 1);
  return b;
}

function a1RangeKey(keyBytes, seq) {
  const b = Buffer.alloc(3 + keyBytes.length + 8);
  b[0] = 0x01;
  b.writeUInt16BE(keyBytes.length, 1);
  keyBytes.copy(b, 3);
  b.writeBigUInt64BE(BigInt(seq), 3 + keyBytes.length);
  return b;
}

function a1RangeStart(keyBytes) {
  return a1RangeKey(keyBytes, 0);
}

function a1RangeEnd(keyBytes) {
  const b = a1RangeKey(keyBytes, 0);
  for (let i = b.length - 1; i >= 3 + keyBytes.length; i--) b[i] = 0xff;
  return b;
}

module.exports = {
  DATA_MAGIC: DATA_MAGIC,
  IDX_MAGIC: IDX_MAGIC,
  VERSION: VERSION,
  DATA_HEADER: DATA_HEADER,
  IDX_HEADER: IDX_HEADER,
  IDX_ENTRY: IDX_ENTRY,
  TAG_ATOM: TAG_ATOM,
  TAG_STRING: TAG_STRING,
  TAG_INT: TAG_INT,
  TAG_FLOAT: TAG_FLOAT,
  trim: trim,
  classifyCellText: classifyCellText,
  readFlatFactRows: readFlatFactRows,
  encodeIndexKey: encodeIndexKey,
  recordPayload: recordPayload,
  writeIndexedStore: writeIndexedStore,
  lmdbMissingError: lmdbMissingError,
  seqKey: seqKey,
  a1RangeKey: a1RangeKey,
  a1RangeStart: a1RangeStart,
  a1RangeEnd: a1RangeEnd
};
