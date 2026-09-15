// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// cost_model.mjs -- helpers for the backend-selection cost model.
//
//   synth <out.p2.jsonl> <keys> <rowsPerKey>
//     Synthesize a multi-row-per-key P/2 store (values are scalar strings) so we
//     can measure how the indexed vs lmdb cold-read scatter grows with
//     rows-per-key. Keys "mrk|kN@v" ... actually keys are "mrk|N", each with
//     rowsPerKey distinct values.
//
//   scatter <idx-prefix>
//     Parse <prefix>.idx (UWIX) and report, over all keys: rows-per-key, and the
//     average number of DISTINCT 4KB .data pages a key's records span. indexed
//     stores records in SOURCE ORDER, so a key with N rows lands at N scattered
//     offsets -> ~N distinct pages -> up to N cold reads under memory pressure.
//     (lmdb keys its B-tree by key, so a key's rows cluster in ~1 leaf page; that
//     is a structural contrast, reported as ~1 in the model.)
//
//   K <args as JSON>  -- print the crossover store/RAM ratio table (see below).

import fs from 'node:fs';

const cmd = process.argv[2];

function u16(b, o) { return b[o] | (b[o + 1] << 8); }
function u32(b, o) { return (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0; }

if (cmd === 'synth') {
  const [, , , out, keysArg, rpkArg] = process.argv;
  const keys = parseInt(keysArg, 10), rpk = parseInt(rpkArg, 10);
  // "grouped" (default) writes a key's rows contiguously; "interleaved" writes
  // round-robin (all keys' row-0, then row-1, ...) so a key's rows land at
  // scattered source-order offsets -- the realistic multi-row case (real ABI
  // dup-keys already scatter this way). Interleaved is the scatter scenario.
  const mode = process.argv[6] || 'interleaved';
  const ws = fs.createWriteStream(out);
  if (mode === 'grouped') {
    for (let k = 0; k < keys; k++)
      for (let r = 0; r < rpk; r++)
        ws.write(JSON.stringify([`mrk|k${k}`, `v${r}#rel${r}#unproven`]) + '\n');
  } else {
    for (let r = 0; r < rpk; r++)
      for (let k = 0; k < keys; k++)
        ws.write(JSON.stringify([`mrk|k${k}`, `v${r}#rel${r}#unproven`]) + '\n');
  }
  ws.end(() => console.error(`synth(${mode}): ${keys} keys x ${rpk} rows = ${keys * rpk} rows -> ${out}`));
} else if (cmd === 'scatter') {
  const prefix = process.argv[3];
  const idx = fs.readFileSync(prefix + '.idx');
  if (idx.slice(0, 4).toString() !== 'UWIX') { console.error('bad idx magic'); process.exit(1); }
  const nKeys = u32(idx, 8), keyblobOff = u32(idx, 12), hitsOff = u32(idx, 16), nRecords = u32(idx, 20);
  const PAGE = 4096;
  let totalRows = 0, totalDistinctPages = 0, maxRows = 0;
  const rpkHist = {};
  for (let i = 0; i < nKeys; i++) {
    const e = 24 + i * 16;
    const nHits = u16(idx, e + 6);
    const hitsRel = u32(idx, e + 8);
    const pages = new Set();
    for (let h = 0; h < nHits; h++) {
      const off = u32(idx, hitsOff + hitsRel + h * 4);
      pages.add(Math.floor(off / PAGE));
    }
    totalRows += nHits;
    totalDistinctPages += pages.size;
    if (nHits > maxRows) maxRows = nHits;
    rpkHist[nHits] = (rpkHist[nHits] || 0) + 1;
  }
  const out = {
    idx: prefix, n_keys: nKeys, n_records: nRecords,
    rows_per_key: +(totalRows / nKeys).toFixed(3),
    distinct_data_pages_per_key: +(totalDistinctPages / nKeys).toFixed(3),
    max_rows_for_one_key: maxRows,
    // indexed cold reads per miss = distinct .data pages the key spans.
    indexed_cold_reads_per_miss: +(totalDistinctPages / nKeys).toFixed(3),
    lmdb_cold_reads_per_miss_structural: 1,  // clustered leaf (see header note)
    rows_per_key_hist_top: Object.entries(rpkHist).sort((a, b) => b[1] - a[1]).slice(0, 6),
  };
  console.log(JSON.stringify(out, null, 2));
} else if (cmd === 'K') {
  // K(rows_per_key): the store/RAM ratio at which switching to lmdb yields a
  // given end-to-end speedup, given measured primitives. Reads a JSON blob of
  // {t_seek_ns, t_mem_ns, t_hit_ns, hit_rate, targets:[...]} from argv[3].
  const p = JSON.parse(process.argv[3]);
  const { t_seek_ns, t_mem_ns, t_hit_ns, hit_rate } = p;
  const h = hit_rate;
  // Per-lookup time as a function of store/RAM ratio r (>=1) and rows_per_key M.
  // P(evicted for a miss) ~= 1 - 1/r (fraction of store not resident).
  // indexed miss touches M distinct .data pages; lmdb touches ~1.
  //   T(backend) = h*t_hit + (1-h)*cold_reads*[ (1/r)*t_mem + (1-1/r)*t_seek ]
  const T = (M, r) => {
    const pageCost = (1 / r) * t_mem_ns + (1 - 1 / r) * t_seek_ns;
    return h * t_hit_ns + (1 - h) * M * pageCost;
  };
  const rows = [];
  for (const M of p.rows_per_key_list || [1, 2, 4, 8, 16]) {
    // smallest r>=1 where lmdb is >= speedup faster (T_indexed/T_lmdb >= speedup)
    let Kmap = {};
    for (const sp of p.targets || [1.2, 1.5, 2.0]) {
      let found = null;
      for (let r = 1.0; r <= 100.0; r += 0.05) {
        const ti = T(M, r), tl = T(1, r);
        if (ti / tl >= sp) { found = +r.toFixed(2); break; }
      }
      Kmap['x' + sp] = found;  // null = never reaches that speedup within r<=100
    }
    rows.push({ rows_per_key: M, K_for_speedup: Kmap,
                ratio_at_r_inf: +(T(M, 1e9) / T(1, 1e9)).toFixed(2) });
  }
  console.log(JSON.stringify({ primitives: p, table: rows }, null, 2));
} else {
  console.error('usage: cost_model.mjs synth|scatter|K ...');
  process.exit(2);
}
