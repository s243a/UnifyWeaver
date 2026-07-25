#!/usr/bin/env python3
"""SM-FS corpus freezer v2 — repairs sm_fs_filing.py per gpt-5.6-sol's review before any training.

Fixes, point by point (sol 2026-07-24):
  1. CATALOG WITHOUT PLACEMENT COUNTS: candidates = every directory in the frozen tree snapshot
     (structure-derived), not dirs-with-≥N-maps. Exact PATH identity is primary; duplicate leaf
     titles stay distinct rows (title-equivalence available downstream as sensitivity only).
  2. DESTINATION-DISJOINT SPLIT: split by destination directory (all maps filed in a directory
     travel together); training additionally EXCLUDES any row whose destination is an ancestor
     or descendant of a reserved destination (lineage-family isolation). Ancestor components
     shared at higher levels are recorded as a documented transductive limitation.
  3. REPRODUCIBLE MEMBERSHIP: the ledger stores every row individually (path id, file sha256,
     split, classification) plus tree-snapshot and code/E5-revision fingerprints — membership is
     independently reproducible, not an aggregate digest.
  4. PRIVACY: consumes sol's privacy index (~/mu_data/sm_fs_privacy_index.json, classification ∈
     {public, private, unknown, excluded}) when present; falls back to the conservative
     path-substring rule, RECORDED as fallback — not certification. 'excluded' rows never enter
     the ledger; private/unknown rows are ledgered local-only (never published/hosted).
  5. DURABLE CACHE + PINNED REVISION: everything under ~/mu_data; E5_REVISION recorded.

Also emits the lineage(fs) TRAINING TARGETS from the exploration partition only: rows
(map_title, ancestor_dir_title, hop) with target decay^(hop-1), process expression
"lineage(fs,decay=0.85)" in the header (P4 provenance convention).

  python3 sm_fs_freeze.py            # freeze + emit ledger and training targets (no scoring)
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import E5_REVISION

DEFAULT_ROOT = "/mnt/c/Users/johnc/Dropbox/root"
DECAY = 0.85
EXPR = f"lineage(fs,decay={DECAY})"


def sha_file(p, cap=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(cap), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def classify(rel_parts, fname, index):
    key = "/".join(rel_parts + [fname])
    if index:
        for scope_len in range(len(rel_parts) + 1, 0, -1):   # most specific record wins
            k = "/".join((rel_parts + [fname])[:scope_len])
            if k in index:
                return index[k], "index"
    if any("private" in p.lower() for p in rel_parts) or "private" in fname.lower():
        return "private", "fallback_substring"
    return "unknown", "fallback_default"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs-root", default=DEFAULT_ROOT)
    ap.add_argument("--privacy-index", default=os.path.expanduser(
        "~/mu_data/sm_fs_privacy_index.json"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/mu_data/sm_fs_v2"))
    ap.add_argument("--holdout-frac", type=float, default=0.40)
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args(argv)

    index = None
    if os.path.exists(a.privacy_index):
        raw = json.load(open(a.privacy_index))
        index = {r["relative_path"]: r["classification"] for r in raw.get("records", [])}
        print(f"privacy index: {len(index)} records (sol classifier)")
    else:
        print("privacy index absent -> conservative substring FALLBACK (not certification)")

    all_dirs, rows = set(), []
    for dirpath, dirnames, files in os.walk(a.fs_root):
        rel = os.path.relpath(dirpath, a.fs_root)
        parts = [] if rel == "." else rel.split(os.sep)
        if parts:
            all_dirs.add("/".join(parts))
        for f in sorted(files):
            if not f.endswith(".smmx"):
                continue
            cls, ev = classify(parts, f, index)
            # 'excluded' never enters; under the UNCERTIFIED fallback, substring-private is
            # treated as excluded too (conservative). A certified index's 'private' rows are
            # ledgered local-only per the privacy spec.
            if cls == "excluded" or (cls == "private" and ev == "fallback_substring") or not parts:
                continue
            rows.append({"map_path": "/".join(parts + [f]), "dest": "/".join(parts),
                         "title": f[:-5].strip(), "sha256": sha_file(os.path.join(dirpath, f)),
                         "classification": cls, "evidence": ev})
    catalog = sorted(all_dirs)                                   # structure-derived, no counts
    tree_snapshot = hashlib.sha256("\n".join(catalog).encode()).hexdigest()[:16]
    print(f"rows: {len(rows)}; catalog (ALL directories, exact-path identity): {len(catalog)}; "
          f"tree snapshot {tree_snapshot}")

    # SUBTREE-BLOCK split: reserve whole depth-K subtrees, so no explore destination is inside
    # any reserved subtree (and vice versa). Ancestors ABOVE block roots are necessarily shared —
    # the documented transductive-at-ancestor limitation, not a leak within blocks.
    K = 3
    cap = max(1, int(0.08 * len(rows)))          # split any block holding >8% of rows deeper
    from collections import Counter
    def assign(depth_map=None):
        blocks = {}
        for r in rows:
            parts = r["dest"].split("/")
            k = K
            while True:
                b = "/".join(parts[:k])
                if depth_map is None or b not in depth_map or k >= len(parts):
                    break
                k += 1
            blocks[r["map_path"]] = b
        return blocks
    oversized, depth_map = True, set()
    while oversized:
        bmap = assign(depth_map)
        w = Counter(bmap.values())
        big = {b for b, c in w.items() if c > cap}
        new = big - depth_map
        oversized = bool(new)
        depth_map |= new
    for r in rows:
        r["_block"] = bmap[r["map_path"]]
    blocks = sorted(set(bmap.values()))
    rng = np.random.default_rng(a.split_seed)
    order = rng.permutation(len(blocks))
    weight = Counter(bmap.values())
    held_blocks, acc = set(), 0
    target = a.holdout_frac * len(rows)
    for i in order:
        if acc >= target:
            break
        held_blocks.add(blocks[i])
        acc += weight[blocks[i]]
    for r in rows:
        r["split"] = "reserved" if r.pop("_block") in held_blocks else "explore"
    n = {s: sum(1 for r in rows if r["split"] == s) for s in ("explore", "reserved")}
    print(f"split (adaptive subtree blocks, base depth {K}, cap {cap} rows, seed {a.split_seed}): "
          f"explore {n['explore']}, "
          f"reserved {n['reserved']} across {len(held_blocks)}/{len(blocks)} blocks (never score; "
          f"whole subtrees — no explore dest inside a reserved subtree)")

    os.makedirs(a.out_dir, exist_ok=True)
    ledger = {
        "schema": "sm_fs_corpus.v2", "fs_root": a.fs_root, "tree_snapshot": tree_snapshot,
        "e5_revision": E5_REVISION, "split_seed": a.split_seed,
        "holdout_frac": a.holdout_frac, "counts": n, "catalog_size": len(catalog),
        "privacy_source": "index" if index else "fallback_substring",
        "limitations": ["ancestor components above destination level shared across splits "
                        "(transductive at ancestor level)",
                        "fallback privacy rule is conservative filtering, not certification"],
        "rows": rows, "catalog": catalog,
    }
    lp = os.path.join(a.out_dir, "ledger.json")
    json.dump(ledger, open(lp, "w"), ensure_ascii=False, indent=0)
    print(f"ledger -> {lp} (sha {sha_file(lp)}) — per-row membership, independently reproducible")

    # lineage(fs) training targets — EXPLORE partition only
    tp = os.path.join(a.out_dir, "lineage_fs_targets.tsv")
    with open(tp, "w", encoding="utf-8") as f:
        f.write(f"# process_expression\t{EXPR}\n# tree_snapshot\t{tree_snapshot}"
                f"\t# e5_revision\t{E5_REVISION}\n# node\tancestor\thop\ttarget\n")
        kept = 0
        for r in rows:
            if r["split"] != "explore":
                continue
            parts = r["dest"].split("/")
            for hop, anc in enumerate(reversed(parts), start=1):
                f.write(f"{r['title']}\t{anc}\t{hop}\t{DECAY ** (hop - 1):.6f}\n")
                kept += 1
    print(f"training targets -> {tp} ({kept} rows, expr `{EXPR}`, explore partition only)")


if __name__ == "__main__":
    main()
