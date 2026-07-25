#!/usr/bin/env python3
"""SimpleMind FILESYSTEM filing corpus: map ↔ its Dropbox directory = a recorded filing decision.

The .smmx maps live inside a real folder hierarchy, so each map's path is ground-truth filing —
the same single-principal-folder task as Pearltrees (map root ≈ bookmark, parent directory ≈
folder), on a corpus the eval program has NEVER touched. Two disciplines applied at first touch:

  FROZEN HOLDOUT — a deterministic 40% of queries is RESERVED at first scan (never scored by this
  script; the file records only their hashes). The Pearltrees manifest lost its confirmatory
  value through adaptive reuse (P1 rigor contract, finding 5); this corpus keeps its own from
  day one. Exploration uses only the 60% split.

  PRIVACY — any path with a component containing 'private' (case-insensitive) is dropped before
  anything is derived from it; per-item outputs stay in ~/mu_data (never committed); the repo
  sees only aggregate metrics.

Task: query = map filename stem; true folder = immediate parent directory (title-equivalence
across duplicate directory names); catalog = directories holding >= --min-maps maps. e5 ranking.
Comparators: PT single-folder R@1 0.203 / MRR 0.291; SM parent-level (in-map) 0.180 / 0.320.

  python3 sm_fs_filing.py            # scan, freeze holdout, score the exploration split
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import build_e5_tables

DEFAULT_ROOT = "/mnt/c/Users/johnc/Dropbox/root"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--min-maps", type=int, default=3, help="catalog floor (= PT min_bm analog)")
    ap.add_argument("--holdout-frac", type=float, default=0.40)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--ledger", default=os.path.expanduser("~/mu_data/sm_fs_ledger.json"))
    ap.add_argument("--e5-cache", default="/tmp/mu_data/sm_fs_e5.pt")
    a = ap.parse_args(argv)

    rows = []
    for dirpath, _, files in os.walk(a.root):
        rel = os.path.relpath(dirpath, a.root)
        parts = [] if rel == "." else rel.split(os.sep)
        if any("private" in p.lower() for p in parts):
            continue
        for f in files:
            if f.endswith(".smmx") and "private" not in f.lower():
                stem = f[:-5].strip()
                if stem and parts:                      # depth-0 maps have no filing folder
                    rows.append({"title": stem, "dir": parts[-1], "path": "/".join(parts)})
    print(f"maps with a filing folder (privacy-filtered): {len(rows)}")

    # catalog: directories holding >= min_maps maps
    from collections import Counter
    per_dir = Counter(r["path"] for r in rows)
    eligible_paths = {p for p, c in per_dir.items() if c >= a.min_maps}
    rows = [r for r in rows if r["path"] in eligible_paths]
    cat_titles = sorted({r["dir"] for r in rows})
    by_title = {t: i for i, t in enumerate(cat_titles)}
    print(f"catalog: {len(cat_titles)} distinct directory names "
          f"({len(eligible_paths)} dirs >= {a.min_maps} maps); queries: {len(rows)}")

    # FROZEN HOLDOUT at first touch — deterministic by content hash, recorded before any scoring
    def qkey(r):
        return hashlib.sha256(f"{r['path']}/{r['title']}".encode()).hexdigest()
    rows.sort(key=qkey)
    rng = np.random.default_rng(a.split_seed)
    perm = rng.permutation(len(rows))
    n_hold = int(len(rows) * a.holdout_frac)
    hold_idx = set(perm[:n_hold].tolist())
    explore = [r for i, r in enumerate(rows) if i not in hold_idx]
    held_hashes = sorted(qkey(rows[i]) for i in hold_idx)
    os.makedirs(os.path.dirname(a.ledger), exist_ok=True)
    ledger = {
        "corpus": "simplemind-fs-filing-v1", "root": a.root, "min_maps": a.min_maps,
        "split_seed": a.split_seed, "holdout_frac": a.holdout_frac,
        "n_total": len(rows), "n_explore": len(explore), "n_reserved": n_hold,
        "reserved_query_hashes_sha256": hashlib.sha256(
            "".join(held_hashes).encode()).hexdigest(),
        "catalog_sha256": hashlib.sha256("\n".join(cat_titles).encode()).hexdigest(),
        "status": "reserved split NEVER scored by sm_fs_filing.py; confirmatory use only",
    }
    json.dump(ledger, open(a.ledger, "w"), indent=1)
    print(f"ledger -> {a.ledger} (reserved {n_hold} queries, digest "
          f"{ledger['reserved_query_hashes_sha256'][:16]})")

    # e5 ranking on the EXPLORATION split only
    q_titles = [r["title"] for r in explore]
    names = sorted(set(q_titles) | set(cat_titles))
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=a.e5_cache, batch_size=128)
    Q, P = qtbl.numpy(), ptbl.numpy()
    cv = np.stack([P[idx[t]] for t in cat_titles])
    ranks = []
    for r in explore:
        cos = Q[idx[r["title"]]] @ cv.T
        tp = by_title[r["dir"]]
        rk = 1 + int(np.sum(cos > cos[tp]))
        # title-equivalence: best rank over identical directory names
        for j, t in enumerate(cat_titles):
            if t == r["dir"] and j != tp:
                rk = min(rk, 1 + int(np.sum(cos > cos[j])))
        ranks.append(rk)
    rk = np.array(ranks, float)
    print(f"\nSimpleMind-FS filing (exploration split, n={len(rk)}, catalog {len(cat_titles)}):")
    print(f"  MRR {np.mean(1 / rk):.3f}  R@1 {np.mean(rk <= 1):.3f}  R@5 {np.mean(rk <= 5):.3f}  "
          f"R@50 {np.mean(rk <= 50):.3f}  med {int(np.median(rk))}")
    print("  comparators: PT 0.203/0.291 (catalog 335); SM in-map parent 0.180/0.320 (catalog 200)")


if __name__ == "__main__":
    main()
