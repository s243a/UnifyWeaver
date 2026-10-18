#!/usr/bin/env python3
"""Emit the wiki-training-cloud GAP-TARGET list for the STEM gap harvest (#3936).

REPORT_hybrid_candidates.md §B′: μ's competence tracks e5 proximity to the wiki training cloud
(MRR_μ 0.074 far-tercile → 0.175 near → 0.219 both-near), so the demonstrated lever for making μ
competitive anywhere is training-cloud COVERAGE. This script turns that finding into a work order:
the bookmarks and folders sitting FAR from the 8,964 campaign training titles, ranked farthest
first — the empirical target list for gap-directed harvesting/augmentation.

Per row: kind (bookmark|folder), title, wiki_density (mean top-5 e5 cosine to the training cloud),
and the 3 nearest training titles (to show what the cloud's closest coverage currently is).

PRIVACY: bookmark/folder titles come from a personal Pearltrees account. Output goes to the
durable PRIVATE data home (~/mu_data), never into the repo. The harvest itself targets PUBLIC
STEM sources near these regions (#3936's privacy-aware framing) — this list only says WHERE the
gaps are, in the model's own input space.

  python3 emit_gap_targets.py                       # writes ~/mu_data/gap_targets.tsv
  python3 emit_gap_targets.py --deciles 3           # keep the bottom-3 density deciles only
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from filing_ranker import load_graph_universe
from mu_attention import build_e5_tables


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/mu_data/gap_targets.tsv"))
    ap.add_argument("--cloud", default=os.path.expanduser("~/mu_data/campaign_100k_e5.pt"))
    ap.add_argument("--e5-cache", default="/tmp/mu_data/pt_ranker_e5.pt")
    ap.add_argument("--max-queries", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7, help="query sampling (= the harness manifest)")
    ap.add_argument("--deciles", type=int, default=10,
                    help="keep the bottom-N density deciles (10 = everything, farthest first)")
    ap.add_argument("--topn", type=int, default=3, help="nearest cloud titles shown per row")
    a = ap.parse_args(argv)

    import torch
    universe, titles, neighbors, parents_dir, queries, cand, cut_ext = load_graph_universe(2)
    queries = sorted(queries)
    if len(queries) > a.max_queries:
        queries = random.Random(a.seed).sample(queries, a.max_queries)
    f_titles = sorted({cand[f] for f in cand})
    q_titles = sorted({q for q, _ in queries})
    ext = sorted({x for xs in cut_ext.values() for x in xs})
    names = sorted(set(q_titles) | set(f_titles) | {titles[n] for n in universe}
                   | {titles[x] for x in ext})
    _, ptbl, idx = build_e5_tables(names, cache_path=a.e5_cache, batch_size=128)
    P = ptbl.numpy()

    z = torch.load(a.cloud, map_location="cpu", weights_only=False)
    W = z["passage"].numpy()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    cloud_names = z["names"]

    rows = []
    for kind, tl in (("bookmark", q_titles), ("folder", f_titles)):
        X = np.stack([P[idx[t]] for t in tl])
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        S = X @ Wn.T
        top = np.argsort(-S, axis=1)[:, :max(5, a.topn)]
        dens = np.sort(S, axis=1)[:, -5:].mean(axis=1)
        for i, t in enumerate(tl):
            near = "; ".join(cloud_names[j] for j in top[i, :a.topn])
            rows.append((kind, t, float(dens[i]), near))

    dens_all = np.array([r[2] for r in rows])
    cut = np.quantile(dens_all, a.deciles / 10.0)
    keep = sorted((r for r in rows if r[2] <= cut), key=lambda r: r[2])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# gap targets (farthest from the wiki training cloud first) — PRIVATE, do not "
                "commit\n# kind\ttitle\twiki_density\tnearest_cloud_titles\n")
        for kind, t, d, near in keep:
            f.write(f"{kind}\t{t}\t{d:.4f}\t{near}\n")
    print(f"cloud: {len(cloud_names)} training titles; scored {len(rows)} targets "
          f"({len(q_titles)} bookmarks + {len(f_titles)} folders)")
    print(f"kept bottom {a.deciles} decile(s): {len(keep)} rows -> {a.out}")
    b = [r for r in keep if r[0] == "bookmark"]
    fo = [r for r in keep if r[0] == "folder"]
    print(f"  bookmarks {len(b)} (density {b[0][2]:.3f}–{b[-1][2]:.3f})" if b else "  bookmarks 0")
    print(f"  folders   {len(fo)} (density {fo[0][2]:.3f}–{fo[-1][2]:.3f})" if fo else "  folders 0")
    print("\nfarthest 10 (the sharpest gaps):")
    for kind, t, d, near in keep[:10]:
        print(f"  {d:.3f} {kind:8s} {t[:48]:48s} | near: {near[:60]}")


if __name__ == "__main__":
    main()
