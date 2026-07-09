#!/usr/bin/env python3
"""Build the e5 cache for the 100k_cats side of the channel campaign.

The Behavior-slice cache (sigma_hop_behavior_slice_e5.pt) already covers every slice node, but the 100k_cats
campaign pairs sample arbitrary graph nodes outside the old 250-pair cache. The Tokenizer embeds each node,
its root, AND its ancestors — so the cache must cover the campaign nodes' ancestor cones too.

  python3 prep_campaign_e5.py --pairs /tmp/mu_data/campaign_pairs.tsv --out /tmp/mu_data/campaign_100k_e5.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import build_e5_tables, load_dag

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
GRAPH_100K = os.path.join(REPO, "data", "benchmark", "100k_cats", "category_parent.tsv")


def ancestor_cone(parents, x, hmax=6):
    out, frontier = set(), {x}
    for _ in range(hmax):
        nxt = set()
        for n in frontier:
            for p in parents.get(n, ()):
                if p not in out:
                    out.add(p); nxt.add(p)
        frontier = nxt
        if not frontier:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/tmp/mu_data/campaign_pairs.tsv")
    ap.add_argument("--out", default="/tmp/mu_data/campaign_100k_e5.pt")
    a = ap.parse_args()

    parents, children, _ = load_dag(GRAPH_100K)
    in_graph = set(parents) | {c for kids in children.values() for c in kids}
    names = set()
    n_rows = 0
    with open(a.pairs, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split("\t")
            if len(c) < 2:
                continue
            x, y = c[0], c[1]
            if x in in_graph and y in in_graph:              # the 100k_cats half of the campaign
                n_rows += 1
                names |= {x, y} | ancestor_cone(parents, x) | ancestor_cone(parents, y)
    print(f"100k_cats campaign rows: {n_rows}; nodes to embed (incl. ancestor cones): {len(names)}")
    build_e5_tables(sorted(names), cache_path=a.out, batch_size=128)
    print(f"cache → {a.out}")


if __name__ == "__main__":
    main()
