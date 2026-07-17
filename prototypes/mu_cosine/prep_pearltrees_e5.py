#!/usr/bin/env python3
"""Build the e5 cache for the Pearltrees labeling campaign (prep_campaign_e5 pattern).

The Tokenizer embeds each pair's node, root, AND sampled ancestors, so the cache must cover the
campaign endpoints' ancestor cones too.  Node identity downstream is the (audited) TITLE — the
scored-TSV join key — so the cache is keyed by title, with the audited Pearltrees title policy
applied by endpoint id (same policy as the sampler).

  python3 prep_pearltrees_e5.py \
      --paths-jsonl ../../.local/data/api_tree_paths_v8.jsonl \
      --titles-tsv ../../.local/data/pearltrees_api/assembled_titles.tsv \
      --pairs /tmp/mu_data/pt_campaign_all_pairs.tsv \
      --out /tmp/mu_data/pt_campaign_e5.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import build_e5_tables
from sample_pearltrees_lateral import ancestors, build_graph, load_policy
from sample_product_kalman_pearltrees_campaign import load_principal_paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths-jsonl", required=True)
    ap.add_argument("--titles-tsv", required=True)
    ap.add_argument("--pairs", default="/tmp/mu_data/pt_campaign_all_pairs.tsv")
    ap.add_argument("--out", default="/tmp/mu_data/pt_campaign_e5.pt")
    ap.add_argument("--hmax", type=int, default=6)
    a = ap.parse_args(argv)

    forest = load_principal_paths(a.paths_jsonl, a.titles_tsv)
    parents, _ = build_graph(forest)
    corrections = load_policy()

    def title_of(node_id):
        return corrections.get(node_id, forest["titles"].get(node_id, ""))

    endpoint_ids = set()
    accounts = {}
    with open(a.pairs, encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < len(header):
                continue
            for key in ("a_id", "b_id"):
                endpoint_ids.add(c[col[key]])
                accounts[c[col[key]]] = c[col["account"]]

    names = set()
    for nid in endpoint_ids:
        node = (accounts[nid], nid)
        cone = ancestors(parents, node, a.hmax)
        for anc_node in list(cone) + [node]:
            t = title_of(anc_node[1])
            if t:
                names.add(t)
    print(f"pearltrees campaign endpoints: {len(endpoint_ids)}; titles to embed (incl. ancestor cones): {len(names)}")
    build_e5_tables(sorted(names), cache_path=a.out, batch_size=128)
    print(f"cache -> {a.out}")


if __name__ == "__main__":
    main()
