#!/usr/bin/env python3
"""Stratified pair sampler for the S/D channel scoring campaign (B1 disposition: blocked on data).

B1's failure analysis: all prior S-labeled pairs were TRANSITIVE ANCESTOR pairs, where `assoc` barely varies —
an S head needs pairs where S VARIES. Strata (per corpus):
  transitive  — h1..5 upward ancestor pairs (the familiar stratum; D varies)
  sibling     — shared parent, neither an ancestor of the other (S should be HIGH-variance here)
  cousin      — shared grandparent, disjoint parents (moderate association)
  random      — random node pairs within the corpus (mostly-unrelated negatives; S low, D low)

Corpora: 100k_cats (TSV) and the Behavior slice (enwiki_cats_correct scoped LMDB, same retained slice as the
confirmatory run). Already-scored pairs (the two 250-pair multihop sets + judge2) are EXCLUDED. Output is the
standard score-input TSV (neighborhood tags campaign_h{k} / campaign_sib / campaign_cous / campaign_rand) plus
a JSON manifest.

  python3 sample_channel_campaign.py --per-corpus 1000 --out /tmp/mu_data/campaign_pairs.tsv
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_product_kalman_realdata import DATASETS
from sigma_hop_confirmatory import FeatureGraphConfig, load_feature_graph, load_scored_pairs

ROOT = os.path.dirname(os.path.abspath(__file__))


def ancestors(parents, x, hmax=6):
    """{node: hop} for ancestors of x up to hmax (BFS on parent links)."""
    out, frontier = {}, {x: 0}
    for h in range(1, hmax + 1):
        nxt = {}
        for n in frontier:
            for p in parents.get(n, ()):  # parents may map to list/set
                if p not in out and p != x:
                    out[p] = h
                    nxt[p] = h
        frontier = nxt
        if not frontier:
            break
    return out


def sample_transitive(parents, nodes, rng, n_per_hop, hmax=5):
    got = {h: [] for h in range(1, hmax + 1)}
    order = rng.permutation(len(nodes))
    for j in order:
        x = nodes[j]
        anc = ancestors(parents, x, hmax)
        by_h = {}
        for a, h in anc.items():
            by_h.setdefault(h, []).append(a)
        for h in range(1, hmax + 1):
            if len(got[h]) < n_per_hop and by_h.get(h):
                got[h].append((x, by_h[h][rng.integers(len(by_h[h]))], f"campaign_h{h}"))
        if all(len(got[h]) >= n_per_hop for h in got):
            break
    return [p for h in got for p in got[h]]


def sample_lateral(parents, children, nodes, rng, n_sib, n_cous):
    sibs, cous, seen = [], [], set()
    order = rng.permutation(len(nodes))
    for j in order:
        x = nodes[j]
        ps = list(parents.get(x, ()))
        if not ps:
            continue
        anc_x = set(ancestors(parents, x, 3))
        if len(sibs) < n_sib:
            p = ps[rng.integers(len(ps))]
            kids = [k for k in children.get(p, ()) if k != x and k not in anc_x and x not in ancestors(parents, k, 3)]
            if kids:
                y = kids[rng.integers(len(kids))]
                key = tuple(sorted((x, y)))
                if key not in seen:
                    seen.add(key); sibs.append((x, y, "campaign_sib"))
        if len(cous) < n_cous:
            p = ps[rng.integers(len(ps))]
            gps = list(parents.get(p, ()))
            if gps:
                gp = gps[rng.integers(len(gps))]
                uncles = [u for u in children.get(gp, ()) if u != p]
                if uncles:
                    u = uncles[rng.integers(len(uncles))]
                    cs = [c for c in children.get(u, ()) if c != x and c not in anc_x]
                    if cs:
                        y = cs[rng.integers(len(cs))]
                        key = tuple(sorted((x, y)))
                        if key not in seen:
                            seen.add(key); cous.append((x, y, "campaign_cous"))
        if len(sibs) >= n_sib and len(cous) >= n_cous:
            break
    return sibs + cous


def sample_random(nodes, rng, n):
    out, seen = [], set()
    while len(out) < n:
        i, j = rng.integers(len(nodes)), rng.integers(len(nodes))
        if i == j:
            continue
        key = tuple(sorted((nodes[i], nodes[j])))
        if key in seen:
            continue
        seen.add(key); out.append((nodes[i], nodes[j], "campaign_rand"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-corpus", type=int, default=1000)
    ap.add_argument("--out", default="/tmp/mu_data/campaign_pairs.tsv")
    ap.add_argument("--manifest", default="/tmp/mu_data/campaign_manifest.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    # exclude everything already scored
    scored = set()
    for name in ("exploratory", "fresh"):
        cfg = DATASETS[name]
        prs, *_ = load_scored_pairs(cfg["score_in"], cfg["responses"], prefix="transitive_h")
        scored |= {tuple(sorted(p)) for p in prs}

    rows, manifest = [], {"per_corpus": a.per_corpus, "seed": a.seed, "excluded_already_scored": len(scored), "strata": {}}
    for name in ("exploratory", "fresh"):
        cfg = DATASETS[name]
        parents, children, deg, _ = load_feature_graph(FeatureGraphConfig(**cfg["graph"]))
        nodes = sorted(set(parents) | {c for kids in children.values() for c in kids})
        n3 = a.per_corpus // 3
        trans = sample_transitive(parents, nodes, rng, n_per_hop=max(1, n3 // 5))
        lat = sample_lateral(parents, children, nodes, rng, n_sib=n3 // 2, n_cous=n3 - n3 // 2)
        rand = sample_random(nodes, rng, a.per_corpus - len(trans) - len(lat))
        pairs = [p for p in trans + lat + rand if tuple(sorted((p[0], p[1]))) not in scored]
        rows += [(x, y, "subcategory", "1.0", tag, "category", "category", "") for x, y, tag in pairs]
        cnt = {}
        for _, _, tag in pairs:
            cnt[tag] = cnt.get(tag, 0) + 1
        manifest["strata"][name] = cnt
        print(f"{name}: {len(pairs)} pairs  {cnt}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# node_title\troot_title\tcur_relation\tconf\tneighborhood\tnode_type\troot_type\traw\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    json.dump(manifest, open(a.manifest, "w"), indent=1)
    print(f"wrote {len(rows)} pairs → {a.out}; manifest → {a.manifest}")


if __name__ == "__main__":
    main()
