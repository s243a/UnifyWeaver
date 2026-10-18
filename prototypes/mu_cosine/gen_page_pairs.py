#!/usr/bin/env python3
"""Turn page-membership edges (from fetch_category_pages.py) into a scored-pair candidate file for the
ELEMENT-OF relation (page in category), tagged so the loader/model can route them to the element-of
operator + node-type=page (DESIGN_calibrated_judges.md §7) — distinct from category subcategory edges.

The membership itself is free+true (a listed page IS a member); Haiku grades the CENTRALITY gradient
(core topic page = 1.0 ... peripheral application = low). Direction: a=category (topic), b=page, scored as
"how central is page B to category A's topic". Free μ=0 negatives pair our pages with clearly off-domain
categories. Extended columns relation/a_type/b_type ride after the standard 5 (load_pairs ignores extras,
so it stays backward-compatible).

    python3 gen_page_pairs.py --members page_members.tsv --out mu_pairs_pages.tsv
"""
import argparse
import os
import random
from collections import defaultdict

from gen_more_sym_pairs import load_existing_keys

ROOT = os.path.dirname(os.path.abspath(__file__))
# clearly off-domain categories for free non-membership negatives (page definitely NOT a member)
OFF_DOMAIN = ["Cooking", "Galaxies", "Feudalism", "Medieval_knights", "Tornado", "Comedians",
              "Television_stations", "Chess_openings", "1978_albums", "Public_toilets_automation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default=os.path.join(ROOT, "page_members.tsv"))
    ap.add_argument("--neg-ratio", type=float, default=2.0)
    ap.add_argument("--dedup-against", default=os.path.join(ROOT, "mu_pairs_scored_cumulative.tsv"))
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs_pages.tsv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cat_pages = defaultdict(list)
    with open(args.members, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 2 and p[0] and p[1]:
                cat_pages[p[1]].append(p[0])     # category -> [pages]

    existing = load_existing_keys(args.dedup_against) if os.path.exists(args.dedup_against) else set()
    pairs = set(existing)
    rows = []   # (a, b, stratum, wl, mu, relation, a_type, b_type)

    # centrality positives: (category, page) element_of — Haiku grades how central the page is
    for cat, pages in sorted(cat_pages.items()):
        if not pages:
            continue
        st = "pos_pageof_" + cat.lower()[:12]
        for pg in sorted(set(pages)):
            k = tuple(sorted((cat, pg)))
            if k in pairs:
                continue
            pairs.add(k)
            rows.append((cat, pg, st, -1, "", "element_of", "category", "page"))

    npos = len(rows)
    all_pages = sorted({pg for pgs in cat_pages.values() for pg in pgs})
    nn = 0
    while nn < args.neg_ratio * npos and all_pages:
        pg = rng.choice(all_pages)
        cat = rng.choice(OFF_DOMAIN)
        k = tuple(sorted((cat, pg)))
        if k in pairs:
            continue
        pairs.add(k)
        rows.append((cat, pg, "neg", -1, "0.0", "element_of", "category", "page"))
        nn += 1
    rng.shuffle(rows)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# page-membership (element_of) candidates (gen_page_pairs.py). a=category b=page; "
                "Haiku grades CENTRALITY of page in category. neg=off-domain non-membership μ0.\n")
        f.write("# cols: a\tb\tstratum\twl\tmu\trelation\ta_type\tb_type  (extras ignored by load_pairs)\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    from collections import Counter
    cnt = Counter(r[2] for r in rows)
    print("wrote " + str(len(rows)) + " -> " + args.out)
    for k, v in sorted(cnt.items()):
        print(f"  {k:28} {v}")
    print(f"to Haiku-score (non-neg): {sum(v for k, v in cnt.items() if k != 'neg')}")


if __name__ == "__main__":
    main()
