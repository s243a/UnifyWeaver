#!/usr/bin/env python3
"""Find PAGE-DATA acquisition targets — the policy: a category node worth supplementing with page data is
one that is (1) THIN on category data (few subcategories, so subcat-sampling is empty), (2) in an AREA OF
INTEREST (the closure of given region roots), and (3) where the model is likely WEAK — low e5
domain-discrimination MARGIN (top1−top2 cosine to the domain roots). e5's low-margin nodes are exactly its
blind spots, and since the model's cross-domain discrimination is e5-driven, that margin is the ex-ante
predictor of where the model is uncertain. Ranks thin∧weak∧relevant nodes so page-harvest budget
(fetch_category_pages.py) goes where it matters most. Streams the full enwiki graph (it OOMs if loaded).

    python3 find_data_gaps.py --region-root Systems_theory --region-root Dynamical_systems \
        --region-root Control_theory --region-root Network_theory --max-subcats 3 --top 30 --page-counts
"""
import argparse
import os
import re
from collections import defaultdict

from gen_multidomain_pairs import e5_cos_to_roots

ROOT = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(os.path.abspath(os.path.join(ROOT, "..", "..")), "data", "benchmark",
                    "enwiki_named", "category_parent.tsv")
DOMAINS = ["Mathematics", "Physics", "Chemistry", "Computer_science", "Engineering",
           "Artificial_intelligence"]
ADMIN = re.compile(r"(Wikipedia|Articles?_|All_|Hidden_|CS1|Pages_|Webarchive|Commons|_stubs?$|Stub|"
                   r"Redirects|Short_desc|Use_|Templates?|Track|_by_|_in_\d|established_in|introductions|"
                   r"disambiguation|_people$|_journals$|_awards$|Wikipedians)")


def stream(path):
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[0] and p[1]:
                yield p[0], p[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-root", action="append", required=True)
    ap.add_argument("--graph", default=FULL)
    ap.add_argument("--depth", type=int, default=2, help="region closure depth")
    ap.add_argument("--max-subcats", type=int, default=3, help="THIN threshold (≤ this = candidate)")
    ap.add_argument("--margin-max", type=float, default=0.20, help="WEAK threshold on e5 domain margin")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--page-counts", action="store_true", help="fetch member-page counts (API) for the top")
    ap.add_argument("--coh-cache", default=os.path.join(ROOT, "e5_gaps_cos.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_gaps.tsv"))
    args = ap.parse_args()

    # --- region closure (area of interest) ---
    region = set(args.region_root)
    frontier = set(args.region_root)
    for d in range(args.depth):
        nx = set()
        for c, par in stream(args.graph):
            if par in frontier and c not in region and not ADMIN.search(c):
                nx.add(c)
        region |= nx
        frontier = nx
        print(f"  region depth {d+1}: +{len(nx)} (total {len(region)})")
        if not frontier:
            break

    # --- subcat counts (THIN signal) ---
    sub = defaultdict(int)
    for c, par in stream(args.graph):
        if par in region and not ADMIN.search(c):
            sub[par] += 1

    # --- e5 domain margin (WEAK signal) ---
    nodes = sorted(region)
    cos = e5_cos_to_roots(nodes, DOMAINS, cache=args.coh_cache)
    C = {r: {n: cos[r][i] for i, n in enumerate(nodes)} for r in DOMAINS}
    rows = []
    for n in nodes:
        vals = sorted(((C[r][n], r) for r in DOMAINS), reverse=True)
        margin = vals[0][0] - vals[1][0]
        rows.append((n, sub.get(n, 0), vals[0][1], round(vals[0][0], 3), round(margin, 3)))

    # candidates: THIN ∧ WEAK, ranked by weakness (ambiguous first), then thinnest
    cand = [r for r in rows if r[1] <= args.max_subcats and r[4] <= args.margin_max]
    cand.sort(key=lambda r: (r[4], r[1]))
    top = cand[:args.top]

    pcounts = {}
    if args.page_counts and top:
        from fetch_category_pages import members
        print(f"  fetching page counts for top {len(top)} …")
        for n, *_ in top:
            try:
                pcounts[n] = sum(1 for _ in members(n, "page"))
            except Exception:
                pcounts[n] = -1

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# page-data gap candidates (find_data_gaps.py). THIN (subcats≤%d) ∧ WEAK (e5 margin≤%.2f) "
                "∧ in region. cols: node\tsubcats\tpages\ttop_domain\tcos\tmargin\n" % (args.max_subcats, args.margin_max))
        for n, sc, dom, cval, mar in top:
            pc = pcounts.get(n, "")
            f.write(f"{n}\t{sc}\t{pc}\t{dom}\t{cval}\t{mar}\n")
    print(f"\nregion {len(region)} nodes; {len(cand)} thin∧weak candidates; top {len(top)} -> {args.out}")
    hdr = f"  {'node':40} {'sub':>3} {'pg':>4} {'domain':10} {'cos':>5} {'margin':>6}"
    print(hdr)
    for n, sc, dom, cval, mar in top:
        pc = pcounts.get(n, "")
        print(f"  {n[:40]:40} {sc:>3} {str(pc):>4} {dom[:10]:10} {cval:>5} {mar:>6}")


if __name__ == "__main__":
    main()
