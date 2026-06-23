#!/usr/bin/env python3
"""Math-deepening round — seed downward from core mathematics subfields (the enwiki graph finally has the
depth). Groups the subfields into AREAS so within-area positives are coherent (both analysis / both
algebra / …) and cross-area pairs are graded (Geometry×Topology high, Analysis×Foundations lower).
Closures are topically coherent by construction (admin-filtered); no e5 needed at generation. Deduped vs
the cumulative scored set; negatives free.

Strata: pos_analysis, pos_algebra, pos_geomtop, pos_foundations, pos_discrete (within-area);
cross_AA (analysis×algebra), cross_GA (geometry×algebra), cross_AF (analysis×foundations),
cross_GT_disc (geom/top × discrete); cross_MP (math×physics, math-of-physics), cross_MS (math×CS).

    UW_MU_GRAPH=../../data/benchmark/wide_enwiki_math/category_parent.tsv python3 gen_math_fields_pairs.py
"""
import argparse
import os
import re
import random

from gen_multidomain_pairs import gen_within, gen_cross, e5_cos_to_roots
from gen_more_sym_pairs import build_children_adj, load_existing_keys
from gen_mu_pairs import load_graph

# e5 μ-coherence: enwiki is densely cross-linked, so even depth-2 closures leak physics/CS/music into the
# math areas. Keep only nodes whose e5 cosine ARGMAX over the domain roots is Mathematics (∩ a floor).
COH_ROOTS = ["Mathematics", "Physics", "Chemistry", "Computer_science", "Engineering",
             "Artificial_intelligence"]

ROOT = os.path.dirname(os.path.abspath(__file__))
SCORED_CUMULATIVE = os.path.join(ROOT, "mu_pairs_scored_cumulative.tsv")
SLICE = os.environ.get("UW_MU_GRAPH", os.path.join(
    os.path.abspath(os.path.join(ROOT, "..", "..")), "data", "benchmark", "wide_enwiki_math",
    "category_parent.tsv"))

AREAS = {
    "analysis": ["Mathematical_analysis", "Real_analysis", "Complex_analysis", "Calculus",
                 "Differential_equations", "Complex_numbers"],
    "algebra": ["Group_theory", "Abstract_algebra", "Linear_algebra", "Number_theory", "Category_theory"],
    "geomtop": ["Geometry", "Topology"],
    "foundations": ["Set_theory", "Mathematical_logic"],
    "discrete": ["Combinatorics", "Probability_theory"],
}
PHYS_ROOTS = ["Physics", "Classical_mechanics", "Quantum_mechanics", "Electromagnetism", "Thermodynamics"]
CS_ROOTS = ["Computer_science", "Theoretical_computer_science", "Algorithms"]
ADMIN = re.compile(r"(Wikipedia|Articles?_|All_|Hidden_|CS1|Pages_|Webarchive|Commons|_stubs?$|Stub|"
                   r"Redirects|Short_desc|Use_|Templates?|Track|_by_|_in_\d|established_in|introductions|"
                   r"disambiguation|mathematicians|_people$|_journals$|_awards$)")


def closure(roots, children, depth, present):
    seen = set(r for r in roots if r in present)
    fr = list(seen)
    for _ in range(depth):
        nx = []
        for n in fr:
            for c in children.get(n, ()):
                if c not in seen and not ADMIN.search(c):
                    seen.add(c); nx.append(c)
        fr = nx
        if not fr:
            break
    return {n for n in seen if not ADMIN.search(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-within", type=int, default=55, help="within-area positives, per area")
    ap.add_argument("--n-cross-area", type=int, default=35, help="per cross-area stratum")
    ap.add_argument("--n-mp", type=int, default=60, help="Math×Physics (math-of-physics)")
    ap.add_argument("--n-ms", type=int, default=60, help="Math×CS (theory)")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--math-floor", type=float, default=0.74, help="cos|Mathematics floor for coherence")
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--dedup-against", default=SCORED_CUMULATIVE)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs_mathfields.tsv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    full = load_graph(SLICE)
    children = build_children_adj(SLICE)
    deg = {n: max(1, len(full.get(n, ()))) for n in full}
    present = set(full.keys())

    # e5 coherence: keep nodes whose argmax over COH_ROOTS is Mathematics (∩ cos floor)
    names_sorted = sorted(present)
    cos = e5_cos_to_roots(names_sorted, COH_ROOTS, cache=os.path.join(ROOT, "e5_mathfields_cos.pt"))
    C = {r: {n: cos[r][i] for i, n in enumerate(names_sorted)} for r in COH_ROOTS}
    def math_coh(n):
        return (max(COH_ROOTS, key=lambda r: C[r][n]) == "Mathematics"
                and C["Mathematics"][n] >= args.math_floor)
    raw = {a: closure(roots, children, args.depth, present) for a, roots in AREAS.items()}
    pools = {a: {n for n in p if math_coh(n)} | {r for r in AREAS[a] if r in present}
             for a, p in raw.items()}
    phys = {n for n in closure(PHYS_ROOTS, children, 2, present)
            if max(COH_ROOTS, key=lambda r: C[r][n]) == "Physics"}
    cs = {n for n in closure(CS_ROOTS, children, 2, present)
          if max(COH_ROOTS, key=lambda r: C[r][n]) == "Computer_science"}
    math_all = set().union(*pools.values())
    print(f"slice {len(present)} nodes; math areas:")
    for a, p in pools.items():
        print(f"  {a:12} {len(p):4d}  {sorted(p)[:6]}")
    print(f"  math_all {len(math_all)}  Physics-pool {len(phys)}  CS-pool {len(cs)}")

    existing = load_existing_keys(args.dedup_against) if os.path.exists(args.dedup_against) else set()
    pairs = set(existing)
    rows = []

    def within(pool, n, st):
        for aa, bb, wl in gen_within(pool, "Mathematics", children, deg, n, pairs, rng):
            rows.append((aa, bb, st, wl))

    def cross(pa, pb, n, st):
        for aa, bb, _ in gen_cross(set(pa), set(pb), n, pairs, rng):
            rows.append((aa, bb, st, -1))

    for a in AREAS:
        within(pools[a], args.n_within, f"pos_{a}")
    cross(pools["analysis"], pools["algebra"], args.n_cross_area, "cross_AA")
    cross(pools["geomtop"], pools["algebra"], args.n_cross_area, "cross_GA")
    cross(pools["analysis"], pools["foundations"], args.n_cross_area, "cross_AF")
    cross(pools["geomtop"], pools["discrete"], args.n_cross_area, "cross_GTd")
    cross(math_all, phys, args.n_mp, "cross_MP")
    cross(math_all, cs, args.n_ms, "cross_MS")

    domain = sorted(math_all)
    names = sorted(present)
    n_pos = len(rows)
    n_neg_target, tries, n_neg = int(round(n_pos * args.neg_ratio)), 0, 0
    while n_neg < n_neg_target and tries < n_neg_target * 80:
        tries += 1
        aa, bb = rng.choice(domain), rng.choice(names)
        if aa == bb or bb in full.get(aa, ()):
            continue
        key = tuple(sorted((aa, bb)))
        if key in pairs:
            continue
        pairs.add(key)
        rows.append((aa, bb, "neg", -1))
        n_neg += 1
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        f.write("# math-deepening candidates (gen_math_fields_pairs.py; corpus=enwiki; deduped vs "
                "cumulative). strata: pos_{analysis,algebra,geomtop,foundations,discrete} (within-area),\n")
        f.write("# cross_AA/GA/AF/GTd (cross-area), cross_MP (math×physics), cross_MS (math×CS), neg=μ0. "
                "BLANK μ to Haiku-score. cols: a\tb\tstratum\twl\tmu\n")
        for aa, bb, st, wl in rows:
            f.write(f"{aa}\t{bb}\t{st}\t{wl}\t{'0.0' if st == 'neg' else ''}\n")
    from collections import Counter
    cnt = Counter(r[2] for r in rows)
    print("wrote " + str(len(rows)) + " rows -> " + args.out + ": "
          + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    print(f"to Haiku-score (non-neg): {sum(v for k, v in cnt.items() if k != 'neg')}")


if __name__ == "__main__":
    main()
