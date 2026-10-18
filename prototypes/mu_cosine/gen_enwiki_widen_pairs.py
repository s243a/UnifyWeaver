#!/usr/bin/env python3
"""Graph-widening round (#3313) — sample the AI + modern-physics subtrees that enwiki finally provides
(scripts/ingest_enwiki_categories.py → data/benchmark/wide_enwiki/). Pools are depth-bounded downward
closures from clean subfield roots (topically coherent by construction; apex/admin cats filtered), so no
e5 is needed at generation time. Deduped vs the cumulative mu_pairs_scored_eng_260621-174251.tsv; negatives free.

Strata: within-AI, within-modern-physics, within-Math(depth); cross AI×CS (expect high-to-both, like
Mechanics×Engineering), modern-physics×classical-physics (intra-physics continuity), Math×{Physics,CS},
AI×Math (theory/ML-math).

    UW_MU_GRAPH=../../data/benchmark/wide_enwiki/category_parent.tsv python3 gen_enwiki_widen_pairs.py
"""
import argparse
import os
import re
import random

from gen_multidomain_pairs import gen_within, gen_cross
from gen_more_sym_pairs import build_children_adj, load_existing_keys
from gen_mu_pairs import load_graph, walk_bidir, GRAPH

ROOT = os.path.dirname(os.path.abspath(__file__))
SCORED_CUMULATIVE = os.path.join(ROOT, "mu_pairs_scored_eng_260621-174251.tsv")
SLICE = os.environ.get("UW_MU_GRAPH", os.path.join(
    os.path.abspath(os.path.join(ROOT, "..", "..")), "data", "benchmark", "wide_enwiki", "category_parent.tsv"))

AI_ROOTS = ["Artificial_intelligence", "Machine_learning", "Neural_networks", "Deep_learning",
            "Natural_language_processing", "Computer_vision", "Robotics", "Reinforcement_learning"]
MODPHYS_ROOTS = ["Quantum_mechanics", "Particle_physics", "Quantum_field_theory", "Theory_of_relativity",
                 "Statistical_mechanics", "Condensed_matter_physics", "Nuclear_physics", "Astrophysics"]
CLASSICAL_PHYS = ["Mechanics", "Thermodynamics", "Electromagnetism", "Optics", "Classical_mechanics",
                  "Motion_(physics)", "Energy", "Heat", "Acoustics", "Waves", "Electricity", "Magnetism"]
CS_ROOTS = ["Computer_science", "Subfields_of_computer_science", "Areas_of_computer_science"]
MATH_ROOT = "Mathematics"

ADMIN = re.compile(r"(Wikipedia|Articles?_|All_|Hidden_|CS1|Pages_|Webarchive|Commons|_stubs?$|Stub|"
                   r"Redirects|Short_desc|Use_|Templates?|Track|_by_|_in_\d|established_in|introductions|"
                   r"_navigational_boxes|disambiguation)")


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
    ap.add_argument("--n-ai", type=int, default=140)
    ap.add_argument("--n-modphys", type=int, default=120)
    ap.add_argument("--n-math", type=int, default=80)
    ap.add_argument("--n-aics", type=int, default=80)
    ap.add_argument("--n-mpcp", type=int, default=80)
    ap.add_argument("--n-mathphys", type=int, default=50)
    ap.add_argument("--n-mathcs", type=int, default=50)
    ap.add_argument("--n-aimath", type=int, default=40)
    ap.add_argument("--depth-sub", type=int, default=2, help="closure depth for leaf subfield roots")
    ap.add_argument("--depth-broad", type=int, default=3, help="closure depth for Math/CS")
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--dedup-against", default=SCORED_CUMULATIVE)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs_enwiki.tsv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    full = load_graph(SLICE)
    children = build_children_adj(SLICE)
    parents = {}
    for par, ch in children.items():
        for c in ch:
            parents.setdefault(c, set()).add(par)
    deg = {n: max(1, len(full.get(n, ()))) for n in full}
    present = set(full.keys())

    ai = closure(AI_ROOTS, children, args.depth_sub, present)
    modphys = closure(MODPHYS_ROOTS, children, args.depth_sub, present)
    math = closure([MATH_ROOT], children, args.depth_broad, present)
    cs = closure(CS_ROOTS, children, args.depth_broad, present)
    classical = [n for n in CLASSICAL_PHYS if n in present]
    phys_all = closure(["Physics"], children, args.depth_broad, present)

    print(f"slice: {len(present)} nodes")
    print(f"  AI pool {len(ai)}: {sorted(ai)[:14]}")
    print(f"  modern-physics pool {len(modphys)}: {sorted(modphys)[:14]}")
    print(f"  Math {len(math)}  CS {len(cs)}  classical-physics {len(classical)}  Physics-all {len(phys_all)}")

    existing = load_existing_keys(args.dedup_against)
    pairs = set(existing)
    rows = []

    def add_within(pool, root, n, st):
        for aa, bb, wl in gen_within(pool, root, children, deg, n, pairs, rng):
            rows.append((aa, bb, st, wl))

    def add_cross(pa, pb, n, st):
        for aa, bb, _ in gen_cross(set(pa), set(pb), n, pairs, rng):
            rows.append((aa, bb, st, -1))

    add_within(ai, "Artificial_intelligence", args.n_ai, "pos_ai")
    add_within(modphys, "Physics", args.n_modphys, "pos_modphys")
    add_within(math, "Mathematics", args.n_math, "pos_math")
    add_cross(ai, cs, args.n_aics, "cross_AICS")             # AI×CS — the high-to-both test
    add_cross(modphys, classical, args.n_mpcp, "cross_MPCP")  # modern×classical physics continuity
    add_cross(math, phys_all, args.n_mathphys, "cross_MP")
    add_cross(math, cs, args.n_mathcs, "cross_MS")
    add_cross(ai, math, args.n_aimath, "cross_AIMath")

    domain = sorted(ai | modphys | math | cs)
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
        f.write("# enwiki graph-widening candidates (gen_enwiki_widen_pairs.py; corpus=enwiki; deduped vs "
                "mu_pairs_scored_eng_260621-174251.tsv). strata: pos_ai, pos_modphys, pos_math, cross_AICS (high-to-both),\n")
        f.write("# cross_MPCP (modern×classical physics), cross_MP/MS (Math×{Phys,CS}), cross_AIMath, "
                "neg=μ0. BLANK μ to Haiku-score. cols: a\tb\tstratum\twl\tmu\n")
        for aa, bb, st, wl in rows:
            f.write(f"{aa}\t{bb}\t{st}\t{wl}\t{'0.0' if st == 'neg' else ''}\n")
    from collections import Counter
    cnt = Counter(r[2] for r in rows)
    print("wrote " + str(len(rows)) + " rows -> " + args.out + ": "
          + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    print(f"to Haiku-score (non-neg): {sum(v for k, v in cnt.items() if k != 'neg')}")


if __name__ == "__main__":
    main()
