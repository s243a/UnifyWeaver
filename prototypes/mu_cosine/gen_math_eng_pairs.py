#!/usr/bin/env python3
"""PART A — DEEPEN Math + add Engineering, building on gen_multidomain_pairs.py (#3310/#3312).

#3312 found Math discrimination 3/5 because Math is data-STARVED: its depth-≤3 closure is only ~12 nodes
AND the strict argmax pool (9) *excluded* the math-of-physics core (`Calculus`, `Differential_equations`,
`Partial_differential_equations`) — those go argmax→Physics because they ARE the maths of physics. The
fix is more DATA in Math (helps a domain) — NOT more breadth (which regressed physics SYM +0.695→+0.570).

So, deduping vs `mu_pairs_scored_4roots_260621-004105.tsv`:
  * **Math pool — DEEPENED & INCLUSIVE**: closure(Mathematics, depth) ∩ (cos|Math ≥ floor), *without* the
    argmax subtraction, so Calculus/Diff-eq/PDE are now trained AS maths (they were never in the Math
    pool before). Within-Math pairs (priority) + the discrimination-critical **Math×Physics** stratum on
    exactly those shared nodes (teach "high to BOTH", the correct cross-domain answer).
  * **Engineering — a MODEST NEW domain** (via the Applied_sciences slice; Engineering closure ∩ argmax
    over the 5 roots, so Mechanics/Thermodynamics→Physics and Computer_engineers→CS are excluded —
    leaving the genuine engineering nodes). Within-Engineering + Eng×{Physics,CS} cross.
  * **Branches_of_science** is the science spine (reaches Physics/Chem/Math/CS/Bio within ~3 hops) — used
    for top-level cross-domain (science×science) pairs.
  * a light **bidir-coinflip** batch at Physics/Chemistry/CS for lateral diversity (#3309).

Physics/Chemistry/CS pools are kept exactly as #3312 (argmax over the 4 original roots). Negatives free.

    python3 gen_math_eng_pairs.py --out mu_pairs_matheng.tsv
"""
import argparse
import os
import random
from itertools import combinations

from gen_multidomain_pairs import (closure, e5_cos_to_roots, calibrate_physics, build_pools_multi,
                                    gen_within, gen_cross, ROOTS as BASE_ROOTS)
from gen_more_sym_pairs import build_children_adj, load_existing_keys
from gen_mu_pairs import load_graph, walk_bidir, GRAPH

ROOT = os.path.dirname(os.path.abspath(__file__))
SCORED_4ROOTS = os.path.join(ROOT, "mu_pairs_scored_4roots_260621-004105.tsv")
ALL_ROOTS = ["Physics", "Chemistry", "Mathematics", "Computer_science", "Engineering"]
SPINE = "Branches_of_science"
# curated clean-physics core for the cross strata (the #3312 Physics pool carries Time/Death leaks; for
# the discrimination-critical Math×Physics pairs we want genuine physics, not those leaks).
PHYS_CORE = ["Mechanics", "Thermodynamics", "Electromagnetism", "Optics", "Motion_(physics)",
             "Classical_mechanics", "Energy", "Wave_mechanics", "Electromagnetic_radiation", "Acoustics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-math", type=int, default=70, help="within-Math (PRIORITY — deepened pool)")
    ap.add_argument("--n-eng", type=int, default=90, help="within-Engineering (modest NEW domain)")
    ap.add_argument("--n-mathphys", type=int, default=80, help="Math×Physics on math-of-physics nodes")
    ap.add_argument("--n-engphys", type=int, default=50)
    ap.add_argument("--n-engcs", type=int, default=50)
    ap.add_argument("--n-spine", type=int, default=60, help="top-level science×science (Branches_of_science)")
    ap.add_argument("--n-bidir", type=int, default=20, help="per seed (Physics/Chemistry/CS)")
    ap.add_argument("--math-depth", type=int, default=4)
    ap.add_argument("--math-floor", type=float, default=0.74)
    ap.add_argument("--eng-depth", type=int, default=4)
    ap.add_argument("--floor", type=float, default=0.78)
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--dedup-against", default=SCORED_4ROOTS)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs_matheng.tsv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    full = load_graph(GRAPH)
    children = build_children_adj(GRAPH)
    parents = {}
    for par, ch in children.items():
        for c in ch:
            parents.setdefault(c, set()).add(par)
    deg = {n: max(1, len(full.get(n, ()))) for n in full}
    names = sorted(full.keys())
    cos = e5_cos_to_roots(names, ALL_ROOTS, cache=os.path.join(ROOT, "e5_5roots_cos.pt"))
    C = {r: {n: cos[r][i] for i, n in enumerate(names)} for r in ALL_ROOTS}
    mu_phys, _ = calibrate_physics(C["Physics"])

    # Physics/Chemistry/CS — EXACTLY as #3312 (argmax over the 4 original roots), kept unchanged.
    base_pools, _ = build_pools_multi(children, {r: C[r] for r in BASE_ROOTS}, BASE_ROOTS, names,
                                      depth=3, floor=args.floor, mu_phys=mu_phys)
    phys, chem, cs = base_pools["Physics"], base_pools["Chemistry"], base_pools["Computer_science"]

    # Math — DEEPENED & INCLUSIVE: closure ∩ cos≥floor ∩ {math-leaning: cos|Math − cos|Phys ≥ −0.01}.
    # The margin (not strict argmax) KEEPS the math-of-physics core (Calculus/Diff-eq/PDE, ~balanced)
    # while dropping the physics-leak (Temperature/Heat/Cold/Wave_mechanics, clearly physics-dominant).
    mcl = closure("Mathematics", children, args.math_depth)
    math = {n for n in mcl if C["Mathematics"][n] >= args.math_floor
            and C["Mathematics"][n] - C["Physics"][n] >= -0.01}
    phys_core = [n for n in PHYS_CORE if n in C["Physics"]]
    # Engineering — argmax over the 5 roots (drops Mechanics→Phys, Computer_engineers→CS).
    ecl = closure("Engineering", children, args.eng_depth)
    eng = {n for n in ecl
           if max(ALL_ROOTS, key=lambda r: C[r][n]) == "Engineering" and C["Engineering"][n] >= args.floor}
    spine = closure(SPINE, children, 3)

    print(f"{len(names)} nodes")
    print(f"  Physics {len(phys)}  Chemistry {len(chem)}  Computer_science {len(cs)} (kept from #3312)")
    print(f"  Mathematics DEEPENED: closure(d≤{args.math_depth}) {len(mcl)} → pool {len(math)} "
          f"(was 9 argmax): {sorted(math, key=lambda n: -C['Mathematics'][n])}")
    print(f"  Engineering NEW: closure(d≤{args.eng_depth}) {len(ecl)} → pool {len(eng)}: "
          f"{sorted(eng, key=lambda n: -C['Engineering'][n])[:12]}")
    print(f"  Branches_of_science spine (d≤3): {len(spine)} nodes")
    for x, y in [("Mathematics", phys), ("Engineering", phys), ("Engineering", cs)]:
        pass
    print(f"  overlap math∩phys {len(math & phys)} (intentional: math-of-physics), eng∩phys "
          f"{len(eng & phys)}, eng∩cs {len(eng & cs)}")

    existing = load_existing_keys(args.dedup_against)
    pairs = set(existing)
    rows = []
    # within-Math (priority) + within-Engineering (modest)
    for aa, bb, wl in gen_within(math, "Mathematics", children, deg, args.n_math, pairs, rng):
        rows.append((aa, bb, "pos_math", wl))
    for aa, bb, wl in gen_within(eng, "Engineering", children, deg, args.n_eng, pairs, rng):
        rows.append((aa, bb, "pos_eng", wl))
    # cross strata — Math/Eng × curated CORE physics (not the leaky #3312 pool)
    for aa, bb, _ in gen_cross(math, set(phys_core), args.n_mathphys, pairs, rng):
        rows.append((aa, bb, "cross_MP", -1))          # the math-of-physics discrimination stratum
    for aa, bb, _ in gen_cross(eng, set(phys_core), args.n_engphys, pairs, rng):
        rows.append((aa, bb, "cross_EP", -1))
    for aa, bb, _ in gen_cross(eng, cs, args.n_engcs, pairs, rng):
        rows.append((aa, bb, "cross_ES", -1))
    # top-level science×science from the spine
    for aa, bb, _ in gen_cross(spine, spine, args.n_spine, pairs, rng):
        rows.append((aa, bb, "cross_spine", -1))
    # bidir-coinflip at Physics/Chemistry/CS (lateral diversity)
    for seed in ["Physics", "Chemistry", "Computer_science"]:
        nb, tries = 0, 0
        while nb < args.n_bidir and tries < args.n_bidir * 200:
            tries += 1
            end, path = walk_bidir(seed, children, parents, deg, 0.4, 1.0, rng, mode="coinflip")
            if end == seed:
                continue
            key = tuple(sorted((seed, end)))
            if key in pairs:
                continue
            pairs.add(key)
            rows.append((seed, end, "bidir", len(path) - 1))
            nb += 1
    # negatives — domain node × uniform-random, free (μ=0)
    domain = sorted(phys | chem | cs | math | eng)
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
        f.write("# Part-A Math-deepening + Engineering candidates (gen_math_eng_pairs.py; deduped vs "
                "mu_pairs_scored_4roots_260621-004105.tsv). strata: pos_math (deepened, incl. math-of-physics), pos_eng,\n")
        f.write("# cross_MP=Math×Physics (math-of-physics), cross_EP/ES=Eng×{Phys,CS}, cross_spine="
                "science×science, bidir. all BLANK μ. neg=μ0. cols: a\tb\tstratum\twl\tmu\n")
        for aa, bb, st, wl in rows:
            f.write(f"{aa}\t{bb}\t{st}\t{wl}\t{'0.0' if st == 'neg' else ''}\n")
    from collections import Counter
    cnt = Counter(r[2] for r in rows)
    print("wrote " + str(len(rows)) + " rows → " + args.out + ": "
          + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    print(f"to Haiku-score (non-neg): {sum(v for k, v in cnt.items() if k != 'neg')}  "
          f"(budget step — confirm first)")


if __name__ == "__main__":
    main()
