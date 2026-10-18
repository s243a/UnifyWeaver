#!/usr/bin/env python3
"""Build out the ENGINEERING domain for the fine-tune-with-replay step. Engineering entered as a *modest*
domain in the math-eng work (#3314, ~13 nodes); here we grow a real Engineering pool from the present
roots and add the discrimination-critical **physics-AND-engineering boundary** (Mechanics/Thermodynamics ×
Engineering — the same high-to-both case as math-of-physics). Built on gen_multidomain_pairs.py /
gen_math_eng_pairs.py; deduped vs the cumulative mu_pairs_scored_matheng_260621-100230.tsv. Negatives free.

Pool: closure(Engineering ∪ Mechanical_engineering ∪ Civil_engineering ∪ Applied_sciences, depth≤3),
μ-coherent to Engineering (argmax over the 5 roots == Engineering, OR within a small margin of it so the
genuinely multi-membered engineering nodes — Telecommunications, Audio_technology — are kept), cos floor.
A bidir-coinflip pass from the engineering seeds adds reach to siblings/cousins WITHIN that clean set.

DATA CEILING (flag for the widening spec #3313): Electrical / Process / Chemical / Software / Aerospace /
Systems / Industrial engineering are ABSENT from this slice — the engineering tree is shallow here.

    python3 gen_engineering_pairs.py --out mu_pairs_eng.tsv
"""
import argparse
import os
import random

from gen_multidomain_pairs import (closure, e5_cos_to_roots, calibrate_physics, build_pools_multi,
                                    gen_within, gen_cross, ROOTS as BASE_ROOTS)
from gen_more_sym_pairs import build_children_adj, load_existing_keys
from gen_mu_pairs import load_graph, walk_bidir, GRAPH

ROOT = os.path.dirname(os.path.abspath(__file__))
SCORED_CUMULATIVE = os.path.join(ROOT, "mu_pairs_scored_matheng_260621-100230.tsv")
ALL_ROOTS = ["Physics", "Chemistry", "Mathematics", "Computer_science", "Engineering"]
# Engineering closure seeds — NOT Applied_sciences (it reaches Medicine→Physiology/Psychiatry, polluting
# the pool); the engineering-proper roots stay clean.
ENG_SEEDS = ["Engineering", "Mechanical_engineering", "Civil_engineering"]
# the physics-AND-engineering boundary nodes (high-to-both) for the cross_EP stratum — curated genuine
# physics that is also the substance of mechanical/civil engineering.
PHYS_ENG_BOUNDARY = ["Mechanics", "Thermodynamics", "Classical_mechanics", "Motion_(physics)", "Energy",
                     "Heat", "Acoustics", "Optics"]
# physics-PRIMARY boundary nodes (belong in cross_EP, not within-Engineering) + medical leaks that sneak
# through the deep closure — blocked from the within-Engineering pool so it stays genuinely engineering.
LEAK_BLOCK = {"Mechanics", "Thermodynamics", "Classical_mechanics", "Motion_(physics)", "Energy", "Heat",
              "Acoustics", "Optics", "Wave_mechanics", "Temperature", "Waves", "Oscillation", "Sound",
              "Electromagnetism", "Physiology", "Psychiatry", "Medical_specialties", "Medicine", "Anatomy",
              "Health", "Nursing", "Diseases", "Nobel_Prize_in_Physiology_or_Medicine"}
# engineering subfields KNOWN absent in this slice (data-ceiling evidence / widening motivation)
ABSENT_ENG = ["Electrical_engineering", "Process_engineering", "Chemical_engineering",
              "Software_engineering", "Aerospace_engineering", "Systems_engineering",
              "Industrial_engineering"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eng", type=int, default=140, help="within-Engineering positives (priority)")
    ap.add_argument("--n-ep", type=int, default=60, help="Eng×Physics — the Mech/Thermo high-to-both boundary")
    ap.add_argument("--n-es", type=int, default=40, help="Eng×Computer_science")
    ap.add_argument("--n-em", type=int, default=30, help="Eng×Mathematics")
    ap.add_argument("--n-ec", type=int, default=30, help="Eng×Chemistry")
    ap.add_argument("--per-seed", type=int, default=30, help="bidir reach collected per engineering seed")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--floor", type=float, default=0.74, help="cos|Engineering floor for membership")
    ap.add_argument("--margin", type=float, default=-0.05, help="keep near-tie eng nodes (multi-membership)")
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--dedup-against", default=SCORED_CUMULATIVE)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs_eng.tsv"))
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

    def argmax5(n):
        return max(ALL_ROOTS, key=lambda r: C[r][n])

    seeds = [s for s in ENG_SEEDS if s in full]
    absent = [s for s in ABSENT_ENG if s not in full]
    boundary = [n for n in PHYS_ENG_BOUNDARY if n in full]
    # ENGINEERING pool — closure-guarded μ-coherence, INCLUSIVE (keep near-tie multi-membered eng nodes).
    ecl = set()
    for s in seeds:
        ecl |= closure(s, children, args.depth)

    def member(n):
        if n in LEAK_BLOCK or n not in ecl or C["Engineering"][n] < args.floor:
            return False
        best_other = max(C[r][n] for r in ALL_ROOTS if r != "Engineering")
        return argmax5(n) == "Engineering" or (C["Engineering"][n] - best_other >= args.margin)

    eng = {n for n in ecl if member(n)}
    # bidir reach within the clean set
    for s in seeds:
        got, tries = 0, 0
        while got < args.per_seed and tries < args.per_seed * 400:
            tries += 1
            end, _ = walk_bidir(s, children, parents, deg, 0.4, 1.0, rng, mode="coinflip")
            if end == s or end in eng or end not in ecl or not member(end):
                continue
            eng.add(end)
            got += 1

    # neighbour pools for the cross strata — Physics/Chemistry/CS from the 4-root argmax pools; Math inclusive
    base_pools, _ = build_pools_multi(children, {r: C[r] for r in BASE_ROOTS}, BASE_ROOTS, names,
                                      depth=3, floor=0.78, mu_phys=mu_phys)
    phys, chem, cs = base_pools["Physics"], base_pools["Chemistry"], base_pools["Computer_science"]
    mcl = closure("Mathematics", children, 4)
    math = {n for n in mcl if C["Mathematics"][n] >= 0.74 and C["Mathematics"][n] - C["Physics"][n] >= -0.01}

    print(f"{len(names)} nodes")
    print(f"  ENGINEERING pool: {len(eng)} nodes (closure(d≤{args.depth}) {len(ecl)} ∩ μ-coherent): "
          f"{sorted(eng, key=lambda n: -C['Engineering'][n])[:18]}")
    print(f"  phys-eng boundary (high-to-both, for cross_EP): {boundary}")
    print(f"  DATA CEILING — engineering subfields ABSENT ({len(absent)}): {absent}")
    print(f"  neighbour pools: Physics {len(phys)}  CS {len(cs)}  Mathematics {len(math)}  Chemistry {len(chem)}")

    existing = load_existing_keys(args.dedup_against)
    pairs = set(existing)
    rows = []
    for aa, bb, wl in gen_within(eng, "Engineering", children, deg, args.n_eng, pairs, rng):
        rows.append((aa, bb, "pos_eng", wl))
    # cross — Eng×Physics on the curated high-to-both boundary FIRST, then the rest of physics
    for aa, bb, _ in gen_cross(eng, set(boundary), args.n_ep, pairs, rng):
        rows.append((aa, bb, "cross_EP", -1))
    for aa, bb, _ in gen_cross(eng, cs, args.n_es, pairs, rng):
        rows.append((aa, bb, "cross_ES", -1))
    for aa, bb, _ in gen_cross(eng, math, args.n_em, pairs, rng):
        rows.append((aa, bb, "cross_EM", -1))
    for aa, bb, _ in gen_cross(eng, chem, args.n_ec, pairs, rng):
        rows.append((aa, bb, "cross_EC", -1))
    # negatives — engineering node × uniform-random, free (μ=0)
    domain = sorted(eng)
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
        f.write("# Engineering build-out candidates (gen_engineering_pairs.py; deduped vs "
                "mu_pairs_scored_matheng_260621-100230.tsv). strata: pos_eng (within-Engineering),\n")
        f.write("# cross_EP=Eng×Physics (Mech/Thermo high-to-both), cross_ES/EM/EC=Eng×{CS,Math,Chem}, "
                "neg=μ0. BLANK μ to Haiku-score. cols: a\tb\tstratum\twl\tmu\n")
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
