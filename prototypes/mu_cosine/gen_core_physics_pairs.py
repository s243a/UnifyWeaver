#!/usr/bin/env python3
"""Targeted CORE-PHYSICS sampling (sharper within-physics data), building on gen_multidomain_pairs.py /
gen_math_eng_pairs.py. The #3314 re-measure (REPORT_phys_discrim_reeval.md) shows physics argmax is
brittle only because physics is the connective spine (high-μ to several roots) — its RANKING stays strong
(top-2 92–100%, margins within ±0.08). This run is the optional polish: feed sharper *within-physics-
subfield* positives so the physics anchor's absolute μ separates a little more from its neighbours.

Method: bidirectional-coinflip walks (the #3309 depth-balanced walk, exact zero-drift) seeded at the
CLASSICAL physics subfields actually present in the 10k slice — {Electromagnetism, Classical_mechanics,
Mechanics, Optics, Thermodynamics, Acoustics} — collecting a μ-coherent core-physics neighbourhood
(argmax over the 5 roots == Physics, cos|Physics ≥ floor). Then within-physics-subfield positives
(subfield×subfield = both physics, high) + a few core×{Math,Chemistry,Engineering} cross pairs. Dedup vs
mu_pairs_scored_matheng_260621-100230.tsv; negatives free.

DATA CEILING (flagged for the widening spec #3313): the modern subfields — Quantum_mechanics,
Statistical_mechanics, Fluid_dynamics, Mathematical_physics, Relativity (and QFT / condensed-matter) —
are ABSENT from this graph. Sharper *classical* data can only do so much; real separation needs the
wider corpus.

    python3 gen_core_physics_pairs.py --out mu_pairs_corephys.tsv
"""
import argparse
import os
import random

from gen_multidomain_pairs import (closure, e5_cos_to_roots, calibrate_physics, build_pools_multi,
                                    gen_within, gen_cross, ROOTS as BASE_ROOTS)
from gen_more_sym_pairs import build_children_adj, load_existing_keys
from gen_mu_pairs import load_graph, walk_bidir, GRAPH

ROOT = os.path.dirname(os.path.abspath(__file__))
SCORED_MATHENG = os.path.join(ROOT, "mu_pairs_scored_matheng_260621-100230.tsv")
ALL_ROOTS = ["Physics", "Chemistry", "Mathematics", "Computer_science", "Engineering"]
PHYS_SEEDS = ["Electromagnetism", "Classical_mechanics", "Mechanics", "Optics", "Thermodynamics",
              "Acoustics"]
# modern subfields we KNOW are absent (data-ceiling evidence for the report / widening motivation)
ABSENT_SUBFIELDS = ["Quantum_mechanics", "Statistical_mechanics", "Fluid_dynamics", "Mathematical_physics",
                    "Relativity", "Special_relativity", "General_relativity", "Quantum_field_theory",
                    "Condensed_matter_physics"]
# recurring philosophy/arts/general leaks that sit 1–2 hops from a physics seed (Acoustics→Sound→Music,
# Optics→Light→Visual_arts, …) and that e5's flat "Physics" baseline cosine cannot reject by margin.
# Blocked explicitly (documented) so the within-physics stratum stays genuinely physics×physics.
LEAK_BLOCK = {"Time", "Future", "Events", "Causality", "Music", "Visual_arts", "Creation_myths",
              "Time_travel", "Chronology", "Reality", "Nature", "Popular_culture", "Internet_celebrities",
              "Brain", "Perception", "History"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-phys", type=int, default=120, help="within-core-physics positives (priority)")
    ap.add_argument("--n-cross-math", type=int, default=24)
    ap.add_argument("--n-cross-chem", type=int, default=24)
    ap.add_argument("--n-cross-eng", type=int, default=24)
    ap.add_argument("--per-seed", type=int, default=40, help="core nodes collected per subfield seed")
    ap.add_argument("--floor", type=float, default=0.74, help="cos|Physics floor for core membership")
    ap.add_argument("--margin", type=float, default=0.005, help="min cos|Phys − next-best-root margin")
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--math-depth", type=int, default=4)
    ap.add_argument("--math-floor", type=float, default=0.74)
    ap.add_argument("--eng-depth", type=int, default=4)
    ap.add_argument("--dedup-against", default=SCORED_MATHENG)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs_corephys.tsv"))
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

    seeds = [s for s in PHYS_SEEDS if s in full]
    absent = [s for s in ABSENT_SUBFIELDS if s not in full]
    # CLOSURE-GUARDED μ-coherence. e5 gives the bare word "Physics" a high baseline cosine to almost
    # everything (Popular_culture=0.82), so argmax-Physics ALONE lets junk in once a bidir walk drifts up
    # through the apex hubs and back down. Guard membership by the depth-bounded DOWNWARD closure (the
    # #3312 method) — genuine physics descendants — plus argmax + a strict positive margin over the
    # next-best root (drops ties like Visual_arts). The bidir coinflip walk then only adds *reach* to
    # siblings/cousins WITHIN that clean set; it cannot pollute it.
    phys_cl = closure("Physics", children, 2)
    for s in seeds:
        phys_cl |= closure(s, children, 1)

    def member(n):
        if n in LEAK_BLOCK or n not in phys_cl or argmax5(n) != "Physics" or C["Physics"][n] < args.floor:
            return False
        best_other = max(C[r][n] for r in ALL_ROOTS if r != "Physics")
        return C["Physics"][n] - best_other >= args.margin

    eligible = {n for n in phys_cl if member(n)}
    core = {s for s in seeds if s in eligible} | set(seeds)
    per_seed_got = {}
    for s in seeds:
        got, tries = 0, 0
        while got < args.per_seed and tries < args.per_seed * 400:
            tries += 1
            end, _ = walk_bidir(s, children, parents, deg, 0.4, 1.0, rng, mode="coinflip")
            if end == s or end in core or end not in eligible:
                continue
            core.add(end)
            got += 1
        per_seed_got[s] = got

    # neighbour-domain pools (for the few cross strata) — chem via the 4-root argmax pool; math inclusive;
    # eng argmax-over-5 — exactly as gen_math_eng_pairs.
    base_pools, _ = build_pools_multi(children, {r: C[r] for r in BASE_ROOTS}, BASE_ROOTS, names,
                                      depth=3, floor=0.78, mu_phys=mu_phys)
    chem = base_pools["Chemistry"]
    mcl = closure("Mathematics", children, args.math_depth)
    math = {n for n in mcl if C["Mathematics"][n] >= args.math_floor
            and C["Mathematics"][n] - C["Physics"][n] >= -0.01}
    ecl = closure("Engineering", children, args.eng_depth)
    eng = {n for n in ecl
           if argmax5(n) == "Engineering" and C["Engineering"][n] >= 0.78}

    print(f"{len(names)} nodes")
    print(f"  seeds present ({len(seeds)}/6): {seeds}")
    print(f"  per-seed core collected: " + "  ".join(f"{s[:5]}={per_seed_got[s]}" for s in seeds))
    print(f"  CORE-PHYSICS pool: {len(core)} nodes "
          f"(argmax==Physics ∩ cos|Phys≥{args.floor})")
    print(f"  DATA CEILING — modern subfields ABSENT ({len(absent)}): {absent}")
    print(f"  neighbour pools: Chemistry {len(chem)}  Mathematics {len(math)}  Engineering {len(eng)}")

    existing = load_existing_keys(args.dedup_against)
    pairs = set(existing)
    rows = []
    for aa, bb, wl in gen_within(core, "Physics", children, deg, args.n_phys, pairs, rng):
        rows.append((aa, bb, "pos_corephys", wl))
    for aa, bb, _ in gen_cross(core, math, args.n_cross_math, pairs, rng):
        rows.append((aa, bb, "cross_PM", -1))
    for aa, bb, _ in gen_cross(core, chem, args.n_cross_chem, pairs, rng):
        rows.append((aa, bb, "cross_PC", -1))
    for aa, bb, _ in gen_cross(core, eng, args.n_cross_eng, pairs, rng):
        rows.append((aa, bb, "cross_PE", -1))
    # negatives — core-physics node × uniform-random, free (μ=0)
    domain = sorted(core)
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
        f.write("# Core-physics targeted candidates (gen_core_physics_pairs.py; deduped vs "
                "mu_pairs_scored_matheng_260621-100230.tsv). strata: pos_corephys (within-subfield, high),\n")
        f.write("# cross_PM/PC/PE = core-physics × {Math,Chem,Eng}, neg=μ0. BLANK μ to Haiku-score. "
                "cols: a\tb\tstratum\twl\tmu\n")
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
