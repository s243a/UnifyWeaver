#!/usr/bin/env python3
"""Generate MORE candidate symmetric positives to enlarge the SYM training set (#3302 follow-up).

Reuses `gen_mu_pairs.py`'s hub-down-weighted RWR mesh, but seeds from a BROADER set of physics
subdomains + adjacent fields (varied anchors — the lever the README says closes the pairwise/Lin gap),
and **dedups against the already-committed `mu_pairs_scored.tsv`** so every emitted positive is NEW.
Negatives (domain × uniform-random) are FREE (μ=0 by construction) and also deduped.

Output: candidate file in the `mu_pairs.tsv` format (`name_a<TAB>name_b<TAB>stratum<TAB>walk_len<TAB>mu`),
positives with a BLANK μ (to be Haiku-scored, the budget step — confirm first), negatives with μ=0.

    python3 gen_more_sym_pairs.py --n-positives 600 --out mu_pairs_more.tsv
"""
import argparse
import os
import random

from gen_mu_pairs import load_graph, walk, GRAPH

ROOT = os.path.dirname(os.path.abspath(__file__))
SCORED = os.path.join(ROOT, "mu_pairs_scored.tsv")

# physics subdomains + adjacent fields (all verified present in the graph) — varied anchors
SEEDS = ["Physics", "Electromagnetism", "Thermodynamics", "Optics", "Mechanics", "Classical_mechanics",
         "Energy", "Matter", "Atoms", "Acoustics", "Astronomy", "Cosmology", "Chemistry", "Mathematics",
         "Geology", "Biology", "Waves", "Electricity", "Light", "Motion_(physics)"]


def build_children_adj(path=GRAPH):
    """Children-only (DOWNWARD) adjacency: parent -> [children], from `child<TAB>parent` edges.

    A downward walk over this adjacency stays inside the seed's subtree, so it **cannot drift up into
    other domains** the way the undirected walk does (the first real run wandered `Physics → Tamils /
    Shinto_shrines` — see the sampler-drift finding in WAM_RUST_BRIDGE_DETECTOR_PHILOSOPHY.md use case 6;
    bridge-guided seeding is the heavier principled fix, child-only walks are the lightweight one)."""
    children = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("child"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            c, par = p[0], p[1]
            children.setdefault(par, set()).add(c)
            children.setdefault(c, children.get(c, set()))
    return {k: sorted(v) for k, v in children.items()}


def load_existing_keys(path):
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                keys.add(tuple(sorted((p[0], p[1]))))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-positives", type=int, default=600, help="NEW positives to emit (after dedup)")
    ap.add_argument("--neg-ratio", type=float, default=5.0)
    ap.add_argument("--stop-prob", type=float, default=0.4)
    ap.add_argument("--hub-beta", type=float, default=1.0)
    ap.add_argument("--restart-alpha", type=float, default=0.3)
    ap.add_argument("--max-frontier", type=int, default=4000)
    ap.add_argument("--child-only", action="store_true",
                    help="DOWNWARD walks over a children-only adjacency (no cross-domain drift)")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds (default: the 20 SEEDS)")
    ap.add_argument("--dedup-against", default=SCORED, help="scored TSV whose pairs to exclude")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs_more.tsv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    full = load_graph(GRAPH)
    deg = {n: max(1, len(full.get(n, ()))) for n in full}     # total degree for hub down-weighting
    adj = build_children_adj(GRAPH) if args.child_only else full
    universe = list(full.keys())
    seed_list = args.seeds.split(",") if args.seeds else SEEDS
    seeds = [s for s in seed_list if s in full]
    existing = load_existing_keys(args.dedup_against)
    print(f"{len(universe)} categories, {len(seeds)} seeds, "
          f"{'CHILD-ONLY (downward)' if args.child_only else 'undirected'} walks; deduping against "
          f"{len(existing)} pairs in {os.path.basename(args.dedup_against)}")

    pairs = set(existing)          # canonical keys already used (existing + this run)
    rows = []
    frontier = list(seeds)

    # POSITIVES — RWR mesh, hub-down-weighted, NEW keys only
    tries = 0
    while sum(1 for r in rows if r[2] == "pos") < args.n_positives and tries < args.n_positives * 80:
        tries += 1
        start = rng.choice(seeds) if rng.random() < args.restart_alpha else rng.choice(frontier)
        end, path = walk(start, adj, deg, args.stop_prob, args.hub_beta, rng)
        if end == start:
            continue
        key = tuple(sorted((start, end)))
        if key in pairs:
            continue
        pairs.add(key)
        rows.append((start, end, "pos", len(path) - 1))
        for n in path[1:]:
            if len(frontier) >= args.max_frontier:
                break
            if n not in frontier:
                frontier.append(n)
    n_pos = sum(1 for r in rows if r[2] == "pos")

    # NEGATIVES — domain node × uniform-random noise, free (μ=0), NEW keys only
    n_neg_target = int(round(n_pos * args.neg_ratio))
    tries = 0
    while sum(1 for r in rows if r[2] == "neg") < n_neg_target and tries < n_neg_target * 80:
        tries += 1
        a, b = rng.choice(frontier), rng.choice(universe)
        if a == b or b in full.get(a, ()):       # exclude undirected neighbours (not negatives)
            continue
        key = tuple(sorted((a, b)))
        if key in pairs:
            continue
        pairs.add(key)
        rows.append((a, b, "neg", -1))
    n_neg = sum(1 for r in rows if r[2] == "neg")
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        f.write("# NEW candidate pairs to enlarge the SYM set (gen_more_sym_pairs.py; deduped against "
                "mu_pairs_scored.tsv). pos = meshed walk (BLANK μ — Haiku-score), neg = noise (μ=0 free).\n")
        f.write("# columns: name_a<TAB>name_b<TAB>stratum<TAB>walk_len<TAB>mu\n")
        for a, b, st, wl in rows:
            f.write(f"{a}\t{b}\t{st}\t{wl}\t{'0.0' if st == 'neg' else ''}\n")

    print(f"wrote {len(rows)} NEW pairs → {args.out}: {n_pos} pos (blank μ) / {n_neg} neg (μ=0 free)")
    print(f"mesh frontier grew to {len(frontier)} nodes around {len(seeds)} seeds")
    print("NEXT: Haiku-score the positives (budget step — confirm with the user first).")


if __name__ == "__main__":
    main()
