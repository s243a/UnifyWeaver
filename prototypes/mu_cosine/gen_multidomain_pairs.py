#!/usr/bin/env python3
"""Generate the FOUR-root multi-domain training set: MORE Physics + MODEST Chemistry / Mathematics /
Computer_science + ALL pairwise cross-domain strata + a small bidirectional-coinflip batch — via
depth-bounded child-only downward closures gated to a μ-coherent (argmax-over-roots) neighbourhood.

Why depth-BOUNDED closure (verified graph fact): the full downward closure from `Physics` is ≈ the whole
graph (7811 nodes — densely cross-linked), so an unbounded closure is useless as a domain pool. A
depth-≤3 closure ∩ μ-coherence stays clean (#3310). `Branches_of_science` is the shared ancestor of all
four roots within ~3 hops, so pools are kept disjoint by **argmax over the four roots** (each node joins
the root it is most e5-coherent with) ∩ that root's depth-bounded closure. AI is absent from this graph.

μ-coherence ≈ #3307's bridge-guided "μ-coherent neighbourhood" seeding (here the e5 prior, not the full
bridge detector which lives in the WAM-Rust core we don't touch). Physics additionally uses the
fixture-calibrated μ floor (we only have a physics fixture); the other roots use argmax + a cosine floor.

Output: candidate file (`name_a<TAB>name_b<TAB>stratum<TAB>walk_len<TAB>mu`); positives/cross/bidir BLANK
μ (Haiku-score), negatives μ=0. Strata: pos_phys/pos_chem/pos_math/pos_cs, cross_PC/PM/PS/CM/CS/MS, bidir.

    python3 gen_multidomain_pairs.py --out mu_pairs_4roots.tsv
"""
import argparse
import os
import random
from collections import deque
from itertools import combinations

import torch

from gen_more_sym_pairs import build_children_adj, load_existing_keys
from gen_mu_pairs import load_graph, walk_bidir, GRAPH

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
SCORED_MULTI = os.path.join(ROOT, "mu_pairs_scored_multidomain.tsv")
FIXTURE = os.path.join(REPO, "tests", "fixtures", "wikipedia_physics_fuzzy_nodes.tsv")
E5 = "intfloat/e5-small-v2"

ROOTS = ["Physics", "Chemistry", "Mathematics", "Computer_science"]
ABBR = {"Physics": "P", "Chemistry": "C", "Mathematics": "M", "Computer_science": "S"}


def e5_cos_to_roots(names, roots, cache=os.path.join(ROOT, "e5_4roots_cos.pt")):
    """Raw e5 cosine cos(node|root) for each root (query: root, passage: node). Cached (regenerable)."""
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        if d["names"] == list(names) and d["roots"] == list(roots):
            return d["cos"]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(E5)
    human = [n.replace("_", " ") for n in names]
    P = model.encode(["passage: " + h for h in human], batch_size=512, convert_to_tensor=True,
                     normalize_embeddings=True, show_progress_bar=False)
    cos = {}
    for r in roots:
        rv = model.encode(["query: " + r.replace("_", " ")], convert_to_tensor=True,
                          normalize_embeddings=True)[0]
        cos[r] = (P @ rv).tolist()
    torch.save({"names": list(names), "roots": list(roots), "cos": cos}, cache)
    return cos


def calibrate_physics(cphys):
    """Fit μ≈a·cos+b on the 90-node Haiku fixture (raw e5 cosine barely separates: Politics≈Thermo)."""
    fx = {}
    for l in open(FIXTURE):
        if l.lstrip().startswith("#"):
            continue
        p = l.rstrip("\n").split("\t")
        if len(p) >= 2:
            try:
                fx[p[0]] = float(p[1])
            except ValueError:
                pass
    xs = [(cphys[n], fx[n]) for n in fx if n in cphys]
    n = len(xs)
    mx, my = sum(x for x, _ in xs) / n, sum(y for _, y in xs) / n
    sxx = sum((x - mx) ** 2 for x, _ in xs)
    a = sum((x - mx) * (y - my) for x, y in xs) / sxx if sxx else 0.0
    b = my - a * mx
    return {nm: max(0.0, min(1.0, a * c + b)) for nm, c in cphys.items()}, (a, b)


def closure(root, children, depth):
    """Downward (children-only) BFS closure to `depth` hops — graph-grounded domain membership."""
    seen = {root: 0}
    q = deque([root])
    while q:
        x = q.popleft()
        if seen[x] >= depth:
            continue
        for c in children.get(x, ()):
            if c not in seen:
                seen[c] = seen[x] + 1
                q.append(c)
    return set(seen)


def build_pools_multi(children, C, roots, names, depth=3, floor=0.78, phys_root="Physics",
                      mu_phys=None, phys_tau=0.40):
    """Per-root pool = depth-bounded closure ∩ {this root is the argmax over all roots} ∩ cos ≥ floor.
    Physics additionally requires calibrated μ ≥ phys_tau. Pools are disjoint by construction (argmax)."""
    cl = {r: closure(r, children, depth) for r in roots}
    union = set().union(*cl.values())
    argmax = {n: max(roots, key=lambda r: C[r][n]) for n in union}
    pools = {}
    for r in roots:
        pool = {n for n in cl[r] if argmax[n] == r and C[r][n] >= floor}
        if r == phys_root and mu_phys is not None:
            pool = {n for n in pool if mu_phys.get(n, 0.0) >= phys_tau}
        pools[r] = pool
    return pools, cl


def walk_in_pool(start, children, deg, pool, rng, stop=0.4, beta=1.0, max_len=6):
    node, path = start, [start]
    while len(path) <= max_len and rng.random() > stop:
        nbrs = [c for c in children.get(node, ()) if c in pool]
        if not nbrs:
            break
        w = [1.0 / (deg.get(c, 1) ** beta) for c in nbrs]
        tot = sum(w) or 1.0
        x, acc, chosen = rng.random() * tot, 0.0, nbrs[-1]
        for c, wi in zip(nbrs, w):
            acc += wi
            if x <= acc:
                chosen = c
                break
        node = chosen
        path.append(node)
    return node, path


def gen_within(pool, root, children, deg, n, pairs, rng, walk_frac=0.6):
    """Within-domain pairs: downward-WALK pairs (locally nested) + RANDOM in-pool pairs (graded range)."""
    rows, pool_l = [], sorted(pool)
    if len(pool_l) < 2:
        return rows
    n_walk, tries = int(n * walk_frac), 0
    while len(rows) < n_walk and tries < n * 200:
        tries += 1
        start = root if (rng.random() < 0.3 and root in pool) else rng.choice(pool_l)
        end, path = walk_in_pool(start, children, deg, pool, rng)
        if end == start:
            continue
        key = tuple(sorted((start, end)))
        if key in pairs:
            continue
        pairs.add(key)
        rows.append((start, end, len(path) - 1))
    tries = 0
    while len(rows) < n and tries < n * 200:
        tries += 1
        a, b = rng.choice(pool_l), rng.choice(pool_l)
        key = tuple(sorted((a, b)))
        if a == b or key in pairs:
            continue
        pairs.add(key)
        rows.append((a, b, -1))
    return rows


def gen_cross(pa, pb, n, pairs, rng):
    rows, la, lb, tries = [], sorted(pa), sorted(pb), 0
    if not la or not lb:
        return rows
    while len(rows) < n and tries < n * 200:
        tries += 1
        a, b = rng.choice(la), rng.choice(lb)
        key = tuple(sorted((a, b)))
        if a == b or key in pairs:
            continue
        pairs.add(key)
        rows.append((a, b, -1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-phys", type=int, default=300, help="more Physics — keep closing SYM")
    ap.add_argument("--n-chem", type=int, default=100)
    ap.add_argument("--n-math", type=int, default=36)
    ap.add_argument("--n-cs", type=int, default=100)
    ap.add_argument("--n-cross-each", type=int, default=50, help="per cross-domain pair (6 strata)")
    ap.add_argument("--n-bidir", type=int, default=60, help="bidirectional-coinflip batch (Physics)")
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--depth", type=int, default=3, help="closure depth bound (full closure ≈ whole graph)")
    ap.add_argument("--floor", type=float, default=0.78, help="min cos(node|root) for pool membership")
    ap.add_argument("--phys-tau", type=float, default=0.40)
    ap.add_argument("--dedup-against", default=SCORED_MULTI)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs_4roots.tsv"))
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
    cos = e5_cos_to_roots(names, ROOTS)
    C = {r: {n: cos[r][i] for i, n in enumerate(names)} for r in ROOTS}
    mu_phys, (a, b) = calibrate_physics(C["Physics"])
    pools, cl = build_pools_multi(children, C, ROOTS, names, depth=args.depth, floor=args.floor,
                                  mu_phys=mu_phys, phys_tau=args.phys_tau)
    print(f"{len(names)} nodes; depth≤{args.depth}; calibrated μ_phys≈{a:.2f}·cos{b:+.2f}")
    for r in ROOTS:
        print(f"  {r:16} closure {len(cl[r]):4}  pool {len(pools[r]):3}  "
              f"e.g. {sorted(pools[r], key=lambda n: -C[r][n])[:6]}")
    for x, y in combinations(ROOTS, 2):
        print(f"  pool overlap {ABBR[x]}∩{ABBR[y]}: {len(pools[x] & pools[y])}")

    existing = load_existing_keys(args.dedup_against)
    pairs = set(existing)
    rows = []
    nmap = {"Physics": args.n_phys, "Chemistry": args.n_chem, "Mathematics": args.n_math,
            "Computer_science": args.n_cs}
    strat = {"Physics": "pos_phys", "Chemistry": "pos_chem", "Mathematics": "pos_math",
             "Computer_science": "pos_cs"}
    for r in ROOTS:
        for aa, bb, wl in gen_within(pools[r], r, children, deg, nmap[r], pairs, rng):
            rows.append((aa, bb, strat[r], wl))
    # ALL pairwise cross-domain strata (discrimination-critical)
    for x, y in combinations(ROOTS, 2):
        st = f"cross_{ABBR[x]}{ABBR[y]}"
        for aa, bb, _ in gen_cross(pools[x], pools[y], args.n_cross_each, pairs, rng):
            rows.append((aa, bb, st, -1))
    # bidirectional-coinflip batch seeded at Physics (lateral diversity beyond the strict pools, #3309)
    nb, tries = 0, 0
    while nb < args.n_bidir and tries < args.n_bidir * 200:
        tries += 1
        end, path = walk_bidir("Physics", children, parents, deg, 0.4, 1.0, rng, mode="coinflip")
        if end == "Physics":
            continue
        key = tuple(sorted(("Physics", end)))
        if key in pairs:
            continue
        pairs.add(key)
        rows.append(("Physics", end, "bidir", len(path) - 1))
        nb += 1
    # NEGATIVES — domain node × uniform-random, free (μ=0)
    domain = sorted(set().union(*pools.values()))
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
        f.write("# Four-root multi-domain candidates (gen_multidomain_pairs.py; depth-bounded child-only "
                "closures ∩ μ-coherence; deduped vs mu_pairs_scored_multidomain.tsv).\n")
        f.write("# strata: pos_phys/pos_chem/pos_math/pos_cs=within-domain, cross_XY=cross-domain "
                "(P/C/M/S), bidir=coinflip(Physics); all BLANK μ. neg=noise μ=0. cols: a\tb\tstratum\twl\tmu\n")
        for aa, bb, st, wl in rows:
            f.write(f"{aa}\t{bb}\t{st}\t{wl}\t{'0.0' if st == 'neg' else ''}\n")
    from collections import Counter
    cnt = Counter(r[2] for r in rows)
    print("wrote " + str(len(rows)) + " rows → " + args.out + ": "
          + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    to_score = sum(v for k, v in cnt.items() if k != "neg")
    print(f"to Haiku-score (non-neg): {to_score}  (budget step — confirm first)")


if __name__ == "__main__":
    main()

