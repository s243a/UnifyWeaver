#!/usr/bin/env python3
"""Generate the multi-domain training set: MORE Physics + a MODEST Chemistry probe + the cross-domain
stratum (Physics-node × Chemistry-node), via CHILD-ONLY downward walks gated to a μ-coherent
neighbourhood of each root.

Why the μ gate (not child-only alone): downward walks reduce drift but the Wikipedia DAG genuinely
nests off-domain subtrees under a root (e.g. `Physics → Cosmology → Creation_myths → Hinduism → Koli_
people`), and the "add every endpoint" frontier then restarts walks from those drifted nodes. So we keep
the frontier inside a **μ-coherent neighbourhood of the root** — the principled fix described in
WAM_RUST_BRIDGE_DETECTOR_PHILOSOPHY.md use case 6 (bridge-guided seeding; here approximated by the e5
prior μ(node|root) instead of the full bridge detector, which lives in the WAM-Rust core we don't touch).

Domain pools (argmax-assigned for clean discrimination): a node is in the Physics pool if
μ(node|Physics) ≥ τ and ≥ μ(node|Chemistry); Chemistry pool symmetrically. Within-domain positives are
pool-restricted downward walks (locally related, in-domain). Cross-domain = Physics-pool × Chemistry-pool.

Output: candidate file (`name_a<TAB>name_b<TAB>stratum<TAB>walk_len<TAB>mu`); positives/cross BLANK μ
(Haiku-score), negatives μ=0. Strata: pos_phys, pos_chem, cross, neg.

    python3 gen_multidomain_pairs.py --n-phys 500 --n-chem 200 --n-cross 150 --out mu_pairs_multidomain.tsv
"""
import argparse
import os
import random
from collections import deque

import torch

from gen_more_sym_pairs import build_children_adj, load_existing_keys
from gen_mu_pairs import load_graph, GRAPH

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
SCORED_LARGE = os.path.join(ROOT, "mu_pairs_scored_large.tsv")
FIXTURE = os.path.join(REPO, "tests", "fixtures", "wikipedia_physics_fuzzy_nodes.tsv")
E5 = "intfloat/e5-small-v2"


def e5_cos_to_roots(names, roots, cache=os.path.join(ROOT, "e5_multidomain_cos.pt")):
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
    """Fit μ≈a·cos+b on the 90-node Haiku fixture (same calibration dense_mu_direct uses) so the raw e5
    cosine to 'Physics' becomes a usable membership μ — raw cosine barely separates (Politics≈Thermo)."""
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
    mx = sum(x for x, _ in xs) / n
    my = sum(y for _, y in xs) / n
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


def build_pools(children, cphys, cchem, mu_phys, phys_root, chem_root,
                phys_depth=5, phys_tau=0.40, chem_depth=6):
    """Graph-closure ∩ μ-coherence pools (clean, in-domain). Physics: closure ∩ calibrated μ≥τ;
    Chemistry: closure ∩ argmax(chem>phys) — grounding in the DAG kills the e5-cosine junk (Literature,
    Politics) that isn't actually a descendant of the root."""
    pc = closure(phys_root, children, phys_depth)
    phys = {n for n in pc if mu_phys.get(n, 0.0) >= phys_tau}
    cc = closure(chem_root, children, chem_depth)
    chem = {n for n in cc if cchem.get(n, 0.0) >= cphys.get(n, 0.0)} - phys
    return phys, chem


def walk_in_pool(start, children, deg, pool, rng, stop=0.4, beta=1.0, max_len=6):
    """Downward walk that only steps into IN-POOL children (hub-down-weighted)."""
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
    """Within-domain pairs: a mix of downward-WALK pairs (locally nested, high relatedness) and
    RANDOM in-pool pairs (same-domain, moderate) for a graded range. Deduped against `pairs`."""
    rows, pool_l = [], sorted(pool)
    n_walk = int(n * walk_frac)
    tries = 0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phys-root", default="Physics")
    ap.add_argument("--chem-root", default="Chemistry")
    ap.add_argument("--n-phys", type=int, default=400)
    ap.add_argument("--n-chem", type=int, default=150)
    ap.add_argument("--n-cross", type=int, default=150)
    ap.add_argument("--neg-ratio", type=float, default=4.0)
    ap.add_argument("--phys-tau", type=float, default=0.40)
    ap.add_argument("--dedup-against", default=SCORED_LARGE)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs_multidomain.tsv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    full = load_graph(GRAPH)
    children = build_children_adj(GRAPH)
    deg = {n: max(1, len(full.get(n, ()))) for n in full}
    names = sorted(full.keys())
    roots = [args.phys_root, args.chem_root]
    cos = e5_cos_to_roots(names, roots)
    cp = {n: cos[args.phys_root][i] for i, n in enumerate(names)}
    cc = {n: cos[args.chem_root][i] for i, n in enumerate(names)}
    mu_phys, (a, b) = calibrate_physics(cp)
    phys, chem = build_pools(children, cp, cc, mu_phys, args.phys_root, args.chem_root,
                             phys_tau=args.phys_tau)
    print(f"{len(names)} nodes; calibrated μ_phys≈{a:.2f}·cos{b:+.2f}; Physics pool {len(phys)}, "
          f"Chemistry pool {len(chem)} (overlap {len(phys & chem)})")
    existing = load_existing_keys(args.dedup_against)
    pairs = set(existing)

    rows = []   # (a, b, stratum, walk_len)
    for a, b, wl in gen_within(phys, args.phys_root, children, deg, args.n_phys, pairs, rng):
        rows.append((a, b, "pos_phys", wl))
    for a, b, wl in gen_within(chem, args.chem_root, children, deg, args.n_chem, pairs, rng):
        rows.append((a, b, "pos_chem", wl))
    # CROSS-DOMAIN: physics-pool node × chemistry-pool node (the discrimination-critical stratum)
    phys_l, chem_l, tries = sorted(phys), sorted(chem), 0
    n_cross = 0
    while n_cross < args.n_cross and tries < args.n_cross * 120:
        tries += 1
        a, b = rng.choice(phys_l), rng.choice(chem_l)
        key = tuple(sorted((a, b)))
        if a == b or key in pairs:
            continue
        pairs.add(key)
        rows.append((a, b, "cross", -1))
        n_cross += 1
    # NEGATIVES — domain node × uniform-random, free (μ=0)
    n_pos = sum(1 for r in rows if r[2] != "neg")
    domain = phys_l + chem_l
    n_neg_target, tries = int(round(n_pos * args.neg_ratio)), 0
    n_neg = 0
    while n_neg < n_neg_target and tries < n_neg_target * 80:
        tries += 1
        a, b = rng.choice(domain), rng.choice(names)
        if a == b or b in full.get(a, ()):
            continue
        key = tuple(sorted((a, b)))
        if key in pairs:
            continue
        pairs.add(key)
        rows.append((a, b, "neg", -1))
        n_neg += 1
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        f.write("# Multi-domain candidates (gen_multidomain_pairs.py; child-only downward walks, "
                "μ-coherent pools; deduped vs mu_pairs_scored_large.tsv).\n")
        f.write("# strata: pos_phys/pos_chem = within-domain walk (BLANK μ), cross = physics×chemistry "
                "(BLANK μ), neg = noise (μ=0). columns: a<TAB>b<TAB>stratum<TAB>walk_len<TAB>mu\n")
        for a, b, st, wl in rows:
            f.write(f"{a}\t{b}\t{st}\t{wl}\t{'0.0' if st == 'neg' else ''}\n")
    from collections import Counter
    c = Counter(r[2] for r in rows)
    print(f"wrote {len(rows)} rows → {args.out}: " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print("NEXT: Haiku-score pos_phys + pos_chem + cross (budget step — confirm first).")


if __name__ == "__main__":
    main()
