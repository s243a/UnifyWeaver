#!/usr/bin/env python3
"""Transitive discrimination target (user 2026-07-05): make the HIER discrimination operator CONTINUOUS across
hops instead of direct-only. For an ancestor-descendant pair at h hops, target

    μ_fwd(desc | anc) = p^h          (forward membership decays with distance)
    μ_rev(anc | desc) = 1 − p^h      (direction confidence degrades toward ambiguous as h grows)

μ_rev = 1−p^h is a DELIBERATE, Wikipedia-calibrated choice: the reverse is the COMPLEMENT (a soft non-membership
that rises toward 0.5 at large h), NOT p^h-in-reverse. A stricter hierarchy (formal ontology) may want reverse ≈ 0
at h=1 — the general two-parameter form is μ_rev = r^h with a separate reverse base r (here r implicitly ties to p).
p = per-source semantic-leakage base. Use p ≈ 0.90 = mean h=1 e5 cosine on enwiki (the geometric grounding); the
trained model's actual h=1 output is ≈0.88, slightly below target from direct-edge (μ≈1) training pressure. LOWER
leakage ⇒ HIGHER p (structure-led, cleaner decay).
Emitted as HIER rows tagged judge=graph (purely graph-structural). Sampled ancestor-descendant chains at exact
hop distances via BFS-up; node-disjoint train/held split so the eval measures the LEARNED decay curve.

  python3 emit_transitive_hops.py --graph .../100k_cats/category_parent.tsv --p 0.9 --hmax 5 --chains 1500 \
      --out /tmp/mu_data/transitive_train.tsv --held-out /tmp/mu_data/transitive_held.json
"""
import argparse, json, os, random, sys
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import load_dag


def anc_by_hop(parents, start, hmax):
    seen, q, byh = {start: 0}, deque([start]), {}
    while q:
        x = q.popleft(); h = seen[x]
        if h >= hmax:
            continue
        for p in (parents.get(x) or []):
            if p not in seen:
                seen[p] = h + 1; byh.setdefault(h + 1, []).append(p); q.append(p)
    return byh


def hit_prob(parents, desc, anc):
    """P(uniform-random UP-walk from desc visits anc) — a graph-native, CONTINUOUS, mean-reverting,
    root-converging effective-membership (needs no embedding/LLM). Two distinct "reverses" (user 2026-07-05):
    (a) INVERT ARGS → μ(anc|desc)=hit_prob(anc,desc)=0 structurally (up-walks never descend) — the correct μ_rev
    for direction; (b) REVERSE THE WALK → a DOWN-walk P(anc→desc) is small-nonzero but is still a *forward*
    containment (top-down, fan-out-diluted), NOT the reverse. See REPORT_multihop_direction.md §(e).

    Exact DP on the DAG (unconditional hitting prob, no depth budget): the memo double-serves as a cycle guard
    (a node mid-computation returns 0). memo is per-(desc,anc) call — not cached across pairs."""
    memo = {}
    def h(x):
        if x == anc:
            return 1.0
        if x in memo:
            return memo[x]
        # Parent maps are commonly sets. Canonical order makes both floating-point summation and the
        # cycle-guard fallback reproducible across PYTHONHASHSEED values/processes.
        ps = sorted(parents.get(x) or [])
        if not ps:
            return 0.0                                   # reached a root without hitting anc
        memo[x] = 0.0                                    # cycle guard: node currently being computed → 0
        v = sum(h(p) for p in ps) / len(ps)
        memo[x] = v
        return v
    return h(desc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--target", choices=["ph", "walk"], default="ph",
                    help="ph: μ_fwd=p^h, μ_rev=1−p^h (exponential, needs p). walk: μ_fwd=up-walk hit-prob, μ_rev=0 "
                    "— graph-native, continuous, mean-reverting, no second model (REPORT_multihop_direction §e).")
    ap.add_argument("--p", type=float, default=0.9, help="per-source leakage base for μ_fwd = p^h (--target ph)")
    ap.add_argument("--hmax", type=int, default=5)
    ap.add_argument("--chains", type=int, default=1500)
    ap.add_argument("--held-frac", type=float, default=0.15)
    ap.add_argument("--corpus", default="enwiki")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--held-out", required=True, help="JSON of held chains for the eval")
    a = ap.parse_args()

    parents, _, _ = load_dag(a.graph)
    rng = random.Random(a.seed)
    nodes = list(parents.keys()); rng.shuffle(nodes)
    held_nodes = set(nodes[:int(a.held_frac * len(nodes))])

    # collect ALL full chains first, then downsample train to the cap — so the held set is NOT biased toward
    # early-shuffle nodes (an earlier version broke out as soon as train filled, starving the held set; review).
    train_src, held = [], {h: [] for h in range(1, a.hmax + 1)}
    for s in nodes:
        byh = anc_by_hop(parents, s, a.hmax)
        if not all(h in byh for h in range(1, a.hmax + 1)):
            continue
        chain = {h: rng.choice(byh[h]) for h in range(1, a.hmax + 1)}
        if s in held_nodes:
            for h in range(1, a.hmax + 1):
                held[h].append((s, chain[h]))
        else:
            train_src.append((s, chain))
    rng.shuffle(train_src)
    train = {h: [(s, ch[h]) for s, ch in train_src[:a.chains]] for h in range(1, a.hmax + 1)}

    # Rows are tagged judge=graph (they ARE graph-structural). This DELIBERATELY extends the graph judge's
    # calibration from direct-edge to transitive membership; if you want to keep those separate, a `graph-transitive`
    # judge row (analogous to dir-blend getting its own row) is the tracked alternative (review 2026-07-05).
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# node\troot\tmu\top\trelation\tnode_type\troot_type\tcorpus\tjudge\tconf\n")
        n = 0
        for h in range(1, a.hmax + 1):
            ph_f, ph_r = a.p ** h, 1.0 - a.p ** h
            for desc, anc in train[h]:
                if a.target == "walk":
                    mf, mr = hit_prob(parents, desc, anc), 0.0     # graph-native; μ_rev=0 by construction
                else:
                    mf, mr = ph_f, ph_r
                f.write(f"{desc}\t{anc}\t{mf:.3f}\tHIER\tsubcategory\tcategory\tcategory\t{a.corpus}\tgraph\t1.0\n")
                f.write(f"{anc}\t{desc}\t{mr:.3f}\tHIER\tsubcategory\tcategory\tcategory\t{a.corpus}\tgraph\t1.0\n")
                n += 2
    json.dump({str(h): held[h] for h in held}, open(a.held_out, "w"))
    print(f"wrote {n} transitive HIER rows (p={a.p}, h=1..{a.hmax}, {len(train_src[:a.chains])} train chains × "
          f"{a.hmax} hops × fwd/rev) → {a.out}; held {len(held[1])} chains → {a.held_out}")
    if a.target == "ph":
        print("  targets μ_fwd=p^h:", {h: round(a.p ** h, 3) for h in range(1, a.hmax + 1)})
    else:
        print("  targets μ_fwd=hit_prob (per-pair up-walk); μ_rev=0")


if __name__ == "__main__":
    main()
