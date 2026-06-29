#!/usr/bin/env python3
"""Transitive-closure generation for the ordinal-constraint training data (DESIGN_transitive_relations.md).

STAGE 1 (this module): compose TAGGED hierarchical edges into transitive pairs, ranked by the **product of
link μ** (= highest-product path = shortest path under cost −log μ; max-product / dominant-path semiring, the
rigorous `⊕ = max` case). Pure graph code — no LLM, no training. Each emitted pair carries:
  product   — Π μ(links): the curriculum ranking key (estimated transitive μ).
  min_link  — min μ(links): the ordinal-constraint BOUND (`μ_transitive ≤ min_link`, §"is the bound product? No").
  hops, rel — chain length and the composed relation (the operator); path for provenance.

Composition (only transitive-compatible relations compose; lateral see_also/assoc do NOT):
  element_of ∘ downward  → element_of      (C∈A, A⊆B ⇒ C∈B)
  downward   ∘ downward  → subcategory/subtopic (containment chain)
  super_category ∘ super_category → super_category
  bridge passes a relation through (identity)
where downward = {subcategory, subtopic}. Deferred (stage 2+): noisy-OR multi-path (needs enumeration, not a
closure), heteroscedastic loss, the soft floor, the multi-factor judge term."""
import math
from collections import defaultdict

DOWNWARD = {"subcategory", "subtopic"}
# forward μ per relation (REL_SPEC); only hierarchical relations compose
REL_MU = {"subcategory": 0.90, "subtopic": 0.85, "element_of": 0.90, "super_category": 0.85, "bridge": 0.90}


def compose_relation(r1, r2):
    """The composed relation for chaining r1 (accumulated from src) then r2 (next edge), or None if the pair
    is not transitive-compatible (⇒ don't extend the chain)."""
    if r1 == "bridge":
        return r2
    if r2 == "bridge":
        return r1
    if r1 == "element_of" and r2 in DOWNWARD:
        return "element_of"
    if r1 in DOWNWARD and r2 in DOWNWARD:
        return "subcategory" if (r1 == "subcategory" and r2 == "subcategory") else "subtopic"
    if r1 == "super_category" and r2 == "super_category":
        return "super_category"
    return None


def transitive_pairs(edges, rel_mu=REL_MU, max_hops=3):
    """edges: iterable of (src, dst, relation) directed, TAGGED hierarchical. Returns transitive pairs
    (hops≥2) as dicts, dominant-path (max-product) per (src,dst), excluding any pair that is itself a direct
    edge, sorted by product descending (the curriculum). Bounded-hop best-product search = Dijkstra-equivalent
    at small depth (optimal substructure: the best path to C extends the best to B)."""
    adj = defaultdict(list)
    direct = set()
    for s, d, r in edges:
        mu = rel_mu.get(r)
        if mu is None:                                 # non-hierarchical (see_also/assoc/...) — never composes
            continue
        adj[s].append((d, r, mu)); direct.add((s, d))
    best = {}                                          # (src,dst) -> dict, keep max product
    for src in list(adj):                              # snapshot — the search below must not mutate adj
        stack = [(src, None, 1.0, 1.0, None, 0, (src,))]   # node, acc_rel, prod, min_link, bound_edge, hops, path
        while stack:
            node, acc_rel, prod, mn, be, hops, path = stack.pop()
            if hops >= max_hops:
                continue
            for d, r, mu in adj.get(node, ()):         # .get → don't auto-create leaf entries
                if d in path:                          # simple paths only (no cycles)
                    continue
                rel = r if acc_rel is None else compose_relation(acc_rel, r)
                if rel is None:                        # incompatible chain — prune
                    continue
                nbe = (node, d, r) if (be is None or mu < mn) else be    # weakest link = the BOUND edge
                np, nmn, nh, npath = prod * mu, min(mn, mu), hops + 1, path + (d,)
                if nh >= 2 and (src, d) not in direct:        # a transitive pair (not a direct edge)
                    key = (src, d)
                    if key not in best or np > best[key]["product"]:
                        best[key] = {"src": src, "dst": d, "product": np, "min_link": nmn,
                                     "hops": nh, "rel": rel, "bound": nbe, "path": npath}
                stack.append((d, rel, np, nmn, nbe, nh, npath))
    return sorted(best.values(), key=lambda x: -x["product"])


def _cli():
    import argparse, os
    ap = argparse.ArgumentParser(description="emit transitive pairs from fused tagged edges (curriculum order)")
    ap.add_argument("--edges", nargs="+", required=True, help="fused *_edges.tsv files (a\\tb\\trel\\tconf[...])")
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--top", type=int, default=20, help="print the top-N by product")
    ap.add_argument("--out", default=None, help="write triples (transitive pair + its bounding direct edge) "
                    "for the trainer's ranking-CE term; sorted by product (curriculum order)")
    a = ap.parse_args()
    edges = []
    for f in a.edges:
        for ln in open(f, encoding="utf-8"):
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split("\t")
            if len(c) >= 4 and c[2] in REL_MU:
                try:
                    if float(c[3]) >= 1.0:                 # TAGGED only (conf=1.0)
                        edges.append((c[0], c[1], c[2]))
                except ValueError:
                    pass
    pairs = transitive_pairs(edges, max_hops=a.max_hops)
    from collections import Counter
    print(f"{len(edges)} tagged hierarchical edges → {len(pairs)} transitive pairs (≤{a.max_hops} hops)")
    print(f"  by hops: {dict(Counter(p['hops'] for p in pairs))}")
    print(f"  by composed rel: {dict(Counter(p['rel'] for p in pairs))}")
    print(f"  top {a.top} by product (the curriculum head):")
    for p in pairs[:a.top]:
        print(f"    {p['product']:.3f}  min={p['min_link']:.2f}  {p['hops']}h  {p['rel']:13} "
              f"{p['src'].split(':')[-1][:24]} -> {p['dst'].split(':')[-1][:24]}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write("# trans_src\ttrans_dst\ttrans_rel\tbound_src\tbound_dst\tbound_rel\tproduct\tmin_link\thops\n")
            for p in pairs:                            # already sorted by product (curriculum order)
                bs, bd, br = p["bound"]
                f.write(f"{p['src']}\t{p['dst']}\t{p['rel']}\t{bs}\t{bd}\t{br}\t"
                        f"{p['product']:.4f}\t{p['min_link']:.4f}\t{p['hops']}\n")
        print(f"  wrote {len(pairs)} triples → {a.out}")


if __name__ == "__main__":
    _cli()
