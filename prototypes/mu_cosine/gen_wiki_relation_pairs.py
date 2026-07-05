#!/usr/bin/env python3
"""Sample stratified Wikipedia category pairs for the graded relation round that validates the JointPosterior
+ 1/d test (DESIGN_sym_estimation_integration.md). Both endpoints must be in the STRUCT EMBEDDING (so 1/d is
defined) and the sample must span relations (vertical + lateral + none) so the classifier has something to do.

Emits the 8-col pairs schema score_inferred_tail.build_prompts expects (key == title, underscores kept, so the
graded _pairs.tsv + e5 cache align without a title↔key map):
  node_title  root_title  cur_relation  conf  neighborhood  node_type  root_type  raw

  python3 gen_wiki_relation_pairs.py --graph ../../data/benchmark/100k_cats/category_parent.tsv \
      --struct-emb /tmp/mu_data/struct_emb_recip.pt --per 220 --out /tmp/mu_data/wiki_rel_pairs.tsv
"""
import argparse, os, random, sys
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import load_dag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--struct-emb", required=True)
    ap.add_argument("--per", type=int, default=220, help="target pairs per stratum")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import torch
    se = torch.load(a.struct_emb, weights_only=False)
    inemb = set(se["nodes"])
    parents, children, deg = load_dag(a.graph)
    print(f"struct-emb nodes {len(inemb)}; graph {len(parents)|len(children) if False else len(set(parents)|set(children))} nodes")

    def anc(x, cap=4):
        out, fr = set(), {x}
        for _ in range(cap):
            nx = set()
            for c in fr:
                for p in (parents.get(c) or []):
                    if p not in out:
                        out.add(p); nx.add(p)
            fr = nx
            if not fr:
                break
        return out

    kids = {p: [c for c in cs if c in inemb] for p, cs in children.items() if p in inemb}
    kids = {p: cs for p, cs in kids.items() if cs}
    parents_l = [p for p in kids]
    seen = set()
    rows = []                                              # (node, root, cur_rel)

    def add(n, r, rel):
        if n != r and n in inemb and r in inemb and (n, r) not in seen and (r, n) not in seen:
            seen.add((n, r)); rows.append((n, r, rel)); return True
        return False

    # 1) SUBCATEGORY: child → parent
    random.Random(a.seed + 1).shuffle(parents_l)
    for p in parents_l:
        if sum(1 for x in rows if x[2] == "subcategory") >= a.per:
            break
        add(rng.choice(kids[p]), p, "subcategory")
    # 2) SUBTOPIC: grandchild → grandparent
    for p in parents_l:
        if sum(1 for x in rows if x[2] == "subtopic") >= a.per:
            break
        gk = [g for c in kids[p] for g in kids.get(c, [])]
        if gk:
            add(rng.choice(gk), p, "subtopic")
    # 3) SEE_ALSO: siblings (share a parent, neither ancestor of the other)
    for p in parents_l:
        if sum(1 for x in rows if x[2] == "see_also") >= a.per:
            break
        cs = kids[p]
        if len(cs) >= 2:
            x, y = rng.sample(cs, 2)
            if y not in anc(x) and x not in anc(y):
                add(x, y, "see_also")
    # 4) NONE: random distant pairs (no shared ancestor within reach, not linked)
    nodes = sorted(inemb)
    tries = 0
    while sum(1 for x in rows if x[2] == "none") < a.per and tries < a.per * 40:
        tries += 1
        x, y = rng.choice(nodes), rng.choice(nodes)
        if x == y:
            continue
        if y in anc(x) or x in anc(y) or (anc(x) & anc(y)):     # share structure ⇒ not "none"
            continue
        add(x, y, "none")

    rng.shuffle(rows)
    from collections import Counter
    print("strata:", dict(Counter(r[2] for r in rows)), " total", len(rows))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# node_title\troot_title\tcur_relation\tconf\tneighborhood\tnode_type\troot_type\traw\n")
        for n, r, rel in rows:
            f.write(f"{n}\t{r}\t{rel}\t1.0\twiki_rel\tcategory\tcategory\t\n")
    print(f"wrote {len(rows)} pairs → {a.out}")


if __name__ == "__main__":
    main()
