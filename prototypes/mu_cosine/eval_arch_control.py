#!/usr/bin/env python3
"""eval_arch_control.py — does μ's win come from its ARCHITECTURE, or just from training a head on frozen e5?

Addresses the PR #3387 review (architecture-control + reproducibility + CIs + sibling false-negative filters):
e5-cos is the *product* baseline (untrained); the *architecture* control is a **trained head on the same frozen
e5 features**. If a trivial logistic probe on the ORDERED pair matches μ, then μ's directional/close-negative
"structural win" is really "any trained head on e5" — not μ-specific. We compare three scorers on held-out edges:

  e5-cos      : cosine(query: a, passage: b)            — untrained, symmetric (the product baseline)
  e5-probe    : logistic on concat(q[a], p[b]) (768-d)  — trained head on FROZEN e5, ORDER-AWARE (the control)
  mu-elem     : μ(a|b) under the element_of operator    — the model

Two paired tasks, on a held-out edge split (probe trained only on the train split):
  DIRECTION  : forward (child,parent) vs reverse (parent,child)  → AUC(fwd > rev)
  CLOSE-NEG  : member (child,parent) vs sibling (child,sibling)  → AUC(parent > sibling)

Sibling negatives are FILTERED (DAG false-negative guard): a sibling is rejected if it is an ancestor or
descendant of the child (so it isn't a true relative via multi-inheritance/transitive paths). Bootstrap CIs on
every AUC.

  python3 eval_arch_control.py --ckpt model_nodetype.pt --graph /tmp/merged_category_parent.tsv
"""
import argparse, random, collections
import torch
from mu_attention import build_e5_tables, Tokenizer, OPS, load_dag
from eval_relatedness import build_model


def reachable_down(children, root, cap=20000):
    seen, fr = set(), [root]
    while fr and len(seen) < cap:
        n = fr.pop()
        for c in children.get(n, ()):
            if c not in seen:
                seen.add(c); fr.append(c)
    return seen


def build_triples(parents, children, n_edges, seed, node_disjoint=False):
    """Deterministic (child, parent, DAG-filtered sibling) triples, split train/test. Shared by the eval and the
    directional fine-tune so μ trains on the SAME train split the probe does and both eval on the held-out test.
    node_disjoint=True: hold out ~30% of NODES; train triples have all endpoints in the train-node set, test
    triples all in the held-out-node set — so neither μ nor the probe ever saw an eval node (the strict control)."""
    rng = random.Random(seed)
    held = None
    if node_disjoint:
        nodes = sorted(set(parents) | set(children))
        held = set(random.Random(seed * 7 + 1).sample(nodes, int(0.30 * len(nodes))))
    triples = []
    for p, kids in children.items():
        kids = [k for k in kids if k]
        if len(kids) < 2:
            continue
        for c in kids:
            desc_c = reachable_down(children, c, 4000)
            cands = [s for s in kids if s != c and s not in desc_c and c not in reachable_down(children, s, 4000)]
            if cands:
                triples.append((c, p, rng.choice(cands)))
    rng.shuffle(triples)
    if node_disjoint:
        tr = [t for t in triples if all(x not in held for x in t)][:int(0.7 * n_edges)]
        te = [t for t in triples if all(x in held for x in t)][:int(0.3 * n_edges)]
        return tr, te
    triples = triples[:n_edges]
    cut = int(0.7 * len(triples))
    return triples[:cut], triples[cut:]


def auc(pos, neg):
    pos = sorted(pos); import bisect
    s = sum(bisect.bisect_left(pos, n) + 0.5 * (bisect.bisect_right(pos, n) - bisect.bisect_left(pos, n)) for n in neg)
    return 1.0 - s / (len(pos) * len(neg))


def boot_auc(pos, neg, rng, B=400):
    import statistics as st
    vals = []
    for _ in range(B):
        p = [rng.choice(pos) for _ in pos]; n = [rng.choice(neg) for _ in neg]
        vals.append(auc(p, n))
    vals.sort()
    return auc(pos, neg), vals[int(0.025 * B)], vals[int(0.975 * B)]


def train_logistic(X, y, dev, steps=400, lr=0.05):
    X = torch.tensor(X, dtype=torch.float32, device=dev); y = torch.tensor(y, dtype=torch.float32, device=dev)
    w = torch.zeros(X.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 1e-3 * (w @ w)
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach(), b.detach()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--graph", required=True)
    ap.add_argument("--n-edges", type=int, default=4000); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", default="/tmp/archctrl_e5.pt")
    a = ap.parse_args(); dev = torch.device(a.device); rng = random.Random(a.seed)
    parents, children, deg = load_dag(a.graph)

    # membership edges + a FILTERED sibling per edge (reject ancestor/descendant siblings — DAG false-neg guard)
    tr, te = build_triples(parents, children, a.n_edges, a.seed)
    triples = tr + te
    print(f"[DATA] {len(triples)} filtered (child,parent,sibling) triples — {len(tr)} train / {len(te)} held-out")

    names = sorted({x for t in triples for x in t})
    qt, pt, idx = build_e5_tables(names, cache_path=a.cache, texts={n: n.replace('_', ' ') for n in names}, device=a.device)
    feat = lambda a_, b_: torch.cat([qt[idx[a_]], pt[idx[b_]]]).tolist()       # ordered-pair e5 feature (768-d)

    model = build_model(a.ckpt, dev); n_ops = model.op_emb.weight.shape[0]
    elem = torch.zeros(1, n_ops).index_fill_(1, torch.tensor([OPS["ELEM"]]), 1.0)
    tok = Tokenizer(qt, pt, idx, parents={}, deg={})

    @torch.no_grad()
    def mu(prs):
        out = []
        for i in range(0, len(prs), 512):
            ch = prs[i:i+512]; bd = tok.build([(x, y, 0) for x, y in ch], train=False)
            bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bd.items()}
            out += model(**bd, op_weights=elem.expand(len(ch), n_ops).to(dev)).cpu().tolist()
        return out
    e5cos = lambda prs: [(qt[idx[x]] * pt[idx[y]]).sum().item() for x, y in prs]

    for task, pos_pairs, neg_pairs in [
        ("DIRECTION (fwd vs rev)", [(c, p) for c, p, s in te], [(p, c) for c, p, s in te]),
        ("CLOSE-NEG (parent vs sibling)", [(c, p) for c, p, s in te], [(c, s) for c, p, s in te])]:
        # train the e5-probe on the TRAIN split for this task
        if "DIRECTION" in task:
            Xtr = [feat(c, p) for c, p, s in tr] + [feat(p, c) for c, p, s in tr]
        else:
            Xtr = [feat(c, p) for c, p, s in tr] + [feat(c, s) for c, p, s in tr]
        ytr = [1.0] * len(tr) + [0.0] * len(tr)
        w, b = train_logistic(Xtr, ytr, dev)
        probe = lambda prs: (torch.tensor([feat(x, y) for x, y in prs], device=dev) @ w + b).cpu().tolist()
        print(f"\n=== {task} ===  (held-out n={len(pos_pairs)})")
        print(f"  {'scorer':10} {'AUC':>6}  {'95% CI':>16}")
        for nm, fn in [("e5-cos", e5cos), ("e5-probe", probe), ("mu-elem", mu)]:
            au, lo, hi = boot_auc(fn(pos_pairs), fn(neg_pairs), rng)
            print(f"  {nm:10} {au:6.3f}  [{lo:.3f}, {hi:.3f}]")
    print("\n  read: if e5-probe ≈ mu-elem, the win is 'trained head on e5' (generic); if mu-elem > e5-probe, it's μ's architecture.")


if __name__ == "__main__":
    main()
