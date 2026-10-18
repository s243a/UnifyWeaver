#!/usr/bin/env python3
"""Node-disjoint, multi-seed hardening of the μ-vs-e5-probe comparison (PR #3387 rigor).

Per seed: hold out 30% of NODES; fine-tune μ (directional ranking loss) on train-node edges only; train the
e5-probe on the same; eval μ / e5-probe / e5-cos on held-out-NODE edges — neither μ nor the probe saw these
nodes as *task examples* (frozen-e5 / base-model pretraining exposure is symmetric). Reports mean±sd across seeds.

  python3 eval_arch_harden.py --ckpt model_nodetype.pt --graph /tmp/merged_category_parent.tsv --seeds 1,2,3
"""
import argparse, random, statistics as st
import torch, torch.nn.functional as F
from mu_attention import build_e5_tables, Tokenizer, OPS, load_dag
from eval_arch_control import build_triples, build_model, auc, train_logistic


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--graph", required=True)
    ap.add_argument("--seeds", default="1,2,3"); ap.add_argument("--n-edges", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device)
    parents, children, deg = load_dag(a.graph)
    seeds = [int(s) for s in a.seeds.split(",")]
    acc = {k: {"dir": [], "cn": []} for k in ("e5-cos", "e5-probe", "mu")}

    for seed in seeds:
        rng = random.Random(seed)
        tr, te = build_triples(parents, children, a.n_edges, seed, node_disjoint=True)
        names = sorted({x for t in tr + te for x in t})
        qt, pt, idx = build_e5_tables(names, cache_path=f"/tmp/harden_{seed}.pt",
                                      texts={n: n.replace('_', ' ') for n in names}, device=a.device)
        tok = Tokenizer(qt, pt, idx, parents={}, deg={})
        model = build_model(a.ckpt, dev); model.train()
        n_ops = model.op_emb.weight.shape[0]
        elem = torch.zeros(1, n_ops, device=dev).index_fill_(1, torch.tensor([OPS["ELEM"]], device=dev), 1.0)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

        def mu(prs):
            out = []
            for i in range(0, len(prs), 512):
                ch = prs[i:i+512]; b = tok.build([(x, y, 0) for x, y in ch], train=False)
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                out += model(**b, op_weights=elem.expand(len(ch), n_ops)).detach().cpu().tolist()
            return out

        edges = [(c, p) for c, p, s in tr]
        for _ in range(a.steps):                                   # fine-tune μ: rank μ(fwd) > μ(rev) on TRAIN nodes
            batch = [rng.choice(edges) for _ in range(a.bs)]
            def mug(prs):
                b = tok.build([(x, y, 0) for x, y in prs], train=False)
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                return model(**b, op_weights=elem.expand(len(prs), n_ops))
            f = mug([(c, p) for c, p in batch]); r = mug([(p, c) for c, p in batch])
            loss = F.softplus(-10.0 * (f - r - 0.1)).mean() + 0.2 * (F.mse_loss(f, torch.full_like(f, 0.9)) + F.mse_loss(r, torch.full_like(r, 0.1)))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()

        feat = lambda x, y: torch.cat([qt[idx[x]], pt[idx[y]]]).tolist()
        e5cos = lambda prs: [(qt[idx[x]] * pt[idx[y]]).sum().item() for x, y in prs]
        for task, pos, neg in [("dir", [(c, p) for c, p, s in te], [(p, c) for c, p, s in te]),
                               ("cn",  [(c, p) for c, p, s in te], [(c, s) for c, p, s in te])]:
            # e5-probe trained on TRAIN nodes for this task
            if task == "dir":
                Xtr = [feat(c, p) for c, p, s in tr] + [feat(p, c) for c, p, s in tr]
            else:
                Xtr = [feat(c, p) for c, p, s in tr] + [feat(c, s) for c, p, s in tr]
            w, b = train_logistic(Xtr, [1.0] * len(tr) + [0.0] * len(tr), dev)
            probe = lambda prs: (torch.tensor([feat(x, y) for x, y in prs], device=dev) @ w + b).cpu().tolist()
            acc["e5-cos"][task].append(auc(e5cos(pos), e5cos(neg)))
            acc["e5-probe"][task].append(auc(probe(pos), probe(neg)))
            acc["mu"][task].append(auc(mu(pos), mu(neg)))
        print(f"  seed {seed}: |tr|={len(tr)} |te|={len(te)}  "
              f"dir μ={acc['mu']['dir'][-1]:.3f}/probe={acc['e5-probe']['dir'][-1]:.3f}  "
              f"cn μ={acc['mu']['cn'][-1]:.3f}/probe={acc['e5-probe']['cn'][-1]:.3f}")

    m = lambda L: (st.mean(L), st.stdev(L) if len(L) > 1 else 0.0)
    print(f"\n[NODE-DISJOINT, {len(seeds)} seeds]  mean ± sd")
    print(f"  {'scorer':9} {'DIRECTION':>16} {'CLOSE-NEG':>16}")
    for k in ("e5-cos", "e5-probe", "mu"):
        d, c = m(acc[k]["dir"]), m(acc[k]["cn"])
        print(f"  {k:9} {d[0]:.3f} ± {d[1]:.3f}    {c[0]:.3f} ± {c[1]:.3f}")


if __name__ == "__main__":
    main()
