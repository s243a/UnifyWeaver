#!/usr/bin/env python3
"""train_dir_rank.py — discriminative (ranking) directional fine-tune.

Closes the residual direction gap (μ 0.84 vs e5-probe 0.92). μ's directional supervision is *regression* to
fixed 0.90/0.10 targets, which caps the forward-vs-reverse separation. This fine-tune instead optimizes the
**comparison** directly — the same ranking-CE form as the transitive constraint:

    L_rank = softplus( -s * ( μ(child|parent) - μ(parent|child) - margin ) )      # reward fwd > rev by a margin

plus a light calibration anchor (keep μ readable, not just separable):

    L_anchor = w * ( MSE(μ_fwd, 0.9) + MSE(μ_rev, 0.1) )

Warm-starts model_dir.pt (the directionally-supervised model), lineage-free to match eval_arch_control.

  python3 train_dir_rank.py --ckpt model_dir.pt --graph /tmp/merged_category_parent.tsv --save model_dir_disc.pt
"""
import argparse, random
import torch
import torch.nn.functional as F
from mu_attention import build_e5_tables, Tokenizer, OPS, load_dag
from eval_relatedness import build_model
from eval_arch_control import build_triples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--graph", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--margin", type=float, default=0.1); ap.add_argument("--anchor-w", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", default="/tmp/dirrank_e5.pt")
    ap.add_argument("--n-edges", type=int, default=4000, help="must match eval_arch_control to share the split")
    a = ap.parse_args(); dev = torch.device(a.device); rng = random.Random(a.seed)

    # train ONLY on the 70% split that eval_arch_control trains its probe on; held-out 30% is never seen (no leak)
    parents, children, deg = load_dag(a.graph)
    tr, te = build_triples(parents, children, a.n_edges, a.seed)
    edges = [(c, p) for c, p, s in tr]                          # (child=member, parent=container)
    print(f"[DATA] fine-tuning on {len(edges)} TRAIN edges (held-out {len(te)} never seen)")
    names = sorted({x for e in (edges + [(c, p) for c, p, s in te]) for x in e})
    qt, pt, idx = build_e5_tables(names, cache_path=a.cache, texts={n: n.replace('_', ' ') for n in names}, device=a.device)
    tok = Tokenizer(qt, pt, idx, parents={}, deg={})           # lineage-free (matches eval)
    model = build_model(a.ckpt, dev); model.train()
    n_ops = model.op_emb.weight.shape[0]
    elem = torch.zeros(1, n_ops, device=dev).index_fill_(1, torch.tensor([OPS["ELEM"]], device=dev), 1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    def mu(prs):
        b = tok.build([(x, y, 0) for x, y in prs], train=False)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        return model(**b, op_weights=elem.expand(len(prs), n_ops))

    for step in range(a.steps):
        batch = [rng.choice(edges) for _ in range(a.bs)]
        fwd = mu([(c, p) for c, p in batch])                   # μ(member|container) — want HIGH
        rev = mu([(p, c) for c, p in batch])                   # μ(container|member) — want LOW
        L_rank = F.softplus(-a.scale * (fwd - rev - a.margin)).mean()
        L_anchor = a.anchor_w * (F.mse_loss(fwd, torch.full_like(fwd, 0.9)) + F.mse_loss(rev, torch.full_like(rev, 0.1)))
        loss = L_rank + L_anchor
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 150 == 0:
            print(f"  step {step+1}: L_rank {L_rank.item():.4f}  mean fwd {fwd.mean().item():.3f}  rev {rev.mean().item():.3f}")
    torch.save({"state": model.state_dict(), "cfg": torch.load(a.ckpt, weights_only=False).get("cfg", {})}, a.save)
    print(f"saved → {a.save}")


if __name__ == "__main__":
    main()
