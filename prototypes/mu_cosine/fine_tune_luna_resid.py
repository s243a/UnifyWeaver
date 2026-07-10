#!/usr/bin/env python3
"""B2 step 3: luna onboarding — the residual-only fine-tune the name architecture was built for.

Luna enters at r=0 (pure name prior — 0.971 card cosine to gpt-5.5-low). Its measured deviation
(+D/−S tilt, REPORT_channel_campaign §7) is exactly what the residual slot exists to hold: train ONLY
`judge_name.resid` on luna's 250 labelled fresh pairs. Gradient flow guarantees isolation — only row 9
appears in the batches, so every other judge's residual gets zero gradient (Adam leaves them untouched);
W and the trunk are frozen, so no other conditioning can move.

Acceptance: held-out corr vs luna's labels must beat the r=0 name prior (does 250 pairs' worth of residual
help?), and the learned offset vs the gpt-5.5-low conditioning must reproduce the measured tilt
(D positive, S negative).

  python3 fine_tune_luna_resid.py --ckpt model_channel_heads_namecond_r0.pt --out model_luna_resid.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_luna_transfer import LUNA_TSV, load_luna
from fine_tune_channel_heads import load_expanded, mu_batch
from mu_attention import CORPORA, JUDGES, OPS, Tokenizer
from run_product_kalman_realdata import DATASETS
from sigma_hop_confirmatory import FeatureGraphConfig, descendant_disjoint_split, load_e5_cache_and_filter, load_feature_graph

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_channel_heads_namecond_r0.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "model_luna_resid.pt"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    a = ap.parse_args()
    dev = "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    model, cfg = load_expanded(a.ckpt, dev=dev)
    assert model.judge_name is not None
    LUNA = JUDGES["gpt-5.6-luna"]

    pairs, D, S = load_luna(LUNA_TSV)
    fcfg = DATASETS["fresh"]
    cache, idx, pairs, _, D, S = load_e5_cache_and_filter(pairs, np.zeros(len(pairs)), D, S, fcfg["e5_cache"])
    parents, _, deg, _ = load_feature_graph(FeatureGraphConfig(**fcfg["graph"]))
    tok = Tokenizer(cache["query"], cache["passage"], idx, parents, deg)
    tr, he = descendant_disjoint_split(list(pairs), 0, held_frac=0.30)
    print(f"luna pairs: {len(pairs)} (train {len(tr)}, held {len(he)})")

    def held_corr():
        model.eval()
        out = {}
        with torch.no_grad():
            for cname, judge, op, tgt in [("D", LUNA, "HIER", D), ("S", LUNA, "SYM", S),
                                          ("D-5.5", JUDGES["gpt-5.5-low"], "HIER", D),
                                          ("S-5.5", JUDGES["gpt-5.5-low"], "SYM", S)]:
                items = [(pairs[i][0], pairs[i][1], OPS[op], CORPORA["enwiki"], judge) for i in he]
                mu = np.array(mu_batch(model, tok, items, dev).cpu())
                out[cname] = (float(np.corrcoef(mu, tgt[he])[0, 1]), float(mu.mean()))
        model.train()
        return out

    c0 = held_corr()
    print(f"BEFORE (r=0 name prior): D {c0['D'][0]:+.3f}  S {c0['S'][0]:+.3f}   "
          f"(5.5-row: D {c0['D-5.5'][0]:+.3f}  S {c0['S-5.5'][0]:+.3f})")

    for p in model.parameters():
        p.requires_grad = False
    model.judge_name.resid.weight.requires_grad = True     # only luna's row appears in batches
    opt = torch.optim.Adam([model.judge_name.resid.weight], lr=a.lr)
    resid_before = model.judge_name.resid.weight.detach().clone()

    rows = []
    for i in tr:
        x, y = pairs[i]
        rows.append(((x, y, OPS["HIER"], CORPORA["enwiki"], LUNA), D[i]))
        rows.append(((x, y, OPS["SYM"], CORPORA["enwiki"], LUNA), S[i]))
    model.train()
    for step in range(1, a.steps + 1):
        sel = rng.choice(len(rows), size=min(a.bs, len(rows)), replace=False)
        items = [rows[j][0] for j in sel]
        tgt = torch.tensor([rows[j][1] for j in sel], dtype=torch.float32, device=dev)
        mu = mu_batch(model, tok, items, dev, train=True, rng=np.random)
        loss = torch.mean((mu - tgt) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            c = held_corr()
            print(f"step {step:3d} loss {loss.item():.4f}  held: D {c['D'][0]:+.3f} S {c['S'][0]:+.3f}")

    drift = (model.judge_name.resid.weight - resid_before).norm(dim=1)
    others = torch.cat([drift[:LUNA], drift[LUNA + 1:]])
    print(f"resid drift: luna {drift[LUNA]:.4f}; max other {others.max():.2e} (must be 0 — isolation)")

    c1 = held_corr()
    print(f"\nAFTER: luna-cond D {c1['D'][0]:+.3f} (was {c0['D'][0]:+.3f})  "
          f"S {c1['S'][0]:+.3f} (was {c0['S'][0]:+.3f})")
    # tilt check: mean readout offset luna − 5.5 per channel (measured judge tilt: D +0.07..0.09, S −0.11..−0.13)
    print(f"tilt (mean μ_luna − μ_5.5 on held): D {c1['D'][1] - c1['D-5.5'][1]:+.3f} (expect +)  "
          f"S {c1['S'][1] - c1['S-5.5'][1]:+.3f} (expect −)")

    torch.save({"state": model.state_dict(), "cfg": cfg}, a.out)
    print(f"saved → {a.out}")


if __name__ == "__main__":
    main()
