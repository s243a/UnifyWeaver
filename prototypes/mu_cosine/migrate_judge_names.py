#!/usr/bin/env python3
"""Behavior-preserving migration: indexed judge_emb → name-function conditioning (REPORT_channel_campaign §6).

The key trick (§6.3): fit W by ridge least squares over the existing judges' rows
(min_W Σ‖W·e_j − judge_emb[j]‖² + λ‖W‖²), then set r_j = judge_emb[j] − W·e_j — the new pathway
cond_j = W·e_j + r_j reproduces the old rows EXACTLY at init regardless of λ (the residual absorbs the fit
error), so λ only shapes how W extrapolates to UNSEEN names (the transfer behavior). Ridge matters because
the card embeddings are highly collinear (all e5 cosines ≳0.8): the unregularized minimum-norm interpolant
extrapolates wildly.

Rows fit vs absorbed: only rows with non-negligible norm inform W (zero rows are unreferenced/zero-init
judges with no calibration to preserve — forcing W·e_j ≈ 0 for them would distort the translation). Their
residual still reproduces cond = 0 exactly at init; training under the ‖r‖→0 regularizer then pulls them
toward their name prior, which is strictly better than the zero row they had.

Acceptance (§6.6, init half): (1) reconstruction max|W·e_j + r_j − judge_emb[j]| at float roundoff;
(2) END-TO-END forward equivalence on a synthetic judge-conditioned batch (catches wiring bugs the
row-level check can't).

  python3 migrate_judge_names.py --ckpt model_prod.pt --out model_prod_namecond.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_cards import judge_card_e5
from mu_attention import JUDGES, OPS, MuAttention

ROOT = os.path.dirname(os.path.abspath(__file__))


def fit_name_function(judge_emb, E, ridge=0.1, min_norm=1e-6):
    """W [d_model, d_e5], resid [J, d_model] from old rows Y=judge_emb and cards E (both torch, float32).
    Fit on informative rows only; residuals guarantee exact reproduction for ALL rows."""
    Y, E = judge_emb.double(), E.double()
    keep = Y.norm(dim=1) > min_norm
    Ek, Yk = E[keep], Y[keep]
    A = Ek.t() @ Ek + ridge * torch.eye(E.shape[1], dtype=torch.float64)
    W = torch.linalg.solve(A, Ek.t() @ Yk).t()               # [d_model, d_e5]
    resid = Y - E @ W.t()
    return W.float(), resid.float(), keep


def build_from_ckpt(ckpt_path, judge_name_e5=None, dev="cpu", n_judge=len(JUDGES)):
    """Load + build, padding judge_emb to n_judge rows (zero-init, unreferenced — the load_expanded
    convention) so pre-expansion checkpoints can be conditioned on every current judge."""
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    sd = dict(ck["state"]); cfg = ck.get("cfg", {"d_model": 384, "heads": 4, "layers": 3})
    Y = sd["judge_emb.weight"]
    if Y.shape[0] < n_judge:
        sd["judge_emb.weight"] = torch.cat([Y, torch.zeros(n_judge - Y.shape[0], Y.shape[1])], 0)
    sz = lambda k, d: sd[k].shape[0] if k in sd else d
    m = MuAttention(d_model=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
                    n_ops=sz("op_emb.weight", len(OPS)), n_corpus=sz("corpus_emb.weight", 2),
                    n_judge=n_judge,
                    n_nodetype=sz("nodetype_emb.weight", 4), judge_name_e5=judge_name_e5).to(dev)
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not unexp, f"unexpected keys: {unexp}"
    return m, sd, cfg


def synthetic_prov_batch(d_model, n_judge, seed=0):
    """A judge-conditioned forward batch with no tokenizer/e5 dependency: B=n_judge examples, T=4 tokens
    (op slot, anchor, content, provenance slot), each example conditioned on a different judge."""
    g = torch.Generator().manual_seed(seed)
    B, T = n_judge, 4
    content = torch.randn(B, T, d_model, generator=g)
    gen_id = torch.full((B, T), -1, dtype=torch.long); gen_id[:, 2] = 1
    is_anchor = torch.zeros(B, T); is_anchor[:, 1] = 1.0
    op_pos = torch.full((B, T), -1, dtype=torch.long); op_pos[:, 0] = OPS["HIER"]
    op_of = torch.full((B,), OPS["HIER"], dtype=torch.long)
    pad = torch.zeros(B, T, dtype=torch.bool)
    is_prov = torch.zeros(B, T); is_prov[:, 3] = 1.0
    corpus_of = torch.full((B, T), -1, dtype=torch.long); corpus_of[:, 3] = 0
    judge_of = torch.full((B, T), -1, dtype=torch.long); judge_of[:, 3] = torch.arange(n_judge)
    return dict(content=content, gen_id=gen_id, is_anchor=is_anchor, op_pos=op_pos, op_of=op_of,
                pad=pad, is_prov=is_prov, corpus_of=corpus_of, judge_of=judge_of)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod.pt"))
    ap.add_argument("--out", default=None, help="default: <ckpt stem>_namecond.pt")
    ap.add_argument("--ridge", type=float, default=0.1)
    a = ap.parse_args()
    out = a.out or a.ckpt.rsplit(".pt", 1)[0] + "_namecond.pt"

    names = sorted(JUDGES, key=JUDGES.get)
    E, _ = judge_card_e5(names)
    old, sd, cfg = build_from_ckpt(a.ckpt)
    Y = sd["judge_emb.weight"]

    W, resid, keep = fit_name_function(Y, E, ridge=a.ridge)
    fitted = [n for n, k in zip(names, keep) if k]
    print(f"fit W on {keep.sum().item()}/{len(names)} informative rows: {fitted}")
    rec_err = (E @ W.t() + resid - Y).abs().max().item()
    print(f"reconstruction max|W·e_j + r_j − judge_emb[j]| = {rec_err:.2e} (must be float roundoff)")
    print(f"residual norms: " + " ".join(f"{n}={resid[i].norm():.3f}" for i, n in enumerate(names)))
    print(f"name-prior norms ‖W·e_j‖: " + " ".join(f"{n}={(E @ W.t())[i].norm():.3f}" for i, n in enumerate(names)))

    sd["judge_name.name_e5"] = E
    sd["judge_name.W.weight"] = W
    sd["judge_name.resid.weight"] = resid

    # end-to-end forward equivalence (acceptance §6.6 init half)
    new = MuAttention(d_model=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
                      n_ops=sd["op_emb.weight"].shape[0], n_corpus=sd["corpus_emb.weight"].shape[0],
                      n_judge=len(names), n_nodetype=sd["nodetype_emb.weight"].shape[0], judge_name_e5=E)
    miss, unexp = new.load_state_dict(sd, strict=False)
    assert not unexp, f"unexpected keys: {unexp}"
    old.eval(); new.eval()
    b = synthetic_prov_batch(cfg["d_model"], len(names))
    with torch.no_grad():
        mu_old, mu_new = old(**b), new(**b)
    fwd_err = (mu_old - mu_new).abs().max().item()
    print(f"forward max|μ_old − μ_new| over all-judges synthetic batch = {fwd_err:.2e}")
    assert rec_err < 1e-5 and fwd_err < 1e-5, "behavior-preserving init FAILED"

    torch.save({"state": new.state_dict(), "cfg": {**cfg, "judge_name": True, "ridge": a.ridge}}, out)
    print(f"saved → {out} (cfg.judge_name=True)")


if __name__ == "__main__":
    main()
