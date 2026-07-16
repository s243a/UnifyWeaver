#!/usr/bin/env python3
"""Luna-fused distillation — the fused head retrained in the regime where it has something to learn.

REPORT_fused_head.md's null: with the reliable judge (R≈0.004) the posterior collapses onto the label.
REPORT_multi_judge_fusion.md's boundary: with LUNA the posterior genuinely mixes channels (pull 0.133).
The stratified luna campaign closes the loop: build Kalman posteriors from [prior, graph, LUNA] against
the 5.5 operating target — luna's error blocks and cross-correlations FIT EMPIRICALLY on the train split
from the pair-matched dual-judge data, no imported R — and distill them into the kalman-fused name head.
Luna is globally affine-calibrated on that train split before the covariance fit, so systematic tilt is
removed as bias rather than charged to measurement variance.

Deployment question this answers: a CHEAP pipeline (luna labels fused with graph+prior, amortized into one
head) — how close does it get to the expensive judge's labels, vs the raw luna channel head?

Also trains the luna CHANNEL head on stratified data (fixing step 3's S starvation) and prints the
analytic fusion ladder (prior | +graph | +graph+luna vs 5.5) before any training.

The channel checkpoint initializes/anchors the trainable model. A separate, campaign-independent checkpoint
supplies the Gaussian prior and P0 residuals; keeping those roles separate avoids an optimistic prior fit.

  python3 fine_tune_fused_head_luna.py \
      --ckpt model_channel_heads_namecond_r0.pt \
      --prior-ckpt model_prod_namecond.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_luna_transfer import load_luna as load_scored_mu_tsv
from eval_within_stratum import decompose, group
from fine_tune_channel_heads import load_campaign_datasets, load_expanded, mu_batch
from fine_tune_fused_head import agnostic_readouts
from mu_attention import CORPORA, JUDGES, OPS
from product_kalman import fit_residual_covariance
from run_judge_channel import H_ALL, correlated_update_H, nll_mahal
from run_product_kalman_logit import dequant
from run_product_kalman_realdata import affine_calibrate

ROOT = os.path.dirname(os.path.abspath(__file__))
LUNA_CAMPAIGN = "/tmp/mu_data/campaign_scored_luna.tsv"


def calibrate_luna_global(luna, y, fit_idx):
    """Train-only global affine calibration, applied to every row before fitting residual covariance."""
    cal_D = affine_calibrate(luna[fit_idx, 0], y[fit_idx, 0], luna[:, 0])
    cal_S = affine_calibrate(luna[fit_idx, 1], y[fit_idx, 1], luna[:, 1])
    return np.column_stack([cal_D, cal_S])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_channel_heads_namecond_r0.pt"),
                    help="channel-head checkpoint used only for model initialization and the training anchor")
    ap.add_argument("--prior-ckpt", default=os.path.join(ROOT, "model_prod_namecond.pt"),
                    help="campaign-independent checkpoint used for prior means and P0")
    ap.add_argument("--out", default=os.path.join(ROOT, "model_fused_head_luna.pt"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0,
                    help="full-pipeline seed: train/held split, covariance fit, torch init, batch order")
    ap.add_argument("--luna-calibration", choices=("global", "none"), default="global",
                    help="global=train-only affine correction; none only reproduces the historical report")
    ap.add_argument("--analytic-only", action="store_true",
                    help="print the corrected analytic ladder and exit without training or writing a checkpoint")
    a = ap.parse_args()
    dev = "cpu"
    torch.set_num_threads(1)  # tiny deterministic campaign batches do not benefit from a large BLAS pool
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    augment_rng = np.random.default_rng(a.seed + 1)   # seed 0 reproduces the pre-flag default (rng 1)

    model, cfg = load_expanded(a.ckpt, dev=dev)
    assert model.judge_name is not None
    ref, _ = load_expanded(a.ckpt, dev=dev)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    prior_ref, _ = load_expanded(a.prior_ckpt, dev=dev)
    prior_ref.eval()
    for p in prior_ref.parameters():
        p.requires_grad = False
    print(f"training init/anchor: {os.path.basename(a.ckpt)}")
    print(f"Gaussian prior:      {os.path.basename(a.prior_ckpt)}")
    print(f"Luna calibration:    {a.luna_calibration}")
    for p in model.parameters():
        p.requires_grad = False
    model.judge_name.W.weight.requires_grad = True
    model.judge_name.resid.weight.requires_grad = True
    last = model.encoder.layers[-1]
    for p in last.parameters():
        p.requires_grad = True
    model.readout_w.requires_grad = True
    model.readout_b.requires_grad = True
    trainable = [model.judge_name.W.weight, model.judge_name.resid.weight,
                 model.readout_w, model.readout_b] + list(last.parameters())
    opt = torch.optim.Adam(trainable, lr=a.lr)

    lp, lD, lS = load_scored_mu_tsv(LUNA_CAMPAIGN)
    luna_by = {p: (lD[i], lS[i]) for i, p in enumerate(lp)}
    dss = load_campaign_datasets()
    posts, train_rows, matched = {}, {}, {}
    for n, ds in dss.items():
        if a.seed != 0:                                       # re-split for multi-seed replication
            from sigma_hop_confirmatory import descendant_disjoint_split
            ds["tr"], ds["he"] = descendant_disjoint_split(list(ds["pairs"]), a.seed, held_frac=0.30)
        mi = [i for i, p in enumerate(ds["pairs"]) if p in luna_by]
        matched[n] = set(mi)
        luna = np.array([luna_by[ds["pairs"][i]] if i in matched[n] else (np.nan, np.nan)
                         for i in range(len(ds["pairs"]))])
        tr_l = [i for i in ds["tr"] if i in matched[n]]
        he_l = [i for i in ds["he"] if i in matched[n]]
        ds["he_l"], ds["luna"] = he_l, luna
        ro = agnostic_readouts(prior_ref, ds, dev)
        y = dequant(np.column_stack([ds["D"], ds["S"]]))
        prior = np.column_stack([ro["prior_D"], ro["prior_S"]])
        m = affine_calibrate(ds["d"][tr_l], y[tr_l, 0], ds["d"])
        luna_c = (calibrate_luna_global(luna, y, tr_l)
                  if a.luna_calibration == "global" else luna.copy())  # bias first (DESIGN §2)
        meas = np.column_stack([m, luna_c[:, 0], luna_c[:, 1]])
        E5 = np.column_stack([y - prior, meas - y[:, [0, 0, 1]]])[tr_l]
        C5 = fit_residual_covariance(E5, shrinkage=a.shrink)
        P0, C_pm, R0 = C5[:2, :2], C5[:2, 2:], C5[2:, 2:]
        print(f"\n{n}: {len(mi)} luna-matched ({len(tr_l)} train / {len(he_l)} held); "
              f"fitted R_luna: D {R0[1, 1]:.4f}  S {R0[2, 2]:.4f}  (5.5 self-consistency R ≈ 0.004)")

        # analytic ladder on held rows (vs the 5.5 target) — the stratified value of the luna channel
        acc = {r: [] for r in ("prior", "prior+graph", "prior+graph+luna")}
        post = np.full((len(y), 2), np.nan)
        pull = []
        for i in range(len(y)):
            if i not in matched[n]:
                continue
            x = prior[i]
            xp_g, Pp_g = correlated_update_H(x, P0, meas[i][[0]], R0[:1, :1], C_pm[:, [0]], H_ALL[:1])
            xp_a, Pp_a = correlated_update_H(x, P0, meas[i], R0, C_pm, H_ALL)
            post[i] = np.clip(xp_a, 0.0, 1.0)
            pull.append(abs(xp_a[0] - meas[i][1]))
            if i in ds["he_l"]:
                acc["prior"].append(nll_mahal(y[i] - x, P0)[0])
                acc["prior+graph"].append(nll_mahal(y[i] - xp_g, Pp_g)[0])
                acc["prior+graph+luna"].append(nll_mahal(y[i] - xp_a, Pp_a)[0])
        posts[n] = post
        g = np.array(acc["prior+graph"]) - np.array(acc["prior+graph+luna"])
        print(f"  ladder (held NLL): prior {np.mean(acc['prior']):+.3f} | +graph "
              f"{np.mean(acc['prior+graph']):+.3f} | +luna {np.mean(acc['prior+graph+luna']):+.3f}  "
              f"(luna channel value {g.mean():+.3f}, descriptive row-SD {g.std(ddof=1):.3f})")
        print(f"  fusion non-degeneracy: mean |post_D − luna_D| = {np.mean(pull):.3f}")

        rows = []
        for i in tr_l:
            x_, y_ = ds["pairs"][i]
            rows.append(((x_, y_, OPS["HIER"], CORPORA["enwiki"], JUDGES["gpt-5.5-low"]), ds["D"][i]))
            rows.append(((x_, y_, OPS["SYM"], CORPORA["enwiki"], JUDGES["gpt-5.5-low"]), ds["S"][i]))
            rows.append(((x_, y_, OPS["HIER"], CORPORA["enwiki"], JUDGES["graph"]), ds["d"][i]))
            rows.append(((x_, y_, OPS["HIER"], CORPORA["enwiki"], JUDGES["gpt-5.6-luna"]), luna[i, 0]))
            rows.append(((x_, y_, OPS["SYM"], CORPORA["enwiki"], JUDGES["gpt-5.6-luna"]), luna[i, 1]))
            rows.append(((x_, y_, OPS["HIER"], CORPORA["enwiki"], JUDGES["kalman-fused"]), post[i, 0]))
            rows.append(((x_, y_, OPS["SYM"], CORPORA["enwiki"], JUDGES["kalman-fused"]), post[i, 1]))
        train_rows[n] = rows
        print(f"  {len(rows)} train rows (5.5 + graph + luna + fused channels)")

    if a.analytic_only:
        print("\nanalytic-only: skipped training and did not write a checkpoint")
        return

    names = list(train_rows)
    model.train()
    for step in range(1, a.steps + 1):
        n = names[step % len(names)]
        rows = train_rows[n]
        sel = rng.choice(len(rows), size=min(a.bs, len(rows)), replace=False)
        items = [rows[j][0] for j in sel]
        tgt = torch.tensor([rows[j][1] for j in sel], dtype=torch.float32, device=dev)
        mu = mu_batch(model, dss[n]["tok"], items, dev, train=True, rng=augment_rng)
        loss = torch.mean((mu - tgt) ** 2)
        ag_items = [(it[0], it[1], it[2]) for it in items]
        mu_ag = mu_batch(model, dss[n]["tok"], ag_items, dev)
        with torch.no_grad():
            mu_ref = mu_batch(ref, dss[n]["tok"], ag_items, dev)
        loss = loss + a.anchor_weight * torch.mean((mu_ag - mu_ref) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"step {step:4d} loss {loss.item():.4f}")

    model.eval()
    for n, ds in dss.items():
        he = ds["he_l"]
        groups = [group(ds["tags"][i]) for i in he]
        mus = {}
        with torch.no_grad():
            for key, judge, op in [("fused_D", "kalman-fused", "HIER"), ("fused_S", "kalman-fused", "SYM"),
                                   ("luna_D", "gpt-5.6-luna", "HIER"), ("luna_S", "gpt-5.6-luna", "SYM"),
                                   ("llm_D", "gpt-5.5-low", "HIER"), ("llm_S", "gpt-5.5-low", "SYM")]:
                items = [(ds["pairs"][i][0], ds["pairs"][i][1], OPS[op], CORPORA["enwiki"], JUDGES[judge])
                         for i in he]
                mus[key] = np.array(mu_batch(model, ds["tok"], items, dev).cpu())
        print(f"\n=== {n} (held {len(he)}) — pooled / between / WITHIN vs the 5.5 labels ===")
        for label, mu, tgt in [
            ("fused vs 5.5 D", mus["fused_D"], ds["D"][he]),
            ("luna  vs 5.5 D", mus["luna_D"], ds["D"][he]),
            ("5.5h  vs 5.5 D", mus["llm_D"], ds["D"][he]),
            ("fused vs 5.5 S", mus["fused_S"], ds["S"][he]),
            ("luna  vs 5.5 S", mus["luna_S"], ds["S"][he]),
            ("5.5h  vs 5.5 S", mus["llm_S"], ds["S"][he]),
            ("fused vs POST D", mus["fused_D"], posts[n][he, 0]),
            ("fused vs POST S", mus["fused_S"], posts[n][he, 1]),
            ("luna-head vs luna labels D", mus["luna_D"], ds["luna"][he, 0]),
            ("luna-head vs luna labels S", mus["luna_S"], ds["luna"][he, 1]),
        ]:
            pooled, between, within, per = decompose(mu, tgt, groups)
            print(f"  {label:28s}: {pooled:+.3f} / {between:+.3f} / {within:+.3f}")

    torch.save({"state": model.state_dict(), "cfg": {**cfg, "judge_name": True}}, a.out)
    print(f"\nsaved → {a.out}")


if __name__ == "__main__":
    main()
