#!/usr/bin/env python3
"""B2 step 2: the FUSED head — distill the Lever-A Kalman posteriors into a name-conditioned readout
(DESIGN_amortized_fusion_heads: the three-way learn's mu_PoE, the amortized filter).

Target construction (per campaign pair, per corpus, covariance blocks fit on the TRAIN split only):
  prior  = frozen base model's AGNOSTIC readouts (mu_D, mu_S)
  meas   = [graph walk d (affine-calibrated to D), judge D, judge S], H rows [1,0],[1,0],[0,1]
  blocks = prior/graph errors fit vs dequantized judge labels on train rows (includes judge noise —
           documented inflation); the judge channel is priced by the MEASURED R_judge from the Lever-A
           self-consistency runs (the campaign has one judge run, so fitting R_judge here would give the
           degenerate 0 — judge-as-measurement-and-truth, the run_judge_channel.py honesty problem)
  target = correlated Kalman posterior (D_post, S_post), clipped to [0,1]

Independent-judge-channel approximation: judge-error cross-correlations with prior/graph are set to 0
(unmeasurable without a second run on THESE pairs; Lever A showed correlation pricing moves error bars far
more than means, and the mean is what we distill).

Training (two-target recipe, DESIGN §three-way): measured anchor (the posterior) + a stop-grad consistency
prior toward the naive fusion of the model's own channel heads (equal-weight mean of graph/LLM heads for D;
the LLM head for S — graph doesn't observe S). Consistency targets come from the FROZEN reference (the
literal stop-grad of the init model; the live channel heads stay pinned by continued channel supervision).
Channel rows keep training alongside so the fused head's trunk updates can't silently repurpose them.
Conditioning: judge="kalman-fused" — a NAME, not a new learned row (r=0 name prior at start): the first
consumer of the §6 migration.

Eval (within-stratum, the honest B2 frame): fused head vs held-out posterior (distillation fidelity) AND
vs held-out judge labels, side-by-side with the llm-head baseline from the same checkpoint.

  python3 fine_tune_fused_head.py --ckpt model_channel_heads_namecond_r0.pt --steps 800 --lr 5e-4
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_luna_transfer import load_luna as load_scored_mu_tsv
from eval_within_stratum import decompose, group
from fine_tune_channel_heads import channel_rows, load_campaign_datasets, load_expanded, mu_batch
from mu_attention import CORPORA, JUDGES, OPS
from product_kalman import fit_residual_covariance
from run_judge_channel import H_ALL, correlated_update_H
from run_product_kalman_logit import dequant
from run_product_kalman_realdata import affine_calibrate

ROOT = os.path.dirname(os.path.abspath(__file__))
J1_TSV = "/tmp/mu_data/sigma_hop_fresh_scored_gpt55low.tsv"
J2_TSV = "/tmp/mu_data/sigma_hop_fresh_scored_j2.tsv"


def measure_r_judge():
    """R_judge per channel from the Lever-A self-consistency runs: Var(j2−j1)/2 (iid assumption)."""
    p1, D1, S1 = load_scored_mu_tsv(J1_TSV)
    p2, D2, S2 = load_scored_mu_tsv(J2_TSV)
    ix = {p: i for i, p in enumerate(p1)}
    m = [(ix[p], j) for j, p in enumerate(p2) if p in ix]
    i1, i2 = zip(*m)
    rD = float(np.var(D2[list(i2)] - D1[list(i1)]) / 2)
    rS = float(np.var(S2[list(i2)] - S1[list(i1)]) / 2)
    print(f"R_judge from self-consistency ({len(m)} pairs): D {rD:.4f}  S {rS:.4f}")
    return rD, rS


def agnostic_readouts(model, ds, dev, bs=256):
    """Frozen agnostic (mu_D, mu_S) priors + the ref channel readouts for the consistency targets."""
    outs = {}
    model.eval()
    with torch.no_grad():
        for key, item_fn in [
            ("prior_D", lambda x, y: (x, y, OPS["HIER"])),
            ("prior_S", lambda x, y: (x, y, OPS["SYM"])),
            ("graph_D", lambda x, y: (x, y, OPS["HIER"], CORPORA["enwiki"], JUDGES["graph"])),
            ("llm_D", lambda x, y: (x, y, OPS["HIER"], CORPORA["enwiki"], JUDGES["gpt-5.5-low"])),
            ("llm_S", lambda x, y: (x, y, OPS["SYM"], CORPORA["enwiki"], JUDGES["gpt-5.5-low"])),
        ]:
            mu = []
            for lo in range(0, len(ds["pairs"]), bs):
                items = [item_fn(x, y) for x, y in ds["pairs"][lo:lo + bs]]
                mu.append(np.array(mu_batch(model, ds["tok"], items, dev).cpu()))
            outs[key] = np.concatenate(mu)
    return outs


def kalman_targets(ds, ro, rD, rS, shrink=0.05):
    """(D_post, S_post) per row + the conflict observable |graph − prior_D| (Lever A's best routing signal)
    — train-split-fit blocks, correlated prior/graph, independent judge rows."""
    y = dequant(np.column_stack([ds["D"], ds["S"]]))
    prior = np.column_stack([ro["prior_D"], ro["prior_S"]])
    tr = ds["tr"]
    m = affine_calibrate(ds["d"][tr], y[tr, 0], ds["d"])
    E3 = np.column_stack([y[:, 0] - prior[:, 0], y[:, 1] - prior[:, 1], m - y[:, 0]])[tr]
    C3 = fit_residual_covariance(E3, shrinkage=shrink)
    P0, C_pg, R_g = C3[:2, :2], C3[:2, 2], C3[2, 2]
    R0 = np.diag([R_g, rD, rS])
    C_pm = np.zeros((2, 3)); C_pm[:, 0] = C_pg
    post = np.zeros((len(y), 2))
    for i in range(len(y)):
        meas = np.array([m[i], y[i, 0], y[i, 1]])
        xp, _ = correlated_update_H(prior[i], P0, meas, R0, C_pm, H_ALL)
        post[i] = np.clip(xp, 0.0, 1.0)
    return post, np.abs(m - prior[:, 0])


def fused_rows(ds, idxs, post):
    rows = []
    for i in idxs:
        x, y = ds["pairs"][i]
        rows.append(((x, y, OPS["HIER"], CORPORA["enwiki"], JUDGES["kalman-fused"]), post[i, 0], i, 0))
        rows.append(((x, y, OPS["SYM"], CORPORA["enwiki"], JUDGES["kalman-fused"]), post[i, 1], i, 1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_channel_heads_namecond_r0.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "model_fused_head.pt"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--consistency-weight", type=float, default=0.25,
                    help="stop-grad prior toward the naive fusion of the model's own channel heads")
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--eval-only", action="store_true", help="skip training; evaluate the --out checkpoint")
    a = ap.parse_args()
    dev = "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    model, cfg = load_expanded(a.ckpt, dev=dev)
    assert model.judge_name is not None, "fused head requires a name-cond checkpoint (migrate_judge_names.py)"
    ref, _ = load_expanded(a.ckpt, dev=dev)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

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
    print(f"trainable params: {sum(p.numel() for p in trainable)} (judge_name + last layer + readout)")

    rD, rS = measure_r_judge()
    dss = load_campaign_datasets()
    posts, cons, confl, train_rows = {}, {}, {}, {}
    for n, ds in dss.items():
        ro = agnostic_readouts(ref, ds, dev)
        posts[n], confl[n] = kalman_targets(ds, ro, rD, rS)
        # stop-grad consistency prior: naive equal-weight fusion of the ref's own channel heads
        cons[n] = np.column_stack([0.5 * (ro["graph_D"] + ro["llm_D"]), ro["llm_S"]])
        train_rows[n] = channel_rows(ds, ds["tr"]) + fused_rows(ds, ds["tr"], posts[n])
        dp = posts[n][:, 0] - ds["D"]; sp = posts[n][:, 1] - ds["S"]
        print(f"{n}: {len(train_rows[n])} rows (channel+fused); posterior pull off the label: "
              f"D mean|Δ| {np.abs(dp).mean():.3f}  S mean|Δ| {np.abs(sp).mean():.3f}")

    if a.eval_only:
        model, _ = load_expanded(a.out, dev=dev)
        a.steps = 0
    names = list(train_rows)
    model.train()
    for step in range(1, a.steps + 1):
        n = names[step % len(names)]
        rows = train_rows[n]
        sel = rng.choice(len(rows), size=min(a.bs, len(rows)), replace=False)
        items = [rows[j][0] for j in sel]
        tgt = torch.tensor([rows[j][1] for j in sel], dtype=torch.float32, device=dev)
        mu = mu_batch(model, dss[n]["tok"], items, dev, train=True, rng=np.random)
        loss = torch.mean((mu - tgt) ** 2)
        # consistency prior on the fused rows of this batch (channel rows carry (item, target) 2-tuples)
        fmask = [k for k, j in enumerate(sel) if len(rows[j]) == 4]
        if fmask and a.consistency_weight > 0:
            ct = torch.tensor([cons[n][rows[sel[k]][2], rows[sel[k]][3]] for k in fmask],
                              dtype=torch.float32, device=dev)
            loss = loss + a.consistency_weight * torch.mean((mu[fmask] - ct) ** 2)
        # agnostic-anchor honesty (B1b): provenance-masked readouts must match the frozen reference
        ag_items = [(it[0], it[1], it[2]) for it in items]
        mu_ag = mu_batch(model, dss[n]["tok"], ag_items, dev)
        with torch.no_grad():
            mu_ref = mu_batch(ref, dss[n]["tok"], ag_items, dev)
        loss = loss + a.anchor_weight * torch.mean((mu_ag - mu_ref) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == 1:
            print(f"step {step:4d} loss {loss.item():.4f}")

    # ---- eval: within-stratum, fused head vs posterior (fidelity) and vs labels (vs llm baseline) ----
    model.eval()
    for n, ds in dss.items():
        he = ds["he"]
        groups = [group(ds["tags"][i]) for i in he]
        mus = {}
        with torch.no_grad():
            for key, judge, op in [("fused_D", "kalman-fused", "HIER"), ("fused_S", "kalman-fused", "SYM"),
                                   ("llm_D", "gpt-5.5-low", "HIER"), ("llm_S", "gpt-5.5-low", "SYM")]:
                items = [(ds["pairs"][i][0], ds["pairs"][i][1], OPS[op], CORPORA["enwiki"], JUDGES[judge])
                         for i in he]
                mus[key] = np.array(mu_batch(model, ds["tok"], items, dev).cpu())
        print(f"\n=== {n} (held: {len(he)}) — pooled / between / WITHIN ===")
        for label, mu, tgt in [
            ("fused vs POSTERIOR D", mus["fused_D"], posts[n][he, 0]),
            ("fused vs POSTERIOR S", mus["fused_S"], posts[n][he, 1]),
            ("fused vs labels D   ", mus["fused_D"], ds["D"][he]),
            ("llm   vs labels D   ", mus["llm_D"], ds["D"][he]),
            ("fused vs labels S   ", mus["fused_S"], ds["S"][he]),
            ("llm   vs labels S   ", mus["llm_S"], ds["S"][he]),
        ]:
            pooled, between, within, per = decompose(mu, tgt, groups)
            per_s = " ".join(f"{g} {v:+.2f}" for g, v in per.items())
            print(f"  {label}: {pooled:+.3f} / {between:+.3f} / {within:+.3f}   [{per_s}]")
        # CONFLICT slice — where the channels disagree is where fusion can differ from the label
        # (Lever A: conflict = the best pre-call routing observable; the posterior deviates most here)
        c_he = confl[n][he]
        hi = c_he >= np.quantile(c_he, 0.75)
        r = lambda x, t: float(np.corrcoef(x, t)[0, 1])
        Dl, Sl = ds["D"][he], ds["S"][he]
        print(f"  conflict slice (top-quartile |graph − prior_D|, n={int(hi.sum())} vs rest {int((~hi).sum())}):")
        print(f"    D vs label : fused {r(mus['fused_D'][hi], Dl[hi]):+.3f} llm {r(mus['llm_D'][hi], Dl[hi]):+.3f}"
              f"   (rest: fused {r(mus['fused_D'][~hi], Dl[~hi]):+.3f} llm {r(mus['llm_D'][~hi], Dl[~hi]):+.3f})")
        print(f"    D vs post  : fused {r(mus['fused_D'][hi], posts[n][he, 0][hi]):+.3f} "
              f"llm {r(mus['llm_D'][hi], posts[n][he, 0][hi]):+.3f}")

    torch.save({"state": model.state_dict(), "cfg": {**cfg, "judge_name": True}}, a.out)
    print(f"\nsaved → {a.out}")


if __name__ == "__main__":
    main()
