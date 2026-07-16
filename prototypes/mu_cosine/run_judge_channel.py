#!/usr/bin/env python3
"""Lever A: the LLM judge as a SECOND MEASUREMENT channel + information-value routing.

Setup honesty: in offline evaluation the judge's labels are the target, so the judge cannot be both measurement
and truth. Fix: a SECOND independent scoring run (same pairs, same judge family, fresh sample) supplies the
measurement channel — and, as a byproduct, the long-requested judge SELF-CONSISTENCY bound (same pairs scored
twice = direct measurement of judge noise; for iid samples Var(j2−j1) = 2·R_judge).

Channels: prior = model readouts (mu_D, mu_S); measurements = [graph walk (observes D), judge2 D, judge2 S]
(H rows [1,0],[1,0],[0,1]); target = judge1 labels (de-quantized). Constant covariance blocks (the judge-value
question is first-order; hop-conditioning enters only the routing IV below). Correlated updates throughout —
the 5x5 joint error covariance carries all cross-channel correlations (judge2 errors are heavily correlated
with the target by construction of self-consistency; the fusion must price that).

Ladder (per descendant-disjoint split): prior | prior+graph | prior+judge | prior+graph+judge.

ROUTING ("when is a judge call worth it"): per-row realized gain g_i = NLL(prior+graph) − NLL(+judge), then
gain-per-call curves under policies, all computable BEFORE the judge call except the oracle:
  random | hop-IV (expected posterior-variance reduction via the bivariate P(h) fit; C=0 approximation) |
  ambiguity (posterior D nearest 0.5 — the epistemic-bimodality proxy) | conflict (|graph − prior_D| innovation)
  | oracle (sort by realized gain; upper bound).

  python3 run_judge_channel.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from product_kalman import fit_residual_covariance, gaussian_condition_update
from run_product_kalman_realdata import DATASETS, affine_calibrate
from run_product_kalman_logit import dequant
from run_product_kalman_sigma_hop import LOG2PI
from sigma_hop_confirmatory import (
    FeatureGraphConfig,
    build_confirmatory_data_from_labels,
    descendant_disjoint_split,
    fit_sigma_of_hop,
    load_scored_pairs,
    sigma_of_hop,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
H_ALL = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])      # graph→D, judgeD→D, judgeS→S


def correlated_update_H(x, P, y, R, C, Hm):
    """Correlated Kalman / Gaussian-conditioning update with prior<->measurement cross-covariance C.

    State x ~ N(x, P); observation y = Hm x + v, Cov(v)=R, Cov(x-x, v)=C (shape state_dim x obs_dim).
    Delegates to product_kalman.gaussian_condition_update (blocker 7): ONE Cholesky-based implementation,
    replacing the former np.linalg.inv path. The algebra is identical --
      S = Hm P Hmᵀ + R + Hm C + Cᵀ Hmᵀ;  K = (P Hmᵀ + C) S⁻¹;  xp = x + K(y - Hm x);  Pp = P - K S Kᵀ
    -- but solved via Cholesky with SPD regularization instead of an explicit inverse. Returns writable
    copies so callers may mutate the results."""
    upd = gaussian_condition_update(x, P, y, R, H=Hm, cross_covariance=C)
    return np.array(upd.mean), np.array(upd.covariance)


def nll_mahal(r, V):
    Vi = np.linalg.inv(V)
    m2 = float(r @ Vi @ r)
    return 0.5 * m2 + 0.5 * np.log(np.linalg.det(V)) + LOG2PI, m2


def run(name, j2_scored, seeds=40, held_frac=0.30, shrink=0.05):
    cfg = DATASETS[name]
    pairs, hop, D, S = load_scored_pairs(cfg["score_in"], cfg["responses"], prefix="transitive_h")
    data = build_confirmatory_data_from_labels(
        pairs, hop, D, S, cfg["e5_cache"], FeatureGraphConfig(**cfg["graph"]),
        os.path.join(ROOT, "model_prod.pt"), "cpu",
    )
    # judge2: second scoring run, matched by (node, root) from its scored TSV mu columns
    j2 = {}
    with open(j2_scored, encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        DIR = ["subcategory", "subtopic", "element_of", "super_category"]; SYM = ["see_also", "assoc"]
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < len(header):
                continue
            d2 = max(float(c[col[f"mu[{r}]"]]) for r in DIR)
            s2 = max(float(c[col[f"mu[{r}]"]]) for r in SYM)
            j2[(c[col["node"]], c[col["root"]])] = (d2, s2)
    keep = [i for i, p in enumerate(data.pairs) if p in j2]
    print(f"\n=== {name}: {len(keep)}/{len(data.pairs)} pairs matched with judge2 ===")
    prior = data.X[keep, :2]; d = data.X[keep, 2]; hopk = data.hop[keep]
    t1 = np.column_stack([data.D[keep], data.S[keep]])       # judge1 = target
    j2v = np.array([j2[data.pairs[i]] for i in keep])        # judge2 = measurement
    y = dequant(t1)
    pk = [data.pairs[i] for i in keep]

    # --- judge self-consistency (the reliability bound the council asked for) ---
    print("judge self-consistency (judge1 vs judge2, same pairs, independent runs):")
    for ch, nm in [(0, "D"), (1, "S")]:
        diff = j2v[:, ch] - t1[:, ch]
        print(f"  {nm}: corr {np.corrcoef(t1[:, ch], j2v[:, ch])[0, 1]:+.3f}  MAE {np.abs(diff).mean():.3f}  "
              f"sd(diff) {diff.std():.3f}  ⇒ R_judge ≈ {diff.var()/2:.4f} (iid assumption)")

    rungs = ["prior", "prior+graph", "prior+judge", "prior+graph+judge"]
    HS = {"prior+graph": H_ALL[:1], "prior+judge": H_ALL[1:], "prior+graph+judge": H_ALL}
    acc = {r: {"nll": [], "m2": []} for r in rungs}
    route_rows = []                                          # (gain, hop, amb, conflict) per held row, pooled
    iv_by_hop = {}
    used = 0
    for seed in range(seeds):
        tr, he = descendant_disjoint_split(pk, seed, held_frac=held_frac)
        if len(tr) < 30 or len(he) < 12:
            continue
        used += 1
        m = affine_calibrate(d[tr], t1[tr, 0], d)
        meas = np.column_stack([m, j2v[:, 0], j2v[:, 1]])
        E = np.column_stack([y - prior, meas - y[:, [0, 0, 1]] * [1, 1, 1]])[tr]   # prior errs (2) + meas errs (3)
        C5 = fit_residual_covariance(E, shrinkage=shrink)
        P0, C_pm, R0 = C5[:2, :2], C5[:2, 2:], C5[2:, 2:]
        # routing IV per hop: expected tr(posterior var) reduction from adding the judge (C=0 approximation)
        ph = fit_sigma_of_hop(y[tr, 0] - prior[tr, 0], y[tr, 1] - prior[tr, 1], hopk[tr])
        for h in sorted(set(hopk.tolist())):
            sD, sS, rho = sigma_of_hop(ph, h)
            Ph = np.array([[sD**2, rho*sD*sS], [rho*sD*sS, sS**2]])
            def post_tr(Hm, R):
                Sx = Hm @ Ph @ Hm.T + R
                K = Ph @ Hm.T @ np.linalg.inv(Sx)
                return np.trace(Ph - K @ Sx @ K.T)
            iv_by_hop[h] = post_tr(H_ALL[:1], R0[:1, :1]) - post_tr(H_ALL, R0)
        for i in he:
            x = prior[i]
            nll, m2 = nll_mahal(y[i] - x, P0)
            acc["prior"]["nll"].append(nll); acc["prior"]["m2"].append(m2)
            row_nll = {}
            for rung in rungs[1:]:
                Hm = HS[rung]
                sel = {"prior+graph": [0], "prior+judge": [1, 2], "prior+graph+judge": [0, 1, 2]}[rung]
                xp, Pp = correlated_update_H(x, P0, meas[i][sel], R0[np.ix_(sel, sel)], C_pm[:, sel], Hm)
                nll, m2 = nll_mahal(y[i] - xp, Pp)
                acc[rung]["nll"].append(nll); acc[rung]["m2"].append(m2)
                row_nll[rung] = nll
            # routing features (all pre-judge-call observables) + realized gain + DECISION flip
            xp_pg, _ = correlated_update_H(x, P0, meas[i][[0]], R0[:1, :1], C_pm[:, [0]], H_ALL[:1])
            xp_pgj, _ = correlated_update_H(x, P0, meas[i], R0, C_pm, H_ALL)
            flip = int((xp_pg[0] > 0.5) != (xp_pgj[0] > 0.5))                 # judge flips the filing decision
            good = int(flip and ((xp_pgj[0] > 0.5) == (y[i][0] > 0.5)))       # ...toward judge1 truth
            route_rows.append((row_nll["prior+graph"] - row_nll["prior+graph+judge"],
                               float(hopk[i]), 0.5 - abs(xp_pg[0] - 0.5), abs(m[i] - x[0]), flip, good))

    print(f"\nfusion ladder ({used} splits, constant blocks, correlated; NLL ↓, Mahal/dim ≈1):")
    print(f"    {'rung':20s} {'NLL':>8s} {'Mahal/dim':>10s}")
    for r in rungs:
        nll = np.array(acc[r]["nll"]); m2 = np.array(acc[r]["m2"])
        print(f"    {r:20s} {nll.mean():+8.4f} {m2.mean()/2:10.2f}")
    g = np.array(acc["prior+graph"]["nll"]) - np.array(acc["prior+graph+judge"]["nll"])
    print(f"    judge-channel value (prior+graph → +judge): {g.mean():+.4f} "
          f"(row-SE {g.std()/np.sqrt(len(g)):.4f} — stability only)")

    rr = np.array(route_rows)                                # gain, hop, amb, conflict
    rng = np.random.default_rng(0)
    pol = {"random": rng.permutation(len(rr)),
           "hop-IV": np.argsort(-np.array([iv_by_hop[h] for h in rr[:, 1]])),
           "ambiguity": np.argsort(-rr[:, 2]),
           "conflict": np.argsort(-rr[:, 3]),
           "oracle": np.argsort(-rr[:, 0])}
    print(f"\nROUTING — cumulative judge-channel NLL gain captured vs fraction of judge calls made:")
    fr = [0.10, 0.25, 0.50, 0.75, 1.00]
    print(f"    {'policy':10s} " + " ".join(f"{f:>7.0%}" for f in fr))
    total = rr[:, 0].sum()
    for pn, order in pol.items():
        caps = [rr[order[:int(f * len(rr))], 0].sum() / total for f in fr]
        print(f"    {pn:10s} " + " ".join(f"{c:>7.0%}" for c in caps))
    print(f"    (100% = the full judge-on-every-row gain of {g.mean():+.4f} NLL/row; a good policy captures")
    print(f"     most of it with few calls — the deployment question 'when is a judge call worth it')")

    # DECISION-utility routing: NLL is the wrong deployment objective — filing is a decision. A judge call
    # matters when it FLIPS the decision (posterior D crossing 0.5), and flips concentrate near the threshold.
    flips = rr[:, 4]; goods = rr[:, 5]
    print(f"\nDECISION routing — flips: {int(flips.sum())}/{len(rr)} rows ({flips.mean():.1%}), "
          f"of which toward judge1 truth: {int(goods.sum())} ({goods.sum()/max(flips.sum(),1):.0%})")
    print(f"    {'policy':10s} " + " ".join(f"{f:>7.0%}" for f in fr) + "   (fraction of FLIPS captured)")
    for pn, order in pol.items():
        caps = [flips[order[:int(f * len(rr))]].sum() / max(flips.sum(), 1) for f in fr]
        print(f"    {pn:10s} " + " ".join(f"{c:>7.0%}" for c in caps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fresh")
    ap.add_argument("--j2-scored", default="/tmp/mu_data/sigma_hop_fresh_scored_j2.tsv")
    ap.add_argument("--seeds", type=int, default=40)
    a = ap.parse_args()
    run(a.dataset, a.j2_scored, seeds=a.seeds)


if __name__ == "__main__":
    main()
