#!/usr/bin/env python3
"""Does the HOP-CONDITIONAL D↔S correlation earn its keep? (DESIGN_two_judge_posterior.md, QDA rung.)

Continuous test on the fuzzy μ (not binarised — binarisation hid the signal). Predict the joint (D,S) LLM
memberships from features [μ_D, μ_S, d], take residuals, and compare the held-out bivariate-Gaussian NLL of the
residuals under three correlation models:
  (a) INDEPENDENT   ρ=0
  (b) CONSTANT ρ    one ρ estimated on train
  (c) ρ(hop)        ρ estimated per hop on train  ← the heteroscedastic / QDA model
If (c) < (b) < (a), the hop-conditional correlation (the cross pseudo-judge coupled to d) improves prediction —
i.e. the D↔S covariance genuinely varies across the space and modelling it helps.

  python3 fit_hetero.py --score-in /tmp/mu_data/multihop_score_in.tsv --responses /tmp/mu_data/multihop_resp.txt \
      --e5-cache /tmp/mu_data/multihop_e5.pt --graph ../../data/benchmark/100k_cats/category_parent.tsv
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import OPS, Tokenizer, load_dag
from eval_relatedness import build_model
from emit_transitive_hops import hit_prob
from emit_direction_blend import parse_responses
from scipy.optimize import minimize

DIR = ["subcategory", "subtopic", "element_of", "super_category"]; SYM = ["see_also", "assoc"]


def sig_of_d(params, d):
    """smooth PARAMETRIC covariance as a function of the conditioning feature d (here = hop): log-linear σ, tanh ρ.
    This is the predictive 'Σ(hop) into the model' — Σ is a learned function of d, not per-hop oracle bins."""
    aD, bD, aS, bS, c, e = params
    return np.exp(aD + bD * d), np.exp(aS + bS * d), np.tanh(c + e * d)


def fit_sig_of_d(rD, rS, d):
    def nll(p):
        sD, sS, rho = sig_of_d(p, d)
        return biv_nll(rD, rS, sD, sS, rho).mean()
    p0 = [np.log(rD.std() + 1e-6), 0.0, np.log(rS.std() + 1e-6), 0.0, np.arctanh(np.clip(np.corrcoef(rD, rS)[0, 1], -0.9, 0.9)), 0.0]
    return minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-7}).x


def biv_nll(rD, rS, sD, sS, rho):
    rho = np.clip(rho, -0.98, 0.98)
    zD, zS = rD / sD, rS / sS
    q = (zD**2 - 2*rho*zD*zS + zS**2) / (1 - rho**2)
    return np.log(2*np.pi) + np.log(sD*sS) + 0.5*np.log(1 - rho**2) + 0.5*q      # per-point NLL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-in", required=True); ap.add_argument("--responses", required=True)
    ap.add_argument("--e5-cache", required=True); ap.add_argument("--graph", required=True)
    ap.add_argument("--model", default="model_prod.pt"); ap.add_argument("--prefix", default="transitive_h")
    ap.add_argument("--seeds", type=int, default=40); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device)

    def gmu(o, rel): return float((o.get(rel, {}) or {}).get("mu_fwd", 0)) if rel in DIR else float((o.get(rel, {}) or {}).get("mu", 0))
    rows = [ln.rstrip("\n").split("\t") for ln in open(a.score_in, encoding="utf-8") if not ln.startswith("#")]
    byid = parse_responses(a.responses)
    pairs, hop, Dl, Sl = [], [], [], []
    for i, r in enumerate(rows):
        if i not in byid or not r[4].startswith(a.prefix): continue
        pairs.append((r[0], r[1])); hop.append(int(r[4][len(a.prefix):]))
        Dl.append(max(gmu(byid[i], x) for x in DIR)); Sl.append(max(gmu(byid[i], x) for x in SYM))
    hop, Dl, Sl = np.array(hop), np.array(Dl), np.array(Sl)

    d = torch.load(a.e5_cache, weights_only=False); idx = {n: i for i, n in enumerate(d["names"])}
    keep = [i for i, (x, y) in enumerate(pairs) if x in idx and y in idx]
    pairs = [pairs[i] for i in keep]; hop, Dl, Sl = hop[keep], Dl[keep], Sl[keep]
    parents, _, deg = load_dag(a.graph); tok = Tokenizer(d["query"], d["passage"], idx, parents, deg)
    m = build_model(a.model, dev)
    def mu(op, ps):
        it = [(x, y, OPS[op]) for x, y in ps]; out = []
        for i in range(0, len(it), 512):
            b = tok.build(it[i:i+512], train=False); b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.no_grad(): out += m(**b).cpu().tolist()
        return np.array(out)
    muD = np.maximum(mu("HIER", pairs), mu("HIER", [(y, x) for x, y in pairs])); muS = mu("SYM", pairs)
    gd = np.array([hit_prob(parents, x, y) for x, y in pairs])
    X = np.column_stack([muD, muS, gd, np.ones(len(pairs))])
    print(f"n={len(pairs)}  hops {sorted(set(hop))}\n")
    # CONFIDENCE vs hop (user): is the direction well-separated at low hop and ambiguous at high hop?
    print(f"{'h':>2} {'μ_D':>5} {'μ_S':>5} {'margin μ_D−μ_S':>14} {'σ(D)':>5} {'σ(S)':>5}   (confidence = margin↑, σ↓)")
    for h in sorted(set(hop)):
        mth = hop == h
        print(f"{h:>2} {Dl[mth].mean():>5.2f} {Sl[mth].mean():>5.2f} {(Dl[mth]-Sl[mth]).mean():>14.2f} "
              f"{Dl[mth].std():>5.2f} {Sl[mth].std():>5.2f}")
    print()

    def resid(tr, he):                                          # fit marginal means on tr, residuals on he
        out = []
        for y in (Dl, Sl):
            beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            out.append((y[tr] - X[tr] @ beta, y[he] - X[he] @ beta))
        return out
    S = {"(a) independent ρ=0": [], "(b) constant ρ": [], "(c) ρ(hop) off-diag": [],
         "(d) σ(hop) CONFIDENCE, const ρ": [], "(e) σ(hop)+ρ(hop) oracle-bin": [],
         "(f) Σ(hop) PREDICTIVE smooth": []}
    heldsizes = []
    for s in range(a.seeds):
        # NODE-DISJOINT split (review): hold out DESCENDANT nodes — the entity that repeats across hops — so a
        # descendant's h=1..5 pairs never straddle train/held (pair-level random splits leaked them).
        rng = np.random.default_rng(s)
        descs = sorted({x for x, y in pairs}); rng.shuffle(descs)
        held_d = set(descs[:max(1, int(0.3 * len(descs)))])
        tr = np.array([i for i, (x, y) in enumerate(pairs) if x not in held_d])
        he = np.array([i for i, (x, y) in enumerate(pairs) if x in held_d])
        if len(he) < 12 or len(tr) < 30:
            continue
        heldsizes.append(len(he))
        (rD_tr, rD_he), (rS_tr, rS_he) = resid(tr, he)
        sD, sS = rD_tr.std() + 1e-6, rS_tr.std() + 1e-6
        rho = np.corrcoef(rD_tr, rS_tr)[0, 1]
        def per_hop(fn, default):                              # estimate a per-hop stat on train, map to held pairs
            hh = {}
            for h in set(hop):
                mth = hop[tr] == h
                hh[h] = fn(mth) if mth.sum() > 5 else default
            return np.array([hh.get(h, default) for h in hop[he]])
        rho_he = np.clip(per_hop(lambda m: np.corrcoef(rD_tr[m], rS_tr[m])[0, 1] if rD_tr[m].std() > 0 else rho, rho), -0.98, 0.98)
        sD_he = per_hop(lambda m: rD_tr[m].std() + 1e-6, sD); sS_he = per_hop(lambda m: rS_tr[m].std() + 1e-6, sS)
        S["(a) independent ρ=0"].append(biv_nll(rD_he, rS_he, sD, sS, 0.0).mean())
        S["(b) constant ρ"].append(biv_nll(rD_he, rS_he, sD, sS, rho).mean())
        S["(c) ρ(hop) off-diag"].append(biv_nll(rD_he, rS_he, sD, sS, rho_he).mean())
        S["(d) σ(hop) CONFIDENCE, const ρ"].append(biv_nll(rD_he, rS_he, sD_he, sS_he, rho).mean())
        S["(e) σ(hop)+ρ(hop) oracle-bin"].append(biv_nll(rD_he, rS_he, sD_he, sS_he, rho_he).mean())
        # (f) the actual build: Σ as a SMOOTH parametric function of hop, fit by MLE on train — predictive, not oracle
        pf = fit_sig_of_d(rD_tr, rS_tr, hop[tr].astype(float))
        sDf, sSf, rhof = sig_of_d(pf, hop[he].astype(float))
        S["(f) Σ(hop) PREDICTIVE smooth"].append(biv_nll(rD_he, rS_he, sDf, sSf, rhof).mean())
    hs = np.array(heldsizes)
    print(f"held-out joint NLL (mean±std over {len(hs)} NODE-DISJOINT splits, ~{hs.mean():.0f} held pairs/split; ↓ better):\n")
    for k, v in S.items():
        v = np.array(v); print(f"  {k:32s}  {v.mean():.4f} ± {v.std():.4f}")
    def gain(base, alt):
        d = np.array(S[base]) - np.array(S[alt]); se = d.std()/np.sqrt(len(d))
        print(f"  gain({base.split()[0]}→{alt.split()[0]}) = {d.mean():+.4f}  (repeated-split SE {se:.4f} — STABILITY "
              "only, NOT a calibrated significance: the 40 resamples share one dataset, so SE understates variance)")
    print()
    gain("(b) constant ρ", "(f) Σ(hop) PREDICTIVE smooth")
    gain("(e) σ(hop)+ρ(hop) oracle-bin", "(f) Σ(hop) PREDICTIVE smooth")

    # CALIBRATED test: permutation on the AVERAGED statistic. Shuffle hop (breaking pair↔hop), recompute the mean
    # Σ(hop)-vs-constant gain over the SAME node-disjoint splits. p = fraction of hop-permutations with gain ≥ observed.
    def mean_gain(hop_arr, nsplits=20):
        gains = []
        for s in range(nsplits):
            rng = np.random.default_rng(s); descs = sorted({x for x, y in pairs}); rng.shuffle(descs)
            hd = set(descs[:max(1, int(0.3 * len(descs)))])
            trI = np.array([i for i, (x, y) in enumerate(pairs) if x not in hd])
            heI = np.array([i for i, (x, y) in enumerate(pairs) if x in hd])
            if len(heI) < 12 or len(trI) < 30:
                continue
            (rD_t, rD_h), (rS_t, rS_h) = resid(trI, heI)
            sD_, sS_ = rD_t.std() + 1e-6, rS_t.std() + 1e-6; rho_ = np.corrcoef(rD_t, rS_t)[0, 1]
            cnll = biv_nll(rD_h, rS_h, sD_, sS_, rho_).mean()
            pf = fit_sig_of_d(rD_t, rS_t, hop_arr[trI].astype(float)); sDf, sSf, rhof = sig_of_d(pf, hop_arr[heI].astype(float))
            gains.append(cnll - biv_nll(rD_h, rS_h, sDf, sSf, rhof).mean())
        return np.mean(gains)
    obs = mean_gain(hop); K = 200; prng = np.random.default_rng(1)
    null = np.array([mean_gain(hop[prng.permutation(len(pairs))]) for _ in range(K)])
    pval = (1 + np.sum(null >= obs)) / (K + 1)
    print(f"\nPERMUTATION TEST — mean Σ(hop)-vs-constant gain, hop SHUFFLED, node-disjoint, K={K} (calibrated):")
    print(f"  observed mean gain {obs:+.4f}; null mean {null.mean():+.4f}, 95%ile {np.percentile(null, 95):+.4f}; "
          f"permutation p = {pval:.3f}  {'← SIGNIFICANT' if pval < 0.05 else '(NOT significant)'}")


if __name__ == "__main__":
    main()
