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

DIR = ["subcategory", "subtopic", "element_of", "super_category"]; SYM = ["see_also", "assoc"]


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

    def resid(tr, he):                                          # fit marginal means on tr, residuals on he
        out = []
        for y in (Dl, Sl):
            beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            out.append((y[tr] - X[tr] @ beta, y[he] - X[he] @ beta))
        return out
    S = {"(a) independent ρ=0": [], "(b) constant ρ": [], "(c) ρ(hop) — heteroscedastic": []}
    for s in range(a.seeds):
        rng = np.random.default_rng(s); p = rng.permutation(len(pairs)); cut = int(0.7*len(pairs))
        tr, he = p[:cut], p[cut:]
        (rD_tr, rD_he), (rS_tr, rS_he) = resid(tr, he)
        sD, sS = rD_tr.std() + 1e-6, rS_tr.std() + 1e-6
        rho = np.corrcoef(rD_tr, rS_tr)[0, 1]
        S["(a) independent ρ=0"].append(biv_nll(rD_he, rS_he, sD, sS, 0.0).mean())
        S["(b) constant ρ"].append(biv_nll(rD_he, rS_he, sD, sS, rho).mean())
        rho_h = {}                                              # per-hop ρ on train (fallback to global)
        for h in set(hop):
            mth = hop[tr] == h
            rho_h[h] = np.corrcoef(rD_tr[mth], rS_tr[mth])[0, 1] if mth.sum() > 5 and rD_tr[mth].std() > 0 else rho
        rho_he = np.array([rho_h.get(h, rho) for h in hop[he]])
        S["(c) ρ(hop) — heteroscedastic"].append(biv_nll(rD_he, rS_he, sD, sS, rho_he).mean())
    print(f"held-out joint NLL (mean±std over {a.seeds} splits; LOWER better):\n")
    for k, v in S.items():
        v = np.array(v); print(f"  {k:32s}  {v.mean():.4f} ± {v.std():.4f}")
    b, c = np.array(S["(b) constant ρ"]), np.array(S["(c) ρ(hop) — heteroscedastic"])
    print(f"\n  Δ (constant − ρ(hop)) = {(b-c).mean():+.4f} ± {(b-c).std():.4f}  "
          f"({'ρ(hop) HELPS' if (b-c).mean()>0 else 'no gain'})")


if __name__ == "__main__":
    main()
