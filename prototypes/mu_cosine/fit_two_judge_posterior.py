#!/usr/bin/env python3
"""First build of the two-judge operator posterior (DESIGN_two_judge_posterior.md, k=2 GLM).

Tests the central claim of the combiner ladder: does the SECOND-ORDER cross pseudo-judge `μ_D·μ_S` (the
directional↔symmetric correlation) improve the joint `P(D,S | features)` over the first-order / product-of-marginals
model? D = directional operator, S = symmetric operator.

- LABELS (fuzzy, so D and S can CO-OCCUR): D = max LLM mu over {subcategory, subtopic, element_of, super_category};
  S = max LLM mu over {see_also, assoc}; binarised at 0.5. (LLM = gpt-5.5-low, wiki_rel_scored.tsv.)
- FEATURES (the judges): model readouts μ_D = max_dir μ_HIER(x|y),μ_HIER(y|x); μ_S = μ_SYM(x,y); graph d = up-walk
  hit-prob. Pseudo-judges: μ_D², μ_S² (self / confidence rung), μ_D·μ_S (cross / correlation rung).
- METRIC: held-out (node-disjoint) log-loss of the JOINT (D,S) ∈ {00,01,10,11}. Ladder:
  product-of-marginals < main-effects joint < +self < +cross. If +cross wins, the correlation term earns its keep.

  python3 fit_two_judge_posterior.py --scored /tmp/mu_data/wiki_rel_scored.tsv --e5-cache /tmp/mu_data/wiki_rel_e5.pt \
      --graph ../../data/benchmark/100k_cats/category_parent.tsv
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import OPS, Tokenizer, load_dag
from eval_relatedness import build_model
from emit_transitive_hops import hit_prob
from sklearn.linear_model import LogisticRegression

DIR = ["subcategory", "subtopic", "element_of", "super_category"]
SYM = ["see_also", "assoc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--e5-cache", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--model", default="model_prod.pt")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=20, help="number of node-disjoint splits to average")
    ap.add_argument("--C", type=float, default=1.0, help="logistic L2 inverse-strength")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    hdr = open(a.scored, encoding="utf-8").readline().lstrip("#").strip().split("\t")
    ci = {c: i for i, c in enumerate(hdr)}
    pairs, D_llm, S_llm = [], [], []
    for ln in open(a.scored, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        c = ln.rstrip("\n").split("\t")
        if len(c) <= ci["mu[assoc]"]:
            continue
        pairs.append((c[0], c[1]))
        D_llm.append(max(float(c[ci[f"mu[{r}]"]]) for r in DIR))
        S_llm.append(max(float(c[ci[f"mu[{r}]"]]) for r in SYM))
    D_llm, S_llm = np.array(D_llm), np.array(S_llm)

    # model readouts μ_D (directional), μ_S (symmetric); graph d (up-walk hit-prob)
    d = torch.load(a.e5_cache, weights_only=False); idx = {n: i for i, n in enumerate(d["names"])}
    keep = [i for i, (x, y) in enumerate(pairs) if x in idx and y in idx]
    pairs = [pairs[i] for i in keep]; D_llm, S_llm = D_llm[keep], S_llm[keep]
    parents, _, deg = load_dag(a.graph)
    tok = Tokenizer(d["query"], d["passage"], idx, parents, deg)
    m = build_model(a.model, dev)

    def mu(op, ps):
        it = [(x, y, OPS[op]) for x, y in ps]; out = []
        for i in range(0, len(it), 512):
            b = tok.build(it[i:i + 512], train=False)
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.no_grad():
                out += m(**b).cpu().tolist()
        return np.array(out)

    muD = np.maximum(mu("HIER", pairs), mu("HIER", [(y, x) for x, y in pairs]))     # directional (either way)
    muS = mu("SYM", pairs)                                                          # symmetric
    gd = np.array([hit_prob(parents, x, y) for x, y in pairs])                      # graph judge

    Dl = (D_llm > a.thresh).astype(int); Sl = (S_llm > a.thresh).astype(int)
    joint = Dl * 2 + Sl                                                             # 0..3 = {00,01,10,11}
    print(f"n={len(pairs)}  D+ {Dl.mean()*100:.0f}%  S+ {Sl.mean()*100:.0f}%  BOTH(1,1) {((Dl&Sl).mean())*100:.0f}%  "
          f"corr(D,S label) {np.corrcoef(Dl,Sl)[0,1]:+.2f}")

    base = np.column_stack([muD, muS, gd])
    rungs = {
        "product-of-marginals": None,
        "joint linear (μ_D,μ_S,d)": base,
        "joint +self (μ_D²,μ_S²)": np.column_stack([base, muD**2, muS**2]),
        "joint +CROSS (μ_D·μ_S)": np.column_stack([base, muD**2, muS**2, muD*muS]),
    }

    def logloss(P, y):
        return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))))

    nodes = sorted({n for p in pairs for n in p})
    scores = {k: [] for k in rungs}
    for s in range(a.seeds):                               # multi-seed node-disjoint splits (n=57/split is noisy)
        rng = np.random.default_rng(a.seed + s); nd = list(nodes); rng.shuffle(nd)
        hn = set(nd[:int(0.25 * len(nd))])
        tr = np.array([i for i, (x, y) in enumerate(pairs) if x not in hn and y not in hn])
        he = np.array([i for i, (x, y) in enumerate(pairs) if x in hn and y in hn])
        if len(he) < 10:
            continue
        lD = LogisticRegression(max_iter=2000).fit(base[tr], Dl[tr])
        lS = LogisticRegression(max_iter=2000).fit(base[tr], Sl[tr])
        pD = lD.predict_proba(base[he])[:, 1]; pS = lS.predict_proba(base[he])[:, 1]
        scores["product-of-marginals"].append(
            logloss(np.column_stack([(1-pD)*(1-pS), (1-pD)*pS, pD*(1-pS), pD*pS]), joint[he]))
        for name, X in rungs.items():
            if X is None:
                continue
            lj = LogisticRegression(max_iter=3000, C=a.C).fit(X[tr], joint[tr])
            Pj = np.zeros((len(he), 4)); Pj[:, lj.classes_] = lj.predict_proba(X[he])
            scores[name].append(logloss(Pj, joint[he]))
    print(f"held-out log-loss (mean ± std over {len(scores['joint linear (μ_D,μ_S,d)'])} node-disjoint splits):\n")
    print(f"{'model':28s}  log-loss")
    for k in rungs:
        v = np.array(scores[k]); print(f"{k:28s}  {v.mean():.4f} ± {v.std():.4f}")


if __name__ == "__main__":
    main()
