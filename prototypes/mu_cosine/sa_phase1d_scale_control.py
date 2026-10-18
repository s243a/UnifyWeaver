#!/usr/bin/env python3
"""SHADOW: A1 FOLLOW-UP — SCALE CONTROL ON THE IDENTITY PROBE (owner challenged the
'corpus difference is topical' framing; the challenge is well-founded).

Confound: probe features are per-row MIN-MAX NORMALIZED across the candidate axis, and the
corpora have very different candidate counts (math 2667 / stats 481 / cyber 208 / simplewiki
2002 / pearltrees 132). Normalized-score distributions differ by pure ORDER STATISTICS as
the candidate count changes, so a probe can separate corpora with zero topical content.
Control: subsample every corpus to a COMMON candidate count K, re-normalize, re-probe.
(Valid on cached normalized matrices: per-row min-max is affine-increasing, and subset +
re-min-max of an affine image equals subset + min-max of the raw scores.)
Arms:
  same_corpus_null  two disjoint K-folder samples from the SAME corpus — must sit at chance,
                    or the probe methodology itself is broken.
  within_enwiki     math/stats/cyber pairwise at common K — the topic question, scale-free.
  cross_corpus      pearltrees vs simplewiki at common K — is the 0.827 also scale?
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch

CACHE = os.path.expanduser("~/mu_data/sa_scores_v2_%s.pt")
CORPORA = ("enwiki_math", "enwiki_stats", "enwiki_cyber", "pearltrees", "simplewiki")
SEED, CAP = 7, 20000

def nzrow(M):
    lo = M.min(dim=1, keepdim=True).values
    hi = M.max(dim=1, keepdim=True).values
    return (M - lo) / (hi - lo + 1e-9)

def feats_at_K(d, K, seed, exclude=None):
    """Subsample K candidate columns, re-normalize, flatten to per-pair features."""
    g = np.random.default_rng(seed)
    n_f = len(d["tid_list"])
    pool = [j for j in range(n_f) if exclude is None or j not in exclude]
    cols = torch.tensor(sorted(g.choice(pool, size=min(K, len(pool)), replace=False)))
    o = np.random.default_rng(SEED).permutation(len(d["queries"]))
    rows = torch.tensor(o[300:])
    F = []
    for key in ("Cz", "Sz", "Az", "Ez"):
        F.append(nzrow(d[key][rows][:, cols]).flatten())
    return torch.stack(F, -1)[:CAP], set(cols.tolist())

def probe(Xa, Xb, seeds=(0, 1, 2)):
    n = min(len(Xa), len(Xb))
    accs = []
    for s in seeds:
        X = torch.cat([Xa[:n], Xb[:n]])
        y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
        g = torch.Generator().manual_seed(s)
        pm = torch.randperm(len(X), generator=g)
        X, y = X[pm], y[pm]
        ntr = int(0.7 * len(X))
        torch.manual_seed(s)
        clf = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(),
                                  torch.nn.Linear(16, 2))
        opt = torch.optim.Adam(clf.parameters(), lr=0.01)
        for _ in range(300):
            loss = torch.nn.functional.cross_entropy(clf(X[:ntr]), y[:ntr])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            accs.append(float((clf(X[ntr:]).argmax(1) == y[ntr:]).float().mean()))
    return {"acc_mean": round(float(np.mean(accs)), 4), "acc_sd": round(float(np.std(accs)), 4)}

def run():
    D = {c: torch.load(CACHE % c, weights_only=False) for c in CORPORA
         if os.path.exists(CACHE % c)}
    K = min(len(d["tid_list"]) for d in D.values())          # 132 (pearltrees)
    Kw = min(len(D[c]["tid_list"]) for c in D if c.startswith("enwiki"))   # 208
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing",
           "question": "does the identity probe read TOPIC, or CANDIDATE-SET SCALE?",
           "candidate_counts": {c: len(d["tid_list"]) for c, d in D.items()},
           "K_all": K, "K_enwiki": Kw}
    # null control: two disjoint samples from the SAME corpus at the same K
    null = {}
    for c in ("enwiki_math", "simplewiki"):
        if c in D:
            Xa, used = feats_at_K(D[c], Kw, seed=101)
            Xb, _ = feats_at_K(D[c], Kw, seed=202, exclude=used)
            null[c] = probe(Xa, Xb)
            print(f"[null {c}] " + json.dumps(null[c]), flush=True)
    res["same_corpus_null"] = null
    # within-enwiki at common K (scale-free topic question)
    within = {}
    for a, b in (("enwiki_math", "enwiki_stats"), ("enwiki_math", "enwiki_cyber"),
                 ("enwiki_stats", "enwiki_cyber")):
        if a in D and b in D:
            Xa, _ = feats_at_K(D[a], Kw, seed=11)
            Xb, _ = feats_at_K(D[b], Kw, seed=11)
            within[f"{a}_vs_{b}"] = probe(Xa, Xb)
            print(f"[within {a} vs {b}] " + json.dumps(within[f"{a}_vs_{b}"]), flush=True)
    res["within_enwiki_common_K"] = within
    # cross-corpus at common K
    if "pearltrees" in D and "simplewiki" in D:
        Xa, _ = feats_at_K(D["pearltrees"], K, seed=11)
        Xb, _ = feats_at_K(D["simplewiki"], K, seed=11)
        res["cross_corpus_common_K"] = probe(Xa, Xb)
        print("[cross common-K] " + json.dumps(res["cross_corpus_common_K"]), flush=True)
    res["uncontrolled_reference"] = {"within_enwiki": [0.6621, 0.6964, 0.7031],
                                     "cross_corpus": 0.8266}
    json.dump(res, open("PHASE1D_SCALE_CONTROL.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)

run()
