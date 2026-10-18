#!/usr/bin/env python3
"""SHADOW: PHASE 2 — cross-corpus regularization, swept against the LOCO TRANSFER NUMBER
(ruling 2: the probe is demoted to a diagnostic; the objective is the failing
simplewiki->Pearltrees cell, baseline 0.150 vs in-domain 0.274).

Method per the owner's ruling: BLENDED INPUTS — convex combinations of corpus-A and
corpus-B channel features with correspondingly aligned targets (manifold-respecting: a
convex combination of two real feature rows stays in the valid hull and perturbs along the
corpus-difference direction specifically). Slate-level alignment keeps the target exact:
the foreign slate is permuted so its true candidate sits at the SAME column as the home
slate's, so the mixed slate's label is unambiguous.
  2a sweep    mixing level m in {0, 0.1, 0.25, 0.5, 0.75}; m=0 reproduces phase-1 baseline
  2b dropout  feature-wise channel dropout as the comparison arm (corpus identity may be
              carried by few features, so it can be more targeted than blending)
Reported per level: LOCO transfer MRR (5 seeds + bootstrap CI vs m=0) and, as an
OBSERVABLE not an objective, the corpus probe on the trained gate's channel-weight outputs
(can you tell the corpus from how the gate BEHAVES?). Probe moving the wrong way while LOCO
improves is itself informative and is reported as such."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from eval_filing import metrics
from sa_phase1_shadow import build, ranks_from

SEEDS = (7, 17, 27, 37, 47)
LEVELS = (0.0, 0.1, 0.25, 0.5, 0.75)
DROPS = (0.0, 0.3, 0.6)
STEPS = 400

def chan(d, ix):
    return torch.stack([d["Sz"][ix], d["Az"][ix], d["Ez"][ix]], -1)

def aligned_foreign(dh, df, ix_h, g, foreign_train):
    """For each home row, a foreign row of the SAME width whose true column matches the
    home true column — so the convex combination has an unambiguous label.
    LEAKAGE GUARD: foreign rows are drawn ONLY from the foreign corpus's TRAIN split;
    the first version sampled all 500 foreign queries, ~40% of which are the eval set."""
    Fh = dh["Cz"].shape[1]
    picks = foreign_train[g.integers(0, len(foreign_train), len(ix_h))]
    out_c, out_s, out_a, out_e = [], [], [], []
    for k, qi in enumerate(ix_h.tolist()):
        j = int(picks[k])
        tf, th = int(df["truepos"][j]), int(dh["truepos"][qi])
        cols = g.permutation(df["Cz"].shape[1])
        cols = cols[cols != tf][:Fh - 1]
        if len(cols) < Fh - 1:                       # foreign corpus narrower: wrap around
            cols = np.resize(cols, Fh - 1)
        full = np.insert(cols, th, tf)[:Fh]
        t = torch.tensor(full)
        out_c.append(df["Cz"][j][t]); out_s.append(df["Sz"][j][t])
        out_a.append(df["Az"][j][t]); out_e.append(df["Ez"][j][t])
    return (torch.stack(out_c), torch.stack(out_s), torch.stack(out_a), torch.stack(out_e))

def train_gate(dh, df, ix_h, m, drop, seed, foreign_train):
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    mlp = torch.nn.Sequential(torch.nn.Linear(3, 12), torch.nn.Tanh(), torch.nn.Linear(12, 3))
    we = torch.nn.Parameter(torch.tensor(0.2))
    opt = torch.optim.Adam(list(mlp.parameters()) + [we], lr=0.05)
    C, S, A, E = dh["Cz"][ix_h], dh["Sz"][ix_h], dh["Az"][ix_h], dh["Ez"][ix_h]
    tgt = dh["truepos"][ix_h]
    if m > 0:
        fC, fS, fA, fE = aligned_foreign(dh, df, ix_h, g, foreign_train)
        C, S, A, E = ((1-m)*C + m*fC, (1-m)*S + m*fS, (1-m)*A + m*fA, (1-m)*E + m*fE)
    X = torch.stack([S, A, E], -1)
    for step in range(STEPS):
        Xs = X
        if drop > 0:                                  # feature-wise channel dropout
            mask = (torch.rand(1, 1, 3) > drop).float() / max(1e-6, (1 - drop))
            Xs = X * mask
        w = torch.softmax(mlp(Xs), -1)
        sc = we * C + (w * Xs).sum(-1)
        loss = torch.nn.functional.cross_entropy(sc * 8.0, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return mlp, we

def apply_gate(mlp, we, d, ix):
    X = chan(d, ix)
    with torch.no_grad():
        w = torch.softmax(mlp(X), -1)
        return we * d["Cz"][ix] + (w * X).sum(-1), w

def behaviour_probe(wa, wb, seeds=(0, 1)):
    """Diagnostic: can the corpus be told from HOW THE GATE BEHAVES (its channel weights)?"""
    Xa, Xb = wa.reshape(-1, 3), wb.reshape(-1, 3)
    n = min(len(Xa), len(Xb), 20000)
    accs = []
    for s in seeds:
        X = torch.cat([Xa[:n], Xb[:n]])
        y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
        gg = torch.Generator().manual_seed(s)
        pm = torch.randperm(len(X), generator=gg)
        X, y = X[pm], y[pm]
        ntr = int(0.7 * len(X))
        torch.manual_seed(s)
        clf = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
        opt = torch.optim.Adam(clf.parameters(), lr=0.01)
        for _ in range(200):
            loss = torch.nn.functional.cross_entropy(clf(X[:ntr]), y[:ntr])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            accs.append(float((clf(X[ntr:]).argmax(1) == y[ntr:]).float().mean()))
    return round(float(np.mean(accs)), 4)

def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cw, cp = build("simplewiki", dev), build("pearltrees", dev)
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing",
           "objective": "LOCO simplewiki->pearltrees MRR (phase-1 baseline 0.150; "
                        "in-domain reference 0.274)",
           "leakage_guard": "foreign blend rows drawn ONLY from the foreign TRAIN split; first run sampled all foreign queries incl. eval (~40%) — that run is void",
           "probe_note": "OBSERVABLE, not optimized; probe floor ~0.543 (same-corpus null)"}
    base_by_seed = None
    sweep = {}
    for m in LEVELS:
        per_seed, ranks_all, probes = [], [], []
        for s in SEEDS:
            o = np.random.default_rng(s).permutation(len(cw["queries"]))
            od = np.random.default_rng(s).permutation(len(cp["queries"]))
            hed = torch.tensor(od[300:])
            mlp, we = train_gate(cw, cp, torch.tensor(o[:300]), m, 0.0, s,
                                 foreign_train=od[:300])
            sc, wts = apply_gate(mlp, we, cp, hed)
            tp = [int(cp["truepos"][i]) for i in hed.tolist()]
            r = ranks_from(sc, tp)
            per_seed.append(round(metrics(r)["MRR"], 4)); ranks_all.append(r)
            hw = torch.tensor(np.random.default_rng(s).permutation(len(cw["queries"]))[300:])
            _, wts_w = apply_gate(mlp, we, cw, hw)
            probes.append(behaviour_probe(wts, wts_w))
        sweep[str(m)] = {"loco_mrr_mean": round(float(np.mean(per_seed)), 4),
                         "loco_mrr_sd": round(float(np.std(per_seed)), 4),
                         "per_seed": per_seed,
                         "gate_behaviour_probe": round(float(np.mean(probes)), 4)}
        if m == 0.0:
            base_by_seed = ranks_all
        else:
            boot = []
            for si, (rb, rm) in enumerate(zip(base_by_seed, ranks_all)):
                g = np.random.default_rng(100 + si)
                for _ in range(300):
                    ix = g.integers(0, len(rm), len(rm))
                    boot.append(float(np.mean([1/rm[i] for i in ix]) -
                                      np.mean([1/rb[i] for i in ix])))
            sweep[str(m)]["delta_vs_m0_CI95"] = [round(float(np.percentile(boot, 2.5)), 4),
                                                 round(float(np.percentile(boot, 97.5)), 4)]
        print(f"[2a m={m}] " + json.dumps(sweep[str(m)]), flush=True)
    res["P2a_blend_sweep"] = sweep
    drop_res = {}
    for dr in DROPS:
        per_seed = []
        for s in SEEDS:
            o = np.random.default_rng(s).permutation(len(cw["queries"]))
            od = np.random.default_rng(s).permutation(len(cp["queries"]))
            hed = torch.tensor(od[300:])
            mlp, we = train_gate(cw, cp, torch.tensor(o[:300]), 0.0, dr, s,
                                 foreign_train=od[:300])
            sc, _ = apply_gate(mlp, we, cp, hed)
            per_seed.append(round(metrics(ranks_from(
                sc, [int(cp["truepos"][i]) for i in hed.tolist()]))["MRR"], 4))
        drop_res[str(dr)] = {"loco_mrr_mean": round(float(np.mean(per_seed)), 4),
                             "per_seed": per_seed}
        print(f"[2b drop={dr}] " + json.dumps(drop_res[str(dr)]), flush=True)
    res["P2b_channel_dropout"] = drop_res
    json.dump(res, open("PHASE2_MIXING_SWEEP.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)

if __name__ == "__main__":
    run()