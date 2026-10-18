#!/usr/bin/env python3
"""SHADOW: PHASE 1 supplement — the three checks Phase 1's own numbers demand.
S1 LOCO variance: 5 seeds + bootstrap CI on every scoreboard contrast (0d discipline —
   a 0.016 margin without a CI is not a result).
S2 ceiling threshold sensitivity: 1a(ii) found ZERO defensible alternatives at e5 cos>0.95;
   e5 cosines are compressed, so that may be a threshold artifact rather than a clean
   corpus property. Sweep 0.80/0.85/0.90/0.95.
S3 TOPIC-MATCHED corpus probe: 1a(i) shows the corpora differ hugely in topic (simplewiki
   49% geography vs Pearltrees 30% society), and 1c's probe reads 82% — but a probe can
   read TOPIC instead of REGIME. Retrain the probe on topic-matched folder subsets; the
   drop is the share of "corpus identity" that is really topic identity. This decides
   whether phase-2 invariance pressure would erase task signal (its stated failure mode)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from eval_filing import metrics
from sa_phase1_shadow import (build, gate_train, gate_apply, ranks_from, REGIONS, region_of)

SEEDS = (7, 17, 27, 37, 47)

def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cp, cw = build("pearltrees", dev), build("simplewiki", dev)
    rnames = list(REGIONS)
    for d in (cp, cw):
        d["folder_region"] = region_of([f"F:{t}" for t in d["tid_list"]], None, rnames,
                                       (d["qtbl"], d["ptbl"], d["idx"]))
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing"}
    # --- S1 LOCO with variance
    loco = {}
    for src, dst, sd, dd in (("simplewiki", "pearltrees", cw, cp),
                             ("pearltrees", "simplewiki", cp, cw)):
        per_seed, boot = [], []
        for s in SEEDS:
            o = np.random.default_rng(s).permutation(len(sd["queries"]))
            m, w_ = gate_train(sd, torch.tensor(o[:300]), seed=s)
            od = np.random.default_rng(s).permutation(len(dd["queries"]))
            hed = torch.tensor(od[300:])
            tpd = [int(dd["truepos"][i]) for i in hed.tolist()]
            r_tr = ranks_from(gate_apply(m, w_, dd, hed), tpd)
            r_e5 = ranks_from(dd["Cz"][hed], tpd)
            mi, wi = gate_train(dd, torch.tensor(od[:300]), seed=s)
            r_in = ranks_from(gate_apply(mi, wi, dd, hed), tpd)
            per_seed.append({"seed": s, "transfer": round(metrics(r_tr)["MRR"], 4),
                             "e5": round(metrics(r_e5)["MRR"], 4),
                             "in_domain": round(metrics(r_in)["MRR"], 4)})
            g = np.random.default_rng(s)
            for _ in range(300):
                ix = g.integers(0, len(r_tr), len(r_tr))
                boot.append((float(np.mean([1/r_tr[i] for i in ix]) -
                                   np.mean([1/r_e5[i] for i in ix])),
                             float(np.mean([1/r_tr[i] for i in ix]) -
                                   np.mean([1/r_in[i] for i in ix]))))
        te = [b[0] for b in boot]; ti = [b[1] for b in boot]
        loco[f"train_{src}_eval_{dst}"] = {
            "per_seed": per_seed,
            "mean": {k: round(float(np.mean([p[k] for p in per_seed])), 4)
                     for k in ("transfer", "e5", "in_domain")},
            "sd_transfer": round(float(np.std([p["transfer"] for p in per_seed])), 4),
            "transfer_minus_e5_CI95": [round(float(np.percentile(te, 2.5)), 4),
                                       round(float(np.percentile(te, 97.5)), 4)],
            "transfer_minus_indomain_CI95": [round(float(np.percentile(ti, 2.5)), 4),
                                             round(float(np.percentile(ti, 97.5)), 4)]}
        print(f"[S1 {src}->{dst}] " + json.dumps(loco[f"train_{src}_eval_{dst}"]["mean"]) +
              f" CIs e5 {loco[f'train_{src}_eval_{dst}']['transfer_minus_e5_CI95']}"
              f" in {loco[f'train_{src}_eval_{dst}']['transfer_minus_indomain_CI95']}", flush=True)
    res["S1_LOCO_with_variance"] = loco
    # --- S2 ceiling threshold sweep (Pearltrees)
    d = cp
    QV = d["qtbl"][[d["idx"][k] for k in d["q_keys"]]]
    SIM = QV @ QV.T
    col_of_title = {}
    for j, tid in enumerate(d["tid_list"]):
        pass
    for i, (b, t) in enumerate(d["queries"]):
        col_of_title.setdefault(b, d["tid_list"].index(t))
    o = np.random.default_rng(7).permutation(len(d["queries"]))
    tr, he = torch.tensor(o[:300]), torch.tensor(o[300:])
    m, w_ = gate_train(d, tr, seed=7)
    gs = gate_apply(m, w_, d, he)
    top1 = gs.argmax(1).tolist()
    tp = [int(d["truepos"][i]) for i in he.tolist()]
    sweep = {}
    for th in (0.80, 0.85, 0.90, 0.95):
        dup = tot = 0; sizes = []
        for r, (qi, t_true) in enumerate(zip(he.tolist(), tp)):
            qt = d["queries"][qi][0]
            T = {t_true}
            for nj in (SIM[qi] > th).nonzero().flatten().tolist():
                bt = d["queries"][nj][0]
                if bt != qt and bt in col_of_title:
                    T.add(col_of_title[bt])
            sizes.append(len(T))
            if top1[r] != t_true:
                tot += 1
                dup += int(top1[r] in T)
        sweep[str(th)] = {"mean_T": round(float(np.mean(sizes)), 3), "n_err": tot,
                          "err_into_alt": dup, "share": round(dup / max(tot, 1), 3)}
    res["S2_ceiling_threshold_sweep"] = sweep
    print("[S2] " + json.dumps(sweep), flush=True)
    # --- S3 topic-matched corpus probe
    def probe(mask_p, mask_w, tag):
        def feats(dd, cols):
            o2 = np.random.default_rng(7).permutation(len(dd["queries"]))
            ix = torch.tensor(o2[300:])
            sub = torch.tensor(cols)
            return torch.stack([dd["Cz"][ix][:, sub].flatten(), dd["Sz"][ix][:, sub].flatten(),
                                dd["Az"][ix][:, sub].flatten(), dd["Ez"][ix][:, sub].flatten()], -1)
        Xp, Xw = feats(cp, mask_p)[:20000], feats(cw, mask_w)[:20000]
        n = min(len(Xp), len(Xw))
        X = torch.cat([Xp[:n], Xw[:n]])
        y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
        pm = torch.randperm(len(X)); X, y = X[pm], y[pm]
        ntr = int(0.7 * len(X))
        torch.manual_seed(0)
        clf = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
        opt = torch.optim.Adam(clf.parameters(), lr=0.01)
        for _ in range(300):
            loss = torch.nn.functional.cross_entropy(clf(X[:ntr]), y[:ntr])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = float((clf(X[ntr:]).argmax(1) == y[ntr:]).float().mean())
        return {"acc": round(acc, 4), "n_pairs": len(X), "tag": tag}
    all_p = list(range(len(cp["tid_list"]))); all_w = list(range(len(cw["tid_list"])))
    res["S3_probe_all_folders"] = probe(all_p, all_w, "all")
    for regset, tag in ((("society", "everyday"), "society+everyday"),
                        (("society",), "society_only")):
        mp = [j for j, r in enumerate(cp["folder_region"]) if r in regset]
        mw = [j for j, r in enumerate(cw["folder_region"]) if r in regset]
        if len(mp) > 5 and len(mw) > 5:
            res[f"S3_probe_topic_matched_{tag}"] = probe(mp, mw, tag) | {
                "n_folders": [len(mp), len(mw)]}
    print("[S3] " + json.dumps({k: v for k, v in res.items() if k.startswith("S3")}), flush=True)
    json.dump(res, open("PHASE1B_SUPPLEMENT.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)

run()
