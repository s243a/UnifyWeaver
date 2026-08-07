#!/usr/bin/env python3
"""SHADOW: ITEMS 1+2 — behavioral ceiling (retracting the e5-geometry proxy) and the gate
architecture arms.

ITEM 1 — BEHAVIORAL CEILING. The old 1a(ii) called a folder a 'defensible alternative' if
it held a bookmark whose TITLE was e5-similar (>0.95) to the query. That is evidence some
OTHER item with a similar title lives there, not that THIS item belonged there — a validity
error, and one that preferentially excuses e5-COHERENT mistakes, inflating scores toward the
very baseline under comparison. Replaced by the owner's own behavior: T(item) = the set of
trees containing the SAME item, keyed by canonical URL. (Pearl ids are unique per copy —
0 shared across trees — so ids cannot detect multi-filing; URLs can. Coverage 100%.)

ITEM 2 — GATE ARCHITECTURE. Current form: sc = we*e5 + (softmax_3(mlp(S,A,E)) * chan).sum,
so the mu channels compete only among themselves for a budget of 1 while e5's weight is
unconstrained — 'e5 is weak on this query' is inexpressible. Arms:
  current      3-way softmax, we outside, single LR (reference)
  softmax4     input-dependent 4-way softmax over {e5,S,A,E} (primary fix)
  twotimescale current form, we in its own param group at lr 0.005 vs MLP 0.05
  softmax4_tt  both
  drop_mu      dropout over mu channels only (phase-2 arm, reference)
  drop_all4    dropout including e5 — tests reversion directly
  hardzero_mu  mu channels zeroed at eval; MUST recover ~e5-alone or the gate plumbing is
               degrading e5 itself (a third explanation beyond robustness/reversion)
Scored on the LOCO simplewiki->pearltrees cell AND in-domain pearltrees, 5 seeds, CIs.
SCOPE (item 4): the gate is a learned combiner of FROZEN cached channel scores; only
combiner weights are learnable. Findings here bound 'the best combination of the channels
as they currently are', NOT the mu approach."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from eval_filing import load_filing, metrics
from sa_phase1_shadow import build, ranks_from

TREES = "/home/s243a/Projects/UnifyWeaver/.local/data/pearltrees_api/trees"
SEEDS = (7, 17, 27, 37, 47)
STEPS = 400

# ---------------- item 1: behavioral ceiling ----------------
def behavioral_ceiling(cp, gate_fn):
    _, cand, identity = load_filing(TREES, 3, return_identity=True)
    tid_list = cp["tid_list"]
    trees_by_url = {}
    for (tid, title), v in identity.items():
        if v["url"]:
            trees_by_url.setdefault(v["url"], set()).add(tid)
    url_of_query = {}
    for i, (title, tid) in enumerate(cp["queries"]):
        v = identity.get((tid, title))
        if v and v["url"]:
            url_of_query[i] = v["url"]
    multi = {u: t for u, t in trees_by_url.items() if len(t) > 1}
    out = {"coverage_query_url": round(len(url_of_query) / len(cp["queries"]), 4),
           "distinct_urls": len(trees_by_url),
           "urls_in_multiple_trees": len(multi),
           "multi_filing_rate": round(len(multi) / max(len(trees_by_url), 1), 4)}
    o = np.random.default_rng(7).permutation(len(cp["queries"]))
    he = torch.tensor(o[300:])
    sc = gate_fn(torch.tensor(o[:300]), he)
    top1 = sc.argmax(1).tolist()
    tp = [int(cp["truepos"][i]) for i in he.tolist()]
    col = {t: j for j, t in enumerate(tid_list)}
    err = amb = err_into_T = 0
    tsizes = []
    for r, qi in enumerate(he.tolist()):
        u = url_of_query.get(qi)
        T = {tp[r]} | ({col[t] for t in trees_by_url.get(u, ()) if t in col} if u else set())
        tsizes.append(len(T))
        if len(T) > 1:
            amb += 1
        if top1[r] != tp[r]:
            err += 1
            if top1[r] in T:
                err_into_T += 1
    out |= {"held_queries": len(he), "ambiguous_labels_in_held": amb,
            "mean_T_size": round(float(np.mean(tsizes)), 4),
            "top1_errors": err, "errors_into_behavioral_T": err_into_T,
            "share_defensible_BEHAVIORAL": round(err_into_T / max(err, 1), 4),
            "retracted_proxy_estimate": "0.6%@0.90 / 4.6%@0.85 / 27.2%@0.80 (e5 geometry)"}
    return out

# ---------------- item 2: gate arms ----------------
def make_gate(kind):
    n_in = 4 if kind.startswith("softmax4") else 3
    n_out = 4 if kind.startswith("softmax4") else 3
    mlp = torch.nn.Sequential(torch.nn.Linear(n_in, 12), torch.nn.Tanh(),
                              torch.nn.Linear(12, n_out))
    we = torch.nn.Parameter(torch.tensor(0.2))
    return mlp, we

def gate_score(kind, mlp, we, d, ix, drop=0.0, zero_mu=False, train=False):
    C = d["Cz"][ix]
    X3 = torch.stack([d["Sz"][ix], d["Az"][ix], d["Ez"][ix]], -1)
    if zero_mu:
        X3 = torch.zeros_like(X3)
    if kind.startswith("softmax4"):
        X4 = torch.cat([C.unsqueeze(-1), X3], -1)
        if train and drop > 0:
            m = (torch.rand(1, 1, 4) > drop).float() / max(1e-6, 1 - drop)
            X4 = X4 * m
        w = torch.softmax(mlp(X4), -1)
        return (w * X4).sum(-1), w
    Xs = X3
    if train and drop > 0:
        nd = 4 if kind == "drop_all4" else 3
        m = (torch.rand(1, 1, nd) > drop).float() / max(1e-6, 1 - drop)
        if kind == "drop_all4":
            Xs = X3 * m[..., 1:]
            C = C * m[..., 0]
        else:
            Xs = X3 * m
    w = torch.softmax(mlp(Xs), -1)
    return we * C + (w * Xs).sum(-1), w

def train_arm(kind, dh, ix, seed, drop=0.0):
    torch.manual_seed(seed)
    mlp, we = make_gate(kind)
    if "tt" in kind or kind == "twotimescale":
        opt = torch.optim.Adam([{"params": mlp.parameters(), "lr": 0.05},
                                {"params": [we], "lr": 0.005}])
    else:
        opt = torch.optim.Adam(list(mlp.parameters()) + [we], lr=0.05)
    for _ in range(STEPS):
        sc, _ = gate_score(kind, mlp, we, dh, ix, drop=drop, train=True)
        loss = torch.nn.functional.cross_entropy(sc * 8.0, dh["truepos"][ix])
        opt.zero_grad(); loss.backward(); opt.step()
    return mlp, we

def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cw, cp = build("simplewiki", dev), build("pearltrees", dev)
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing",
           "scope_item4": "gate = learned combiner of FROZEN cached channel scores; only "
                          "combiner weights learnable. Bounds 'best combination of channels "
                          "as they currently are', NOT the mu approach."}
    ARMS = [("current", 0.0), ("softmax4", 0.0), ("twotimescale", 0.0),
            ("softmax4_tt", 0.0), ("drop_mu", 0.6), ("drop_all4", 0.6)]
    table = {}
    for kind, drop in ARMS:
        loco, indom, we_end = [], [], []
        for s in SEEDS:
            o = np.random.default_rng(s).permutation(len(cw["queries"]))
            od = np.random.default_rng(s).permutation(len(cp["queries"]))
            hed = torch.tensor(od[300:])
            tpd = [int(cp["truepos"][i]) for i in hed.tolist()]
            k = "current" if kind in ("drop_mu",) else ("current" if kind == "drop_all4" else kind)
            kk = kind if kind in ("softmax4", "twotimescale", "softmax4_tt") else \
                ("drop_all4" if kind == "drop_all4" else "current")
            mlp, we = train_arm(kk, cw, torch.tensor(o[:300]), s, drop=drop)
            with torch.no_grad():
                sc, _ = gate_score(kk, mlp, we, cp, hed)
            loco.append(metrics(ranks_from(sc, tpd))["MRR"])
            mlp2, we2 = train_arm(kk, cp, torch.tensor(od[:300]), s, drop=drop)
            with torch.no_grad():
                sc2, _ = gate_score(kk, mlp2, we2, cp, hed)
            indom.append(metrics(ranks_from(sc2, tpd))["MRR"])
            we_end.append(float(we2))
        table[kind] = {"loco": round(float(np.mean(loco)), 4),
                       "loco_sd": round(float(np.std(loco)), 4),
                       "in_domain": round(float(np.mean(indom)), 4),
                       "we_final_in_domain": round(float(np.mean(we_end)), 4)}
        print(f"[arm {kind}] " + json.dumps(table[kind]), flush=True)
    # hard-zero ablation: mu channels zeroed at eval — must recover ~e5-alone
    hz = []
    for s in SEEDS:
        od = np.random.default_rng(s).permutation(len(cp["queries"]))
        hed = torch.tensor(od[300:])
        tpd = [int(cp["truepos"][i]) for i in hed.tolist()]
        mlp, we = train_arm("current", cp, torch.tensor(od[:300]), s)
        with torch.no_grad():
            sc, _ = gate_score("current", mlp, we, cp, hed, zero_mu=True)
        hz.append(metrics(ranks_from(sc, tpd))["MRR"])
    e5_only = []
    for s in SEEDS:
        od = np.random.default_rng(s).permutation(len(cp["queries"]))
        hed = torch.tensor(od[300:])
        e5_only.append(metrics(ranks_from(cp["Cz"][hed],
                                          [int(cp["truepos"][i]) for i in hed.tolist()]))["MRR"])
    table["hardzero_mu"] = {"in_domain": round(float(np.mean(hz)), 4),
                            "e5_alone_reference": round(float(np.mean(e5_only)), 4),
                            "recovers_e5": bool(abs(np.mean(hz) - np.mean(e5_only)) < 0.02)}
    print("[hardzero] " + json.dumps(table["hardzero_mu"]), flush=True)
    res["P3_gate_arms"] = table
    # item 1 using the best available in-domain gate (current form, for comparability)
    def gate_fn(tr, he):
        mlp, we = train_arm("current", cp, tr, 7)
        with torch.no_grad():
            sc, _ = gate_score("current", mlp, we, cp, he)
        return sc
    res["P3_behavioral_ceiling"] = behavioral_ceiling(cp, gate_fn)
    print("[item1] " + json.dumps(res["P3_behavioral_ceiling"]), flush=True)
    json.dump(res, open("PHASE3_CEILING_AND_GATE.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)

run()
