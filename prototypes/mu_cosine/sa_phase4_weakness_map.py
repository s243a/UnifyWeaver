#!/usr/bin/env python3
"""SHADOW: PHASE 4 — REGION WEAKNESS MAP (the harvest-targeting loop) + the owed region
labels and DAG cross-check.

Loop (owner + Kimi): let the gate say where the mu channels are untrustworthy, cross-
reference with graph COVERAGE, and spend the enwiki harvest exactly there.
  signal 1  channel gap: e5 MRR - best mu-channel MRR per region (size of the prize)
  signal 2  the softmax4 gate's learned mu weight per region (the model's own measured
            trust — the more sensitive instrument)
  signal 3  coverage: how many categories the enwiki graph already has in that region
  rule      weak + THIN coverage -> harvest; weak + DENSE coverage -> capability/
            granularity problem, bulk enwiki data is the wrong remedy (the widen_enwiki
            saturation lesson: AI was coverage-poor and data fixed it; Chem/CS/Eng were
            saturated and data did nothing)
CROSS-CHECK (Kimi's caution, and the reason it matters here): region labels are currently
e5-assigned by argmax over region-keyword embeddings — but the whole point is finding places
e5's geometry misleads. A DAG-based label (shallow ancestor in the category graph) is
computed alongside and their AGREEMENT is reported; regions where the two disagree are
exactly where an e5-only stratification would mislead.
SCOPE (item 4): the gate is a learned combiner of FROZEN cached channel scores. This map
says where the CHANNELS' INPUTS are weak — it is not a claim about the mu approach."""
import os, sys, json
from collections import defaultdict, deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from eval_filing import metrics
from sa_phase1_shadow import build, ranks_from, REGIONS, region_of
from sa_phase3_shadow import train_arm, gate_score
from mu_attention import load_dag, GRAPH, build_e5_tables, E5_REVISION

REPO = "/home/s243a/Projects/UnifyWeaver"
ENWIKI = os.path.join(REPO, "data", "benchmark", "wide_enwiki", "category_parent.tsv")
SEEDS = (7, 17, 27)

def dag_region_labels(tid_list, rnames):
    """Structural label: walk up to a shallow ancestor, then name that ancestor's region
    by e5 (the STRUCTURE comes from the user's graph; e5 only names the bucket)."""
    # NO single-parent choice: the alphabetical sorted(parents)[0] rule was WITHDRAWN by
    # owner ruling in phase 0 (zero semantic content) and must not return. Use the full DAG:
    # BFS up 2 hops over ALL parents and take the MAJORITY region of the ancestor multiset.
    # Walking to a near-root ancestor also collapses everything onto a handful of hubs
    # (the first version's 6-of-7 'everyday' degeneracy), so depth is capped at 2.
    parents, children, _ = load_dag(GRAPH)
    anc = {}
    for t in tid_list:
        seen, frontier, acc = {t}, [t], []
        for _ in range(2):
            nxt = []
            for x in frontier:
                for p in (parents.get(x, ()) or ()):
                    if p not in seen:
                        seen.add(p); nxt.append(p); acc.append(p)
            frontier = nxt
        anc[t] = acc or [t]
    uniq = sorted({a for v in anc.values() for a in v})
    texts = {f"T:{u}": u.replace("_", " ") for u in uniq} | \
            {f"R:{r}": v for r, v in REGIONS.items()}
    _, pt, ix = build_e5_tables(sorted(texts), cache_path=None, texts=texts,
                               model_revision=E5_REVISION)
    V = torch.stack([pt[ix[f"R:{r}"]] for r in rnames])
    reg_of_top = {u: rnames[int(torch.argmax(V @ pt[ix[f"T:{u}"]]))] for u in uniq}
    out = []
    for t in tid_list:
        votes = {}
        for a in anc[t]:
            r = reg_of_top[a]; votes[r] = votes.get(r, 0) + 1
        out.append(max(votes.items(), key=lambda kv: kv[1])[0])
    return out, len(uniq)

def enwiki_coverage(rnames):
    """How many enwiki categories sit in each region (e5-argmax over category titles)."""
    cats = set()
    with open(ENWIKI, encoding="utf-8") as fh:
        next(fh)
        for ln in fh:
            c, _, p = ln.rstrip("\n").partition("\t")
            cats.add(c); cats.add(p)
    cats = sorted(cats)[:20000]
    texts = {f"C:{c}": c.replace("_", " ") for c in cats} | \
            {f"R:{r}": v for r, v in REGIONS.items()}
    _, pt, ix = build_e5_tables(sorted(texts), cache_path=None, texts=texts,
                               model_revision=E5_REVISION)
    V = torch.stack([pt[ix[f"R:{r}"]] for r in rnames])
    M = torch.stack([pt[ix[f"C:{c}"]] for c in cats])
    lab = (M @ V.T).argmax(1).tolist()
    out = defaultdict(int)
    for l in lab:
        out[rnames[l]] += 1
    return dict(out), len(cats)

def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rnames = list(REGIONS)
    cw = build("simplewiki", dev)
    cpt = build("pearltrees", dev)
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing",
           "scope_item4": "gate = combiner of FROZEN channels; this maps where the CHANNEL "
                          "INPUTS are weak, not the mu approach"}
    e5_reg = region_of([f"F:{t}" for t in cw["tid_list"]], None, rnames,
                       (cw["qtbl"], cw["ptbl"], cw["idx"]))
    dag_reg, n_tops = dag_region_labels(cw["tid_list"], rnames)
    agree = float(np.mean([a == b for a, b in zip(e5_reg, dag_reg)]))
    res["region_label_crosscheck"] = {
        "e5_vs_dag_agreement": round(agree, 4), "n_shallow_ancestors": n_tops,
        "note": "regions where the two disagree are where an e5-only stratification misleads"}
    conf = defaultdict(lambda: defaultdict(int))
    for a, b in zip(e5_reg, dag_reg):
        conf[a][b] += 1
    res["region_label_crosscheck"]["e5_region_top_dag_match"] = {
        r: max(v.items(), key=lambda kv: kv[1])[0] for r, v in conf.items()}
    print("[crosscheck] " + json.dumps(res["region_label_crosscheck"]), flush=True)
    cov, n_cat = enwiki_coverage(rnames)
    res["enwiki_coverage"] = {"sampled_categories": n_cat, "by_region": cov}
    print("[coverage] " + json.dumps(cov), flush=True)
    # per-region weakness: channel gap + gate trust, on the in-domain simplewiki gate
    qreg = [e5_reg[int(cw["truepos"][i])] for i in range(len(cw["queries"]))]
    rows = {}
    per_seed_w = defaultdict(list)
    for s in SEEDS:
        o = np.random.default_rng(s).permutation(len(cw["queries"]))
        tr, he = torch.tensor(o[:300]), torch.tensor(o[300:])
        mlp, we = train_arm("softmax4", cw, tr, s)
        with torch.no_grad():
            sc, w = gate_score("softmax4", mlp, we, cw, he)
        for r in rnames:
            sel = [k for k, i in enumerate(he.tolist()) if qreg[i] == r]
            if len(sel) < 8:
                continue
            ixs = he[torch.tensor(sel)]
            tps = [int(cw["truepos"][i]) for i in ixs.tolist()]
            gap_e5 = metrics(ranks_from(cw["Cz"][ixs], tps))["MRR"]
            best_mu = max(metrics(ranks_from(cw[k][ixs], tps))["MRR"]
                          for k in ("Sz", "Az", "Ez"))
            gate_mrr = metrics(ranks_from(sc[torch.tensor(sel)], tps))["MRR"]
            wsel = w[torch.tensor(sel)]
            per_seed_w[r].append({
                "n": len(sel), "e5": gap_e5, "best_mu": best_mu, "gate": gate_mrr,
                "gap": gap_e5 - best_mu,
                "w_e5": float(wsel[..., 0].mean()),
                "w_mu": float(wsel[..., 1:].sum(-1).mean())})
    for r, lst in per_seed_w.items():
        agg = {k: round(float(np.mean([x[k] for x in lst])), 4)
               for k in ("e5", "best_mu", "gate", "gap", "w_e5", "w_mu")}
        agg["n"] = int(np.mean([x["n"] for x in lst]))
        agg["enwiki_categories"] = cov.get(r, 0)
        thin = cov.get(r, 0) < np.median(list(cov.values()))
        weak = agg["gap"] > 0 or agg["w_mu"] < 0.5
        agg["verdict"] = ("HARVEST (weak + thin coverage)" if weak and thin else
                          "saturated/capability — bulk data likely wrong remedy" if weak
                          else "mu already competitive")
        rows[r] = agg
    res["region_weakness_map_simplewiki"] = rows
    # same map on pearltrees (e5 regions only — no category DAG there; caveat recorded)
    pt_reg = region_of([f"F:{t}" for t in cpt["tid_list"]], None, rnames,
                       (cpt["qtbl"], cpt["ptbl"], cpt["idx"]))
    pqreg = [pt_reg[int(cpt["truepos"][i])] for i in range(len(cpt["queries"]))]
    prows = defaultdict(list)
    for s2 in SEEDS:
        o = np.random.default_rng(s2).permutation(len(cpt["queries"]))
        tr, he = torch.tensor(o[:300]), torch.tensor(o[300:])
        mlp, we = train_arm("softmax4", cpt, tr, s2)
        with torch.no_grad():
            sc, w = gate_score("softmax4", mlp, we, cpt, he)
        for r in rnames:
            sel = [k for k, i in enumerate(he.tolist()) if pqreg[i] == r]
            if len(sel) < 8:
                continue
            ixs = he[torch.tensor(sel)]
            tps = [int(cpt["truepos"][i]) for i in ixs.tolist()]
            e5m = metrics(ranks_from(cpt["Cz"][ixs], tps))["MRR"]
            bmu = max(metrics(ranks_from(cpt[k][ixs], tps))["MRR"] for k in ("Sz","Az","Ez"))
            gm = metrics(ranks_from(sc[torch.tensor(sel)], tps))["MRR"]
            wsel = w[torch.tensor(sel)]
            prows[r].append({"n": len(sel), "e5": e5m, "best_mu": bmu, "gate": gm,
                             "gap": e5m - bmu, "w_e5": float(wsel[..., 0].mean()),
                             "w_mu": float(wsel[..., 1:].sum(-1).mean())})
    pm = {}
    for r, lst in prows.items():
        agg = {k: round(float(np.mean([x[k] for x in lst])), 4)
               for k in ("e5","best_mu","gate","gap","w_e5","w_mu")}
        agg["n"] = int(np.mean([x["n"] for x in lst]))
        agg["enwiki_categories"] = cov.get(r, 0)
        thin = cov.get(r, 0) < np.median(list(cov.values()))
        weak = agg["gap"] > 0 or agg["w_mu"] < 0.5
        agg["verdict"] = ("HARVEST (weak + thin coverage)" if weak and thin else
                          "saturated/capability — bulk data likely wrong remedy" if weak
                          else "mu already competitive")
        pm[r] = agg
    res["region_weakness_map_pearltrees"] = pm
    print("[map-pt] " + json.dumps(pm, indent=1), flush=True)
    print("[map] " + json.dumps(rows, indent=1), flush=True)
    json.dump(res, open("PHASE4_WEAKNESS_MAP.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)

run()
