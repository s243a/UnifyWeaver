#!/usr/bin/env python3
"""SHADOW: S/A fusion PHASE 0 — D0 controls (encoder-lane ruling, supersedes v2 follow-up).
Blocks the explicit-distance arm until the D0 curve survives its confounds.

0a(i)  shortest-path distance in the FULL DAG (no parent choice). The v2 full_dag BFS
       already assigned first-visit hop = shortest path, so that curve stands; the
       principal_tree contrast — and the "multi-parent phenomenon" conclusion drawn from
       it — is WITHDRAWN (alphabetical sorted(set)[0] carries zero semantic content).
0a(iii) random-chain NULL CONTROL: N seeded random ancestor chains, report the band. Any
       chain-based claim must sit outside it.
0a(iv) DIRECT multi-parent test: at FIXED hop, A/S for candidates reachable by ONE upward
       path vs SEVERAL (path multiplicity counted over the BFS frontier).
0b     DEGREE CONFOUND: regress log(A/S) on (hop, log deg) — does hop's coefficient
       survive? plus within-degree-band stratification.
0c     n per hop reported everywhere.
0d     A2 headline variance: 5 seeds x per-seed bootstrap CI of gate − mumax on held.
Caveat carried: typos masquerade as drift (§7 decay fork) — every curve is an UPPER BOUND
on true semantic drift. Scope: simplewiki (GRAPH) only; no Pearltrees claim.
Writes a score-matrix cache for later phases."""
import os, sys, json, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from eval_filing import load_membership, score_mu, metrics
from fine_tune_channel_heads import load_expanded
from mu_attention import OPS, Tokenizer, build_e5_tables, E5_REVISION, load_dag, GRAPH

CKPT, SEED, MAXQ = "model_pt_filing.pt", 7, 500
CACHE = os.path.expanduser("~/mu_data/sa_scores_simplewiki.pt")
MAXHOP, N_RANDOM_CHAINS = 6, 20

def nzrow(M):
    lo = M.min(dim=1, keepdim=True).values
    hi = M.max(dim=1, keepdim=True).values
    return (M - lo) / (hi - lo + 1e-9)

def ranks_from(M, truepos):
    return [1 + int(((M[r] > M[r][truepos[r]]) |
                     ((M[r] == M[r][truepos[r]]) & (torch.arange(M.shape[1]) < truepos[r]))).sum().item())
            for r in range(M.shape[0])]

def build_scores(dev):
    if os.path.exists(CACHE):
        d = torch.load(CACHE, weights_only=False)
        print(f"[cache] loaded {CACHE}", flush=True)
        return d
    queries, cand = load_membership(GRAPH, 3)
    rng = random.Random(SEED)
    if len(queries) > MAXQ:
        queries = rng.sample(queries, MAXQ)
    f_keys = [f"F:{t}" for t in cand]
    q_keys = [f"B:{i}" for i in range(len(queries))]
    texts = {f"F:{t}": cand[t] for t in cand} | {f"B:{i}": q for i, (q, _) in enumerate(queries)}
    qtbl, ptbl, idx = build_e5_tables(sorted(texts), cache_path=None, texts=texts,
                                      model_revision=E5_REVISION)
    tok = Tokenizer(qtbl, ptbl, idx, {}, {})
    model, _ = load_expanded(os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT),
                             dev=dev)
    model.eval()
    ow = lambda op: torch.zeros(1, model.op_emb.weight.shape[0]).index_fill_(
        1, torch.tensor([OPS[op]]), 1.0)
    sm = lambda o: torch.tensor(score_mu(model, tok, idx, q_keys, f_keys, o, dev))
    E, A, S = sm(ow("ELEM")), sm(ow("HIER")), sm(ow("SYM"))
    C = (qtbl[[idx[k] for k in q_keys]] @ ptbl[[idx[k] for k in f_keys]].T).cpu()
    tid_list = list(cand)
    d = dict(queries=queries, tid_list=tid_list,
             truepos=torch.tensor([tid_list.index(t) for _, t in queries]),
             Cz=nzrow(C), Sz=nzrow(S), Az=nzrow(A), Ez=nzrow(E))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save(d, CACHE)
    return d

def bfs_levels(parents, start, tid_ix, maxhop, rng=None, random_chain=False):
    """→ {hop: [(cand_col, n_paths_from_frontier)]}, shortest-path hops (BFS first visit)."""
    seen, frontier, out = {start}, [start], {}
    for hop in range(1, maxhop + 1):
        counts = {}
        for x in frontier:
            ps = sorted(parents.get(x, []))
            if random_chain and ps:
                ps = [ps[rng.randrange(len(ps))]]
            for p in ps:
                if p not in seen:
                    counts[p] = counts.get(p, 0) + 1
        if not counts:
            break
        seen |= set(counts)
        frontier = list(counts)
        rows = [(tid_ix[p], n) for p, n in counts.items() if p in tid_ix]
        if rows:
            out[hop] = rows
    return out

def curve(d, parents, tid_ix, random_chain=False, seed=0):
    rng = random.Random(seed)
    acc = {}
    for r, (_, true_t) in enumerate(d["queries"]):
        for hop, rows in bfs_levels(parents, true_t, tid_ix, MAXHOP, rng, random_chain).items():
            for col, npath in rows:
                acc.setdefault(hop, []).append((float(d["Az"][r, col]),
                                                float(d["Sz"][r, col]), npath))
    return {h: {"n": len(v), "A": round(float(np.mean([a for a, _, _ in v])), 4),
                "S": round(float(np.mean([s for _, s, _ in v])), 4),
                "A_over_S": round(float(np.mean([a for a, _, _ in v]) /
                                        max(np.mean([s for _, s, _ in v]), 1e-6)), 4)}
            for h, v in sorted(acc.items())}, acc

def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = build_scores(dev)
    parents, children, deg = load_dag(GRAPH)
    tid_ix = {t: j for j, t in enumerate(d["tid_list"])}
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing",
           "scope": "simplewiki only; no Pearltrees claim",
           "caveat": "typos masquerade as drift (§7) — curves are UPPER BOUNDS on drift",
           "withdrawn": "v2 principal_tree contrast + its multi-parent conclusion "
                        "(alphabetical sorted(set)[0] = zero semantic content)"}
    # 0a(i)+0c full-DAG shortest-path curve
    main, acc = curve(d, parents, tid_ix)
    res["D0_shortest_path_full_dag"] = main
    print("[0a-i] " + json.dumps(main), flush=True)
    # 0a(iii) random-chain null band
    bands = {}
    for s in range(N_RANDOM_CHAINS):
        c, _ = curve(d, parents, tid_ix, random_chain=True, seed=1000 + s)
        for h, v in c.items():
            bands.setdefault(h, []).append(v["A_over_S"])
    res["D0_random_chain_null"] = {
        h: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4),
            "min": round(float(np.min(v)), 4), "max": round(float(np.max(v)), 4),
            "full_dag_outside_band": not (float(np.min(v)) <= main[h]["A_over_S"] <=
                                          float(np.max(v)))}
        for h, v in sorted(bands.items()) if h in main}
    print("[0a-iii] " + json.dumps(res["D0_random_chain_null"]), flush=True)
    # 0a(iv) direct multi-parent test at fixed hop
    mp = {}
    for h, v in sorted(acc.items()):
        one = [(a, s) for a, s, n in v if n == 1]
        many = [(a, s) for a, s, n in v if n > 1]
        if len(one) > 30 and len(many) > 30:
            f = lambda xs: float(np.mean([a for a, _ in xs]) /
                                 max(np.mean([s for _, s in xs]), 1e-6))
            mp[h] = {"n_one": len(one), "n_many": len(many),
                     "A_over_S_one_path": round(f(one), 4),
                     "A_over_S_multi_path": round(f(many), 4)}
    res["D0_multipath_at_fixed_hop"] = mp
    print("[0a-iv] " + json.dumps(mp), flush=True)
    # 0b degree confound
    rows = []
    for h, v in acc.items():
        for a, s, _ in v:
            rows.append((h, a, s))
    cols = {c: j for j, c in enumerate(d["tid_list"])}
    degs = {}
    for r, (_, true_t) in enumerate(d["queries"]):
        for hop, rr in bfs_levels(parents, true_t, tid_ix, MAXHOP).items():
            for col, _ in rr:
                t = d["tid_list"][col]
                degs.setdefault(hop, []).append(deg.get(t, len(children.get(t, ()))) or 1)
    X, y, dg, hp = [], [], [], []
    for h in sorted(acc):
        for (a, s, _), dv in zip(acc[h], degs.get(h, [])):
            ratio = (a + 1e-4) / (s + 1e-4)
            X.append([1.0, h, math.log(max(dv, 1))]); y.append(math.log(ratio))
            dg.append(dv); hp.append(h)
    X, y = np.array(X), np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (resid @ resid) / (len(y) - 3))
    res["D0b_degree_regression"] = {
        "n": len(y), "model": "log(A/S) ~ 1 + hop + log(deg)",
        "coef": {"intercept": round(float(beta[0]), 4), "hop": round(float(beta[1]), 4),
                 "log_deg": round(float(beta[2]), 4)},
        "se": {"hop": round(float(se[1]), 4), "log_deg": round(float(se[2]), 4)},
        "hop_t": round(float(beta[1] / se[1]), 2), "log_deg_t": round(float(beta[2] / se[2]), 2),
        "hop_only_coef": round(float(np.linalg.lstsq(X[:, :2], y, rcond=None)[0][1]), 4)}
    dgq = np.quantile(dg, [0.33, 0.66])
    bandsd = {}
    for lo, hi, nm in ((0, dgq[0], "lowdeg"), (dgq[0], dgq[1], "middeg"), (dgq[1], 1e18, "highdeg")):
        sel = [(h, yy) for h, yy, dv in zip(hp, y, dg) if lo <= dv < hi]
        per = {}
        for h in sorted(set(hh for hh, _ in sel)):
            vs = [yy for hh, yy in sel if hh == h]
            if len(vs) > 30:
                per[h] = round(float(np.exp(np.mean(vs))), 4)
        bandsd[nm] = per
    res["D0b_within_degree_band_A_over_S"] = bandsd
    print("[0b] " + json.dumps(res["D0b_degree_regression"]) + " " + json.dumps(bandsd), flush=True)
    # 0d A2 headline variance
    def gate_arm(seed):
        g = np.random.default_rng(seed)
        order = g.permutation(len(d["queries"]))
        tr, he = torch.tensor(order[:300]), torch.tensor(order[300:])
        torch.manual_seed(seed)
        mlp = torch.nn.Sequential(torch.nn.Linear(3, 12), torch.nn.Tanh(), torch.nn.Linear(12, 3))
        we = torch.nn.Parameter(torch.tensor(0.2))
        opt = torch.optim.Adam(list(mlp.parameters()) + [we], lr=0.05)
        chan = lambda ix: torch.stack([d["Sz"][ix], d["Az"][ix], d["Ez"][ix]], -1)
        for _ in range(400):
            w = torch.softmax(mlp(chan(tr)), -1)
            sc = we * d["Cz"][tr] + (w * chan(tr)).sum(-1)
            loss = torch.nn.functional.cross_entropy(sc * 8.0, d["truepos"][tr])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            w = torch.softmax(mlp(chan(he)), -1)
            gs = we * d["Cz"][he] + (w * chan(he)).sum(-1)
        mumax = nzrow(torch.maximum(torch.maximum(d["Sz"], d["Az"]), d["Ez"]))
        mm = (0.1 * d["Cz"] + 0.9 * mumax)[he]
        tp = [int(d["truepos"][i]) for i in he.tolist()]
        rg, rm = ranks_from(gs, tp), ranks_from(mm, tp)
        return rg, rm
    seeds, gaps, boots = [], [], []
    for s in (7, 17, 27, 37, 47):
        rg, rm = gate_arm(s)
        mg, mm_ = metrics(rg)["MRR"], metrics(rm)["MRR"]
        seeds.append({"seed": s, "gate_MRR": round(mg, 4), "mumax_MRR": round(mm_, 4),
                      "gap": round(mg - mm_, 4)})
        gaps.append(mg - mm_)
        bs = np.random.default_rng(s)
        for _ in range(400):
            ix = bs.integers(0, len(rg), len(rg))
            boots.append(float(np.mean([1/rg[i] for i in ix]) - np.mean([1/rm[i] for i in ix])))
    res["D0d_A2_headline_variance"] = {
        "per_seed": seeds, "mean_gap": round(float(np.mean(gaps)), 4),
        "sd_gap": round(float(np.std(gaps)), 4),
        "pooled_bootstrap_CI95": [round(float(np.percentile(boots, 2.5)), 4),
                                  round(float(np.percentile(boots, 97.5)), 4)],
        "n_held": 200}
    print("[0d] " + json.dumps(res["D0d_A2_headline_variance"]), flush=True)
    json.dump(res, open("PHASE0_D0_CONTROLS.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)

if __name__ == "__main__":
    run()