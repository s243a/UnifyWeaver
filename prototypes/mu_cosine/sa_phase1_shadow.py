#!/usr/bin/env python3
"""SHADOW: S/A fusion PHASE 1 — corpus audit + transfer scoreboard (encoder-lane ruling).
1a(i)   topical composition of Pearltrees vs simplewiki candidate sets, plus BOTH skew
        mechanisms: (a) privacy-filter removal rate BY REGION, (b) min_bm>=3 tail drop
        by region. PRIVACY: private titles are embedded in-memory for aggregate rates
        ONLY — no private text is printed, written, or committed.
1a(ii)  ground-truth ceiling: share of top-1 errors where the predicted folder holds a
        near-duplicate title (plausibly correct-but-differently-filed) + rank-to-capture
        normalized by |T| (diagnostic only, NOT the objective).
1a(iii) D1 standalone channels stratified by topical region — is e5 dominance a
        technical-vocabulary effect rather than a general OOD property?
1b      LEAVE-ONE-CORPUS-OUT: train gate on one corpus, evaluate on the OTHER, vs chance
        and vs e5 alone. This is the scoreboard (A3's joint-vs-specialist measured
        multi-task interference, a different question — recorded as superseded).
1c      corpus-identity probe on the learned representation → tuning instrument for ph2.
"""
import os, sys, json, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from eval_filing import load_filing, load_membership, score_mu, metrics
from fine_tune_channel_heads import load_expanded
from mu_attention import OPS, Tokenizer, build_e5_tables, E5_REVISION, load_dag, GRAPH

TREES = "/home/s243a/Projects/UnifyWeaver/.local/data/pearltrees_api/trees"
CKPT, SEED, MAXQ = "model_pt_filing.pt", 7, 500
CACHE = os.path.expanduser("~/mu_data/sa_scores_%s.pt")
REGIONS = {
    "stem_physical": "physics chemistry mathematics astronomy thermodynamics engineering",
    "stem_computing": "computer science software programming algorithms networks security",
    "life_health": "biology medicine health nutrition food fitness psychology",
    "society": "politics law economics government history war military",
    "culture": "music film art literature religion philosophy language",
    "geography": "country city region geography places travel maps",
    "everyday": "shopping hobbies home sports games entertainment personal",
}

def nzrow(M):
    lo = M.min(dim=1, keepdim=True).values
    hi = M.max(dim=1, keepdim=True).values
    return (M - lo) / (hi - lo + 1e-9)

def ranks_from(M, truepos):
    return [1 + int(((M[r] > M[r][truepos[r]]) |
                     ((M[r] == M[r][truepos[r]]) & (torch.arange(M.shape[1]) < truepos[r]))).sum().item())
            for r in range(M.shape[0])]

def region_of(titles, rtbl, rnames, model_tbls):
    qt, pt, idx = model_tbls
    out = []
    V = torch.stack([pt[idx[f"R:{r}"]] for r in rnames])
    for t in titles:
        v = pt[idx[t]] if t in idx else None
        out.append(rnames[int(torch.argmax(V @ v))] if v is not None else "unassigned")
    return out

def build(source, dev):
    cp = CACHE % source
    if os.path.exists(cp):
        return torch.load(cp, weights_only=False)
    if source == "simplewiki":
        queries, cand = load_membership(GRAPH, 3)
    else:
        queries, cand = load_filing(TREES, 3)
    rng = random.Random(SEED)
    if len(queries) > MAXQ:
        queries = rng.sample(queries, MAXQ)
    f_keys = [f"F:{t}" for t in cand]
    q_keys = [f"B:{i}" for i in range(len(queries))]
    texts = {f"F:{t}": cand[t] for t in cand} | {f"B:{i}": q for i, (q, _) in enumerate(queries)}
    texts |= {f"R:{r}": v for r, v in REGIONS.items()}
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
    d = dict(queries=queries, cand=cand, tid_list=tid_list, q_keys=q_keys, f_keys=f_keys,
             truepos=torch.tensor([tid_list.index(t) for _, t in queries]),
             Cz=nzrow(C), Sz=nzrow(S), Az=nzrow(A), Ez=nzrow(E),
             qtbl=qtbl, ptbl=ptbl, idx=idx)
    torch.save(d, cp)
    return d

def gate_train(d, tr, seed=7, steps=400):
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(torch.nn.Linear(3, 12), torch.nn.Tanh(), torch.nn.Linear(12, 3))
    we = torch.nn.Parameter(torch.tensor(0.2))
    opt = torch.optim.Adam(list(mlp.parameters()) + [we], lr=0.05)
    chan = lambda ix: torch.stack([d["Sz"][ix], d["Az"][ix], d["Ez"][ix]], -1)
    for _ in range(steps):
        w = torch.softmax(mlp(chan(tr)), -1)
        sc = we * d["Cz"][tr] + (w * chan(tr)).sum(-1)
        loss = torch.nn.functional.cross_entropy(sc * 8.0, d["truepos"][tr])
        opt.zero_grad(); loss.backward(); opt.step()
    return mlp, we

def gate_apply(mlp, we, d, ix):
    chan = torch.stack([d["Sz"][ix], d["Az"][ix], d["Ez"][ix]], -1)
    with torch.no_grad():
        w = torch.softmax(mlp(chan), -1)
        return we * d["Cz"][ix] + (w * chan).sum(-1)

def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing",
           "privacy_note": "private titles embedded in-memory for aggregate rates only; "
                           "no private text printed or written"}
    cp, cw = build("pearltrees", dev), build("simplewiki", dev)
    rnames = list(REGIONS)
    # --- 1a(i) topical composition + skew mechanisms
    comp = {}
    for tag, d in (("pearltrees", cp), ("simplewiki", cw)):
        regs = region_of([f"F:{t}" for t in d["tid_list"]], None, rnames,
                         (d["qtbl"], d["ptbl"], d["idx"]))
        n = len(regs)
        comp[tag] = {r: round(regs.count(r) / n, 3) for r in rnames}
        comp[tag]["n_folders"] = n
        d["folder_region"] = regs
    res["P1a_i_topical_composition"] = comp
    print("[1a-i] " + json.dumps(comp), flush=True)
    # (a) privacy removal by region + (b) min_bm tail, Pearltrees only
    q_all, cand_all, privacy = load_filing(TREES, 1, return_privacy=True)
    pub_ids = set(privacy.public_ids)
    removed_titles, kept_titles = [], []
    for k, t in privacy.tree_payloads.items():
        ttl = t.get("title")
        if not ttl:
            continue
        (kept_titles if k in pub_ids else removed_titles).append(ttl)
    tex = {f"X:{i}": t for i, t in enumerate(removed_titles + kept_titles)}
    tex |= {f"R:{r}": v for r, v in REGIONS.items()}
    _, pt2, idx2 = build_e5_tables(sorted(tex), cache_path=None, texts=tex,
                                   model_revision=E5_REVISION)
    V = torch.stack([pt2[idx2[f"R:{r}"]] for r in rnames])
    assign = lambda keys: [rnames[int(torch.argmax(V @ pt2[idx2[k]]))] for k in keys]
    rem_r = assign([f"X:{i}" for i in range(len(removed_titles))])
    kept_r = assign([f"X:{i}" for i in range(len(removed_titles), len(removed_titles) + len(kept_titles))])
    res["P1a_i_a_privacy_removal"] = {
        "n_trees_total": len(removed_titles) + len(kept_titles),
        "n_removed_nonpublic": len(removed_titles),
        "removal_rate_overall": round(len(removed_titles) /
                                      max(len(removed_titles) + len(kept_titles), 1), 3),
        "removal_rate_by_region": {
            r: round(rem_r.count(r) / max(rem_r.count(r) + kept_r.count(r), 1), 3)
            for r in rnames}}
    small = {tid: ttl for tid, ttl in cand_all.items() if tid not in cp["cand"]}
    tex2 = {f"Y:{i}": t for i, t in enumerate(small.values())} | \
           {f"R:{r}": v for r, v in REGIONS.items()}
    if len(small) > 1:
        _, pt3, idx3 = build_e5_tables(sorted(tex2), cache_path=None, texts=tex2,
                                       model_revision=E5_REVISION)
        V3 = torch.stack([pt3[idx3[f"R:{r}"]] for r in rnames])
        drop_r = [rnames[int(torch.argmax(V3 @ pt3[idx3[f"Y:{i}"]]))] for i in range(len(small))]
        res["P1a_i_b_min_bm_tail"] = {
            "n_dropped_folders": len(small),
            "dropped_region_share": {r: round(drop_r.count(r) / len(drop_r), 3) for r in rnames},
            "kept_region_share": comp["pearltrees"]}
    print("[1a-i-ab] " + json.dumps({k: res[k] for k in res if k.startswith("P1a_i_")}), flush=True)
    # --- 1a(ii) ground-truth ceiling on Pearltrees
    by_folder = {}
    for (bt, tid) in load_filing(TREES, 3)[0]:
        by_folder.setdefault(tid, []).append(bt)
    d = cp
    order = np.random.default_rng(SEED).permutation(len(d["queries"]))
    tr, he = torch.tensor(order[:300]), torch.tensor(order[300:])
    mlp, we = gate_train(d, tr)
    gs = gate_apply(mlp, we, d, he)
    tp = [int(d["truepos"][i]) for i in he.tolist()]
    top1 = gs.argmax(1).tolist()
    # vectorized: query-title similarity matrix, then title -> owning folder column
    title_to_qi = {}
    for i, (b, _) in enumerate(d["queries"]):
        title_to_qi.setdefault(b, i)
    QV = d["qtbl"][[d["idx"][k] for k in d["q_keys"]]]          # [Q, 384] unit-normed
    SIM = QV @ QV.T                                            # [Q, Q] title similarity
    col_of_title = {}
    for j, tid in enumerate(d["tid_list"]):
        for bt in by_folder.get(tid, []):
            col_of_title.setdefault(bt, j)
    dup, tot, tset_sizes, capture = 0, 0, [], []
    for r, (qi, t_true) in enumerate(zip(he.tolist(), tp)):
        qtitle = d["queries"][qi][0]
        near = (SIM[qi] > 0.95).nonzero().flatten().tolist()
        T = {t_true}
        for nj in near:
            bt = d["queries"][nj][0]
            if bt != qtitle and bt in col_of_title:
                T.add(col_of_title[bt])
        tset_sizes.append(len(T))
        if top1[r] != t_true:
            tot += 1
            if top1[r] in T:
                dup += 1
        srt = gs[r].argsort(descending=True).tolist()
        capture.append((max(srt.index(t) for t in T) + 1) / len(T))
    res["P1a_ii_ground_truth_ceiling"] = {
        "n_top1_errors": tot, "errors_into_defensible_alt": dup,
        "share_defensible": round(dup / max(tot, 1), 3),
        "mean_T_size": round(float(np.mean(tset_sizes)), 3),
        "mean_rank_to_capture_over_T": round(float(np.mean(capture)), 2),
        "note": "diagnostic only; set-valued view, NOT the training objective"}
    print("[1a-ii] " + json.dumps(res["P1a_ii_ground_truth_ceiling"]), flush=True)
    # --- 1a(iii) D1 stratified by region
    qreg = [d["folder_region"][int(d["truepos"][i])] for i in range(len(d["queries"]))]
    strat = {}
    for r in rnames:
        ix = torch.tensor([i for i in he.tolist() if qreg[i] == r])
        if len(ix) < 15:
            continue
        tpx = [int(d["truepos"][i]) for i in ix.tolist()]
        strat[r] = {"n": len(ix)}
        for nm, key in (("S", "Sz"), ("A", "Az"), ("E", "Ez"), ("e5", "Cz")):
            strat[r][nm] = round(metrics(ranks_from(d[key][ix], tpx))["MRR"], 4)
        strat[r]["e5_minus_best_mu"] = round(
            strat[r]["e5"] - max(strat[r]["S"], strat[r]["A"], strat[r]["E"]), 4)
    res["P1a_iii_D1_by_region"] = strat
    print("[1a-iii] " + json.dumps(strat), flush=True)
    # --- 1b LOCO scoreboard
    loco = {}
    for src, dst, sd, dd in (("simplewiki", "pearltrees", cw, cp),
                             ("pearltrees", "simplewiki", cp, cw)):
        o = np.random.default_rng(SEED).permutation(len(sd["queries"]))
        m, w_ = gate_train(sd, torch.tensor(o[:300]))
        od = np.random.default_rng(SEED).permutation(len(dd["queries"]))
        hed = torch.tensor(od[300:])
        tpd = [int(dd["truepos"][i]) for i in hed.tolist()]
        nf = len(dd["tid_list"])
        loco[f"train_{src}_eval_{dst}"] = {
            "transfer_gate_MRR": round(metrics(ranks_from(
                gate_apply(m, w_, dd, hed), tpd))["MRR"], 4),
            "e5_only_MRR": round(metrics(ranks_from(dd["Cz"][hed], tpd))["MRR"], 4),
            "chance_MRR": round(sum(1.0 / r for r in range(1, nf + 1)) / nf, 4),
            "in_domain_gate_MRR": round(metrics(ranks_from(
                gate_apply(*gate_train(dd, torch.tensor(od[:300])), dd, hed), tpd))["MRR"], 4)}
    res["P1b_LOCO_scoreboard"] = loco
    print("[1b] " + json.dumps(loco), flush=True)
    # --- 1c corpus-identity probe on the learned representation
    def feats(d, ix):
        return torch.stack([d["Cz"][ix].flatten(), d["Sz"][ix].flatten(),
                            d["Az"][ix].flatten(), d["Ez"][ix].flatten()], -1)
    n = 20000
    Xp, Xw = feats(cp, he)[:n], feats(cw, torch.tensor(
        np.random.default_rng(SEED).permutation(len(cw["queries"]))[300:]))[:n]
    X = torch.cat([Xp, Xw]); y = torch.cat([torch.zeros(len(Xp)), torch.ones(len(Xw))]).long()
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]
    ntr = int(0.7 * len(X))
    clf = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    opt = torch.optim.Adam(clf.parameters(), lr=0.01)
    for _ in range(300):
        loss = torch.nn.functional.cross_entropy(clf(X[:ntr]), y[:ntr])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((clf(X[ntr:]).argmax(1) == y[ntr:]).float().mean())
    res["P1c_corpus_probe"] = {"pair_level_accuracy": round(acc, 4), "chance": 0.5,
                               "n_pairs": len(X),
                               "readout": "instrument for phase-2 tuning (target: -> chance)"}
    print("[1c] " + json.dumps(res["P1c_corpus_probe"]), flush=True)
    json.dump(res, open("PHASE1_AUDIT_SCOREBOARD.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)

run()
