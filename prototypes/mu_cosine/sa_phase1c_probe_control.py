#!/usr/bin/env python3
"""SHADOW: A1 — WITHIN-CORPUS TOPICAL CONTROL ON THE IDENTITY PROBE. Phase 2's go-ahead
rests on 'corpus identity is structural, not topical' (82.2% -> 80.8% topic-matched), but
that control compared corpora differing in EVERYTHING. wide_enwiki_{math,stats,cyber} are
the same corpus and construction differing essentially only in TOPIC, so a probe run
pairwise across them isolates the variable:
  probe >> chance  => the probe reads TOPIC; the 2-point estimate is wrong; phase-2 failure
                      mode is LIVE -> report and STOP before the sweep.
  probe ~ chance   => structural conclusion confirmed on the right control -> sweep proceeds.
Same probe as 1c/S3 (4 channel features per (query,candidate) pair, 16-unit MLP) and the
same cross-corpus probe re-run here as the positive control on identical machinery."""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch

REPO = "/home/s243a/Projects/UnifyWeaver"
SUBS = {"math": "wide_enwiki_math", "stats": "wide_enwiki_stats", "cyber": "wide_enwiki_cyber"}
SEED, MAXQ, CKPT = 7, 500, "model_pt_filing.pt"
CACHE = os.path.expanduser("~/mu_data/sa_scores_v2_%s.pt")

def nzrow(M):
    lo = M.min(dim=1, keepdim=True).values
    hi = M.max(dim=1, keepdim=True).values
    return (M - lo) / (hi - lo + 1e-9)

def build_graph_corpus(tag, graph_path, dev):
    cp = CACHE % tag
    if os.path.exists(cp):
        d = torch.load(cp, weights_only=False)
        if "Cz" in d:
            print(f"[cache] {tag}", flush=True)
            return d
    os.environ["UW_MU_GRAPH"] = graph_path
    import importlib
    import mu_attention, eval_filing
    importlib.reload(mu_attention); importlib.reload(eval_filing)
    from eval_filing import load_membership, score_mu
    from fine_tune_channel_heads import load_expanded
    from mu_attention import OPS, Tokenizer, build_e5_tables, E5_REVISION
    queries, cand = load_membership(graph_path, 3)
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
    d = dict(queries=queries, cand=cand, tid_list=tid_list, q_keys=q_keys, f_keys=f_keys,
             truepos=torch.tensor([tid_list.index(t) for _, t in queries]),
             Cz=nzrow(C), Sz=nzrow(S), Az=nzrow(A), Ez=nzrow(E),
             qtbl=qtbl, ptbl=ptbl, idx=idx)
    torch.save(d, cp)
    return d

def feats(d, cap=20000):
    o = np.random.default_rng(SEED).permutation(len(d["queries"]))
    ix = torch.tensor(o[300:]) if len(o) > 300 else torch.tensor(o)
    return torch.stack([d["Cz"][ix].flatten(), d["Sz"][ix].flatten(),
                        d["Az"][ix].flatten(), d["Ez"][ix].flatten()], -1)[:cap]

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
    return {"acc_mean": round(float(np.mean(accs)), 4), "acc_sd": round(float(np.std(accs)), 4),
            "n_pairs": 2 * n}

def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing",
           "question": "does the identity probe read TOPIC or STRUCTURE?"}
    subs = {}
    for tag, folder in SUBS.items():
        gp = os.path.join(REPO, "data", "benchmark", folder, "category_parent.tsv")
        subs[tag] = build_graph_corpus(f"enwiki_{tag}", gp, dev)
        print(f"[built] enwiki_{tag}: {len(subs[tag]['queries'])} q x "
              f"{len(subs[tag]['tid_list'])} folders", flush=True)
    F = {k: feats(v) for k, v in subs.items()}
    within = {}
    for a, b in (("math", "stats"), ("math", "cyber"), ("stats", "cyber")):
        within[f"{a}_vs_{b}"] = probe(F[a], F[b])
        print(f"[A1 within-corpus] {a} vs {b}: " + json.dumps(within[f"{a}_vs_{b}"]), flush=True)
    res["A1_within_corpus_topic_probe"] = within
    # positive control on identical machinery: the cross-corpus pair from phase 1
    from sa_phase1_shadow import build as build_p1
    cp, cw = build_p1("pearltrees", dev), build_p1("simplewiki", dev)
    res["A1_positive_control_pearltrees_vs_simplewiki"] = probe(feats(cp), feats(cw))
    print("[A1 cross-corpus control] " + json.dumps(
        res["A1_positive_control_pearltrees_vs_simplewiki"]), flush=True)
    mx = max(v["acc_mean"] for v in within.values())
    res["verdict"] = ("PROBE READS TOPIC — phase-2 failure mode LIVE, stop before sweep"
                      if mx > 0.65 else
                      "topic separation weak — structural conclusion confirmed; sweep proceeds")
    res["max_within_corpus_acc"] = mx
    json.dump(res, open("PHASE1C_PROBE_CONTROL.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)

run()
