#!/usr/bin/env python3
"""SHADOW: PHASE 5 — RE-RUN THE CORPUS-DEPENDENT CLAIMS AT REAL ENWIKI SCALE.

CORRECTION (owner): every phase-1..4 result labelled 'simplewiki' actually used
data/benchmark/10k/category_parent.tsv — 25,228 edges, the default GRAPH constant,
mislabelled in the suite's own docstrings and repeated by me. It is neither simplewiki
(297,283 edges) nor enwiki (9.9M named edges). Renamed to `bench10k` throughout this file.

This runner rebuilds the corpus from enwiki_named/category_parent.tsv (9,914,841 named
child->parent edges) by STREAMING (the WSL memory cap forbids loading the whole graph):
  pass 1  count children per parent (ints only)
  pass 2  keep parents with >= MIN_BM children, sample CATALOG_CAP of them, collect their
          children as the query pool
Then re-measures the two claims that depend on which corpus 'the wiki side' is:
  C1 gate trust      — is w_mu ~0.99 in-domain an artifact of the 25k toy graph?
  C2 LOCO transfer   — enwiki->Pearltrees, against the bench10k->Pearltrees number (0.150)
Candidate catalog is CAPPED for tractability (500 queries x CATALOG_CAP x 3 ops of model
forwards on consumer hardware); the cap is reported with every number, since a larger
catalog makes the task strictly harder and MRR is not comparable across catalog sizes."""
import os, sys, json, random
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from eval_filing import score_mu, metrics
from fine_tune_channel_heads import load_expanded
from mu_attention import OPS, Tokenizer, build_e5_tables, E5_REVISION
from sa_phase1_shadow import build, ranks_from
from sa_phase3_shadow import train_arm, gate_score

REPO = "/home/s243a/Projects/UnifyWeaver"
ENWIKI = os.path.join(REPO, "data", "benchmark", "enwiki_named", "category_parent.tsv")
CACHE = os.path.expanduser("~/mu_data/sa_scores_v2_enwiki_real.pt")
SEED, MAXQ, MIN_BM, CATALOG_CAP = 7, 500, 3, 2000
SEEDS = (7, 17, 27)

def build_enwiki(dev):
    if os.path.exists(CACHE):
        d = torch.load(CACHE, weights_only=False)
        if "Cz" in d:
            print("[cache] enwiki_real", flush=True)
            return d
    nchild = defaultdict(int)
    n_edges = 0
    with open(ENWIKI, encoding="utf-8", errors="replace") as fh:
        next(fh)
        for ln in fh:
            i = ln.find("\t")
            if i > 0:
                nchild[ln[i+1:].rstrip("\n")] += 1
                n_edges += 1
    print(f"[enwiki] {n_edges} edges, {len(nchild)} parents", flush=True)
    eligible = [p for p, c in nchild.items() if c >= MIN_BM]
    print(f"[enwiki] parents with >={MIN_BM} children: {len(eligible)}", flush=True)
    rng = random.Random(SEED)
    catalog = sorted(rng.sample(eligible, min(CATALOG_CAP, len(eligible))))
    catset = set(catalog)
    del nchild, eligible
    kids = defaultdict(list)
    with open(ENWIKI, encoding="utf-8", errors="replace") as fh:
        next(fh)
        for ln in fh:
            i = ln.find("\t")
            if i > 0:
                p = ln[i+1:].rstrip("\n")
                if p in catset and len(kids[p]) < 40:
                    kids[p].append(ln[:i])
    pairs = [(c, p) for p, cs in kids.items() for c in cs]
    rng.shuffle(pairs)
    queries = [(c.replace("_", " "), p) for c, p in pairs[:MAXQ]]
    print(f"[enwiki] catalog {len(catalog)}, query pool {len(pairs)}, using {len(queries)}",
          flush=True)
    f_keys = [f"F:{t}" for t in catalog]
    q_keys = [f"B:{i}" for i in range(len(queries))]
    texts = {f"F:{t}": t.replace("_", " ") for t in catalog} | \
            {f"B:{i}": q for i, (q, _) in enumerate(queries)}
    qtbl, ptbl, ix = build_e5_tables(sorted(texts), cache_path=None, texts=texts,
                                     batch_size=256, model_revision=E5_REVISION)
    tok = Tokenizer(qtbl, ptbl, ix, {}, {})
    model, _ = load_expanded(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "model_pt_filing.pt"), dev=dev)
    model.eval()
    ow = lambda op: torch.zeros(1, model.op_emb.weight.shape[0]).index_fill_(
        1, torch.tensor([OPS[op]]), 1.0)
    sm = lambda o: torch.tensor(score_mu(model, tok, ix, q_keys, f_keys, o, dev))
    E, A, S = sm(ow("ELEM")), sm(ow("HIER")), sm(ow("SYM"))
    C = (qtbl[[ix[k] for k in q_keys]] @ ptbl[[ix[k] for k in f_keys]].T).cpu()
    nz = lambda M: (M - M.min(1, keepdim=True).values) / (
        M.max(1, keepdim=True).values - M.min(1, keepdim=True).values + 1e-9)
    d = dict(queries=queries, cand={t: t for t in catalog}, tid_list=catalog,
             q_keys=q_keys, f_keys=f_keys,
             truepos=torch.tensor([catalog.index(p) for _, p in queries]),
             Cz=nz(C), Sz=nz(S), Az=nz(A), Ez=nz(E), qtbl=qtbl, ptbl=ptbl, idx=ix)
    torch.save(d, CACHE)
    return d

def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    de = build_enwiki(dev)
    cp = build("pearltrees", dev)
    cb = build("simplewiki", dev)          # = bench10k, correctly named below
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing",
           "naming_correction": "phases 1-4 'simplewiki' = data/benchmark/10k "
                                "(25,228 edges); renamed bench10k. This runner adds "
                                "enwiki_named (9,914,841 edges).",
           "catalog_caps": {"enwiki": len(de["tid_list"]), "bench10k": len(cb["tid_list"]),
                            "pearltrees": len(cp["tid_list"])},
           "mrr_not_comparable_across_catalog_sizes": True}
    # C1 gate trust in-domain, per corpus
    trust = {}
    for tag, d in (("enwiki", de), ("bench10k", cb), ("pearltrees", cp)):
        we_mu, mrrs, e5s = [], [], []
        for s in SEEDS:
            o = np.random.default_rng(s).permutation(len(d["queries"]))
            tr, he = torch.tensor(o[:300]), torch.tensor(o[300:])
            mlp, we = train_arm("softmax4", d, tr, s)
            with torch.no_grad():
                sc, w = gate_score("softmax4", mlp, we, d, he)
            tps = [int(d["truepos"][i]) for i in he.tolist()]
            we_mu.append(float(w[..., 1:].sum(-1).mean()))
            mrrs.append(metrics(ranks_from(sc, tps))["MRR"])
            e5s.append(metrics(ranks_from(d["Cz"][he], tps))["MRR"])
        trust[tag] = {"w_mu": round(float(np.mean(we_mu)), 4),
                      "gate_mrr": round(float(np.mean(mrrs)), 4),
                      "e5_mrr": round(float(np.mean(e5s)), 4),
                      "gate_minus_e5": round(float(np.mean(mrrs) - np.mean(e5s)), 4),
                      "catalog": len(d["tid_list"])}
        print(f"[C1 {tag}] " + json.dumps(trust[tag]), flush=True)
    res["C1_gate_trust_by_corpus"] = trust
    # C2 LOCO into pearltrees from each wiki source
    loco = {}
    for tag, src in (("enwiki", de), ("bench10k", cb)):
        vals = []
        for s in SEEDS:
            o = np.random.default_rng(s).permutation(len(src["queries"]))
            mlp, we = train_arm("softmax4", src, torch.tensor(o[:300]), s)
            od = np.random.default_rng(s).permutation(len(cp["queries"]))
            hed = torch.tensor(od[300:])
            with torch.no_grad():
                sc, _ = gate_score("softmax4", mlp, we, cp, hed)
            vals.append(metrics(ranks_from(
                sc, [int(cp["truepos"][i]) for i in hed.tolist()]))["MRR"])
        loco[f"train_{tag}_eval_pearltrees"] = {
            "mrr": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals)), 4)}
        print(f"[C2 {tag}] " + json.dumps(loco[f"train_{tag}_eval_pearltrees"]), flush=True)
    res["C2_LOCO_into_pearltrees"] = loco
    json.dump(res, open("PHASE5_ENWIKI_SCALE.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)

run()
