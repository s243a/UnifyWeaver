#!/usr/bin/env python3
"""SimpleMind OOD arc — STAGE 1: measure the per-query gap field on the SM-FS SHADOW folds
ONLY (the reserve of 1,481 rows stays sealed; it is the transfer eval, touched later under
sol's spending protocol — NOT here).

For each shadow-fold query (mindmap title), rank the full candidate-folder catalog by e5
cosine and by the best mu channel (ELEM/HIER/SYM) using the current best trunk (replay-s3),
and record gap_q = clip(RR_e5 - RR_bestmu, 0). This is the OOD analog of the enwiki gap
field: where the trained model still loses to frozen e5 on SimpleMind filing. Cached for the
smear stage. Reports the distribution — if SM-FS gaps are broad (they should be: e5 was the
SM-FS champion at 0.573 while trained arms trailed), the field is informative and the arc
proceeds; if near-zero like enwiki, the front has moved and we say so."""
import os, sys, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from eval_filing import score_mu
from mu_attention import CORPORA, OPS, Tokenizer, build_e5_tables, E5_REVISION, JUDGES, NODETYPE

CK = os.path.expanduser(os.environ.get("SM_CK", "~/mu_data/enwiki_transitive_v1/trunk_stem_replay_s3997003.pt"))
OUT = os.path.expanduser("~/mu_data/sm_gap_v1")
CM, J, MM = CORPORA["mindmap"], JUDGES["graph"], NODETYPE["mindmap_node"]

def run():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, mode=0o700, exist_ok=True)
    titles = sh._titles()
    man, pairs, _ = pl.load_pairs_verified()
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    catalog = sorted({p["candidate"] for p in pairs})
    ci = {c: j for j, c in enumerate(catalog)}
    queries = sorted(dest_of)                      # shadow-fold queries only
    print(f"[s1] SM-FS shadow queries {len(queries)}, catalog {len(catalog)}", flush=True)
    qt, pt, ix = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                 model_revision=E5_REVISION)
    tok = Tokenizer(qt, pt, ix, {}, {})
    ck = pl.read_bound(os.path.join(pl.RANK_DIR, "init_seed3997001.pt"),
                       expect_sha=pl.INIT_SHA[3997001], private=True)
    model, _ = pl.load_checkpoint_bytes(ck, dev)
    model.load_state_dict(torch.load(CK, map_location=dev))
    model.eval()
    q_keys = queries
    f_keys = catalog
    ow = lambda op: torch.zeros(1, model.op_emb.weight.shape[0]).index_fill_(
        1, torch.tensor([OPS[op]]), 1.0)
    sm = lambda o: torch.tensor(score_mu(model, tok, ix, q_keys, f_keys, o, dev))
    E, A, S = sm(ow("ELEM")), sm(ow("HIER")), sm(ow("SYM"))
    C = (qt[[ix[k] for k in q_keys]] @ pt[[ix[k] for k in f_keys]].T).cpu()
    tp = torch.tensor([ci[dest_of[q]] for q in queries])
    def rr(M):
        out = []
        for r in range(M.shape[0]):
            s = M[r]
            gt = (s > s[tp[r]]).sum().item()
            eq = (s == s[tp[r]]).sum().item() - 1
            out.append(1.0 / (1 + gt + eq / 2.0))
        return np.array(out)
    rr_e5 = rr(C)
    rr_mu = np.maximum.reduce([rr(E), rr(A), rr(S)])
    gap = np.clip(rr_e5 - rr_mu, 0, None)
    d = {"queries": queries, "dest": {q: dest_of[q] for q in queries},
         "gap": gap, "qvec": qt[[ix[k] for k in q_keys]].numpy(),
         "e5_mrr": float(rr_e5.mean()), "bestmu_mrr": float(rr_mu.mean())}
    torch.save(d, f"{OUT}/{os.environ.get('SM_GAPOUT', 'gapfield.pt')}")
    print(json.dumps({
        "stamp": "shadow-exploratory-tier1-not-decision-bearing",
        "e5_mrr": round(d["e5_mrr"], 4), "bestmu_mrr": round(d["bestmu_mrr"], 4),
        "gap_mean": round(float(gap.mean()), 4), "gap_nonzero_frac": round(float((gap > 0).mean()), 4),
        "gap_p90": round(float(np.percentile(gap, 90)), 4),
        "verdict": ("INFORMATIVE — SM-FS gaps broad; smear stage proceeds"
                    if gap.mean() > 0.02 else
                    "NEAR-ZERO — front has moved off SM-FS too; report and reconsider"),
        "note": "reserve (1481 rows) NOT touched; shadow queries only"}, indent=1), flush=True)

run()
