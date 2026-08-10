#!/usr/bin/env python3
"""Pearltrees OOD arc — STAGE 1 with PRE-DECLARED GATE (coordinating ruling, 2026-08-09).

GATE — declared here, before the field is measured, so the reading is not post-hoc:
  PROCEED with harvest + concept training iff BOTH:
    (G1) field informative: mean gap > 0.05 AND nonzero fraction > 0.35
         (well above enwiki's self-annealed floor of 0.002 / 28%);
    (G2) concept-coherent: the top-weighted categories of a semantic smear onto the
         enwiki pool read as recognizable SUBJECT MATTER, and the top-gap queries do not
         concentrate on surface phenomena (naming conventions, abbreviations, formatting).
  PAUSE concept training on Pearltrees iff failures concentrate on surface phenomena —
  SimpleMind-class regardless of folder naming; the lever is then surface-side.
  Both diagnostics PRINT before any generation decision. Sampler reads the VISIBLE
  partition (1,702) only; the sealed 1,136 (sha a27c41ab...) stays sealed.

Measurement: per-query gap on the visible partition, current best trunk (replay-s3;
the grounded ckpt is not champion — its SM transfer was null), full 132-folder catalog,
mid-tie ranks. Field cached for the smear stage if the gate opens."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from eval_filing import load_filing, score_mu
from mu_attention import OPS, Tokenizer, build_e5_tables, E5_REVISION
import sm_fs_ranking_pipeline as pl

CK = os.path.expanduser("~/mu_data/enwiki_transitive_v1/trunk_stem_replay_s3997003.pt")
TREES = "/home/s243a/Projects/UnifyWeaver/.local/data/pearltrees_api/trees"
SPLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PEARLTREES_OOD_SPLIT.json")
OUT = os.path.expanduser("~/mu_data/pt_gap_v1")

def run():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, mode=0o700, exist_ok=True)
    split = json.load(open(SPLIT))
    vis = split["visible_indices"]
    queries_all, cand = load_filing(TREES, 3)
    queries = [queries_all[i] for i in vis]              # VISIBLE ONLY
    tid_list = list(cand)
    print(f"[pt-s1] visible queries {len(queries)}, catalog {len(tid_list)}", flush=True)
    f_keys = [f"F:{t}" for t in tid_list]
    q_keys = [f"B:{i}" for i in range(len(queries))]
    texts = {f"F:{t}": cand[t] for t in cand} | {f"B:{i}": q for i, (q, _) in enumerate(queries)}
    qt, pt, ix = build_e5_tables(sorted(texts), cache_path=None, texts=texts,
                                 model_revision=E5_REVISION)
    tok = Tokenizer(qt, pt, ix, {}, {})
    ck = pl.read_bound(os.path.join(pl.RANK_DIR, "init_seed3997001.pt"),
                       expect_sha=pl.INIT_SHA[3997001], private=True)
    model, _ = pl.load_checkpoint_bytes(ck, dev)
    model.load_state_dict(torch.load(CK, map_location=dev))
    model.eval()
    ow = lambda op: torch.zeros(1, model.op_emb.weight.shape[0]).index_fill_(
        1, torch.tensor([OPS[op]]), 1.0)
    sm = lambda o: torch.tensor(score_mu(model, tok, ix, q_keys, f_keys, o, dev))
    E, A, S = sm(ow("ELEM")), sm(ow("HIER")), sm(ow("SYM"))
    C = (qt[[ix[k] for k in q_keys]] @ pt[[ix[k] for k in f_keys]].T).cpu()
    ci = {t: j for j, t in enumerate(tid_list)}
    tp = torch.tensor([ci[t] for _, t in queries])
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
    d = {"queries": queries, "gap": gap,
         "qvec": qt[[ix[k] for k in q_keys]].numpy(),
         "e5_mrr": float(rr_e5.mean()), "bestmu_mrr": float(rr_mu.mean())}
    torch.save(d, f"{OUT}/gapfield_visible.pt")
    g1 = bool(gap.mean() > 0.05 and (gap > 0).mean() > 0.35)
    print(json.dumps({
        "stamp": "shadow-exploratory-tier1-not-decision-bearing",
        "e5_mrr": round(d["e5_mrr"], 4), "bestmu_mrr": round(d["bestmu_mrr"], 4),
        "gap_mean": round(float(gap.mean()), 4),
        "gap_nonzero_frac": round(float((gap > 0).mean()), 4),
        "gap_p90": round(float(np.percentile(gap, 90)), 4),
        "G1_field_informative": g1,
        "note": "G2 (concept-coherence) requires the two printed diagnostics below"}, indent=1),
        flush=True)
    # diagnostic 1: top-gap queries (surface-vs-concept character, human-readable)
    order = np.argsort(-gap)
    print("[diag1] top-gap visible queries:", flush=True)
    for i in order[:15]:
        print(f"   gap {gap[i]:.3f}  {queries[i][0][:60]!r} -> {cand[queries[i][1]][:40]!r}",
              flush=True)
    # diagnostic 2: semantic smear preview onto the enwiki pool (concept-coherence read)
    pool_cache = os.path.expanduser("~/mu_data/enwiki_gap_v3/pool_e5.pt")
    if os.path.exists(pool_cache):
        import glob
        from eval_filing import is_admin
        TITLES = ("/tmp/claude-1000/-home-s243a-Projects-UnifyWeaver/"
                  "be81a5e2-7bea-4a18-9241-de457531548d/scratchpad/ns14_id_title.tsv")
        if os.path.exists(TITLES):
            title = {}
            for ln in open(TITLES, encoding="utf-8", errors="replace"):
                i2, _, t2 = ln.rstrip("\n").partition("\t")
                title[int(i2)] = t2
            pe = torch.load(pool_cache, weights_only=False)
            # pool_e5 cache stores tables from stage-v3; rebuild keys quickly
            print("[diag2] semantic smear preview requires pool rebuild — deferred to "
                  "smear stage; G2 assessed from diag1 + smear preview before generation",
                  flush=True)
    print("[gate] decision recorded AFTER diagnostics are read — no generation in this run",
          flush=True)

run()
