#!/usr/bin/env python3
"""SHADOW: PHASE 6 — the two transfer readouts for the STEM multi-op checkpoint.
The within-subtree 1.59x result is within-distribution FOR mu and within-blind-spot FOR e5
(the most favorable framing — stated per the coordinating ruling; it bounds the claim, and
it confirms the e5 bar is REGIONAL: 0.87-0.96 mixed-region vs 0.234 within-subtree, i.e.
0.324 was the average bar, not the bar). The real test:
  R1  re-score S/A/E channels on the UNCHANGED enwiki_real eval (mixed-region, 500q x 2000
      cats) with the new trunk; per-region e5-vs-mu gaps before/after — does stem_computing
      (+0.566, the largest) close?
  R2  gate-changes-its-mind: retrain softmax4 gates on both channel sets (old model / new
      model scores); does the gate shift weight toward mu in stem regions specifically?
Same queries, catalog, folds, ranks — only the channel scorer changes."""
import os, sys, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
from eval_filing import score_mu, metrics
from fine_tune_channel_heads import mu_batch
from mu_attention import OPS, Tokenizer, build_e5_tables, E5_REVISION
from sa_phase1_shadow import ranks_from, REGIONS, region_of
from sa_phase3_shadow import train_arm, gate_score

CK = os.path.expanduser("~/mu_data/enwiki_transitive_v1/trunk_stem_multiop.pt")
OLD = os.path.expanduser("~/mu_data/sa_scores_v2_enwiki_real.pt")
NEW = os.path.expanduser("~/mu_data/sa_scores_v2_enwiki_real_stemtrunk.pt")
SEED = 7

def nzrow(M):
    lo = M.min(dim=1, keepdim=True).values
    hi = M.max(dim=1, keepdim=True).values
    return (M - lo) / (hi - lo + 1e-9)

def run():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(OLD, weights_only=False)
    if os.path.exists(NEW):
        dn = torch.load(NEW, weights_only=False)
    else:
        ck = pl.read_bound(os.path.join(pl.RANK_DIR, "init_seed3997001.pt"),
                           expect_sha=pl.INIT_SHA[3997001], private=True)
        model, _ = pl.load_checkpoint_bytes(ck, dev)
        model.load_state_dict(torch.load(CK, map_location=dev))
        model.eval()
        tok = Tokenizer(d["qtbl"], d["ptbl"], d["idx"], {}, {})
        ow = lambda op: torch.zeros(1, model.op_emb.weight.shape[0]).index_fill_(
            1, torch.tensor([OPS[op]]), 1.0)
        sm = lambda o: torch.tensor(score_mu(model, tok, d["idx"], d["q_keys"],
                                             d["f_keys"], o, dev))
        E, A, S = sm(ow("ELEM")), sm(ow("HIER")), sm(ow("SYM"))
        dn = dict(d)
        dn["Sz"], dn["Az"], dn["Ez"] = nzrow(S), nzrow(A), nzrow(E)
        torch.save(dn, NEW)
    rnames = list(REGIONS)
    # region vectors built independently — the cached tables lack R: keys
    rtex = {f"R:{r}": v for r, v in REGIONS.items()} | \
           {f"F:{t}": t.replace("_", " ") for t in d["tid_list"]}
    _, rpt, rix = build_e5_tables(sorted(rtex), cache_path=None, texts=rtex,
                                  model_revision=E5_REVISION)
    V = torch.stack([rpt[rix[f"R:{r}"]] for r in rnames])
    freg = [rnames[int(torch.argmax(V @ rpt[rix[f"F:{t}"]]))] for t in d["tid_list"]]
    qreg = [freg[int(d["truepos"][i])] for i in range(len(d["queries"]))]
    o = np.random.default_rng(SEED).permutation(len(d["queries"]))
    tr, he = torch.tensor(o[:300]), torch.tensor(o[300:])
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing",
           "frame": "eval unchanged (mixed-region enwiki_real); only channel scorer swapped"}
    out = {}
    for tag, dd in (("old_trunk", d), ("stem_trunk", dn)):
        mlp, we = train_arm("softmax4", dd, tr, SEED)
        with torch.no_grad():
            sc, w = gate_score("softmax4", mlp, we, dd, he)
        rows = {}
        for r in rnames:
            sel = [k for k, i in enumerate(he.tolist()) if qreg[i] == r]
            if len(sel) < 8:
                continue
            ixs = he[torch.tensor(sel)]
            tps = [int(dd["truepos"][i]) for i in ixs.tolist()]
            m = {k: round(metrics(ranks_from(dd[key][ixs], tps))["MRR"], 4)
                 for k, key in (("e5", "Cz"), ("S", "Sz"), ("A", "Az"), ("E", "Ez"))}
            wsel = w[torch.tensor(sel)]
            rows[r] = m | {"n": len(sel),
                           "gap": round(m["e5"] - max(m["S"], m["A"], m["E"]), 4),
                           "gate": round(metrics(ranks_from(sc[torch.tensor(sel)],
                                                            tps))["MRR"], 4),
                           "w_mu": round(float(wsel[..., 1:].sum(-1).mean()), 4)}
        out[tag] = rows
        print(f"[{tag}] " + json.dumps(rows), flush=True)
    res["R1_R2_by_region"] = out
    delta = {r: {"gap_delta": round(out["stem_trunk"][r]["gap"] - out["old_trunk"][r]["gap"], 4),
                 "w_mu_delta": round(out["stem_trunk"][r]["w_mu"] - out["old_trunk"][r]["w_mu"], 4),
                 "gate_delta": round(out["stem_trunk"][r]["gate"] - out["old_trunk"][r]["gate"], 4)}
             for r in out["old_trunk"] if r in out["stem_trunk"]}
    res["deltas"] = delta
    print("[deltas] " + json.dumps(delta, indent=1), flush=True)
    json.dump(res, open("PHASE6_TRANSFER_READOUT.json", "w"), indent=1)

run()
