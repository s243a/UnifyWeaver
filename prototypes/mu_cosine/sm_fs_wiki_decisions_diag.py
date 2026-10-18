#!/usr/bin/env python3
"""SHADOW: WIKI-SCALE FILING DECISIONS on the anchored head (cross-corpus lever 1) — the
SM-FS verdict was 'no correction beats e5 at <=288 decisions/fold; crossover-n out of
sight'. This is the crossover test at scale: 30,000 direct-parent decisions (targets.tsv
pos rows with target 1.0) against a 19,838-category catalog. Node-disjoint 27k/3k split.
Anchored bilinear CE-only (identity-frozen base + low-rank delta), hard negatives = top-32
cosine categories per train node. Reports: (a) wiki-held full-catalog MRR vs the wiki e5
floor at 20k/40k/60k steps — does the correction EVER cross its floor when decisions are
plentiful; (b) zero-shot SM-FS transfer of the wiki-trained delta (corpus-agnostic head).
Ranks are strictly-greater torch ranks (ties broken optimistically; e5 cos ties negligible).
Stamped shadow-exploratory; SM-FS reserve untouched."""
import json, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from sm_fs_bilinear_ceonly_shadow import BilinearHead
from mu_attention import E5_REVISION, build_e5_tables

OUT = os.path.expanduser("~/mu_data/wiki_lineage_v1")
SEED, SLATE, D = 3997001, 32, 384
STEPS, EVAL_AT = 16000, (500, 1000, 2000, 4000, 8000, 16000)
N_HELD, N_HARD = 3000, 32

def full_catalog_mrr(head, Q, P, q_ids, dest_col, cat_ids, dev, chunk=256):
    ranks = []
    with torch.no_grad():
        cv = P[cat_ids]                                   # [C, D]
        cB = cv @ head.B                                  # [C, R]
        for s in range(0, len(q_ids), chunk):
            qs = Q[q_ids[s:s + chunk]]                    # [b, D]
            base = qs @ cv.T
            delta = (qs @ head.A) @ cB.T
            sc = base + delta
            dcol = dest_col[s:s + chunk]
            dsc = sc.gather(1, dcol.unsqueeze(1))
            ranks.append((sc > dsc).sum(1) + 1)
    r = torch.cat(ranks).float()
    return float((1.0 / r).mean())

def run():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ln in open(os.path.join(OUT, "targets.tsv"), encoding="utf-8"):
        if ln.startswith("#"):
            continue
        n, a, t, kind = ln.rstrip("\n").split("\t")
        if kind == "pos" and t == "1.0":
            rows.append((n, a))
    names = sorted({x for r in rows for x in r})
    qtbl, ptbl, ix = build_e5_tables(names, cache_path=os.path.join(OUT, "wiki_e5.pt"),
                                     batch_size=512, model_revision=E5_REVISION)
    Q = qtbl.to(dev); P = ptbl.to(dev)
    catalog = sorted({a for _, a in rows})
    ci = {c: j for j, c in enumerate(catalog)}
    cat_ids = torch.tensor([ix[c] for c in catalog], device=dev)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    held = [rows[i] for i in order[:N_HELD]]
    train = [rows[i] for i in order[N_HELD:]]
    print(f"decisions train {len(train)} held {len(held)} catalog {len(catalog)}", flush=True)
    hq_ids = torch.tensor([ix[n] for n, _ in held], device=dev)
    hd_col = torch.tensor([ci[a] for _, a in held], device=dev)
    tq_ids = torch.tensor([ix[n] for n, _ in train], device=dev)
    td_col = torch.tensor([ci[a] for _, a in train], device=dev)
    # e5 floor on wiki held
    ident = BilinearHead().to(dev)          # delta is exactly zero at init
    floor = full_catalog_mrr(ident, Q, P, hq_ids, hd_col, cat_ids, dev)
    print(f"[wiki] e5 floor (held): {floor:.4f}", flush=True)
    # hard negatives: top-N_HARD cosine categories per train node (excluding the destination)
    cv = P[cat_ids]
    hard = torch.empty(len(train), N_HARD, dtype=torch.long)
    with torch.no_grad():
        for s in range(0, len(train), 1024):
            sc = Q[tq_ids[s:s + 1024]] @ cv.T
            sc.scatter_(1, td_col[s:s + 1024].unsqueeze(1), -1e9)
            hard[s:s + 1024] = sc.topk(N_HARD, dim=1).indices.cpu()
    hard = hard.numpy()
    res = {"e5_floor_wiki_held": floor}
    for tag, lr, wd in (("lr5e5_wd0", 5e-5, 0.0), ("lr5e5_wd01", 5e-5, 0.1),
                        ("lr1e5_wd01", 1e-5, 0.1)):
      torch.manual_seed(SEED)
      rng = np.random.default_rng(SEED)
      head = BilinearHead().to(dev)
      opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
      t0 = time.time()
      for step in range(1, STEPS + 1):
          i = int(rng.integers(len(train)))
          negs = list(hard[i][rng.integers(0, N_HARD, 16)]) + \
                 list(rng.integers(0, len(catalog), SLATE - 17))
          cols = [int(td_col[i])] + [int(x) for x in negs]
          qv = Q[tq_ids[i]].expand(len(cols), D)
          cvs = cv[torch.tensor(cols, device=dev)]
          cond = torch.zeros(len(cols), dtype=torch.long, device=dev)
          s = head(qv, cvs, cond)
          loss = torch.nn.functional.cross_entropy(
              (s * 8).unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
          opt.zero_grad(); loss.backward()
          torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
          if step in EVAL_AT:
              m = full_catalog_mrr(head, Q, P, hq_ids, hd_col, cat_ids, dev)
              dn = float((head.A @ head.B.T).norm())
              res[f"{tag}_step{step}"] = m
              print(f"[wiki] {tag} step {step}: held MRR {m:.4f} deltaN {dn:.3f} "
                    f"(floor {floor:.4f}, {(time.time()-t0)/60:.1f} min)", flush=True)
    # zero-shot SM-FS transfer of the wiki-trained delta
    titles = sh._titles()
    sqt, spt, six = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                    model_revision=E5_REVISION)
    SQ = sqt.to(dev); SP = spt.to(dev)
    man, pairs, _ = pl.load_pairs_verified()
    scat = sorted({p["candidate"] for p in pairs})
    sci = {c: j for j, c in enumerate(scat)}
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    queries = sorted(dest_of)
    sq_ids = torch.tensor([six[q] for q in queries], device=dev)
    sd_col = torch.tensor([sci[dest_of[q]] for q in queries], device=dev)
    scat_ids = torch.tensor([six[c] for c in scat], device=dev)
    res["smfs_zeroshot_last_cfg"] = full_catalog_mrr(head, SQ, SP, sq_ids, sd_col,
                                                      scat_ids, dev)
    res["smfs_e5_floor_same_ranker"] = full_catalog_mrr(ident, SQ, SP, sq_ids, sd_col,
                                                        scat_ids, dev)
    print(json.dumps(res | {"stamp": "shadow-exploratory-tier1-not-decision-bearing",
                            "comparators": {"smfs_e5_recompute_rank": 0.573,
                                            "smfs_anchored_ce_n288": 0.552}},
                     indent=1), flush=True)

run()
