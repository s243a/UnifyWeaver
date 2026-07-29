#!/usr/bin/env python3
"""SHADOW: SUBSAMPLE-N SCALING CURVE on the anchored bilinear CE-only head (the 0.552
recipe) — Kimi's cheap instrument for sol's crossover-n question, now running on validated
inputs (title-source settled: filename stems). Trains the identical recipe on n_train in
{36, 72, 144, 288} queries per fold (deterministic subsample, seed-fixed shuffle), 5 folds.
The head starts AT the e5 floor by construction, so the curve reads directly as 'how many
filing decisions before the learned correction stops hurting' — extrapolate toward the
crossover where it starts helping."""
import json, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from sm_fs_bilinear_ceonly_shadow import BilinearHead
from mu_attention import E5_REVISION, build_e5_tables

SEED, STEPS_PER_Q, SLATE, D = 3997001, 8, 32, 384  # epoch-matched: steps = 8*n (2400/288 ratio held)
SIZES = (36, 72, 144, 288)

def main():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    titles = sh._titles()
    qt, pt, ix = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                 model_revision=E5_REVISION)
    Q = torch.tensor(qt.numpy(), device=dev)
    P = torch.tensor(pt.numpy(), device=dev)
    man, pairs, _ = pl.load_pairs_verified()
    catalog = sorted({p["candidate"] for p in pairs})
    ci = {c: j for j, c in enumerate(catalog)}
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    held = defaultdict(list)
    for p in pairs:
        held[p["fold"]].append(p)
    cat_ids = torch.tensor([ix[c] for c in catalog], device=dev)
    res = defaultdict(dict)
    for f in range(5):
        rows = sh._projection(f)
        pos, buckets = defaultdict(list), defaultdict(lambda: defaultdict(list))
        for i, p in enumerate(rows):
            (pos[p["query"]].append(i) if p["class"].startswith("positive")
             else buckets[p["query"]][p["hardness"]].append(i))
        dest_idx = {}
        for q, idxs in pos.items():
            for i in idxs:
                if rows[i]["candidate"] == dest_of[q]:
                    dest_idx[q] = i
        full_q = sorted(q for q in pos if q in dest_idx)
        order = np.random.default_rng(SEED + 7).permutation(len(full_q))
        for n in SIZES:
            train_q = [full_q[i] for i in order[:min(n, len(full_q))]]
            torch.manual_seed(SEED)
            rng = np.random.default_rng(SEED)
            model = BilinearHead().to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=5e-5)
            t0 = time.time()
            for step in range(STEPS_PER_Q * len(train_q)):
                q = train_q[rng.integers(len(train_q))]
                negs = []
                for b, k in (("hard", 12), ("medium", 10), ("easy", 9)):
                    mem = buckets[q].get(b, [])
                    if mem:
                        negs += [mem[i] for i in rng.integers(0, len(mem), min(k, len(mem)))]
                slate = [dest_idx[q]] + negs[:SLATE - 1]
                qv = Q[ix[q]].expand(len(slate), D)
                cv = P[torch.tensor([ix[rows[i]["candidate"]] for i in slate], device=dev)]
                cond = torch.zeros(len(slate), dtype=torch.long, device=dev)
                s = model(qv, cv, cond)
                loss = torch.nn.functional.cross_entropy(
                    (s * 8).unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval()
            rrs = []
            with torch.no_grad():
                hq = sorted({p["query"] for p in held[f]})
                cv_all = P[cat_ids]
                for q in hq:
                    s = model(Q[ix[q]].expand(len(catalog), D), cv_all,
                              torch.zeros(len(catalog), dtype=torch.long, device=dev))
                    rrs.append(1.0 / pl.recompute_rank([float(v) for v in s.cpu()],
                                                       catalog, ci[dest_of[q]]))
            res[f"n{n}"][f] = float(np.mean(rrs))
            print(f"[scale] fold {f} n={n}: MRR {res[f'n{n}'][f]:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(json.dumps({k: float(np.mean(list(v.values()))) for k, v in res.items()} |
                     {"comparators": {"e5_frozen": 0.573,
                                      "anchored_ce_full288_prior": 0.552}}, indent=1),
          flush=True)

main()
