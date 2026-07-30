#!/usr/bin/env python3
"""SHADOW: FULL-CAPACITY listwise rank-CE (all 4.0M params trainable, LR 2e-4) — the owner's framing:
'a function trained on cross entropy based on sorting by mu — one more function it learns
to generalize across.' Slate CE is invariant to target compression (the e5 calibration issue),
so it optimizes ordering directly. Two arms per fold (seed 3997001):
  R1 rank-CE alone; R2 rank-CE mixed 50/50 with the graded LINEAGE regression channel."""
import copy, io, json, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from fine_tune_channel_heads import mu_batch
from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, build_e5_tables

SEED, STEPS, SLATE = 3997001, 1600, 32
CM, J, MM = CORPORA["mindmap"], JUDGES["graph"], NODETYPE["mindmap_node"]

def run():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    titles = sh._titles()
    qt, pt, ix = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                 model_revision=E5_REVISION)
    tok = Tokenizer(qt, pt, ix, {}, {})
    man, pairs, _ = pl.load_pairs_verified()
    catalog = sorted({p["candidate"] for p in pairs})
    ci = {c: j for j, c in enumerate(catalog)}
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    held = defaultdict(lambda: defaultdict(dict))
    for p in pairs:
        held[p["fold"]][p["query"]][p["candidate"]] = p
    res = defaultdict(dict)
    for arm in ("full_rank_ce", "full_rank_ce_plus_graded"):
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
            train_q = sorted(q for q in pos if q in dest_idx)
            torch.manual_seed(SEED); aug = np.random.default_rng(SEED + 1)
            rng = np.random.default_rng(SEED)
            ck = pl.read_bound(os.path.join(pl.RANK_DIR, f"init_seed{SEED}.pt"),
                               expect_sha=pl.INIT_SHA[SEED], private=True)
            model, cfg = pl.load_checkpoint_bytes(ck, dev)
            ref = copy.deepcopy(model); ref.eval()
            for p in ref.parameters(): p.requires_grad = False
            for p in model.parameters():
                p.requires_grad = True          # FULL capacity: residual paths
            tensors = list(model.parameters())  # feed gradient to lower layers
            opt = torch.optim.Adam(tensors, lr=2e-4)
            model.train(); t0 = time.time()
            for step in range(STEPS):
                if arm == "full_rank_ce_plus_graded" and step % 2 == 1:
                    c, k = pl.sample_step(f, SEED, step, train_q, pos, buckets, "graded_negative")
                    pl.fit_one_step(model, ref, tensors, opt, tok, rows, c, k, aug, dev)
                    continue
                q = train_q[rng.integers(len(train_q))]
                negs = []
                for b, n in (("hard", 12), ("medium", 10), ("easy", 9)):
                    mem = buckets[q].get(b, [])
                    if mem:
                        negs += [mem[i] for i in rng.integers(0, len(mem), min(n, len(mem)))]
                slate = [dest_idx[q]] + negs[:SLATE - 1]
                items = [(rows[i]["query"], rows[i]["candidate"], OPS["LINEAGE_RANK"],
                          CM, J, MM, MM) for i in slate]
                mu = mu_batch(model, tok, items, dev, train=True, rng=aug)
                loss = torch.nn.functional.cross_entropy(
                    (mu * 8.0).unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
                ag = [(it[0], it[1], it[2]) for it in items[:8]]
                mu_ag = mu_batch(model, tok, ag, dev)
                with torch.no_grad():
                    mu_ref = mu_batch(ref, tok, ag, dev)
                loss = loss + torch.mean((mu_ag - mu_ref) ** 2)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(tensors, 1.0); opt.step()
            model.eval()
            rrs = []
            with torch.no_grad():
                for q in sorted(held[f]):
                    items = [(q, c, OPS["LINEAGE_RANK"], CM, J, MM, MM) for c in catalog]
                    mu = [float(v) for v in mu_batch(model, tok, items, dev).cpu()]
                    rrs.append(1.0 / pl.recompute_rank(mu, catalog, ci[dest_of[q]]))
            res[arm][f] = float(np.mean(rrs))
            print(f"[rank-ce] {arm} fold {f}: MRR {res[arm][f]:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(json.dumps({a: float(np.mean(list(v.values()))) for a, v in res.items()} |
                     {"comparators": {"graded_mse": 0.280, "partial_rank_ce": 0.291, "e5_frozen": 0.573}}, indent=1),
          flush=True)

run()
