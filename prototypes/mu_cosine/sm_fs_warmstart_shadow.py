#!/usr/bin/env python3
"""SHADOW: E5-IMITATION WARM-START (unlocked by the sigmoid diagnostic) — phase 1 distills
the attention stack toward e5's own top-1 on random slates (free supervision, de-sigmoided
CE over logit(mu)); phase 2 fine-tunes on true destinations (rank-CE + graded channel, also
de-sigmoided). The attention-stack analog of the anchored-bilinear 'start at the floor'
trick: from-scratch training caps ~0.35 regardless of loss space, distillation alone reaches
~0.355 fold-0 — question is whether imitation + truth composes above both. Reports the
phase-1 (distill-only) MRR per fold as its own arm. 5 folds, seed 3997001."""
import copy, json, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from fine_tune_channel_heads import mu_batch
from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, build_e5_tables

SEED, STEPS, SLATE = 3997001, 1600, 32
CM, J, MM = CORPORA["mindmap"], JUDGES["graph"], NODETYPE["mindmap_node"]

def logit_ce(mu, label, dev):
    return torch.nn.functional.cross_entropy(
        torch.logit(mu.clamp(1e-6, 1 - 1e-6)).unsqueeze(0),
        torch.tensor([label], dtype=torch.long, device=dev))

def evaluate(model, tok, held_f, catalog, ci, dest_of, dev):
    model.eval(); rrs = []
    with torch.no_grad():
        for q in sorted(held_f):
            items = [(q, c, OPS["LINEAGE_RANK"], CM, J, MM, MM) for c in catalog]
            mu = [float(v) for v in mu_batch(model, tok, items, dev).cpu()]
            rrs.append(1.0 / pl.recompute_rank(mu, catalog, ci[dest_of[q]]))
    model.train()
    return float(np.mean(rrs))

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
        for p in model.parameters(): p.requires_grad = True
        tensors = list(model.parameters())
        opt = torch.optim.Adam(tensors, lr=2e-4)
        model.train(); t0 = time.time()
        # phase 1: distill toward e5 top-1 on random slates (free supervision)
        for step in range(STEPS):
            q = train_q[rng.integers(len(train_q))]
            slate_c = [catalog[i] for i in rng.integers(0, len(catalog), SLATE)]
            qv = qt.numpy()[ix[q]]
            cosv = np.array([float(qv @ pt.numpy()[ix[c]]) for c in slate_c])
            items = [(q, c, OPS["LINEAGE_RANK"], CM, J, MM, MM) for c in slate_c]
            mu = mu_batch(model, tok, items, dev, train=True, rng=aug)
            loss = logit_ce(mu, int(np.argmax(cosv)), dev)
            ag = [(it[0], it[1], it[2]) for it in items[:8]]
            mu_ag = mu_batch(model, tok, ag, dev)
            with torch.no_grad():
                mu_ref = mu_batch(ref, tok, ag, dev)
            loss = loss + torch.mean((mu_ag - mu_ref) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(tensors, 1.0); opt.step()
        res["distill_only"][f] = evaluate(model, tok, held[f], catalog, ci, dest_of, dev)
        # phase 2: fine-tune on true destinations (rank-CE alternating with graded channel)
        for step in range(STEPS):
            if step % 2 == 1:
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
            loss = logit_ce(mu, 0, dev)
            ag = [(it[0], it[1], it[2]) for it in items[:8]]
            mu_ag = mu_batch(model, tok, ag, dev)
            with torch.no_grad():
                mu_ref = mu_batch(ref, tok, ag, dev)
            loss = loss + torch.mean((mu_ag - mu_ref) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(tensors, 1.0); opt.step()
        res["warmstart_finetune"][f] = evaluate(model, tok, held[f], catalog, ci, dest_of, dev)
        print(f"[warmstart] fold {f}: distill {res['distill_only'][f]:.4f} -> "
              f"finetune {res['warmstart_finetune'][f]:.4f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(json.dumps({a: float(np.mean(list(v.values()))) for a, v in res.items()} |
                     {"comparators": {"scratch_rank_ce_graded_desig": 0.349,
                                      "scratch_rank_ce_graded_sigmoid": 0.347,
                                      "e5_frozen": 0.573}}, indent=1), flush=True)

run()
