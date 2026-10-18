#!/usr/bin/env python3
"""SHADOW: DISTILLATION-PATHOLOGY DIAGNOSTIC (Kimi caveat 1) — the 1.4% top-1 agreement
from sm_fs_selfdistill_shadow is too low for a function-class story; before "attention is the
wrong function class" hardens, isolate the optimization suspects. Four arms, fold 0 only,
each changing EXACTLY ONE thing from the original recipe:
  desigmoid    CE over logit(mu) instead of (mu*8) — kills sigmoid-tail gradient compression
  noanchor     drop the anchor-to-init MSE term — the original pinned the model to its RANDOM
               init's behavior on 8 items/step, directly opposing the copy-e5 objective
  readout_only train only readout_w/readout_b at lr 1e-3 — pure optimization-surface probe
  lowlr        full capacity at lr 2e-5 instead of 2e-4
If any arm jumps far above 1.4%, the pathology was optimization, not function class."""
import copy, json, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from fine_tune_channel_heads import mu_batch
from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, build_e5_tables

SEED, STEPS, SLATE, FOLD = 3997001, 1600, 32, 0
CM, J, MM = CORPORA["mindmap"], JUDGES["graph"], NODETYPE["mindmap_node"]
ARMS = ("desigmoid", "noanchor", "readout_only", "lowlr")

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
    rows = sh._projection(FOLD)
    pos = defaultdict(list)
    for i, p in enumerate(rows):
        if p["class"].startswith("positive"):
            pos[p["query"]].append(i)
    train_q = sorted(q for q in pos if q in dest_of)
    res = {}
    for arm in ARMS:
        torch.manual_seed(SEED); aug = np.random.default_rng(SEED + 1)
        rng = np.random.default_rng(SEED)
        ck = pl.read_bound(os.path.join(pl.RANK_DIR, f"init_seed{SEED}.pt"),
                           expect_sha=pl.INIT_SHA[SEED], private=True)
        model, cfg = pl.load_checkpoint_bytes(ck, dev)
        ref = copy.deepcopy(model); ref.eval()
        for p in ref.parameters(): p.requires_grad = False
        if arm == "readout_only":
            for p in model.parameters(): p.requires_grad = False
            model.readout_w.requires_grad = True
            model.readout_b.requires_grad = True
            tensors = [model.readout_w, model.readout_b]
            lr = 1e-3
        else:
            for p in model.parameters(): p.requires_grad = True
            tensors = list(model.parameters())
            lr = 2e-5 if arm == "lowlr" else 2e-4
        opt = torch.optim.Adam(tensors, lr=lr)
        model.train(); t0 = time.time()
        for step in range(STEPS):
            q = train_q[rng.integers(len(train_q))]
            slate_c = [catalog[i] for i in rng.integers(0, len(catalog), SLATE)]
            qv = qt.numpy()[ix[q]]
            cosv = np.array([float(qv @ pt.numpy()[ix[c]]) for c in slate_c])
            label = int(np.argmax(cosv))
            items = [(q, c, OPS["LINEAGE_RANK"], CM, J, MM, MM) for c in slate_c]
            mu = mu_batch(model, tok, items, dev, train=True, rng=aug)
            if arm == "desigmoid":
                logits = torch.logit(mu.clamp(1e-6, 1 - 1e-6))
            else:
                logits = mu * 8.0
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([label], dtype=torch.long, device=dev))
            if arm != "noanchor":
                ag = [(it[0], it[1], it[2]) for it in items[:8]]
                mu_ag = mu_batch(model, tok, ag, dev)
                with torch.no_grad():
                    mu_ref = mu_batch(ref, tok, ag, dev)
                loss = loss + torch.mean((mu_ag - mu_ref) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(tensors, 1.0); opt.step()
        model.eval()
        rrs, agree = [], []
        cv = np.stack([pt.numpy()[ix[c]] for c in catalog])
        with torch.no_grad():
            for q in sorted(held[FOLD]):
                items = [(q, c, OPS["LINEAGE_RANK"], CM, J, MM, MM) for c in catalog]
                mu = np.array([float(v) for v in mu_batch(model, tok, items, dev).cpu()])
                rrs.append(1.0 / pl.recompute_rank(list(mu), catalog, ci[dest_of[q]]))
                e5top = int(np.argmax(qt.numpy()[ix[q]] @ cv.T))
                agree.append(1.0 if int(np.argmax(mu)) == e5top else 0.0)
        res[arm] = {"mrr": float(np.mean(rrs)),
                    "top1_agreement_with_e5": float(np.mean(agree)),
                    "min": round((time.time() - t0) / 60, 1)}
        print(f"[diag] {arm}: agree {res[arm]['top1_agreement_with_e5']:.3f} "
              f"MRR {res[arm]['mrr']:.4f}", flush=True)
    print(json.dumps(res | {"comparators": {"original_recipe_agreement_5fold": 0.014,
                                            "e5_frozen": 0.573}}, indent=1), flush=True)

run()
