#!/usr/bin/env python3
"""TIER-1 SHADOW RUN — actually train the SM-FS ranking experiment, honestly labeled.

Owner decision (2026-07-27): the Tier-2 authorization cathedral (chain of custody, final locks,
trust-rooted reviews) was blocking the feedback loop that tells us whether the model is any good.
This runner executes the SAME frozen experiment the pipeline defines — 5 folds x 3 seeds x
2 arms, sol's counter sampler, the frozen fold blocks, the frozen bootstrap — as an EXPLORATORY
SHADOW run:

  - results are descriptive/exploratory, catalog-transductive; NO confirmatory claim;
  - the 1,481 reserved rows remain untouched (shadow uses only the exploration folds);
  - preregistrations are NOT amended; the Tier-2 machinery stays intact for a later
    authoritative rerun of whatever shadow finds;
  - every artifact is stamped "shadow-exploratory" so it can never be passed off as Tier 2.

  python3 sm_fs_ranking_shadow.py run-all      # 30 fits + 30 evals + decide, sequential (GPU)
  python3 sm_fs_ranking_shadow.py decide
"""
import argparse
import copy
import io
import json
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sm_fs_ranking_pipeline as pl

SHADOW_DIR = os.path.expanduser("~/mu_data/sm_fs_ranking_shadow_v1")
STAMP = "shadow-exploratory-tier1-not-decision-bearing"


def _titles():
    return json.loads(pl.read_bound(
        os.path.join(os.path.expanduser("~/mu_data/sm_fs_ranking_run_v4"), "titles.json"),
        private=True, description="title table"))


def _projection(fold):
    plan = json.loads(pl.read_bound(
        os.path.join(os.path.expanduser("~/mu_data/sm_fs_ranking_run_v4"),
                     "training_plan.json"), private=True, description="plan"))
    proj = plan["projections"][str(fold)]
    rows_bytes = pl.read_bound(
        os.path.join(os.path.expanduser("~/mu_data/sm_fs_ranking_run_v4"),
                     f"fold{fold}", "train_projection.jsonl"),
        expect_sha=proj["projection_sha256"], private=True, description="projection")
    return [json.loads(l) for l in rows_bytes.decode().splitlines()]


def fit_job(fold, arm, seed, tok_cache):
    import numpy as np
    import torch
    env = pl.enforce_environment()
    rows = _projection(fold)
    for p in rows:
        pl.need(p["fold"] != fold, "held-fold row inside projection")
    torch.manual_seed(seed)
    aug_rng = np.random.default_rng(seed + 1)
    ckpt_bytes = pl.read_bound(os.path.join(pl.RANK_DIR, f"init_seed{seed}.pt"),
                               expect_sha=pl.INIT_SHA[seed], private=True,
                               description="initialized checkpoint")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = pl.load_checkpoint_bytes(ckpt_bytes, dev)
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    names, shapes, tensors = pl.resolve_allowlist(model)
    opt = torch.optim.Adam(tensors, lr=pl.ADAM["lr"], betas=tuple(pl.ADAM["betas"]),
                           eps=pl.ADAM["eps"], weight_decay=pl.ADAM["weight_decay"],
                           amsgrad=pl.ADAM["amsgrad"])
    tok = tok_cache["tok"]
    pos, buckets = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for i, p in enumerate(rows):
        (pos[p["query"]].append(i) if p["class"].startswith("positive")
         else buckets[p["query"]][p["hardness"]].append(i))
    train_q = sorted(pos)
    model.train()
    losses = []
    t0 = time.time()
    for step in range(pl.STEPS):
        c, k = pl.sample_step(fold, seed, step, train_q, pos, buckets, arm)
        losses.append(pl.fit_one_step(model, ref, tensors, opt, tok, rows, c, k,
                                      aug_rng, dev))
    model.eval()
    buf = io.BytesIO()
    torch.save({"state": model.state_dict(), "cfg": cfg}, buf)
    out_dir = os.path.join(SHADOW_DIR, f"fold{fold}")
    ck_sha = pl.install_private(os.path.join(out_dir, f"fit_{arm}_{seed}.pt"), buf.getvalue())
    rec = {"stamp": STAMP, "fold": fold, "arm": arm, "seed": seed,
           "init_sha256": pl.INIT_SHA[seed], "checkpoint_sha256": ck_sha,
           "steps": pl.STEPS, "loss_first_last": [losses[0], losses[-1]],
           "wall_seconds": round(time.time() - t0, 1), "environment": env}
    pl.install_private(os.path.join(out_dir, f"fit_{arm}_{seed}.receipt.json"), pl.canon(rec))
    print(f"[shadow] fit fold={fold} arm={arm} seed={seed} "
          f"loss {losses[0]:.4f}->{losses[-1]:.4f} ({rec['wall_seconds']}s)", flush=True)


def eval_job(fold, arm, seed, tok_cache, pairs, catalog, held_by_fold):
    import torch

    from fine_tune_channel_heads import mu_batch
    from mu_attention import CORPORA, JUDGES, NODETYPE, OPS
    out_dir = os.path.join(SHADOW_DIR, f"fold{fold}")
    rec = json.loads(pl.read_bound(os.path.join(out_dir, f"fit_{arm}_{seed}.receipt.json"),
                                   private=True, description="shadow fit receipt"))
    ck_bytes = pl.read_bound(os.path.join(out_dir, f"fit_{arm}_{seed}.pt"),
                             expect_sha=rec["checkpoint_sha256"], private=True,
                             description="shadow checkpoint")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = pl.load_checkpoint_bytes(ck_bytes, dev)
    model.eval()
    tok = tok_cache["tok"]
    held = held_by_fold[fold]
    C, J, MM = CORPORA["mindmap"], JUDGES["graph"], NODETYPE["mindmap_node"]
    out_rows = []
    with torch.no_grad():
        for q in sorted(held):
            items = [(q, c, OPS["LINEAGE"], C, J, MM, MM) for c in catalog]
            mu = [float(v) for v in mu_batch(model, tok, items, dev).cpu()]
            dest = next(c for c, p in held[q].items() if p["class"] == "positive_parent")
            rank = pl.recompute_rank(mu, catalog, catalog.index(dest))
            out_rows.append({"query": q, "destination": dest, "rank": rank,
                             "rr": 1.0 / rank, "scores": mu})
    pred = {"stamp": STAMP, "fold": fold, "arm": arm, "seed": seed,
            "checkpoint_sha256": rec["checkpoint_sha256"],
            "catalog_size": len(catalog), "rows": out_rows}
    pl.install_private(os.path.join(out_dir, f"eval_{arm}_{seed}.receipt.json"),
                       pl.canon(pred))
    mrr = sum(r["rr"] for r in out_rows) / len(out_rows)
    print(f"[shadow] eval fold={fold} arm={arm} seed={seed} MRR {mrr:.4f}", flush=True)
    return mrr


def cmd_decide(_a=None):
    from sm_fs_bootstrap import decide as boot_decide
    man, pairs, fold_txt = pl.load_pairs_verified()
    catalog = sorted({p["candidate"] for p in pairs})
    cat_index = {c: j for j, c in enumerate(catalog)}
    fold_of = {p["query"]: p["fold"] for p in pairs}
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    rr = {arm: defaultdict(dict) for arm in pl.ARMS}
    for f in range(5):
        for arm in pl.ARMS:
            for seed in pl.SEEDS:
                rec = json.loads(pl.read_bound(
                    os.path.join(SHADOW_DIR, f"fold{f}", f"eval_{arm}_{seed}.receipt.json"),
                    private=True, description="shadow eval receipt"))
                for row in rec["rows"]:
                    q = row["query"]
                    rank = pl.recompute_rank(row["scores"], catalog, cat_index[dest_of[q]])
                    rr[arm][q][seed] = 1.0 / rank
    all_held = set(fold_of)
    d = {q: (sum(rr["graded_negative"][q].values()) / 3.0
             - sum(rr["positive_only"][q].values()) / 3.0) for q in sorted(all_held)}
    blocks = [ln.split("\t")[0] for ln in fold_txt.splitlines()]
    block_values = {b: [] for b in blocks}
    for q, val in d.items():
        cands = [b for b in blocks if dest_of[q] == b or dest_of[q].startswith(b + "/")]
        block_values[max(cands, key=len)].append(val)
    result = boot_decide(block_values)
    result.update({
        "stamp": STAMP,
        "seed_specific_mrr": {arm: {str(s): sum(rr[arm][q][s] for q in all_held)
                                    / len(all_held) for s in pl.SEEDS} for arm in pl.ARMS},
        "arm_mrr": {arm: sum(sum(v.values()) / 3.0 for v in rr[arm].values())
                    / len(all_held) for arm in pl.ARMS},
    })
    out = os.path.join(SHADOW_DIR, "shadow_decision.json")
    if os.path.exists(out):
        os.unlink(out)                                    # shadow results are re-runnable
    pl.install_private(out, pl.canon(result))
    print(json.dumps({k: result[k] for k in
                      ("delta_mrr", "ci95", "arm_mrr", "seed_specific_mrr",
                       "passed_exploratory_gate")}, indent=1))


def cmd_run_all(_a=None):
    from mu_attention import E5_REVISION, Tokenizer, build_e5_tables
    titles = _titles()
    qtbl, ptbl, idx = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                      model_revision=E5_REVISION)
    tok_cache = {"tok": Tokenizer(qtbl, ptbl, idx, {}, {})}
    man, pairs, _ = pl.load_pairs_verified()
    catalog = sorted({p["candidate"] for p in pairs})
    held_by_fold = defaultdict(lambda: defaultdict(dict))
    for p in pairs:
        held_by_fold[p["fold"]][p["query"]][p["candidate"]] = p
    t0 = time.time()
    for f in range(5):
        for seed in pl.SEEDS:
            for arm in pl.ARMS:
                done = os.path.join(SHADOW_DIR, f"fold{f}", f"eval_{arm}_{seed}.receipt.json")
                if os.path.exists(done):
                    print(f"[shadow] skip fold={f} arm={arm} seed={seed} (done)", flush=True)
                    continue
                fit_job(f, arm, seed, tok_cache)
                eval_job(f, arm, seed, tok_cache, pairs, catalog, held_by_fold)
    print(f"[shadow] all 30 jobs complete in {(time.time()-t0)/60:.1f} min", flush=True)
    cmd_decide()


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run-all")
    sub.add_parser("decide")
    a = ap.parse_args(argv)
    return {"run-all": cmd_run_all, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()
