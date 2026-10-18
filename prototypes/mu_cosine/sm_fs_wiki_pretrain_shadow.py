#!/usr/bin/env python3
"""SHADOW: does wiki-scale lineage data lift SM-FS ranking? (owner's more-data hypothesis)

Pretrain the LINEAGE head on ~180k enwiki lineage rows (corpus=enwiki token), then:
  A. zero-shot evaluate on all SM-FS held folds (corpus=mindmap conditioning);
  B. fine-tune per fold on the SM-FS graded arm (seed 3997001) and evaluate.
Comparators: SM-FS-only graded 0.280; e5 frozen 0.573. Stamped shadow-exploratory."""
import copy
import io
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from fine_tune_channel_heads import mu_batch
from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, \
    build_e5_tables

OUT = os.path.expanduser("~/mu_data/wiki_lineage_v1")
SEED = 3997001
WIKI_STEPS, WIKI_BS = 3000, 64
CE, CM = CORPORA["enwiki"], CORPORA["mindmap"]
J, MM, CAT = JUDGES["graph"], NODETYPE["mindmap_node"], NODETYPE["category"]

rows = []
for ln in open(os.path.join(OUT, "targets.tsv"), encoding="utf-8"):
    if ln.startswith("#"):
        continue
    n, a, t, kind = ln.rstrip("\n").split("\t")
    rows.append((n, a, float(t)))
names = sorted({x for r in rows for x in r[:2]})
print(f"wiki rows {len(rows)}, names {len(names)}", flush=True)
qtbl, ptbl, idx = build_e5_tables(names, cache_path=os.path.join(OUT, "wiki_e5.pt"),
                                  batch_size=512, model_revision=E5_REVISION)
tok = Tokenizer(qtbl, ptbl, idx, {}, {})
env = pl.enforce_environment()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
aug = np.random.default_rng(SEED + 1)
ck = pl.read_bound(os.path.join(pl.RANK_DIR, f"init_seed{SEED}.pt"),
                   expect_sha=pl.INIT_SHA[SEED], private=True)
model, cfg = pl.load_checkpoint_bytes(ck, dev)
ref = copy.deepcopy(model)
ref.eval()
for p in ref.parameters():
    p.requires_grad = False
names18, shapes, tensors = pl.resolve_allowlist(model)
opt = torch.optim.Adam(tensors, lr=pl.ADAM["lr"])
rng = np.random.default_rng(SEED)
model.train()
t0 = time.time()
for step in range(WIKI_STEPS):
    sel = rng.choice(len(rows), size=WIKI_BS, replace=False)
    items = [(rows[i][0], rows[i][1], OPS["LINEAGE"], CE, J, CAT, CAT) for i in sel]
    tgt = torch.tensor([rows[i][2] for i in sel], dtype=torch.float32, device=dev)
    mu = mu_batch(model, tok, items, dev, train=True, rng=aug)
    loss = torch.mean((mu - tgt) ** 2)
    ag = [(it[0], it[1], it[2]) for it in items[:16]]
    mu_ag = mu_batch(model, tok, ag, dev)
    with torch.no_grad():
        mu_ref = mu_batch(ref, tok, ag, dev)
    loss = loss + torch.mean((mu_ag - mu_ref) ** 2)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tensors, 1.0)
    opt.step()
    if step % 500 == 0:
        print(f"  wiki step {step} loss {loss.item():.4f}", flush=True)
print(f"wiki pretrain done ({(time.time()-t0)/60:.1f} min)", flush=True)
buf = io.BytesIO()
torch.save({"state": model.state_dict(), "cfg": cfg}, buf)
wiki_ck = buf.getvalue()
open(os.path.join(OUT, "wiki_pretrained.pt"), "wb").write(wiki_ck)
os.chmod(os.path.join(OUT, "wiki_pretrained.pt"), 0o600)

# SM-FS evaluation machinery
titles = sh._titles()
q2, p2, i2 = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                             model_revision=E5_REVISION)
tok_sm = Tokenizer(q2, p2, i2, {}, {})
man, pairs, _ = pl.load_pairs_verified()
catalog = sorted({p["candidate"] for p in pairs})
cat_index = {c: j for j, c in enumerate(catalog)}
dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
held_by_fold = defaultdict(lambda: defaultdict(dict))
for p in pairs:
    held_by_fold[p["fold"]][p["query"]][p["candidate"]] = p


def eval_model(m, label):
    m.eval()
    rrs = []
    with torch.no_grad():
        for f in range(5):
            for q in sorted(held_by_fold[f]):
                items = [(q, c, OPS["LINEAGE"], CM, J, MM, MM) for c in catalog]
                mu = [float(v) for v in mu_batch(m, tok_sm, items, dev).cpu()]
                rank = pl.recompute_rank(mu, catalog, cat_index[dest_of[q]])
                rrs.append(1.0 / rank)
    print(f"[{label}] SM-FS whole-population MRR {np.mean(rrs):.4f}  "
          f"R@1 {np.mean([r==1.0 for r in rrs]):.3f}", flush=True)
    return float(np.mean(rrs))


mA = eval_model(model, "A zero-shot wiki-pretrained")

# B: per-fold graded fine-tune from the wiki checkpoint
tok_cache = {"tok": tok_sm, "qtbl": q2, "ptbl": p2, "idx": i2}
rr = defaultdict(dict)
for f in range(5):
    proj_rows = sh._projection(f)
    torch.manual_seed(SEED)
    aug2 = np.random.default_rng(SEED + 1)
    m2, cfg2 = pl.load_checkpoint_bytes(wiki_ck, dev)
    ref2 = copy.deepcopy(m2)
    ref2.eval()
    for p in ref2.parameters():
        p.requires_grad = False
    _, _, tns = pl.resolve_allowlist(m2)
    o2 = torch.optim.Adam(tns, lr=pl.ADAM["lr"])
    pos, buckets = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for i, p in enumerate(proj_rows):
        (pos[p["query"]].append(i) if p["class"].startswith("positive")
         else buckets[p["query"]][p["hardness"]].append(i))
    train_q = sorted(pos)
    m2.train()
    for step in range(pl.STEPS):
        c, k = pl.sample_step(f, SEED, step, train_q, pos, buckets, "graded_negative")
        pl.fit_one_step(m2, ref2, tns, o2, tok_sm, proj_rows, c, k, aug2, dev)
    m2.eval()
    with torch.no_grad():
        for q in sorted(held_by_fold[f]):
            items = [(q, c, OPS["LINEAGE"], CM, J, MM, MM) for c in catalog]
            mu = [float(v) for v in mu_batch(m2, tok_sm, items, dev).cpu()]
            rr[q][SEED] = 1.0 / pl.recompute_rank(mu, catalog, cat_index[dest_of[q]])
    print(f"[B] fold {f} done", flush=True)
mB = float(np.mean([list(v.values())[0] for v in rr.values()]))
print(json.dumps({"A_zero_shot_wiki": mA, "B_wiki_then_smfs_graded": mB,
                  "comparators": {"smfs_only_graded": 0.280, "e5_frozen": 0.573}}, indent=1),
      flush=True)
