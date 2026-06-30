#!/usr/bin/env python3
"""train_filing.py — filing fine-tune LEARNING CURVE (DESIGN_model_applications.md, build-plan).

Tests the prediction "with enough in-domain data, the attention μ crosses the e5-cosine filing bar."
Warm-start `model_nodetype.pt`, fine-tune on `element_of(bookmark→folder)` with **in-batch contrastive**
negatives (a B×B μ matrix; same-folder = positive, else negative), at increasing **data fractions**, and eval
MRR/recall@k on a **FIXED held-out bookmark set** (never trained, same across fractions) against the flat
`e5-cos` baseline. The curve is μ-MRR vs fraction; the question is where (if) it crosses e5-cos.

Split is **bookmark-holdout** (folders shared): folders are a stable taxonomy, new bookmarks get filed — so we
hold out *bookmarks*, not folders. The model carries no per-folder parameters (a folder is only its e5 title),
so a shared folder is not leakage — the gain is the readout adapting to the filing distribution.

Data lives in .local (gitignored); this script is committable, the data is not.

Usage:
  python3 train_filing.py --ckpt model_nodetype.pt --trees ../../.local/data/pearltrees_api/trees \
      --fracs 0.1,0.3,1.0 --steps 300 --bs 48
"""
import argparse, collections, os, random
import torch
import torch.nn.functional as F
from mu_attention import build_e5_tables, Tokenizer, MuAttention, OPS
from eval_filing import load_filing, score_mu, metrics


def build_model(ckpt, dev):
    ck = torch.load(ckpt, weights_only=False)
    sd, cfg = ck["state"], ck.get("cfg", {"d_model": 384, "heads": 4, "layers": 3})
    sz = lambda k, d: sd[k].shape[0] if k in sd else d
    m = MuAttention(d_model=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
                    n_ops=sz("op_emb.weight", len(OPS)), n_corpus=sz("corpus_emb.weight", 2),
                    n_judge=sz("judge_emb.weight", 2), n_nodetype=sz("nodetype_emb.weight", 4)).to(dev)
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not unexp and all(("account" in k or "prefix" in k) for k in miss), (miss, unexp)
    return m, cfg


def train_one(ckpt, tok, idx, bm_key, bm_folder, train_idx, f_order, eval_keys, eval_truepos,
              f_keys, steps, bs, lr, dev, seed):
    """Warm-start a fresh model, fine-tune on `train_idx` bookmarks, return MRR on the fixed eval set."""
    model, _ = build_model(ckpt, dev)
    n_ops = model.op_emb.weight.shape[0]
    elem = torch.zeros(1, n_ops, device=dev).index_fill_(1, torch.tensor([OPS["ELEM"]], device=dev), 1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    model.train()
    for step in range(steps):
        batch = rng.sample(train_idx, min(bs, len(train_idx)))
        bk = [bm_key[i] for i in batch]
        fid = [bm_folder[i] for i in batch]
        # B×B grid: μ(bookmark_i | folder_of_j); same-folder = positive (supervised in-batch contrastive)
        fkeys_b = [f"F:{f}" for f in fid]
        items = [(bk[i], fkeys_b[j], 0) for i in range(len(batch)) for j in range(len(batch))]
        b = tok.build(items, train=False)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        ow = elem.expand(len(items), n_ops)
        mu = model(**b, op_weights=ow).view(len(batch), len(batch))
        fi = torch.tensor([hash(f) for f in fid], device=dev)
        target = (fi[:, None] == fi[None, :]).float()
        bce = F.binary_cross_entropy(mu.clamp(1e-6, 1 - 1e-6), target, reduction="none")
        pos, neg = target > 0.5, target <= 0.5                       # balance pos/neg (negatives dominate B×B)
        loss = bce[pos].mean() + bce[neg].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # eval on the fixed held-out set
    S = score_mu(model, tok, idx, eval_keys, f_keys, elem.cpu(), dev)
    ranks = [1 + sum(1 for j, s in enumerate(row) if s > row[eval_truepos[r]]
                     or (s == row[eval_truepos[r]] and j < eval_truepos[r])) for r, row in enumerate(S)]
    return metrics(ranks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--trees", required=True)
    ap.add_argument("--min-bm", type=int, default=3)
    ap.add_argument("--fracs", default="0.1,0.3,1.0", help="comma-sep training-data fractions")
    ap.add_argument("--eval-frac", type=float, default=0.3, help="fraction of bookmarks per folder held out for eval")
    ap.add_argument("--max-eval", type=int, default=400)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7, help="split seed (FIXED across training seeds for comparability)")
    ap.add_argument("--seeds", default=None, help="comma-sep TRAINING seeds (multi-seed lock); split stays at --seed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", default="/tmp/trainfiling_e5.pt")
    a = ap.parse_args()
    dev = torch.device(a.device)

    queries, cand = load_filing(a.trees, a.min_bm)
    rng = random.Random(a.seed)
    byf = collections.defaultdict(list)
    for q in queries:
        byf[q[1]].append(q)
    eval_q, pool = [], []
    for fid, qs in byf.items():                                      # per-folder split: eval holdout vs train pool
        qs = qs[:]; rng.shuffle(qs)
        k = max(1, int(a.eval_frac * len(qs))) if len(qs) >= 2 else 0
        eval_q += qs[:k]; pool += qs[k:]
    rng.shuffle(eval_q); eval_q = eval_q[:a.max_eval]
    print(f"[DATA] {len(cand)} folders, {len(queries)} bookmarks → {len(pool)} train-pool, "
          f"{len(eval_q)} fixed eval (held-out)")

    # e5 over folders + ALL bookmarks (eval first so eval keys are stable B:0..)
    bm_list = eval_q + pool
    f_keys = [f"F:{t}" for t in cand]
    bm_key = [f"B:{i}" for i in range(len(bm_list))]
    bm_folder = [bm_list[i][1] for i in range(len(bm_list))]
    names = f_keys + bm_key
    texts = {**{f"F:{t}": cand[t] for t in cand}, **{bm_key[i]: bm_list[i][0] for i in range(len(bm_list))}}
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=a.cache, texts=texts, device=a.device)
    tok = Tokenizer(qtbl, ptbl, idx, parents={}, deg={})

    f_order = list(cand)
    eval_keys = [f"B:{i}" for i in range(len(eval_q))]
    eval_truepos = [f_order.index(eval_q[i][1]) for i in range(len(eval_q))]
    train_idx = list(range(len(eval_q), len(bm_list)))

    # e5-cos baseline on the fixed eval set (the flat bar)
    qn = qtbl[[idx[k] for k in eval_keys]]; fn = ptbl[[idx[k] for k in f_keys]]
    C = qn @ fn.T
    base_ranks = [1 + int((C[r] > C[r][eval_truepos[r]]).sum().item()) for r in range(C.shape[0])]
    base = metrics(base_ranks)
    print(f"\n[BASELINE] e5-cos (no training): MRR {base['MRR']:.3f}  recall@10 {base['recall@10']:.3f}  "
          f"med.rank {base['median_rank']}")

    import statistics as stt
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed]
    print(f"\n[SEEDS] training seeds {seeds} (split fixed at seed {a.seed}); ✓ = mean−sd above the e5-cos bar")
    print(f"\n{'frac':>6} {'n_train':>8} {'MRR (mean±sd)':>15} {'recall@10 (m±sd)':>18} {'med.rank':>9}  vs e5-cos")
    fracs = [float(x) for x in a.fracs.split(",")]
    for fr in fracs:
        n = min(max(a.bs, int(fr * len(train_idx))), len(train_idx))
        mrrs, r10s, meds = [], [], []
        for sd in seeds:
            sub = random.Random(sd + 1).sample(train_idx, n)
            m = train_one(a.ckpt, tok, idx, bm_key, bm_folder, sub, f_order, eval_keys, eval_truepos,
                          f_keys, a.steps, a.bs, a.lr, dev, sd)
            mrrs.append(m["MRR"]); r10s.append(m["recall@10"]); meds.append(m["median_rank"])
        mm, ms = stt.mean(mrrs), (stt.stdev(mrrs) if len(mrrs) > 1 else 0.0)
        rm, rs = stt.mean(r10s), (stt.stdev(r10s) if len(r10s) > 1 else 0.0)
        delta = mm - base["MRR"]
        flag = "  ✓ CROSSED" if mm - ms > base["MRR"] else ("  ~at-bar" if mm > base["MRR"] else "")
        print(f"{fr:6.2f} {n:8d}   {mm:.3f}±{ms:.3f}     {rm:.3f}±{rs:.3f}    {stt.median(meds):7.1f}  {delta:+.3f}{flag}")


if __name__ == "__main__":
    main()
