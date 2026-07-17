#!/usr/bin/env python3
"""Filing v1 acceptance eval: rank REAL Pearltrees folders for held-out pearls, base vs fine-tuned.

Reuses eval_filing.py's data path (harvested tree JSONs; ground truth = each bookmark's actual
treeId — a real human filing decision) and metrics (recall@k / MRR / median rank), and adds what
the Pearltrees fine-tune specifically needs:

  * CONDITIONED rankers — the fine-tune trains the pearltrees corpus card, judge heads, and
    nodetype embedding while the agnostic-anchor loss deliberately pins the agnostic readouts to
    the base model; eval_filing's stock (agnostic) rankers therefore CANNOT see the fine-tune.
    Items here are 7-tuples (bookmark, folder, op, pearltrees, judge, page, pearltrees_collection)
    scored under ELEM / HIER / SYM (judge kalman-fused) and LINEAGE (judge graph).
  * A held-folder subset — queries whose TRUE folder title never appeared in the fine-tune's
    node-disjoint train node set (campaign split seed 0): the honest generalization slice.
  * The ESCALATION curve — the deployment policy: route a filing decision to the judge when the
    top-2 margin of the operating ranker is below threshold; report fraction-routed and recall@1
    among the kept decisions at each threshold.

  python3 eval_pearltrees_filing.py --base model_prod_namecond_full.pt --tuned model_pt_filing.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_filing import load_filing, metrics, rank_all
from fine_tune_pearltrees_filing import load_with_lineage_ops
from mu_attention import CORPORA, JUDGES, NODETYPE, OPS, Tokenizer, build_e5_tables

ROOT = os.path.dirname(os.path.abspath(__file__))
TREES = os.path.join(ROOT, "..", "..", ".local", "data", "pearltrees_api", "trees")


def score_cond(model, tok, q_keys, f_keys, op, judge, dev="cpu", batch=512):
    """μ(bookmark|folder) under an explicit (op, corpus=pearltrees, judge, nodetypes) conditioning."""
    items = [
        (qk, fk, OPS[op], CORPORA["pearltrees"], JUDGES[judge],
         NODETYPE["page"], NODETYPE["pearltrees_collection"])
        for qk in q_keys for fk in f_keys
    ]
    out = []
    with torch.no_grad():
        for i in range(0, len(items), batch):
            b = tok.build(items[i:i + batch], train=False)
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            out += model(**b).cpu().tolist()
    F = len(f_keys)
    return torch.tensor([out[r * F:(r + 1) * F] for r in range(len(q_keys))])


def ranks_from(M, truepos):
    return [1 + int(((M[r] > M[r][truepos[r]]) |
                     ((M[r] == M[r][truepos[r]]) & (torch.arange(M.shape[1]) < truepos[r]))).sum().item())
            for r in range(M.shape[0])]


def escalation_curve(M, truepos, thresholds=(0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)):
    """Deployment policy: escalate when top1−top2 margin < t. Returns rows (t, routed_frac, kept_recall@1)."""
    top2 = M.topk(min(2, M.shape[1]), dim=1).values
    margin = (top2[:, 0] - top2[:, 1]) if top2.shape[1] > 1 else torch.zeros(M.shape[0])
    ranks = torch.tensor(ranks_from(M, truepos), dtype=torch.float32)
    rows = []
    for t in thresholds:
        kept = margin >= t
        routed = 1.0 - kept.float().mean().item()
        kept_r1 = float((ranks[kept] <= 1).float().mean().item()) if kept.any() else float("nan")
        rows.append((t, routed, kept_r1))
    return rows


def campaign_train_nodes(split_seed=0):
    """Node titles on the TRAIN side of the fine-tune's node-disjoint campaign split."""
    from node_disjoint_eval import node_disjoint_pair_split
    from run_pearltrees_fusion import load_pearltrees_campaign
    ds = load_pearltrees_campaign(require_55=False)
    strata = [("principal" if t.startswith("principal") else t) for t in ds["tags"]]
    split = node_disjoint_pair_split(ds["pairs"], split_seed, strata=strata)
    return set(split.train_nodes)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "model_prod_namecond_full.pt"))
    ap.add_argument("--tuned", default=os.path.join(ROOT, "model_pt_filing.pt"))
    ap.add_argument("--trees", default=TREES)
    ap.add_argument("--min-bm", type=int, default=3)
    ap.add_argument("--max-queries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", default="/tmp/mu_data/pt_filing_eval_e5.pt")
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args(argv)
    dev = "cpu"
    torch.set_num_threads(4)

    queries, cand = load_filing(a.trees, a.min_bm)
    import random
    if len(queries) > a.max_queries:
        queries = random.Random(a.seed).sample(queries, a.max_queries)
    f_ids = sorted(cand)
    f_titles = [cand[fid] for fid in f_ids]
    q_titles = [q for q, _ in queries]
    truepos = [f_ids.index(fid) for _, fid in queries]
    print(f"filing eval: {len(queries)} bookmark queries over {len(f_ids)} candidate folders "
          f"(min_bm {a.min_bm}, seed {a.seed})")

    train_nodes = campaign_train_nodes(a.split_seed)
    held_q = [i for i in range(len(queries)) if f_titles[truepos[i]] not in train_nodes]
    print(f"held-folder subset (true folder unseen in fine-tune train nodes): {len(held_q)}/{len(queries)}")

    names = sorted(set(q_titles) | set(f_titles))
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=a.cache, batch_size=128)
    tok = Tokenizer(qtbl, ptbl, idx, {}, {})

    results = {}
    for label, ckpt in (("base", a.base), ("tuned", a.tuned)):
        torch.manual_seed(0)   # deterministic fresh LINEAGE readout rows when the base ckpt lacks them
        model, _ = load_with_lineage_ops(ckpt, dev=dev)
        model.eval()
        # stock agnostic rankers (prior art; anchor loss should keep base≈tuned here)
        rank_of, order = rank_all(model, tok, qtbl, ptbl, idx, q_titles, f_titles, truepos, dev)
        # conditioned rankers — where the fine-tune is allowed to show up
        S_elem = score_cond(model, tok, q_titles, f_titles, "ELEM", "kalman-fused", dev)
        S_hier = score_cond(model, tok, q_titles, f_titles, "HIER", "kalman-fused", dev)
        S_sym = score_cond(model, tok, q_titles, f_titles, "SYM", "kalman-fused", dev)
        S_lin = score_cond(model, tok, q_titles, f_titles, "LINEAGE", "graph", dev)
        S_maxc = torch.maximum(torch.maximum(S_elem, S_hier), S_sym)
        S_maxl = torch.maximum(S_maxc, S_lin)
        for nm, M in (("mu-elem-cond", S_elem), ("mu-lineage", S_lin),
                      ("mu-max-cond", S_maxc), ("mu-max+lineage", S_maxl)):
            rank_of[nm] = ranks_from(M, truepos)
        order = tuple(order) + ("mu-elem-cond", "mu-lineage", "mu-max-cond", "mu-max+lineage")
        results[label] = dict(rank_of=rank_of, order=order, S_maxl=S_maxl)

        print(f"\n=== {label}: {os.path.basename(ckpt)} ===")
        print(f"  {'ranker':16} {'recall@1':>9} {'recall@5':>9} {'recall@10':>10} {'MRR':>7} {'med.rank':>9}")
        for nm in order:
            m = metrics(rank_of[nm])
            print(f"  {nm:16} {m['recall@1']:9.3f} {m['recall@5']:9.3f} {m['recall@10']:10.3f} "
                  f"{m['MRR']:7.3f} {m['median_rank']:9d}")
        if held_q:
            print(f"  held-folder subset (n={len(held_q)}):")
            for nm in ("mu-max-cond", "mu-max+lineage", "e5-cos"):
                m = metrics([rank_of[nm][i] for i in held_q])
                print(f"    {nm:16} recall@1 {m['recall@1']:.3f}  recall@5 {m['recall@5']:.3f}  "
                      f"MRR {m['MRR']:.3f}")

    print("\nescalation curve (mu-max+lineage margin; routed = sent to the judge):")
    print(f"  {'threshold':>9} | {'base routed':>11} {'base kept R@1':>13} | {'tuned routed':>12} {'tuned kept R@1':>14}")
    curve_b = escalation_curve(results["base"]["S_maxl"], truepos)
    curve_t = escalation_curve(results["tuned"]["S_maxl"], truepos)
    for (t, rb, kb), (_, rt, kt) in zip(curve_b, curve_t):
        print(f"  {t:9.2f} | {rb:11.3f} {kb:13.3f} | {rt:12.3f} {kt:14.3f}")

    dmrr = metrics(results["tuned"]["rank_of"]["mu-max+lineage"])["MRR"] - \
        metrics(results["base"]["rank_of"]["mu-max+lineage"])["MRR"]
    print(f"\npaired MRR delta (tuned - base, mu-max+lineage): {dmrr:+.4f}")


if __name__ == "__main__":
    main()
