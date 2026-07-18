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
from eval_filing import load_filing, metrics, score_mu
from fine_tune_pearltrees_filing import load_with_lineage_ops
from mu_attention import CORPORA, JUDGES, NODETYPE, OPS, Tokenizer, build_e5_tables

ROOT = os.path.dirname(os.path.abspath(__file__))
PT_API = os.path.join(ROOT, "..", "..", ".local", "data", "pearltrees_api")
TREES = os.path.join(PT_API, "trees")
DAG = os.path.join(PT_API, "assembled_dag.tsv")
TITLES = os.path.join(PT_API, "assembled_titles.tsv")


PATHS_JSONL = os.path.join(PT_API, "..", "api_tree_paths_v8.jsonl")


def folder_lineage(cand, depth=5, shuffle_seed=None):
    """Build {folder_title: [parent_title,...]} PRINCIPAL paths for candidate folders.

    Principal parent = the OBSERVATION-MAJORITY parent across the account's recorded path_ids (the
    true filing lineage), falling back to the assembled DAG's edge only when a folder appears in no
    record. The earlier first-DAG-parent walk contradicted the record-majority parent on 61% of
    multi-parent folders (external statistical audit finding 1). Returns (parents_title,
    ancestor_titles). Only the folder's pre-existing folder→parent lineage is used (the §7 leakage
    boundary — never the evaluated bookmark's placement or the folder's bookmark children).
    `shuffle_seed` permutes which folder receives which lineage, SUPPORT-PRESERVING: chains permute
    only among folders that HAVE a chain (audit finding 6 — a naive permutation also changes which
    folders get any lineage at all, confounding the control)."""
    import json as _json
    from collections import Counter, defaultdict

    votes = defaultdict(Counter)
    with open(PATHS_JSONL, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = _json.loads(ln)
            ids = [str(x).split(":")[-1] for x in (r.get("path_ids") or [])]
            for p, c in zip(ids, ids[1:]):
                if p != c:
                    votes[c][p] += 1
    principal = {c: max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0] for c, cnt in votes.items()}
    dag_first = {}
    for ln in open(DAG, encoding="utf-8"):
        p, c = ln.split()
        dag_first.setdefault(c, p)
    titles = {}
    for ln in open(TITLES, encoding="utf-8"):
        parts = ln.rstrip("\n").split("\t")
        if len(parts) >= 2:
            titles[parts[0]] = parts[1]

    n_rec, n_dag = 0, 0

    def chain(tid):
        nonlocal n_rec, n_dag
        out, cur, seen = [], str(tid), {str(tid)}
        for _ in range(depth):
            if cur in principal:
                nxt = principal[cur]
                n_rec += 1
            elif cur in dag_first:
                nxt = dag_first[cur]
                n_dag += 1
            else:
                break
            if nxt in seen:
                break
            seen.add(nxt)
            t = titles.get(nxt)
            if t:
                out.append(t)
            cur = nxt
        return out

    tids = list(cand)
    chains = [chain(t) for t in tids]
    print(f"    lineage edges: {n_rec} record-majority (principal), {n_dag} DAG-fallback")
    if shuffle_seed is not None:
        import numpy as np

        has = [i for i, ch in enumerate(chains) if ch]        # support-preserving permutation
        perm = np.random.default_rng(shuffle_seed).permutation(len(has))
        remapped = list(chains)
        for slot, src in zip(has, [has[j] for j in perm]):
            remapped[slot] = chains[src]
        chains = remapped
    parents_title, anc_titles = {}, set()
    for tid, ch in zip(tids, chains):
        ft = cand[tid]
        # parents map is a chain: folder→p1→p2… (first-parent walk), keyed by title
        prev = ft
        for p in ch:
            parents_title.setdefault(prev, [p])
            anc_titles.add(p)
            prev = p
    return parents_title, anc_titles


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


def ranks_from(M, truepos_sets):
    """1-based rank of the BEST-ranked acceptable folder per query.

    `truepos_sets[r]` is the set of column indices counted correct — all folders sharing the true
    folder's title (duplicate titles are indistinguishable to a title-keyed model; breaking their
    ties by candidate order would misgrade ~10% of queries — external review)."""
    out = []
    arange = torch.arange(M.shape[1])
    for r in range(M.shape[0]):
        best = None
        for tp in truepos_sets[r]:
            rank = 1 + int(((M[r] > M[r][tp]) | ((M[r] == M[r][tp]) & (arange < tp))).sum().item())
            best = rank if best is None else min(best, rank)
        out.append(best)
    return out


def escalation_curve(M, truepos_sets, thresholds=(0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)):
    """Escalate when top1−top2 margin < t → rows (t, routed_frac, kept_n, kept_recall@1).

    DESCRIPTIVE margin diagnostic only: full policy evaluation (judge rescue accuracy, AURC,
    cluster bootstrap, cost curve) needs judge labels on the routed queries — future spend."""
    top2 = M.topk(min(2, M.shape[1]), dim=1).values
    margin = (top2[:, 0] - top2[:, 1]) if top2.shape[1] > 1 else torch.zeros(M.shape[0])
    ranks = torch.tensor(ranks_from(M, truepos_sets), dtype=torch.float32)
    rows = []
    for t in thresholds:
        kept = margin >= t
        routed = 1.0 - kept.float().mean().item()
        kept_r1 = float((ranks[kept] <= 1).float().mean().item()) if kept.any() else float("nan")
        rows.append((t, routed, int(kept.sum().item()), kept_r1))
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
    ap.add_argument("--cand-lineage", action="store_true",
                    help="§7: supply each candidate folder's principal path (assembled DAG) as anc "
                         "tokens — evaluate a checkpoint trained with --cand-lineage in its regime")
    ap.add_argument("--shuffle-lineage", action="store_true",
                    help="control: permute which folder gets which lineage (right vs any lineage)")
    ap.add_argument("--lineage-depth", type=int, default=5)
    a = ap.parse_args(argv)
    dev = "cpu"
    torch.set_num_threads(4)

    queries, cand = load_filing(a.trees, a.min_bm)
    import hashlib
    import random
    queries = sorted(queries)          # glob() order is fs-dependent; sort BEFORE seeded sampling
    if len(queries) > a.max_queries:
        queries = random.Random(a.seed).sample(queries, a.max_queries)
    f_ids = sorted(cand)
    f_titles = [cand[fid] for fid in f_ids]
    q_titles = [q for q, _ in queries]
    # duplicate folder titles are indistinguishable to a title-keyed model → equivalence sets
    by_title = {}
    for j, t in enumerate(f_titles):
        by_title.setdefault(t, []).append(j)
    truepos = [sorted(by_title[cand[fid]]) for _, fid in queries]
    n_alias = sum(1 for tp in truepos if len(tp) > 1)
    qman = hashlib.sha256("\n".join(f"{q}\t{fid}" for q, fid in queries).encode()).hexdigest()
    print(f"filing eval: {len(queries)} bookmark queries over {len(f_ids)} candidate folders "
          f"(min_bm {a.min_bm}, seed {a.seed}); {n_alias} queries hit duplicate-title folder sets; "
          f"query manifest sha256 {qman[:16]}")

    train_nodes = campaign_train_nodes(a.split_seed)
    held_q = [i for i in range(len(queries)) if f_titles[truepos[i][0]] not in train_nodes]
    print(f"held-folder subset (true folder unseen in fine-tune train nodes): {len(held_q)}/{len(queries)}")

    parents_title, anc_titles = {}, set()
    if a.cand_lineage:
        parents_title, anc_titles = folder_lineage(
            cand, depth=a.lineage_depth,
            shuffle_seed=(a.seed if a.shuffle_lineage else None))
        covered = sum(1 for ft in set(f_titles) if ft in parents_title)
        print(f"candidate lineage: {covered}/{len(set(f_titles))} folders have a principal path; "
              f"{len(anc_titles)} ancestor titles"
              + (" [SHUFFLED control]" if a.shuffle_lineage else ""))
    names = sorted(set(q_titles) | set(f_titles) | anc_titles)
    cache_path = a.cache.rsplit(".pt", 1)[0] + ("_lin.pt" if a.cand_lineage else ".pt")
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=cache_path, batch_size=128)
    tok = Tokenizer(qtbl, ptbl, idx, parents_title, {},
                    root_lineage=a.cand_lineage, root_lineage_depth=a.lineage_depth)

    results = {}
    for label, ckpt in (("base", a.base), ("tuned", a.tuned)):
        torch.manual_seed(0)   # deterministic fresh LINEAGE readout rows when the base ckpt lacks them
        model, _ = load_with_lineage_ops(ckpt, dev=dev)
        model.eval()
        # stock agnostic matrices (rank_all's definitions, graded here with equivalence sets so
        # duplicate-title folders never lose to candidate-order tie-breaking)
        n_ops = model.op_emb.weight.shape[0]
        ow_of = lambda op: torch.zeros(1, n_ops).index_fill_(1, torch.tensor([OPS[op]]), 1.0)
        sm = lambda ow: torch.tensor(score_mu(model, tok, idx, q_titles, f_titles, ow, dev))
        A_elem, A_hier, A_sym = sm(ow_of("ELEM")), sm(ow_of("HIER")), sm(ow_of("SYM"))
        A_super = sm(torch.full((1, n_ops), 1.0 / n_ops))
        A_max = torch.maximum(torch.maximum(A_elem, A_hier), A_sym)
        C = (qtbl[[idx[k] for k in q_titles]] @ ptbl[[idx[k] for k in f_titles]].T).cpu()

        def nzrow(M):
            lo = M.min(dim=1, keepdim=True).values
            hi = M.max(dim=1, keepdim=True).values
            return (M - lo) / (hi - lo + 1e-9)

        Cz, Sz = nzrow(C), nzrow(A_max)
        A_blend = 0.1 * Cz + 0.9 * Sz
        top2 = A_max.topk(min(2, A_max.shape[1]), dim=1).values
        marg = (top2[:, 0] - top2[:, 1]).clamp(min=0)
        mq = marg.argsort().argsort().float() / max(1, len(marg) - 1)
        alpha = (0.3 + 0.6 * mq).unsqueeze(1)
        A_gate = (1 - alpha) * Cz + alpha * Sz
        # conditioned rankers — where the fine-tune is allowed to show up
        S_elem = score_cond(model, tok, q_titles, f_titles, "ELEM", "kalman-fused", dev)
        S_hier = score_cond(model, tok, q_titles, f_titles, "HIER", "kalman-fused", dev)
        S_sym = score_cond(model, tok, q_titles, f_titles, "SYM", "kalman-fused", dev)
        S_lin = score_cond(model, tok, q_titles, f_titles, "LINEAGE", "graph", dev)
        S_maxc = torch.maximum(torch.maximum(S_elem, S_hier), S_sym)
        S_maxl = torch.maximum(S_maxc, S_lin)
        matrices = (("e5-cos", C), ("mu-super", A_super), ("mu-elem", A_elem), ("mu-max", A_max),
                    ("e5+mu-max", A_blend), ("margin-gate", A_gate),
                    ("mu-elem-cond", S_elem), ("mu-lineage", S_lin),
                    ("mu-max-cond", S_maxc), ("mu-max+lineage", S_maxl))
        rank_of = {nm: ranks_from(M, truepos) for nm, M in matrices}
        order = tuple(nm for nm, _ in matrices)
        results[label] = dict(rank_of=rank_of, order=order, S_maxc=S_maxc, C=C)

        print(f"\n=== {label}: {os.path.basename(ckpt)} ===")
        print(f"  {'ranker':16} {'recall@1':>9} {'recall@5':>9} {'recall@10':>10} {'MRR':>7} {'med.rank':>9}")
        for nm in order:
            m = metrics(rank_of[nm])
            print(f"  {nm:16} {m['recall@1']:9.3f} {m['recall@5']:9.3f} {m['recall@10']:10.3f} "
                  f"{m['MRR']:7.3f} {m['median_rank']:9d}")
        if held_q:
            print(f"  transductive held-folder subset (n={len(held_q)}; endpoint-only definition — "
                  "some true folders still appear as ancestor CONTEXT in training):")
            for nm in ("mu-max-cond", "mu-max+lineage", "e5-cos"):
                m = metrics([rank_of[nm][i] for i in held_q])
                print(f"    {nm:16} recall@1 {m['recall@1']:.3f}  recall@5 {m['recall@5']:.3f}  "
                      f"MRR {m['MRR']:.3f}")

    # escalation margins on the DEPLOYED ranker (e5) and the like-for-like conditioned head —
    # descriptive only; judge-rescue/AURC evaluation needs judge labels on routed queries
    for head, key in (("e5-cos (deployed ranker)", "C"), ("mu-max-cond", "S_maxc")):
        print(f"\nescalation margins [{head}] (routed = margin < t; kept_n in parens):")
        print(f"  {'threshold':>9} | {'base routed':>11} {'base kept R@1':>16} | "
              f"{'tuned routed':>12} {'tuned kept R@1':>17}")
        curve_b = escalation_curve(results["base"][key], truepos)
        curve_t = escalation_curve(results["tuned"][key], truepos)
        for (t, rb, nb, kb), (_, rt, nt, kt) in zip(curve_b, curve_t):
            print(f"  {t:9.2f} | {rb:11.3f} {kb:10.3f} ({nb:3d}) | {rt:12.3f} {kt:11.3f} ({nt:3d})")

    for nm in ("mu-max-cond", "mu-max+lineage"):
        d = metrics(results["tuned"]["rank_of"][nm])["MRR"] - metrics(results["base"]["rank_of"][nm])["MRR"]
        print(f"paired MRR delta (tuned - base, {nm}): {d:+.4f}")
    print("NOTE: the base LINEAGE readout is a fresh random row (the base ckpt predates the op) — "
          "mu-max+lineage deltas mostly measure training-a-random-head; mu-max-cond is the "
          "like-for-like conditioned comparison.")


if __name__ == "__main__":
    main()
