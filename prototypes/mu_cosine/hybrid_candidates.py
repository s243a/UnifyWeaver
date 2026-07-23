#!/usr/bin/env python3
"""Hybrid candidate-generation harness: can μ / graph sources lift the e5 recall@50 ceiling?

MEASUREMENT ONLY — no fitted combiner, no deployment claim. Two questions, in order:

  A (recall ceiling)  candidate-recall@50 = 0.680 caps every ranker (filing_ranker.py). Do μ- or
     graph-sourced candidates ADD true folders the e5 top-50 misses, at a MATCHED budget of 50?
     Sources: e5 top-K (baseline) | μ_LINEAGE top-K (trained filing head) | graph lineage
     neighborhood of the bookmark's anchor. Unions are budget-matched (e5 top-(50−m) + m fill
     slots from the source, deduped) so recall gains are never an artifact of a bigger pool.
     Unmatched full unions are reported separately as upper bounds.

  B (precondition, per the project owner): for μ to deserve ranking weight anywhere, μ must beat
     e5's distance at least in regions with dense nearby training. Stratify queries by
     (i) the graph region bin of (anchor → true folder) — hop h1..h5 / sib / cous / rand /
         missing, via fit_bias_states.pair_distance_features + soft_bin_weights (hard argmax);
     (ii) the TRUE folder's training mass — # of campaign TRAIN-side rows (node-disjoint seed-0
          split, the fine-tune's own split) touching the folder title or a 1-hop DAG neighbor.
     Report per-stratum MRR/median-rank of e5 vs μ over the FULL folder catalog (identical pools).
     The campaign sampled hop bins ~uniformly (~50 rows each), so BIN-level density is flat by
     design — the node-level training mass is the honest "closely trained neighbours" axis.

  Stratification uses the true folder (outcome-AWARE) — legitimate for a descriptive diagnostic,
  never for feature construction. All candidate sources are outcome-blind. Ranks are graded by
  title-equivalence (eval_pearltrees_filing.ranks_from). Query sampling matches filing_ranker.py
  (seed 7, ≤1200) so recall@50 is comparable with the standing 0.680.

  python3 hybrid_candidates.py                 # one torch job (μ scoring; cached after first run)
  python3 hybrid_candidates.py --skip-mu       # torch-free: graph source + e5 baseline only
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_pearltrees_filing import campaign_train_nodes, ranks_from
from filing_ranker import load_graph_universe
from fit_bias_states import BINS, pair_distance_features, soft_bin_weights
from mu_attention import build_e5_tables

ROOT = os.path.dirname(os.path.abspath(__file__))


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def d_sym_to_candidates(parents_dir, anchor, f_node_ids, hmax=6, cap=13):
    """d_sym(anchor, f) for every candidate folder id, np.inf when no common ancestor."""
    from sample_channel_campaign import ancestors

    anc_a = ancestors(parents_dir, anchor, hmax)
    anc_a[anchor] = 0
    out = np.full(len(f_node_ids), np.inf)
    for j, f in enumerate(f_node_ids):
        anc_f = ancestors(parents_dir, f, hmax)
        anc_f[f] = 0
        common = set(anc_a) & set(anc_f)
        if common:
            out[j] = min(min(anc_a[c] + anc_f[c] for c in common), cap)
    return out


def budget_union(e5_order, fill_orders, K, m_fill):
    """e5 top-(K − Σm) plus, per source, its top-m unseen candidates; any slots a sparse source
    can't fill are backfilled from e5's continuation (budget always exactly K)."""
    base = K - sum(m_fill.values())
    sel = list(e5_order[:base])
    have = set(sel)
    for name, order in fill_orders.items():
        m = m_fill[name]
        for c in order:
            if m == 0:
                break
            if c not in have:
                sel.append(c)
                have.add(c)
                m -= 1
    for c in e5_order[base:]:
        if len(sel) >= K:
            break
        if c not in have:
            sel.append(c)
            have.add(c)
    return sel


def recall_of(sel_lists, truepos):
    hits = [any(c in set(tp) for c in sel) for sel, tp in zip(sel_lists, truepos)]
    return float(np.mean(hits)), np.array(hits)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-queries", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7, help="query-sampling seed (= filing_ranker)")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_pt_filing_lin.pt"),
                    help="trained μ checkpoint (LINEAGE head; the Pearltrees single-path decision)")
    ap.add_argument("--judge", default="graph", help="judge conditioning for the LINEAGE head")
    ap.add_argument("--skip-mu", action="store_true", help="torch-free: skip the μ source")
    ap.add_argument("--mu-cache", default="/tmp/mu_data/hybrid_cand_mu.npz")
    ap.add_argument("--e5-cache", default="/tmp/mu_data/pt_ranker_e5.pt")
    ap.add_argument("--fill", type=int, default=10, help="budget slots granted to each fill source")
    a = ap.parse_args(argv)

    universe, titles, neighbors, parents_dir, queries, cand, cut_ext = load_graph_universe(a.hops)

    import random
    queries = sorted(queries)
    if len(queries) > a.max_queries:
        queries = random.Random(a.seed).sample(queries, a.max_queries)
    f_ids = sorted(cand)
    f_titles = [cand[fid] for fid in f_ids]
    by_title = {}
    for j, t in enumerate(f_titles):
        by_title.setdefault(t, []).append(j)
    q_titles = [q for q, _ in queries]
    truepos = [sorted(by_title[cand[fid]]) for _, fid in queries]
    qman = hashlib.sha256("\n".join(f"{q}\t{fid}" for q, fid in queries).encode()).hexdigest()
    B, K = len(queries), a.top_k
    print(f"queries: {B} (manifest sha {qman[:16]}); catalog: {len(f_ids)} folders; budget K={K}")

    # e5 tables — names constructed exactly as filing_ranker.py so its cache is reusable
    ext_nodes = sorted({x for xs in cut_ext.values() for x in xs})
    names = sorted(set(q_titles) | set(f_titles) | {titles[n] for n in universe}
                   | {titles[x] for x in ext_nodes})
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=a.e5_cache, batch_size=128)
    Q, P = qtbl.numpy(), ptbl.numpy()
    uni_vec = np.stack([P[idx[titles[n]]] for n in universe])
    cand_vec = np.stack([P[idx[t]] for t in f_titles])
    qv = np.stack([Q[idx[t]] for t in q_titles])
    cos_cand = qv @ cand_vec.T                                   # [B, n_cand]
    cos_uni = qv @ uni_vec.T
    anchors = [universe[int(i)] for i in np.argmax(cos_uni, axis=1)]   # outcome-blind anchors
    e5_orders = np.argsort(-cos_cand, axis=1)

    # ---- μ_LINEAGE scores over the full catalog (one torch job; cached) ----
    M_mu = None
    if not a.skip_mu:
        mu_key = hashlib.sha256(json.dumps(
            [qman, len(f_ids), file_sha(a.ckpt), a.judge, "LINEAGE"]).encode()).hexdigest()[:16]
        if os.path.exists(a.mu_cache) and str(np.load(a.mu_cache)["key"]) == mu_key:
            M_mu = np.load(a.mu_cache)["M"]
            print(f"μ scores loaded from cache ({a.mu_cache})")
        else:
            import torch

            from eval_pearltrees_filing import score_cond
            from fine_tune_pearltrees_filing import load_with_lineage_ops
            from mu_attention import Tokenizer
            torch.set_num_threads(4)
            torch.manual_seed(0)
            mdl, _ = load_with_lineage_ops(a.ckpt, dev="cpu")
            mdl.eval()
            tok = Tokenizer(qtbl, ptbl, idx, {}, {})
            print(f"scoring μ_LINEAGE({os.path.basename(a.ckpt)}, judge={a.judge}) "
                  f"for {B}×{len(f_ids)} pairs …")
            M_mu = np.array(score_cond(mdl, tok, q_titles, f_titles, "LINEAGE", a.judge))
            os.makedirs(os.path.dirname(a.mu_cache), exist_ok=True)
            np.savez_compressed(a.mu_cache, M=M_mu, key=mu_key)
            print(f"μ scores cached -> {a.mu_cache}")
        mu_orders = np.argsort(-M_mu, axis=1)

    # ---- graph source: candidates by d_sym to the anchor (outcome-blind), e5 tie-break ----
    print("graph source: d_sym(anchor → candidate) …")
    fid_strs = [str(f) for f in f_ids]
    dsym_cache = {}
    graph_orders = np.zeros((B, len(f_ids)), dtype=int)
    graph_dsym = np.zeros((B, len(f_ids)))
    for b in range(B):
        anc = anchors[b]
        if anc not in dsym_cache:
            dsym_cache[anc] = d_sym_to_candidates(parents_dir, anc, fid_strs)
        d = dsym_cache[anc]
        graph_dsym[b] = d
        graph_orders[b] = np.lexsort((-cos_cand[b], d))          # near first, e5 breaks ties
        if (b + 1) % 300 == 0:
            print(f"  {b + 1}/{B}")

    # ================= A: budget-matched recall =================
    print("\n=== A. candidate-recall@%d (budget-matched; baseline must reproduce 0.680) ===" % K)
    r_e5, hit_e5 = recall_of([e5_orders[b][:K] for b in range(B)], truepos)
    print(f"  e5@{K}                          : {r_e5:.3f}")
    rows = []
    variants = {}
    graph_fill = [[c for c in graph_orders[b] if np.isfinite(graph_dsym[b][c])] for b in range(B)]
    variants[f"e5@{K - a.fill} + graph@{a.fill}"] = [
        budget_union(e5_orders[b], {"g": graph_fill[b]}, K, {"g": a.fill}) for b in range(B)]
    if M_mu is not None:
        variants[f"e5@{K - a.fill} + mu@{a.fill}"] = [
            budget_union(e5_orders[b], {"m": mu_orders[b]}, K, {"m": a.fill}) for b in range(B)]
        variants[f"e5@{K - a.fill} + mu@{a.fill // 2} + graph@{a.fill - a.fill // 2}"] = [
            budget_union(e5_orders[b],
                         {"m": mu_orders[b], "g": graph_fill[b]}, K,
                         {"m": a.fill // 2, "g": a.fill - a.fill // 2}) for b in range(B)]
    for name, sels in variants.items():
        r, hit = recall_of(sels, truepos)
        rescues = int((hit & ~hit_e5).sum())
        lost = int((hit_e5 & ~hit).sum())
        rows.append((name, r, rescues, lost))
        print(f"  {name:32s}: {r:.3f}  (+{rescues} rescued, −{lost} displaced vs e5@{K})")
    # unmatched upper bounds — pool-size artifacts allowed, labeled as such
    ub = [list(e5_orders[b][:K]) + [c for c in graph_fill[b]] for b in range(B)]
    r_ubg, _ = recall_of(ub, truepos)
    print(f"  UPPER BOUND e5@{K} ∪ graph(all finite d_sym): {r_ubg:.3f}  (unmatched pool)")
    if M_mu is not None:
        ub2 = [list(e5_orders[b][:K]) + list(mu_orders[b][:K]) for b in range(B)]
        r_ubm, _ = recall_of(ub2, truepos)
        print(f"  UPPER BOUND e5@{K} ∪ mu@{K}              : {r_ubm:.3f}  (unmatched pool)")

    # ================= B: per-region μ vs e5 precondition =================
    print("\n=== B. precondition (DESCRIPTIVE): does μ beat e5 where training is dense? ===")
    train_nodes = campaign_train_nodes(0)
    # node-level training mass: campaign TRAIN rows touching the folder title or a 1-hop neighbor
    from node_disjoint_eval import node_disjoint_pair_split
    from run_pearltrees_fusion import load_pearltrees_campaign
    ds = load_pearltrees_campaign(require_55=False)
    strata = [("principal" if t.startswith("principal") else t) for t in ds["tags"]]
    split = node_disjoint_pair_split(ds["pairs"], 0, strata=strata)
    train_rows = [ds["pairs"][i] for i in split.train]
    touch = Counter()
    for x, y in train_rows:
        touch[x] += 1
        touch[y] += 1
    id_by_title = defaultdict(list)
    for n in universe:
        id_by_title[titles[n]].append(n)

    def train_mass(fid):
        t = cand[fid]
        mass = touch.get(t, 0)
        for n in id_by_title.get(t, []):
            for nb in neighbors.get(n, []):
                mass += touch.get(titles[nb], 0)
        return mass

    # region bin of (anchor → true folder), via the bias-state binning (hard argmax)
    pairs_at = [(anchors[b], str(queries[b][1])) for b in range(B)]
    feats = pair_distance_features(parents_dir, pairs_at)
    Wb = soft_bin_weights(feats, tau=0.25)
    bin_of = [BINS[int(i)] for i in np.argmax(Wb, axis=1)]

    strat_defs = {"region bin (anchor→true)": bin_of}
    mass_all = np.array([train_mass(queries[b][1]) for b in range(B)])
    qm = np.quantile(mass_all[mass_all > 0], [0.5]) if (mass_all > 0).any() else [1]
    mass_bin = ["zero" if m == 0 else ("low" if m <= qm[0] else "high") for m in mass_all]
    strat_defs["train mass of true folder (0 / ≤med / >med)"] = mass_bin
    held_bin = ["train-node" if cand[queries[b][1]] in train_nodes else "held-node"
                for b in range(B)]
    strat_defs["true folder in μ's train split?"] = held_bin

    if M_mu is None:
        print("  (μ skipped — rerun without --skip-mu for the comparison)")
    else:
        import torch
        ranks_e5 = ranks_from(torch.tensor(cos_cand), truepos)
        ranks_mu = ranks_from(torch.tensor(M_mu), truepos)
        re5, rmu = np.array(ranks_e5, float), np.array(ranks_mu, float)
        for title_, lab in strat_defs.items():
            print(f"  -- by {title_} --")
            print(f"  {'stratum':12s} {'n':>5s} {'MRR_e5':>8s} {'MRR_mu':>8s} "
                  f"{'med_e5':>7s} {'med_mu':>7s} {'mu wins':>8s}")
            order = [b_ for b_ in BINS] + ["zero", "low", "high", "train-node", "held-node"]
            seen_s = sorted(set(lab), key=lambda s: order.index(s) if s in order else 99)
            for s in seen_s:
                m = np.array([x == s for x in lab])
                if m.sum() == 0:
                    continue
                wins = float(np.mean(rmu[m] < re5[m]))
                print(f"  {s:12s} {int(m.sum()):5d} {np.mean(1 / re5[m]):8.3f} "
                      f"{np.mean(1 / rmu[m]):8.3f} {np.median(re5[m]):7.0f} "
                      f"{np.median(rmu[m]):7.0f} {wins:8.2f}")
        print("\n  Read: μ earns ranking weight only in strata where MRR_mu ≥ MRR_e5 (esp. "
              "high-mass / train-node rows). All DESCRIPTIVE — stratification uses the true "
              "folder; no combiner is fit here.")

    print("\nDone. (Phase A budget-matched numbers are the ceiling result; Phase B is the "
          "μ-certainty precondition.)")


if __name__ == "__main__":
    main()
