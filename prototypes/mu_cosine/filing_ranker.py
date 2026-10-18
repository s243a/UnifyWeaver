#!/usr/bin/env python3
"""Practical Pearltrees filing ranker: a thin, honest feature blend over recorded placements.

The standing result (REPORT_pearltrees_candidate_lineage.md): e5-cos ranks filing at MRR 0.294 and
every learned μ head sits ≤0.11 — so the deliverable is e5 + graph features with a small blend, not
a learned ranker. No LLM scoring: ground truth = real recorded placements (each bookmark's treeId).

Features per (bookmark, candidate folder) — ALL OUTCOME-BLIND (no placement labels anywhere in
feature construction; labels appear only in the blend fit and the metrics):
  e5_cos    query-bookmark → passage-folder cosine (the champion baseline, kept as a feature)
  h_s       grounded-diffusion screening score (docs/design/LEAKY_GRAPH_DIFFUSION.md): unit source
            current injected at the bookmark's top-M e5-nearest GRAPH folders (weights ∝ cosine),
            diffused through the folder topology with e5-RBF semantic conductance and uniform
            leakage α; h_s(f) = equilibrium response at candidate f. α is calibrated OUTCOME-BLIND
            by the e-fold recipe on a frozen 2-hop shell; ℓ = median edge e5 distance; ε floor
            recorded. Dense float64 reference at this scale (~6k nodes), per the design doc.
  hit_fwd   hit_prob(anchor → candidate) on the DIRECTED record-majority principal-parent graph
  hit_rev   hit_prob(candidate → anchor)      (anchor = bookmark's e5-nearest graph folder)
  sym4      sym_graph_features(anchor, candidate): 1/(1+d_sym), shared parent, shared gp, is_anc
  mu3       (optional, --mu-feats) the UNTRAINED base model's agnostic μ under ELEM/HIER/SYM —
            losers as rankers (≤0.11) but may earn their keep as features

Blend: weighted ridge (closed-form, deterministic; positives up-weighted by the candidate ratio)
over standardized features, trained on recorded placements with NODE-DISJOINT folds — the held
pairs' bookmarks AND true folders are unseen as training identities (node_disjoint_eval
conventions; training rows additionally exclude any candidate that is a held node).

Report: MRR/recall@1/@5 vs the e5-cos-only baseline on IDENTICAL candidate lists and folds,
drop-one-family ablations, and the escalation curve (fraction routed vs top-2 margin). Duplicate
folder titles are graded by title-equivalence (best-alias rank). An honest null is reportable.

  python3 filing_ranker.py --mu-feats            # full feature set (one torch job)
  python3 filing_ranker.py                       # torch-free feature set
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from emit_transitive_hops import hit_prob
from eval_filing import load_filing, metrics
from mu_attention import build_e5_tables
from node_disjoint_eval import node_disjoint_pair_split, paired_node_bootstrap_ci
from run_sym_channel_fusion import sym_graph_features
from unifyweaver.graph.leaky_diffusion import (
    build_grounded_semantic_diffusion,
    combinatorial_laplacian,
    semantic_conductance_matrix,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
PT_API = os.path.join(ROOT, "..", "..", ".local", "data", "pearltrees_api")
TREES = os.path.join(PT_API, "trees")
DAG = os.path.join(PT_API, "assembled_dag.tsv")
TITLES = os.path.join(PT_API, "assembled_titles.tsv")
PATHS_JSONL = os.path.join(PT_API, "..", "api_tree_paths_v8.jsonl")

FAMILIES = {
    "e5": ["e5_cos"],
    "diffusion": ["h_s"],
    "walk": ["hit_fwd", "hit_rev"],
    "sym": ["inv_d_sym", "shared_par", "shared_gp", "is_anc"],
    "mu": ["mu_elem", "mu_hier", "mu_sym"],
}


def load_graph_universe(hops=2):
    """Candidate folders' `hops`-hop titled undirected neighborhood of the assembled DAG.

    The diffusion operator requires a symmetric network (design doc §directed graphs): we take the
    undirected union of DAG edges — appropriate here because screening asks "is this candidate in
    the bookmark's semantic-topological neighborhood", not a directional membership question (the
    directional features are hit_prob/sym4 on the DIRECTED record-majority graph)."""
    titles = {}
    for ln in open(TITLES, encoding="utf-8"):
        parts = ln.rstrip("\n").split("\t")
        if len(parts) >= 2:
            titles[parts[0]] = parts[1]
    edges = []
    adj = defaultdict(set)
    for ln in open(DAG, encoding="utf-8"):
        p, c = ln.split()
        edges.append((p, c))
        adj[p].add(c)
        adj[c].add(p)
    queries, cand = load_filing(TREES, 3)
    seeds = {str(t) for t in cand}
    seen, frontier = set(seeds), set(seeds)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt |= adj[n]
        nxt -= seen
        seen |= nxt
        frontier = nxt
    universe = sorted(n for n in seen if n in titles)
    uset = set(universe)
    neighbors = {n: sorted(adj[n] & uset) for n in universe}
    # DIRICHLET boundary (audit finding 1): edges cut by the 2-hop truncation must become shunts to
    # the bath, not vanish (an insulating boundary changes the operator, α calibration, and h_s).
    # Record each boundary node's titled exterior neighbors; their cut-edge conductance is added to
    # that node's leakage in main().
    cut_ext = {n: sorted((adj[n] - uset) & set(titles)) for n in universe if adj[n] - uset}

    # DIRECTED principal-parent map: record-majority first (the audit rule), DAG-edge fallback
    votes = defaultdict(Counter)
    with open(PATHS_JSONL, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            ids = [str(x).split(":")[-1] for x in (r.get("path_ids") or [])]
            for p, c in zip(ids, ids[1:]):
                if p != c:
                    votes[c][p] += 1
    principal = {c: max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0] for c, cnt in votes.items()}
    dag_first = {}
    for p, c in edges:
        dag_first.setdefault(c, p)
    parents_dir = {}
    for n in universe:
        par = principal.get(n, dag_first.get(n))
        if par is not None and par in uset:
            parents_dir[n] = [par]
    return universe, titles, neighbors, parents_dir, queries, cand, cut_ext


def calibrate_alpha_efold(universe, neighbors, emb, ell, eps, base_leak=None, shell_hops=2,
                          target=np.exp(-1.0), probe_stride=25, lo=1e-4, hi=10.0, iters=14):
    """OUTCOME-BLIND e-fold calibration of the uniform leakage α on a frozen shell.

    Pick α so that the MEDIAN (over a fixed probe set of source folders) of
    [median equilibrium response on the shell at `shell_hops` graph hops] / [response at source]
    equals 1/e. `base_leak` (node-aligned) carries the FIXED Dirichlet boundary shunts (cut-edge
    conductance); the bisected uniform α sits ON TOP of it. Uses graph + embeddings only; bisection
    with a light Cholesky path (the full fail-closed contract is applied once at the final α)."""
    nodes, cond = semantic_conductance_matrix(universe, neighbors, emb, length_scale=ell,
                                              conductance_floor=eps)
    L = combinatorial_laplacian(cond)
    if base_leak is not None:
        L = L + np.diag(base_leak)
    index = {n: i for i, n in enumerate(nodes)}
    adj = {n: set(neighbors[n]) for n in universe}
    probes = universe[::probe_stride]
    shells = {}
    for s in probes:
        ring, seen = {s}, {s}
        for _ in range(shell_hops):
            ring = set().union(*(adj[n] for n in ring)) - seen
            seen |= ring
        if ring:
            shells[s] = sorted(ring)

    def ratio(alpha):
        J = L + alpha * np.eye(len(nodes))
        cho = np.linalg.cholesky(J)
        rs = []
        for s, ring in shells.items():
            q = np.zeros(len(nodes))
            q[index[s]] = 1.0
            u = np.linalg.solve(cho.T, np.linalg.solve(cho, q))
            rs.append(float(np.median(u[[index[r] for r in ring]]) / u[index[s]]))
        return float(np.median(rs))

    trace = []
    for _ in range(iters):
        mid = np.sqrt(lo * hi)                    # log-space bisection (α spans decades)
        r = ratio(mid)
        trace.append((mid, r))
        if r > target:                            # too little decay → more leakage
            lo = mid
        else:
            hi = mid
    alpha = float(np.sqrt(lo * hi))
    return alpha, {"shell_hops": shell_hops, "target": float(target), "probes": len(shells),
                   "probe_stride": probe_stride, "iterations": iters,
                   "final_ratio": ratio(alpha), "trace": [(round(a, 6), round(r, 4)) for a, r in trace]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=50, help="e5 candidate folders per bookmark")
    ap.add_argument("--inject-m", type=int, default=20, help="graph nodes receiving source current")
    ap.add_argument("--max-queries", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--folds", type=int, default=5, help="node-disjoint fold seeds 0..folds-1")
    ap.add_argument("--eps", type=float, default=0.05, help="conductance floor (recorded)")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--hops", type=int, default=2, help="graph-universe radius around candidates")
    ap.add_argument("--mu-feats", action="store_true", help="add the base model's agnostic μ features")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod_namecond_full.pt"))
    ap.add_argument("--cache", default="/tmp/mu_data/pt_ranker_features.npz")
    ap.add_argument("--e5-cache", default="/tmp/mu_data/pt_ranker_e5.pt")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)

    universe, titles, neighbors, parents_dir, queries, cand, cut_ext = load_graph_universe(a.hops)
    print(f"graph universe: {len(universe)} titled folders (radius {a.hops}); "
          f"candidate folders: {len(cand)}; principal-parent edges: {len(parents_dir)}; "
          f"boundary nodes with cut edges: {len(cut_ext)}")

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
    print(f"queries: {len(queries)} (manifest sha {qman[:16]})")

    ext_nodes = sorted({x for xs in cut_ext.values() for x in xs})
    names = sorted(set(q_titles) | set(f_titles) | {titles[n] for n in universe}
                   | {titles[x] for x in ext_nodes})
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=a.e5_cache, batch_size=128)
    Q = qtbl.numpy()
    P = ptbl.numpy()
    uni_vec = np.stack([P[idx[titles[n]]] for n in universe])            # graph nodes, passage space
    cand_vec = np.stack([P[idx[t]] for t in f_titles])

    # semantic length scale: median e5 distance over realized edges (outcome-blind, recorded)
    dists = []
    ui = {n: i for i, n in enumerate(universe)}
    for n in universe:
        for m in neighbors[n]:
            if n < m:
                dists.append(float(np.linalg.norm(uni_vec[ui[n]] - uni_vec[ui[m]])))
    ell = float(np.median(dists))
    emb = {n: uni_vec[ui[n]] for n in universe}
    # DIRICHLET boundary shunts (audit finding 1): each truncated edge's semantic conductance —
    # same RBF-with-floor formula as interior edges — leaks to the bath at its boundary node,
    # replacing the (wrong) insulating truncation.
    ext_vec = {x: P[idx[titles[x]]] for x in ext_nodes}
    cut_leak = np.zeros(len(universe))
    n_cut = 0
    for n, xs in cut_ext.items():
        zi = emb[n]
        for x in xs:
            d2 = float(np.sum((zi - ext_vec[x]) ** 2))
            cut_leak[ui[n]] += a.eps + (1.0 - a.eps) * np.exp(-d2 / (2.0 * ell * ell))
            n_cut += 1
    print(f"Dirichlet boundary: {n_cut} cut edges → shunt mass {cut_leak.sum():.1f} over "
          f"{int((cut_leak > 0).sum())} boundary nodes")
    alpha, alpha_log = calibrate_alpha_efold(universe, neighbors, emb, ell, a.eps,
                                             base_leak=cut_leak)
    print(f"diffusion params (outcome-blind): ell={ell:.4f} (median edge e5 dist), eps={a.eps}, "
          f"uniform alpha={alpha:.5f} ON TOP of boundary shunts, via e-fold on frozen "
          f"{alpha_log['shell_hops']}-hop shell ({alpha_log['probes']} probes, "
          f"final ratio {alpha_log['final_ratio']:.3f})")

    leak_map = {n: float(alpha + cut_leak[ui[n]]) for n in universe}
    model = build_grounded_semantic_diffusion(
        universe, neighbors, leakage_conductance=leak_map, node_embeddings=emb,
        length_scale=ell, conductance_floor=a.eps)
    print(f"grounded precision: cond {model.condition_number:.1f} "
          f"(reciprocal {model.reciprocal_condition_number:.2e}; float64 contract passed)")
    root = model.precision_root                       # J = rootᵀ root

    def file_sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    # full provenance key (audit finding 5): source data, e5 cache, checkpoint, feature schema
    feat_names = FAMILIES["e5"] + FAMILIES["diffusion"] + FAMILIES["walk"] + FAMILIES["sym"] + (
        FAMILIES["mu"] if a.mu_feats else [])
    cache_key = hashlib.sha256(json.dumps(
        [qman, a.top_k, a.inject_m, a.hops, a.eps, round(ell, 6), round(alpha, 6),
         round(float(cut_leak.sum()), 3), a.mu_feats, len(universe), feat_names,
         file_sha(DAG), file_sha(TITLES), file_sha(PATHS_JSONL),
         file_sha(a.e5_cache) if os.path.exists(a.e5_cache) else "",
         file_sha(a.ckpt) if a.mu_feats else ""]).encode()).hexdigest()[:16]
    if os.path.exists(a.cache) and np.load(a.cache, allow_pickle=True)["key"] == cache_key:
        z = np.load(a.cache, allow_pickle=True)
        X, cand_lists = z["X"], z["cand_lists"]
        print(f"features loaded from cache ({a.cache})")
    else:
        qv = np.stack([Q[idx[t]] for t in q_titles])
        cos_cand = qv @ cand_vec.T                                       # [B, n_cand]
        cos_uni = qv @ uni_vec.T                                         # [B, n_universe]
        cand_lists = np.argsort(-cos_cand, axis=1)[:, :a.top_k]          # e5 top-K candidate ids
        B, K = len(q_titles), a.top_k
        X = np.zeros((B, K, len(feat_names)))
        cand_ui = np.array([ui.get(str(fid), -1) for fid in f_ids])      # candidate → universe row
        anc_cache = {}
        for b in range(B):
            # diffusion source: unit current split over the bookmark's top-M graph nodes ∝ cosine⁺
            top = np.argsort(-cos_uni[b])[:a.inject_m]
            w = np.clip(cos_uni[b][top], 0.0, None)
            q_vec = np.zeros(len(universe))
            if w.sum() > 0:
                q_vec[top] = w / w.sum()
            u = np.linalg.solve(root, np.linalg.solve(root.T, q_vec))    # J u = q via the root
            anchor = universe[int(np.argmax(cos_uni[b]))]
            for k in range(K):
                ci = cand_lists[b, k]
                fnode = str(f_ids[ci])
                row = [float(cos_cand[b, ci]),
                       float(u[cand_ui[ci]]) if cand_ui[ci] >= 0 else 0.0]
                key = (anchor, fnode)
                if key not in anc_cache:
                    hf = hit_prob(parents_dir, anchor, fnode) if fnode in ui else 0.0
                    hr = hit_prob(parents_dir, fnode, anchor) if fnode in ui else 0.0
                    s4 = sym_graph_features(parents_dir, [(anchor, fnode)])[0] if fnode in ui \
                        else np.zeros(4)
                    anc_cache[key] = (hf, hr, s4)
                hf, hr, s4 = anc_cache[key]
                row += [hf, hr, *s4.tolist()]
                X[b, k, :len(row)] = row
            if (b + 1) % 200 == 0:
                print(f"  features {b + 1}/{B}")
        if a.mu_feats:
            import torch
            from fine_tune_channel_heads import mu_batch
            from fine_tune_pearltrees_filing import load_with_lineage_ops
            from mu_attention import OPS, Tokenizer
            torch.set_num_threads(4)
            torch.manual_seed(0)
            mdl, _ = load_with_lineage_ops(a.ckpt, dev="cpu")
            mdl.eval()
            tok = Tokenizer(qtbl, ptbl, idx, {}, {})
            col0 = len(feat_names) - 3
            with torch.no_grad():
                for oi, op in enumerate(("ELEM", "HIER", "SYM")):
                    for b in range(B):
                        items = [(q_titles[b], f_titles[cand_lists[b, k]], OPS[op])
                                 for k in range(K)]
                        X[b, :, col0 + oi] = np.array(mu_batch(mdl, tok, items, "cpu").cpu())
                    print(f"  mu[{op}] scored")
        np.savez_compressed(a.cache, X=X, cand_lists=cand_lists, key=cache_key)
        print(f"features cached -> {a.cache}")

    # labels: candidate list position of the true folder (title-equivalence)
    B, K, F = X.shape
    y = np.zeros((B, K), dtype=bool)
    true_in_k = np.zeros(B, dtype=bool)
    for b in range(B):
        tp = set(truepos[b])
        for k in range(K):
            if int(cand_lists[b, k]) in tp:
                y[b, k] = True
                true_in_k[b] = True
    print(f"candidate-recall@{K}: {true_in_k.mean():.3f} "
          f"(true folder inside the e5 top-{K} for this share of queries)")

    def rank_metrics(scores, sel):
        """MRR / recall@k over selected queries; miss (true outside top-K) scores rank ∞."""
        ranks = []
        for b in sel:
            if not true_in_k[b]:
                ranks.append(10**6)
                continue
            s = scores[b]
            t = max((s[k] for k in range(K) if y[b, k]))
            ranks.append(1 + int(np.sum(s > t)))
        m = metrics(ranks)
        return m

    split_pairs = [(q_titles[b], f_titles[truepos[b][0]]) for b in range(B)]
    results = defaultdict(list)
    ablations = defaultdict(list)
    weights_log = []
    esc_rows = None
    for fold_seed in range(a.folds):
        split = node_disjoint_pair_split(split_pairs, fold_seed)
        tr_b, he_b = split.train, split.held
        held_nodes = split.held_nodes

        def fit_blend(cols):
            rows_X, rows_y, rows_w = [], [], []
            for b in tr_b:
                for k in range(K):
                    cand_title = f_titles[int(cand_lists[b, k])]
                    if cand_title in held_nodes:
                        continue                     # held identities never train, even as negatives
                    rows_X.append(X[b, k, cols])
                    rows_y.append(1.0 if y[b, k] else 0.0)
                    rows_w.append(float(K) if y[b, k] else 1.0)
            Xm = np.array(rows_X)
            ym = np.array(rows_y)
            wm = np.array(rows_w)
            mu_ = Xm.mean(0)
            sd_ = Xm.std(0) + 1e-9
            Xs = (Xm - mu_) / sd_
            A = Xs.T @ (Xs * wm[:, None]) + a.ridge * np.eye(len(cols))
            beta = np.linalg.solve(A, Xs.T @ (ym * wm))
            return beta, mu_, sd_

        all_cols = list(range(F))
        beta, mu_, sd_ = fit_blend(all_cols)
        weights_log.append(beta)
        scores_blend = ((X - mu_) / sd_) @ beta
        scores_e5 = X[:, :, 0]
        results["blend"].append(rank_metrics(scores_blend, he_b))
        results["e5-only"].append(rank_metrics(scores_e5, he_b))
        for fam, cols_drop in FAMILIES.items():
            keep = [i for i, nm in enumerate(feat_names) if nm not in cols_drop]
            if len(keep) == F:
                continue
            b2, m2, s2 = fit_blend(keep)
            sc = ((X[:, :, keep] - m2) / s2) @ b2
            ablations[fam].append(rank_metrics(sc, he_b))
        if fold_seed == 0:
            top2 = np.sort(scores_blend, axis=1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
            esc_rows = []
            for t in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
                kept = [b for b in he_b if margin[b] >= t]
                m = rank_metrics(scores_blend, kept) if kept else {"recall@1": float("nan")}
                esc_rows.append((t, 1 - len(kept) / max(len(he_b), 1), len(kept), m["recall@1"]))

            # CONFIRMATORY quantity (frozen seed-0 split): paired per-query Δ(1/rank), two-endpoint
            # node-block bootstrap (the repeated splits below are OVERLAPPING and only descriptive)
            def rr(scores, b):
                if not true_in_k[b]:
                    return 0.0
                t_ = max((scores[b][k] for k in range(K) if y[b, k]))
                return 1.0 / (1 + int(np.sum(scores[b] > t_)))

            keep_dd = [i for i, nm in enumerate(feat_names) if nm not in FAMILIES["diffusion"]]
            b_dd, m_dd, s_dd = fit_blend(keep_dd)
            sc_dd = ((X[:, :, keep_dd] - m_dd) / s_dd) @ b_dd
            held_pairs = [split_pairs[b] for b in he_b]
            frozen = {
                "blend - e5 [primary]": np.array([rr(scores_blend, b) - rr(scores_e5, b) for b in he_b]),
                "full - drop-diffusion": np.array([rr(scores_blend, b) - rr(sc_dd, b) for b in he_b]),
            }
            print(f"\nfrozen seed-0 split ({len(he_b)} held): paired two-endpoint node-block "
                  "bootstrap on per-query Δ(1/rank) (95% CI):")
            for ei, (nm, vals) in enumerate(frozen.items()):
                ci = paired_node_bootstrap_ci(held_pairs, vals, n_resamples=2000, seed=1729 + ei)
                print(f"    {nm:24s}: {ci.estimate:+.4f} [{ci.low:+.4f}, {ci.high:+.4f}]")

    def summarize(ms):
        return {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])))
                for k in ("MRR", "recall@1", "recall@5")}

    print(f"\n=== held metrics across {a.folds} REPEATED node-disjoint splits "
          "(OVERLAPPING — descriptive stability, NOT an uncertainty interval) ===")
    print(f"  {'ranker':12s} {'MRR':>16s} {'recall@1':>16s} {'recall@5':>16s}")
    for nm in ("e5-only", "blend"):
        s = summarize(results[nm])
        print(f"  {nm:12s} " + " ".join(f"{s[k][0]:8.3f}±{s[k][1]:.3f}" for k in ("MRR", "recall@1", "recall@5")))
    deltas = [results["blend"][i]["MRR"] - results["e5-only"][i]["MRR"] for i in range(a.folds)]
    print(f"  per-split ΔMRR (blend − e5): mean {np.mean(deltas):+.4f}; "
          f"per-split {[f'{d:+.3f}' for d in deltas]}; {sum(d > 0 for d in deltas)}/{a.folds} splits + "
          f"(descriptive; the frozen-split bootstrap above is the confirmatory quantity)")
    print("\n  ablations (blend minus family; ΔMRR vs full blend):")
    full_mrr = np.mean([m["MRR"] for m in results["blend"]])
    for fam, ms in ablations.items():
        d = np.mean([m["MRR"] for m in ms]) - full_mrr
        print(f"    - {fam:10s}: MRR {np.mean([m['MRR'] for m in ms]):.3f} (Δ {d:+.4f})")
    wbar = np.mean(weights_log, axis=0)
    print("\n  mean standardized blend weights: " +
          ", ".join(f"{nm}={w:+.3f}" for nm, w in zip(feat_names, wbar)))
    print("\n  escalation (fold-0 held; routed = top-2 margin < t; DESCRIPTIVE — post-hoc "
          "thresholds, no AURC/bootstrap/matched-coverage comparator):")
    print(f"  {'t':>6s} {'routed':>8s} {'kept_n':>7s} {'kept R@1':>9s}")
    for t, routed, kn, r1 in esc_rows:
        print(f"  {t:6.3f} {routed:8.3f} {kn:7d} {r1:9.3f}")


if __name__ == "__main__":
    main()
