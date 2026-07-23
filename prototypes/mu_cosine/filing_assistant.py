#!/usr/bin/env python3
"""Filing assistant v1: e5 ranking @ K=100 + margin routing — the honest, deployable loop.

Grounding (REPORT_hybrid_candidates.md): the recall@50 = 0.680 "ceiling" was pure K-truncation
(e5@100 = 0.801, e5@150 = 0.870); at matched pool size pure e5 beats every μ/graph-augmented pool;
and no μ head beats e5 in any stratum. So the deployable v1 is the champion ranker used honestly:
e5-cos over the folder catalog, a longer candidate list (K=100), and margin-gated escalation —
confident placements auto-suggested, low-margin ones routed to review (a human or judge).

  eval mode     python3 filing_assistant.py eval [--top-k 100] [--min-bm 3]
      Honest numbers on the standing harness population (seed 7, ≤1200 queries, ≥3-bookmark
      catalog — identical manifest to filing_ranker.py): MRR / recall@1 / @5 with misses (true
      folder outside top-K) scored rank ∞, for K ∈ {50, K}, plus the margin-routing table
      (fraction routed vs kept recall@1). Title-equivalence grading throughout.

  suggest mode  python3 filing_assistant.py suggest "Some new bookmark title" [--top-n 5]
      Rank folders for new bookmark title(s) (repeatable arg, or --from-file, one per line).
      Catalog = folders with ≥ --min-bm recorded bookmarks (default 1: every folder actually in
      use as a filing destination). Folder IDs remain distinct even when titles match, so exact
      score ties have the same ascending-catalog-column rule as evaluation and correctly produce
      zero margin. Each suggestion prints the exact tree ID and e5 cosine; the query prints its
      top-2 ID-level margin and a routing verdict at --margin (default 0.05 — see the eval routing
      table for the measured kept-R@1 tradeoff).

No μ, no blend, no diffusion — those are nulls as rankers on this task (REPORTs); revisit only if
a future head beats e5 somewhere (gap-directed training is the demonstrated path — §B′/B″).
"""
import argparse
import hashlib
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_filing import load_filing, metrics
from filing_privacy import public_catalog_title_eligible
from mu_attention import E5_MODEL, build_e5_tables

ROOT = os.path.dirname(os.path.abspath(__file__))
TREES = os.path.join(ROOT, "..", "..", ".local", "data", "pearltrees_api", "trees")


def apply_public_catalog_policy(queries, candidates):
    """Apply the one shared outcome-blind folder eligibility rule."""
    eligible = {
        folder_id: title
        for folder_id, title in candidates.items()
        if public_catalog_title_eligible(title)
    }
    filtered_queries = [
        (bookmark, folder_id)
        for bookmark, folder_id in queries
        if folder_id in eligible
    ]
    return filtered_queries, eligible


def catalog_tables(min_bm, cache_tag, *, return_privacy=False):
    """Certified-public folder catalog + unit e5 passage vectors.

    ``return_privacy`` exposes the exact privacy index used to construct the
    catalog so hosted-task emitters can bind it into their provenance.
    """
    queries, cand, privacy = load_filing(TREES, min_bm, return_privacy=True)
    queries, cand = apply_public_catalog_policy(queries, cand)
    f_ids = sorted(cand)
    f_titles = [cand[fid] for fid in f_ids]
    names = sorted(set(f_titles))
    cache_root = os.environ.get("MU_COSINE_CACHE_DIR", "/tmp/mu_data")
    os.makedirs(cache_root, exist_ok=True)
    cache = os.path.join(
        cache_root,
        f"filing_assistant_e5_{cache_tag}_{privacy.manifest_sha256[:12]}.pt",
    )
    _, ptbl, idx = build_e5_tables(names, cache_path=cache, batch_size=128)
    P = ptbl.numpy()
    cand_vec = np.stack([P[idx[t]] for t in f_titles])
    result = (queries, f_ids, f_titles, cand_vec)
    return (*result, privacy) if return_privacy else result


def encode_queries(titles):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(E5_MODEL)
    q = model.encode(["query: " + t.replace("_", " ") for t in titles], batch_size=64,
                     convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return q


def ranks_np(cos, truepos):
    """1-based rank of the best supplied acceptable column.

    ``truepos`` determines exact-ID versus title-equivalence grading. Exact
    ties favor the lower catalog column, matching ``stable_score_order`` and
    ``eval_pearltrees_filing.ranks_from``.
    """
    out = []
    for r in range(cos.shape[0]):
        best = None
        for tp in truepos[r]:
            rank = 1 + int(((cos[r] > cos[r][tp]) |
                            ((cos[r] == cos[r][tp]) & (np.arange(cos.shape[1]) < tp))).sum())
            best = rank if best is None else min(best, rank)
        out.append(best)
    return np.array(out)


def stable_score_order(cos):
    """Descending score order with exact ties broken by lower catalog column."""
    return np.argsort(-np.asarray(cos), axis=-1, kind="stable")


def run_eval(a):
    queries, f_ids, f_titles, cand_vec = catalog_tables(a.min_bm, f"eval{a.min_bm}")
    by_title = {}
    for j, t in enumerate(f_titles):
        by_title.setdefault(t, []).append(j)
    queries = sorted(queries)
    if len(queries) > a.max_queries:
        queries = random.Random(a.seed).sample(queries, a.max_queries)
    q_titles = [q for q, _ in queries]
    cand_by_id = dict(zip(f_ids, f_titles))
    truepos = [sorted(by_title[cand_by_id[fid]]) for _, fid in queries]
    qman = hashlib.sha256("\n".join(f"{q}\t{fid}" for q, fid in queries).encode()).hexdigest()
    print(f"eval: {len(queries)} queries (manifest sha {qman[:16]}) over "
          f"{len(f_ids)} folders (min_bm={a.min_bm})")

    qv = encode_queries(q_titles)
    cos = qv @ cand_vec.T
    ranks = ranks_np(cos, truepos)

    for K in sorted({50, a.top_k}):
        rk = np.where(ranks <= K, ranks, 10**6)
        m = metrics(list(rk))
        print(f"  e5 @ K={K:<4d} recall@K {np.mean(ranks <= K):.3f}  "
              f"MRR {m['MRR']:.3f}  R@1 {m['recall@1']:.3f}  R@5 {m['recall@5']:.3f}")

    # margin routing on the K=top_k list (miss = rank ∞ stays a miss when kept)
    srt = np.sort(cos, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    rk = np.where(ranks <= a.top_k, ranks, 10**6)
    print(f"\n  margin routing (top1−top2 e5 cosine; K={a.top_k}):")
    print(f"  {'t':>7s} {'routed':>7s} {'kept_n':>7s} {'kept R@1':>9s} {'kept R@5':>9s}")
    for t in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        kept = margin >= t
        if kept.sum() == 0:
            continue
        print(f"  {t:7.3f} {1 - kept.mean():7.2f} {int(kept.sum()):7d} "
              f"{np.mean(rk[kept] <= 1):9.3f} {np.mean(rk[kept] <= 5):9.3f}")
    print("\n  DESCRIPTIVE thresholds (post-hoc grid, no bootstrap) — pick t for the "
          "auto-file/review tradeoff you want; suggest mode defaults to t=0.05.")


def run_suggest(a):
    titles = list(a.title or [])
    if a.from_file:
        titles += [ln.strip() for ln in open(a.from_file, encoding="utf-8") if ln.strip()]
    if not titles:
        print("no titles given — pass positional titles or --from-file", file=sys.stderr)
        sys.exit(2)
    _, f_ids, f_titles, cand_vec = catalog_tables(a.min_bm, f"sugg{a.min_bm}")
    qv = encode_queries(titles)
    cos = qv @ cand_vec.T
    for i, bt in enumerate(titles):
        order = stable_score_order(cos[i])
        margin = float(cos[i][order[0]] - cos[i][order[1]]) if len(order) > 1 else float("inf")
        verdict = "AUTO-FILE candidate" if margin >= a.margin else "ROUTE TO REVIEW (low margin)"
        print(f"\n» {bt}")
        print(f"  margin {margin:.3f} → {verdict} (t={a.margin})")
        for r, j in enumerate(order[:a.top_n], 1):
            print(
                f"  {r}. {cos[i][j]:.3f}  {f_titles[j]}   "
                f"[tree {f_ids[j]}]"
            )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    ev = sub.add_parser("eval", help="honest metrics on the standing harness population")
    ev.add_argument("--top-k", type=int, default=100)
    ev.add_argument("--min-bm", type=int, default=3, help="catalog floor (3 = standing harness)")
    ev.add_argument("--max-queries", type=int, default=1200)
    ev.add_argument("--seed", type=int, default=7)
    sg = sub.add_parser("suggest", help="file new bookmark title(s)")
    sg.add_argument("title", nargs="*", help="bookmark title(s) to file")
    sg.add_argument("--from-file", help="file with one bookmark title per line")
    sg.add_argument("--top-n", type=int, default=5)
    sg.add_argument("--min-bm", type=int, default=1, help="catalog floor (1 = all in-use folders)")
    sg.add_argument("--margin", type=float, default=0.05, help="auto-file vs review threshold")
    a = ap.parse_args(argv)
    (run_eval if a.mode == "eval" else run_suggest)(a)


if __name__ == "__main__":
    main()
