#!/usr/bin/env python3
"""Routed-queries loop, step 1: rescue ceilings + the judge task file.

filing_assistant.py routes low-margin queries to review. This is the policy evaluation that the
escalation tables always deferred ("needs judge labels on the routed queries"). Three modes:

  ceilings   python3 routed_queries.py ceilings
      No judge needed: for a grid of margin thresholds t and judge menus of size N, report the
      POLICY CEILING — overall R@1 if the judge picked perfectly whenever the true folder is in
      the top-N menu of a routed query (auto-filed kept queries keep their e5 top-1). This bounds
      what any judge spend can buy and locates the useful (t, N) region.

  emit       python3 routed_queries.py emit --margin 0.02 --menu 10
      Write the OUTCOME-BLIND judge task file for the routed set at t: one JSONL row per routed
      query {qid, bookmark, menu:[{pos, title}]} (no truth marked, menu in e5 order) →
      ~/mu_data/routed_tasks_t{t}_n{N}.jsonl (PRIVATE — personal titles). A judge (cheap LLM or
      human) fills {qid, pick: pos|null}. Contract mirrors the campaign judge files: strict rows,
      one pick per qid.

  score      python3 routed_queries.py score --picks <file> --margin 0.02 --menu 10
      Ingest judge picks (JSONL {qid, pick}) fail-closed (every routed qid exactly once, pick in
      menu range or null) and report the POLICY result: R@1 of [auto-filed kept + judge-picked
      routed] vs the no-routing baseline, with a paired bootstrap CI on the delta.

Population/manifest identical to filing_assistant eval (seed 7, ≤1200 queries, min_bm=3 catalog,
manifest sha printed). Ranking = e5 @ K=100. Title-equivalence grading.
"""
import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_filing import load_filing
from filing_assistant import TREES, catalog_tables, encode_queries, ranks_np


def build(a):
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
    qv = encode_queries(q_titles)
    cos = qv @ cand_vec.T
    ranks = ranks_np(cos, truepos)
    srt = np.sort(cos, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    order = np.argsort(-cos, axis=1)
    print(f"population: {len(queries)} queries (manifest sha {qman[:16]}), "
          f"{len(f_ids)} folders, K={a.top_k}")
    return q_titles, f_titles, truepos, cos, ranks, margin, order, qman


def menu_hit(order, truepos, b, N):
    """True folder (title-equivalent) inside the top-N menu?"""
    tp = set(truepos[b])
    return any(int(c) in tp for c in order[b][:N])


def run_ceilings(a):
    q_titles, f_titles, truepos, cos, ranks, margin, order, _ = build(a)
    B = len(q_titles)
    base_r1 = float(np.mean(ranks <= 1))
    print(f"\nno-routing baseline R@1: {base_r1:.3f}")
    print(f"{'t':>7s} {'routed':>7s} {'kept R@1':>9s} " +
          " ".join(f"ceil@N={n:<3d}" for n in a.menus))
    for t in a.grid:
        routed = margin < t
        kept = ~routed
        kept_r1 = float(np.mean(ranks[kept] <= 1)) if kept.any() else float("nan")
        cells = []
        for N in a.menus:
            resc = sum(menu_hit(order, truepos, b, N) for b in np.where(routed)[0])
            pol = (int((ranks[kept] <= 1).sum()) + resc) / B
            cells.append(f"{pol:9.3f}")
        print(f"{t:7.3f} {routed.mean():7.2f} {kept_r1:9.3f} " + " ".join(cells))
    print("\nCEILINGS assume a perfect judge over the menu (upper bounds, not predictions). "
          "Judge value = ceiling − baseline at the chosen (t, N); real judges land in between — "
          "emit + score to measure.")


def task_path(a):
    suf = "_lin" if getattr(a, "lineage", False) else ""
    return os.path.expanduser(f"~/mu_data/routed_tasks_t{a.margin}_n{a.menu}{suf}.jsonl")


def run_emit(a):
    q_titles, f_titles, truepos, cos, ranks, margin, order, qman = build(a)
    routed = np.where(margin < a.margin)[0]
    lin = {}
    if a.lineage:
        # folder principal-path context (§7 machinery): outcome-blind — uses only the folder's own
        # pre-existing folder→parent lineage, never the bookmark's placement
        from eval_pearltrees_filing import folder_lineage
        _, cand = load_filing(TREES, a.min_bm)
        parents_title, _ = folder_lineage(cand, depth=a.lineage_depth)
        lin = parents_title
    out = task_path(a)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"manifest": qman, "margin": a.margin, "menu": a.menu,
                            "n_routed": len(routed), "lineage": bool(a.lineage)}) + "\n")
        for b in routed:
            menu = [{"pos": p, "title": f_titles[int(order[b][p])],
                     **({"path": " > ".join(reversed(lin.get(f_titles[int(order[b][p])], [])))}
                        if a.lineage else {})} for p in range(a.menu)]
            f.write(json.dumps({"qid": int(b), "bookmark": q_titles[b], "menu": menu},
                               ensure_ascii=False) + "\n")
    hit = sum(menu_hit(order, truepos, b, a.menu) for b in routed)
    print(f"emitted {len(routed)} routed tasks -> {out} (PRIVATE)")
    print(f"rescue ceiling on this set: {hit}/{len(routed)} menus contain the true folder")
    print("judge contract: JSONL rows {\"qid\": int, \"pick\": pos-int or null}; one row per qid; "
          "pick = the menu position to file into, null = none fits.")


def run_score(a):
    q_titles, f_titles, truepos, cos, ranks, margin, order, qman = build(a)
    routed = np.where(margin < a.margin)[0]
    picks = {}
    with open(a.picks, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if "manifest" in r:
                if r["manifest"] != qman:
                    sys.exit(f"FAIL-CLOSED: picks manifest {r['manifest'][:16]} != population "
                             f"{qman[:16]}")
                continue
            q = int(r["qid"])
            if q in picks:
                sys.exit(f"FAIL-CLOSED: duplicate pick for qid {q}")
            p = r["pick"]
            if p is not None and not (0 <= int(p) < a.menu):
                sys.exit(f"FAIL-CLOSED: pick {p} out of menu range for qid {q}")
            picks[q] = None if p is None else int(p)
    missing = [int(b) for b in routed if int(b) not in picks]
    if missing:
        sys.exit(f"FAIL-CLOSED: {len(missing)} routed qids unscored (first: {missing[:5]})")

    kept = np.ones(len(q_titles), bool)
    kept[routed] = False
    correct = (ranks <= 1) & kept
    n_resc = 0
    for b in routed:
        p = picks[int(b)]
        if p is not None and int(order[b][p]) in set(truepos[b]):
            correct[b] = True
            n_resc += 1
    base = (ranks <= 1)
    pol_r1, base_r1 = float(correct.mean()), float(base.mean())
    print(f"\njudge policy: routed {len(routed)}, judge-rescued {n_resc} "
          f"(menus containing truth: {sum(menu_hit(order, truepos, b, a.menu) for b in routed)})")
    print(f"policy R@1 {pol_r1:.3f} vs no-routing baseline {base_r1:.3f} "
          f"(delta {pol_r1 - base_r1:+.3f})")
    rng = np.random.default_rng(0)
    d = correct.astype(float) - base.astype(float)
    boots = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"paired bootstrap 95% CI on delta: [{lo:+.3f}, {hi:+.3f}] "
          f"({'EXCLUDES' if lo > 0 or hi < 0 else 'includes'} zero)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-bm", type=int, default=3)
    ap.add_argument("--max-queries", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--top-k", type=int, default=100)
    sub = ap.add_subparsers(dest="mode", required=True)
    ce = sub.add_parser("ceilings")
    ce.add_argument("--grid", type=float, nargs="*",
                    default=(0.005, 0.01, 0.02, 0.03, 0.05, 0.08))
    ce.add_argument("--menus", type=int, nargs="*", default=(5, 10, 20, 50))
    em = sub.add_parser("emit")
    em.add_argument("--margin", type=float, default=0.02)
    em.add_argument("--menu", type=int, default=10)
    em.add_argument("--lineage", action="store_true",
                    help="include each folder's principal-path context in the menu")
    em.add_argument("--lineage-depth", type=int, default=3)
    sc = sub.add_parser("score")
    sc.add_argument("--picks", required=True)
    sc.add_argument("--margin", type=float, default=0.02)
    sc.add_argument("--menu", type=int, default=10)
    a = ap.parse_args(argv)
    {"ceilings": run_ceilings, "emit": run_emit, "score": run_score}[a.mode](a)


if __name__ == "__main__":
    main()
