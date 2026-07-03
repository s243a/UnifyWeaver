#!/usr/bin/env python3
"""recall_curve.py — how much does widening the shortlist raise recall, and is max(μ,e5) worth it?

For N real bookmarks (true folder known), score e5 AND μ over ALL folders, find the true folder's rank under
three cutoffs — **e5**, **μ**, and **max(μ,e5)** (union) — and report recall@K for K ∈ {15,30,50,100,200,500}.
Pure embedding computation, NO LLM / no quota (μ-over-all is CPU-heavy: ~seconds/bookmark).

Reads:
  * steep e5/μ curve ⇒ just widen K (and by how much);
  * max(μ,e5) above both ⇒ the union cutoff earns its extra μ cost;
  * flat curves ⇒ the ceiling is the embedding/μ, not list size (widening won't help — improve μ/e5).

  python3 recall_curve.py --model models/pearltrees_federated_s243a.pkl --mu-ckpt prototypes/mu_cosine/model_prod.pt --n 100
"""
import argparse, pickle, random, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mu_filer_ranker import MuRanker

KS = [15, 30, 50, 100, 200, 500]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--mu-ckpt", required=True)
    ap.add_argument("--trees", default=".local/data/pearltrees_api/trees")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    meta = pickle.load(open(a.model, "rb"))
    titles, ids = meta["global_target_titles"], [str(x) for x in meta["global_target_ids"]]
    id_to_idx = {t: i for i, t in enumerate(ids)}
    ranker = MuRanker(a.mu_ckpt, titles, ids, device="cpu",
                      cache_path=str(a.model) + ".mu_folder_e5_%d.pt" % len(titles))

    sys.path.insert(0, str(Path(__file__).parent.parent / "prototypes" / "mu_cosine"))
    from eval_filing import load_filing
    pairs, cand = load_filing(a.trees, min_bm=3)
    rng = random.Random(a.seed); rng.shuffle(pairs)
    eng = set(ids)
    sample = [(bm, str(f)) for bm, f in pairs if cand.get(f) and str(f) in eng][:a.n]
    print(f"[RECALL] {len(sample)} bookmarks (true folder in-model); μ+e5 over all {len(ids)} folders\n")

    def nz(v):
        return (v - v.min()) / (v.max() - v.min() + 1e-9)

    rank = {"e5": [], "mu": [], "max": []}
    for i, (bm, tid) in enumerate(sample):
        ti = id_to_idx[tid]
        e5, mu = ranker.score_components(bm)
        comb = np.maximum(nz(e5), nz(mu))
        for name, s in (("e5", e5), ("mu", mu), ("max", comb)):
            rank[name].append(1 + int((s > s[ti]).sum()))       # 1-based rank of the true folder
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(sample)}")

    print("\n== recall@K (true folder within top-K) ==")
    print("  cutoff  " + "  ".join("@%-4d" % k for k in KS))
    for name in ("e5", "mu", "max"):
        r = np.array(rank[name])
        row = "  ".join("%4.0f%%" % (100 * (r <= k).mean()) for k in KS)
        print(f"  {name:6}  {row}")
    print("\n  (max = max(μ,e5) union. If it beats e5 at your target K, the wider-μ cutoff is worth it; "
          "if e5≈μ≈max and all are flat, widen won't help — improve the embedding/μ.)")


if __name__ == "__main__":
    main()
