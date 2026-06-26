#!/usr/bin/env python3
"""Label-data μ→relation distribution estimator — the foundation for inferred-operator assignment
(DESIGN_inferred_operator_superposition.md) and multi-source combination
(DESIGN_mu_sources_and_estimation.md).

From the TAGGED pairs (whose relation we know) we estimate, per μ SOURCE (frozen e5, the training model,
later an LLM), the generative `P(μ | relation)` — one smoothed histogram per relation — and Bayes-invert to
the posterior `P(relation | μ)`. Two uses:
  1. **Inferred-operator assignment**: for an untagged pair, measure μ and read the posterior over relations
     (→ operator), instead of v1's fixed-breadth heuristic.
  2. **Label-anomaly review** (the side-note rule): a tagged label keeps confidence 1.0 UNLESS its measured μ
     is OUTSIDE the expected band for that relation — then it is flagged for LLM/human review.

Sources are NOT independent (the model consumes e5) — combine with weighted product-of-experts, not a naive
product; estimate the e5↔model dependence before trusting a tight combined posterior (see the design doc).

This module is pure-Python + numpy (no torch); the static **e5** source runs now from an e5 cache.

    python3 mu_posterior.py --pairs /tmp/graded_pairs.tsv --e5-cache e5_tables_graded.pt
"""
import argparse
import math
import os
from collections import Counter, defaultdict

import numpy as np


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[m] - xs[m].mean(), ys[m] - ys[m].mean()
    d = (np.linalg.norm(xs) * np.linalg.norm(ys))
    return float((xs @ ys) / d) if d else float("nan")


class MuPosterior:
    """Per-(source, relation) μ histograms; Bayes posterior P(relation | μ_1..μ_K); per-relation expected band."""

    def __init__(self, nbins=20, lo=0.0, hi=1.0, smoothing=1.0):
        self.nbins, self.lo, self.hi, self.smoothing = nbins, lo, hi, smoothing
        self.edges = np.linspace(lo, hi, nbins + 1)
        self.dens = {}                                    # (source, relation) -> density array (sums to 1)
        self.raw = defaultdict(list)                      # (source, relation) -> list of μ (for quantile bands)
        self.prior = {}                                   # relation -> P(relation)
        self.weights = {}                                 # source -> weight in the product-of-experts
        self.sources = []

    def _bin(self, mu):
        return int(np.clip(np.searchsorted(self.edges, mu, side="right") - 1, 0, self.nbins - 1))

    def fit_source(self, source, rel_mu, weight=1.0):
        """rel_mu: iterable of (relation, μ). Builds a smoothed density per relation for this source."""
        if source not in self.sources:
            self.sources.append(source)
        self.weights[source] = weight
        by_rel = defaultdict(list)
        for rel, mu in rel_mu:
            if mu == mu:                                  # drop NaN
                by_rel[rel].append(float(mu))
        for rel, mus in by_rel.items():
            h = np.histogram(mus, bins=self.edges)[0].astype(float) + self.smoothing
            self.dens[(source, rel)] = h / h.sum()
            self.raw[(source, rel)] = sorted(mus)
        # prior from the (first-source) relation frequencies
        if not self.prior:
            tot = sum(len(v) for v in by_rel.values())
            self.prior = {rel: len(v) / tot for rel, v in by_rel.items()}

    def relations(self):
        return sorted(self.prior)

    def posterior(self, mu_by_source, candidates=None):
        """mu_by_source: {source: μ}. Returns {relation: P(relation | μ's)} over `candidates` (default all)."""
        rels = candidates or self.relations()
        logp = {}
        for rel in rels:
            lp = math.log(self.prior.get(rel, 1e-9))
            for src, mu in mu_by_source.items():
                if mu != mu or (src, rel) not in self.dens:
                    continue
                lp += self.weights.get(src, 1.0) * math.log(self.dens[(src, rel)][self._bin(mu)] + 1e-12)
            logp[rel] = lp
        m = max(logp.values())
        exp = {r: math.exp(lp - m) for r, lp in logp.items()}
        z = sum(exp.values()) or 1.0
        return {r: v / z for r, v in exp.items()}

    def band(self, source, relation, q=0.05):
        """Expected μ band for a relation under a source = its [q, 1−q] quantiles on the tagged data."""
        xs = self.raw.get((source, relation))
        if not xs:
            return (self.lo, self.hi)
        return (float(np.quantile(xs, q)), float(np.quantile(xs, 1 - q)))

    def is_anomalous(self, source, relation, mu, q=0.05):
        """True if a TAGGED pair's measured μ is outside its relation's expected band ⇒ needs review."""
        if mu != mu:
            return False
        lo, hi = self.band(source, relation, q)
        return not (lo <= mu <= hi)

    def separability(self, source):
        """How well this source's μ separates the relations: pairwise band overlap summary + the mutual-info-ish
        spread of per-relation means. Low ⇒ a weak source ⇒ small weight."""
        means = {rel: float(np.mean(self.raw[(source, rel)])) for rel in self.relations() if self.raw.get((source, rel))}
        if len(means) < 2:
            return 0.0, means
        return float(np.std(list(means.values()))), means


# ---------- static e5 source: μ_e5(node, root) = cos(query[root], passage[node]) -------------------------
def e5_mu_fn(cache_path):
    import torch
    d = torch.load(cache_path, weights_only=False)
    idx = {n: i for i, n in enumerate(d["names"])}
    q, p = d["query"].numpy(), d["passage"].numpy()       # unit-normed

    def mu(node, root):
        ni, ri = idx.get(node), idx.get(root)
        return float("nan") if ni is None or ri is None else float(p[ni] @ q[ri])
    return mu


def load_pairs(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split("\t")               # node root mu op relation ... corpus judge conf
            if len(c) >= 5:
                conf = float(c[9]) if len(c) > 9 and c[9] else 1.0
                rows.append({"node": c[0], "root": c[1], "mu": float(c[2]), "op": c[3], "rel": c[4], "conf": conf})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="graded _pairs.tsv (uses the TAGGED rows, conf≥1.0)")
    ap.add_argument("--e5-cache", required=True)
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--q", type=float, default=0.05, help="anomaly band quantile")
    ap.add_argument("--model", default=None, help="a trained MuAttention checkpoint → adds the DYNAMIC model "
                    "μ source (symmetric SYM μ, masked provenance) and reports e5↔model correlation")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_pairs(args.pairs)
    tagged = [r for r in rows if r["conf"] >= 1.0]
    e5mu = e5_mu_fn(args.e5_cache)
    mu_e5 = {(r["node"], r["root"]): e5mu(r["node"], r["root"]) for r in tagged}

    post = MuPosterior(nbins=args.nbins)
    post.fit_source("e5", ((r["rel"], mu_e5[(r["node"], r["root"])]) for r in tagged), weight=1.0)
    sp_e5, means = post.separability("e5")
    print(f"tagged pairs: {len(tagged)} / {len(rows)} total")
    print(f"e5 μ per-relation means (separability spread {sp_e5:.3f}):")
    for rel in sorted(means, key=lambda r: -means[r]):
        lo, hi = post.band("e5", rel, args.q)
        print(f"  {rel:14s} mean {means[rel]:.3f}  band[{args.q:.2f}] [{lo:.3f},{hi:.3f}]  n={len(post.raw[('e5',rel)])}")

    # DYNAMIC model source + the non-independence measurement (design: set weights from e5↔model correlation)
    if args.model:
        from bridge_ensemble import model_scorer
        _, mmu = model_scorer(args.model, args.e5_cache, device=args.device)
        mu_md = {(r["node"], r["root"]): mmu(r["node"], r["root"]) for r in tagged}
        post.fit_source("model", ((r["rel"], mu_md[(r["node"], r["root"])]) for r in tagged))
        sp_md, mmeans = post.separability("model")
        print(f"\nmodel μ per-relation means (separability spread {sp_md:.3f}  vs e5 {sp_e5:.3f}):")
        for rel in sorted(mmeans, key=lambda r: -mmeans[r]):
            print(f"  {rel:14s} mean {mmeans[rel]:.3f}  n={len(post.raw[('model',rel)])}")
        ce5 = [mu_e5[k] for k in mu_e5]
        cmd = [mu_md[k] for k in mu_e5]
        r_all = pearson(ce5, cmd)
        print(f"\ne5 ↔ model μ correlation (overall): {r_all:+.3f}  "
              f"⇒ they are {'strongly' if r_all > 0.6 else 'moderately' if r_all > 0.3 else 'weakly'} "
              f"correlated; a naive product over-counts the shared evidence by ~this much.")
        # set product-of-experts weights: down-weight the less-separating source AND the shared (correlated)
        # part — give the model the full weight, e5 only its UNIQUE contribution ≈ (1 − r²).
        w_e5 = max(0.0, 1.0 - r_all * r_all) * (sp_e5 / (sp_e5 + sp_md + 1e-9))
        post.weights["e5"], post.weights["model"] = round(w_e5, 3), 1.0
        print(f"  → suggested product-of-experts weights: model 1.0, e5 {post.weights['e5']:.3f} "
              f"(its (1−r²)·relative-separability share)")

    print("\nP(relation | μ_e5) at sample μ values:")
    for mu in (0.78, 0.84, 0.90, 0.96):
        pst = post.posterior({"e5": mu})
        top = sorted(pst.items(), key=lambda x: -x[1])[:3]
        print(f"  μ_e5={mu:.2f} → " + ", ".join(f"{r} {p:.2f}" for r, p in top))

    # the side-note rule: TAGGED labels whose measured μ is out of band ⇒ flag for LLM/human review
    flagged = [(r, e5mu(r["node"], r["root"])) for r in tagged
               if post.is_anomalous("e5", r["rel"], e5mu(r["node"], r["root"]), args.q)]
    print(f"\nLABEL-ANOMALY review (tagged, μ_e5 outside its relation band): {len(flagged)}/{len(tagged)}")
    for r, mu in sorted(flagged, key=lambda x: x[1])[:15]:
        print(f"  μ_e5={mu:.3f}  [{r['rel']}]  {r['node']} ∈/~ {r['root']}")
    if args.out and flagged:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# node\troot\trelation\tmu_e5\treason=out_of_band\n")
            for r, mu in sorted(flagged, key=lambda x: x[1]):
                f.write(f"{r['node']}\t{r['root']}\t{r['rel']}\t{mu:.3f}\n")
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
