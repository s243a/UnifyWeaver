#!/usr/bin/env python3
"""Dense μ map by **direct asymmetric embedding** — no training (README "prompt A", model-comparison
variant). For every category in `category_parent.tsv`, cosine its embedding to the domain root and
clamp to [0,1]; emit `name<TAB>μ` (names verbatim) in the fixture format.

**Asymmetric** retrieval embedders score a *query* against *documents*, which is exactly the μ
question ("how much does this category belong to the <root> domain?"): encode the root as the query
and each category as a document, with the model's prefixes. Presets:

  * `minilm` — `sentence-transformers/all-MiniLM-L6-v2` (384-d, symmetric, no prefix) — the baseline.
  * `e5`     — `intfloat/e5-small-v2` (384-d, `query: ` / `passage: `).
  * `nomic`  — `nomic-ai/nomic-embed-text-v1.5` (768-d, `search_query: ` / `search_document: `,
               needs `trust_remote_code=True` + `einops`).

**Budget metric:** the **decision band** = #categories with μ ∈ [lo, hi] straddling the 0.3 gate
(default [0.2, 0.45]) — these are the ambiguous ones a later Haiku pass would have to re-score, so the
model with the *smallest* band (that still discriminates — watch the sanity cases) is budget-optimal.

    python3 dense_mu_direct.py --compare                       # band sizes + sanity for all presets
    python3 dense_mu_direct.py --model e5 --out dense_mu_e5.tsv   # write the chosen map
"""
from __future__ import annotations

import argparse
import os

import torch

from emit_dense_mu import load_graph_names, GRAPH, REPO
from mu_encoder_torch import to_membership
from train_cosine_mu_torch import load_mu, FIXTURE, pearson


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    return pearson(ranks(xs), ranks(ys))

PRESETS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", "", False),
    "e5":     ("intfloat/e5-small-v2", "query: ", "passage: ", False),
    "nomic":  ("nomic-ai/nomic-embed-text-v1.5", "search_query: ", "search_document: ", True),
}
SANITY = ["Music", "Optics", "Thermodynamics", "Quantum_mechanics", "Cooking", "Religious_buildings"]


def humanize(n):
    return n.replace("_", " ")


def embed_mu(preset, names, root, batch_size=512):
    """Return the **raw cosine** (∈[-1,1], one per name) of cos(doc(name), query(root)). Calibration /
    clamping happen in `report` so the raw scale (which differs per model) stays inspectable."""
    from sentence_transformers import SentenceTransformer

    model_name, qpre, dpre, trust = PRESETS[preset]
    model = SentenceTransformer(model_name, trust_remote_code=trust)
    root_vec = model.encode([qpre + humanize(root)], normalize_embeddings=True, convert_to_tensor=True)
    root_vec = root_vec[0]
    cos_all = []
    for i in range(0, len(names), batch_size):
        chunk = [dpre + humanize(n) for n in names[i:i + batch_size]]
        docs = model.encode(chunk, normalize_embeddings=True, convert_to_tensor=True,
                            show_progress_bar=False)
        cos_all.extend((docs @ root_vec).tolist())     # normalized ⇒ dot = cosine
    return cos_all


def lstsq_fit(x, y):
    """Least-squares a,b for y ≈ a·x + b."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    a = sxy / sxx if sxx else 0.0
    return a, my - a * mx


def band_count(mus, lo, hi):
    return (sum(1 for m in mus if m < lo), sum(1 for m in mus if lo <= m <= hi),
            sum(1 for m in mus if m > hi))


def report(preset, names, cos_all, lo, hi, fixture_mu):
    idx = {n: i for i, n in enumerate(names)}
    raw_mu = [to_membership(c) for c in cos_all]       # guard #1: clamp to [0,1]
    below, band, above = band_count(raw_mu, lo, hi)

    # discrimination + calibration against the 90-node Haiku fixture (no LLM budget):
    fx = [(idx[n], m) for n, m in fixture_mu.items() if n in idx]
    fcos = [cos_all[i] for i, _ in fx]
    fmu = [m for _, m in fx]
    r, rho = pearson(fcos, fmu), spearman(fcos, fmu)
    a, b = lstsq_fit(fcos, fmu)
    cal_mu = [min(1.0, max(0.0, a * c + b)) for c in cos_all]
    cb_below, cb_band, cb_above = band_count(cal_mu, lo, hi)

    print(f"\n[{preset}] {PRESETS[preset][0]}")
    print(f"  RAW   band μ∈[{lo},{hi}]: {band:5d}  (below {below} / band {band} / above {above})")
    print(f"  CALIB band μ∈[{lo},{hi}]: {cb_band:5d}  (below {cb_below} / band {cb_band} / above "
          f"{cb_above})   [fit μ≈{a:.2f}·cos{b:+.2f}]")
    print(f"  fixture discrimination ({len(fx)} nodes): Pearson r={r:+.3f}  Spearman ρ={rho:+.3f}")
    print(f"  sanity RAW μ: " + "  ".join(
        f"{n.split('_')[0]}={raw_mu[idx[n]]:.2f}" for n in SANITY if n in idx))
    return {"raw_band": band, "cal_band": cb_band, "r": r, "rho": rho,
            "raw_mu": raw_mu, "cal_mu": cal_mu, "fit": (a, b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(PRESETS), default=None, help="write this preset's map")
    ap.add_argument("--compare", action="store_true", help="report band sizes for all presets")
    ap.add_argument("--root", default="Physics")
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.45)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = load_graph_names(GRAPH)
    fixture_mu = load_mu(FIXTURE)
    print(f"{len(names)} categories; root '{args.root}'; decision band [{args.lo},{args.hi}]; "
          f"{len(fixture_mu)} fixture nodes for discrimination/calibration")
    if args.root not in names:
        raise SystemExit(f"root '{args.root}' is not a graph name")

    presets = list(PRESETS) if args.compare else [args.model]
    if presets == [None]:
        raise SystemExit("pass --compare or --model <preset>")

    results = {}
    for p in presets:
        cos_all = embed_mu(p, names, args.root)
        results[p] = report(p, names, cos_all, args.lo, args.hi, fixture_mu)

    if args.compare:
        # Budget-optimal = smallest band, BUT only among models that actually discriminate (the raw
        # band rewards a model that piles everything above the gate — useless). Rank by CALIBRATED band
        # (comparable across cosine scales), tie-broken by fixture correlation.
        print("\n  model   raw_band  calib_band   fixture_r   fixture_ρ")
        for p in presets:
            x = results[p]
            print(f"  {p:6s}  {x['raw_band']:7d}  {x['cal_band']:9d}   {x['r']:+.3f}      {x['rho']:+.3f}")
        best = min(presets, key=lambda p: (results[p]["cal_band"], -results[p]["r"]))
        print(f"\nbudget-optimal (smallest CALIBRATED band, then best fixture corr): {best}. "
              f"NOTE: the RAW band is misleading — e5/nomic pile everything above {args.hi} (cosine "
              f"floor), so their tiny/huge raw band does not reflect discrimination; calibrating each "
              f"model's cosine→μ against the fixture makes the band comparable.")

    if args.out and args.model:
        # emit the CALIBRATED μ (raw cosine scale is not a membership; calibration aligns it with μ)
        mus = results[args.model]["cal_mu"]
        a, b = results[args.model]["fit"]
        with open(args.out, "w") as f:
            f.write(f"# Dense μ(category | {args.root}) — direct {PRESETS[args.model][0]} cosine "
                    f"(asymmetric, no training), calibrated μ≈{a:.3f}·cos{b:+.3f} on the Haiku fixture. "
                    f"clamp [0,1], names verbatim. Format: name<TAB>μ.\n")
            for n, m in zip(names, mus):
                f.write(f"{n}\t{m:.4f}\n")
        print(f"\nwrote {len(names)} calibrated rows → {args.out}")


if __name__ == "__main__":
    main()
