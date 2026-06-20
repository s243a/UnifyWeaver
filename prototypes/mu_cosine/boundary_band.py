#!/usr/bin/env python3
"""Stream V, step 2 — identify the DECISION BAND and surface the ambiguous boundary PAIRS.

The active-learning premise (README Prompt C): a label's value ≈ its probability of changing a
decision, so the *only* categories worth a Haiku rescore are those whose prior μ sits close enough to
the `0.3` gate that a rescore could **flip** the in/out call. Far-from-cutoff nodes can't flip → stay
on the prior (μ unchanged, free). This selects μ ∈ [lo, hi] (default [0.2, 0.45]) from the calibrated
e5 prior, splits it around the gate, and writes:

  * `boundary_band_e5.tsv`  — name<TAB>prior_mu for the band (the Haiku rescore worklist), μ-desc.
  * `boundary_pairs_e5.tsv` — band category × Physics-subdomain pairs for the *tight* straddle
    [0.25,0.35], where single-node membership is most ambiguous (a secondary, pairwise surfacing that
    feeds stream T; NOT Haiku-scored here — budget goes to the single-category membership labels).

Physics-subdomains are graph-derived: the children of `Physics` and of `Subfields_of_physics`.

    python3 boundary_band.py --mu-file dense_mu_e5.tsv
"""
from __future__ import annotations

import argparse
import os

from emit_dense_mu import GRAPH

ROOT = os.path.dirname(os.path.abspath(__file__))
GATE = 0.3


def load_mu_map(path):
    mu = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, val = line.rstrip("\n").split("\t")
            mu[name] = float(val)
    return mu


def physics_subdomains(graph_path):
    """Children of `Physics` and of `Subfields_of_physics` (graph-derived domain axes)."""
    roots = {"Physics", "Subfields_of_physics"}
    subs = []
    seen = set()
    with open(graph_path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("child"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            child, parent = parts[0], parts[1]
            if parent in roots and child not in roots and child not in seen:
                seen.add(child)
                subs.append(child)
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu-file", default=os.path.join(ROOT, "dense_mu_e5.tsv"))
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.45)
    ap.add_argument("--tight-lo", type=float, default=0.25)
    ap.add_argument("--tight-hi", type=float, default=0.35)
    ap.add_argument("--band-out", default=os.path.join(ROOT, "boundary_band_e5.tsv"))
    ap.add_argument("--pairs-out", default=os.path.join(ROOT, "boundary_pairs_e5.tsv"))
    args = ap.parse_args()

    mu = load_mu_map(args.mu_file)
    band = {n: m for n, m in mu.items() if args.lo <= m <= args.hi}
    below = {n: m for n, m in band.items() if m <= GATE}          # [lo, 0.30] currently OUT
    above = {n: m for n, m in band.items() if m > GATE}           # (0.30, hi] currently IN
    tight = {n: m for n, m in mu.items() if args.tight_lo <= m <= args.tight_hi}

    band_sorted = sorted(band.items(), key=lambda kv: -kv[1])
    with open(args.band_out, "w") as f:
        f.write(f"# Decision band μ∈[{args.lo},{args.hi}] from {os.path.basename(args.mu_file)} "
                f"(gate {GATE}). name<TAB>prior_mu, μ-desc. The Haiku-rescore worklist.\n")
        for n, m in band_sorted:
            f.write(f"{n}\t{m:.4f}\n")

    subs = physics_subdomains(GRAPH)
    tight_sorted = sorted(tight.items(), key=lambda kv: -kv[1])
    n_pairs = 0
    with open(args.pairs_out, "w") as f:
        f.write(f"# Ambiguous boundary pairs: tight-straddle band cats μ∈[{args.tight_lo},"
                f"{args.tight_hi}] × Physics-subdomains (graph-derived). category<TAB>subdomain<TAB>"
                f"prior_mu. Secondary pairwise surfacing for stream T; not Haiku-scored here.\n")
        for n, m in tight_sorted:
            if n in subs:
                continue
            for s in subs:
                f.write(f"{n}\t{s}\t{m:.4f}\n")
                n_pairs += 1

    print(f"prior map: {len(mu)} cats; gate {GATE}")
    print(f"DECISION BAND μ∈[{args.lo},{args.hi}]: {len(band)} cats  "
          f"(below-gate [{args.lo},{GATE}]: {len(below)} currently OUT;  "
          f"above-gate ({GATE},{args.hi}]: {len(above)} currently IN — the leak side)")
    print(f"  tight straddle μ∈[{args.tight_lo},{args.tight_hi}]: {len(tight)} cats")
    print(f"  → wrote band worklist → {os.path.basename(args.band_out)} ({len(band)} rows)")
    print(f"  → wrote {n_pairs} ambiguous pairs ({len(subs)} subdomains) → "
          f"{os.path.basename(args.pairs_out)}")
    print(f"  physics-subdomains: {', '.join(subs)}")
    print("\n  above-gate leak side (top 25 by prior μ — the false-positive candidates to veto):")
    for n, m in [kv for kv in band_sorted if kv[1] > GATE][:25]:
        print(f"    {m:.3f}  {n}")


if __name__ == "__main__":
    main()
