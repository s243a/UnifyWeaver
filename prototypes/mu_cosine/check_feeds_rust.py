#!/usr/bin/env python3
"""Sanity-check that a dense μ map feeds the Rust core's `gated_ic` / `lin_from_ic` (README step 4b /
the chosen-map check). Faithful python port of those functions (reused from `validate_lin_agreement`),
run on a `name<TAB>μ` file (e.g. `dense_mu_direct.py --model e5 --out ...`).

Two layers, kept separate on purpose:
  * **Mechanical integration (asserted)** — the load-bearing guards + that the map actually drives the
    pipeline: every name resolves verbatim (else silent μ=0), μ ∈ [0,1] (clamp guard — mass is summed),
    `gated_ic` is finite for in-domain anchors, and IC orders general→specific (big cone ⇒ low IC).
  * **Discrimination (diagnostic, not asserted)** — membership μ and gated Lin for physics vs
    non-physics probes. NOT a hard pass/fail because it is a property of the *map's quality*, not the
    integration, and because pairwise Lin on this graph saturates toward 1.0 for many pairs (the
    membership μ is the cleaner separator; an out-of-domain node only drives Lin→0 once it gates OUT).

    python3 check_feeds_rust.py --mu-file dense_mu_e5.tsv
"""
from __future__ import annotations

import argparse
import math

from validate_lin_agreement import (load_graph, load_mu, lin_from_ic, gated_ic_for, GRAPH)

PHYSICS = ["Physics", "Optics", "Thermodynamics", "Electromagnetism", "Quantum_mechanics", "Atoms"]
NONPHYS = ["Music", "Cooking", "Religious_buildings", "Politics", "Fashion"]
IN_PAIRS = [("Optics", "Thermodynamics"), ("Optics", "Electromagnetism")]
OUT_PAIRS = [("Optics", "Music"), ("Physics", "Cooking"), ("Thermodynamics", "Religious_buildings")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu-file", required=True, help="dense μ map (name<TAB>μ)")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    parents, children = load_graph(GRAPH)
    graph_names = set(parents)
    mu = load_mu(args.mu_file)
    total_mu = sum(mu.values())
    cache = {}

    # ---- mechanical integration (asserted) ----
    resolved = sum(1 for n in mu if n in graph_names)
    out_of_range = [(n, m) for n, m in mu.items() if m < 0.0 or m > 1.0]
    print(f"{len(mu)} μ rows; {resolved}/{len(mu)} resolve against the graph; Σμ={total_mu:.1f}")
    assert resolved == len(mu), f"{len(mu)-resolved} names do NOT resolve (would silently become μ=0)"
    assert not out_of_range, f"{len(out_of_range)} μ outside [0,1] (corrupts mass): {out_of_range[:3]}"

    ics = {n: gated_ic_for(n, children, mu, args.threshold, total_mu, cache)
           for n in PHYSICS if n in graph_names}
    finite = {n: ic for n, ic in ics.items() if math.isfinite(ic)}
    assert finite, "no in-domain anchor has a finite gated IC — the map gated everything out"
    assert ics.get("Physics", math.inf) <= min(v for n, v in finite.items() if n != "Physics"), \
        "IC(Physics) should be the lowest (most general) among in-domain anchors"
    print("  guards OK: names resolve verbatim; μ∈[0,1]; gated_ic finite + general→specific ordered")
    print("  gated IC (general→specific): " +
          "  ".join(f"{n}={ic:.2f}" for n, ic in sorted(finite.items(), key=lambda kv: kv[1])))

    # ---- discrimination (diagnostic) ----
    print(f"\nmembership μ — physics:  " +
          "  ".join(f"{n.split('_')[0]}={mu.get(n,0):.2f}" for n in PHYSICS if n in graph_names))
    print(f"membership μ — non-phys: " +
          "  ".join(f"{n.split('_')[0]}={mu.get(n,0):.2f}" for n in NONPHYS if n in graph_names))
    in_above = sum(1 for n in PHYSICS if mu.get(n, 0) >= args.threshold and n in graph_names)
    out_above = sum(1 for n in NONPHYS if mu.get(n, 0) >= args.threshold and n in graph_names)
    print(f"  at the {args.threshold} gate: {in_above}/{sum(1 for n in PHYSICS if n in graph_names)} "
          f"physics pass, {out_above}/{sum(1 for n in NONPHYS if n in graph_names)} non-physics pass "
          f"(fewer non-physics passing = cleaner absolute separation)")

    def lin(a, b):
        return lin_from_ic(a, b, parents, children, mu, args.threshold, total_mu, cache) \
            if a in graph_names and b in graph_names else None
    print("  gated Lin in-domain:  " + "  ".join(f"{a[:4]}~{b[:4]}={lin(a,b)}" for a, b in IN_PAIRS))
    print("  gated Lin out-domain: " + "  ".join(f"{a[:4]}~{b[:4]}={lin(a,b)}" for a, b in OUT_PAIRS))
    print("\nmechanical integration: PASS (the map feeds gated_ic / lin_from_ic in the Rust core).")


if __name__ == "__main__":
    main()
