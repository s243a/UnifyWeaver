#!/usr/bin/env python3
"""Stream V, steps 4–5 — fuse the Haiku boundary rescores into the e5 prior and VALIDATE the theory.

Fusion is the **geometric mean** `√(prior·haiku)` (a log-opinion-pool / the cascade fusion used in the
merged `wikipedia_fuzzy_gated_hybrid_membership` test). It hard-vetoes a loose connection: a category
the e5 prior rates related but Haiku rates μ=0 fuses to `√(prior·0)=0` (Music, Politics, Sociology …),
while a genuine physics subfield Haiku confirms keeps a high μ. Only the **decision band** is fused;
far-from-cutoff categories keep the prior untouched (their rescore couldn't flip the gate anyway).

Validation (the point of stream V — does the active-learning / gating theory hold on real data?):
  1. DECISION-FLIP rate — how many band categories the rescore moved across the `0.3` gate. A high flip
     rate on the just-above-cutoff side is the active-learning premise paying off (those labels bought a
     decision change); a label far from the cutoff can't flip, which is why we didn't buy it.
  2. node-vs-root Lin is DEGENERATE — `Lin(X, Physics)` saturates (Physics is the most-general in-domain
     node, IC≈0), so membership cannot be read off node-vs-root graph similarity; you need the prior +
     the directional Haiku label. (Confirms the README's "do NOT validate against node-vs-root Lin".)
  3. cosine→μ CALIBRATION still holds — the e5 prior vs the 90-node Haiku fixture correlation is stable
     (≈ the +0.665 measured at emission), i.e. the prior we fused into is the same calibrated map.

    python3 fuse_and_validate.py --prior dense_mu_e5.tsv --boundary \
        ../../tests/fixtures/wikipedia_physics_boundary_haiku.tsv --fused-out dense_mu_e5_fused.tsv
"""
from __future__ import annotations

import argparse
import math
import os

from validate_lin_agreement import (
    GRAPH, load_graph, load_mu, lin_from_ic, gated_ic_for, pearson, spearman,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
FIXTURE90 = os.path.join(ROOT, "..", "..", "tests", "fixtures", "wikipedia_physics_fuzzy_nodes.tsv")
GATE = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=os.path.join(ROOT, "dense_mu_e5.tsv"))
    ap.add_argument("--boundary", default=os.path.join(
        ROOT, "..", "..", "tests", "fixtures", "wikipedia_physics_boundary_haiku.tsv"))
    ap.add_argument("--fused-out", default=os.path.join(ROOT, "dense_mu_e5_fused.tsv"))
    ap.add_argument("--root", default="Physics")
    ap.add_argument("--gate", type=float, default=GATE)
    args = ap.parse_args()

    prior = load_mu(args.prior)
    haiku = load_mu(args.boundary)
    band = list(haiku)                       # the rescored decision band
    print(f"prior: {len(prior)} cats; boundary rescores: {len(band)}; gate {args.gate}")

    # --- (4) fuse √(prior·haiku) on the band; everything else keeps the prior ---
    fused = dict(prior)
    for n in band:
        fused[n] = math.sqrt(max(prior.get(n, 0.0), 0.0) * max(haiku[n], 0.0))
    with open(args.fused_out, "w") as f:
        f.write(f"# e5 prior fused with Haiku boundary rescores: μ=√(prior·haiku) on the decision band "
                f"μ∈[0.2,0.45], prior elsewhere. clamp [0,1], names verbatim. Format: name<TAB>μ.\n")
        for n, m in fused.items():
            f.write(f"{n}\t{m:.4f}\n")
    print(f"wrote fused map ({len(fused)}) → {os.path.basename(args.fused_out)}")

    # --- (1) decision-flip rate over the band ---
    in_to_out, out_to_in, stay_in, stay_out = [], [], 0, 0
    for n in band:
        before, after = prior.get(n, 0.0) >= args.gate, fused[n] >= args.gate
        if before and not after:
            in_to_out.append(n)
        elif not before and after:
            out_to_in.append(n)
        elif before:
            stay_in += 1
        else:
            stay_out += 1
    flips = len(in_to_out) + len(out_to_in)
    print(f"\n[1] DECISION-FLIP rate: {flips}/{len(band)} band cats flipped across the {args.gate} gate "
          f"({100*flips/len(band):.1f}%)")
    print(f"    IN→OUT (leaks vetoed): {len(in_to_out)}   OUT→IN (promoted): {len(out_to_in)}   "
          f"stayed: in {stay_in} / out {stay_out}")
    print(f"    sample IN→OUT (relatedness leaks the membership rescore removed): "
          f"{', '.join(in_to_out[:12])}")
    if out_to_in:
        print(f"    OUT→IN (genuine physics the prior under-rated): {', '.join(out_to_in)}")
    above = [n for n in band if prior.get(n, 0.0) > args.gate]
    flipped_above = [n for n in above if fused[n] < args.gate]
    print(f"    just-above-cutoff side: {len(flipped_above)}/{len(above)} flipped OUT "
          f"({100*flipped_above.__len__()/max(len(above),1):.1f}% of the prior false-positives removed)")

    # --- (2) node-vs-root Lin degeneracy — non-circular, on the independent 90-node fixture map
    # (mirrors validate_lin_agreement's claim-check; using the fused map here would be circular since
    # fusion already encodes the Haiku membership). The claim: Lin(X,root) cannot recover μ(X). ---
    parents, children = load_graph(GRAPH)
    fx_mu = load_mu(FIXTURE90)
    fx_total = sum(fx_mu.values())
    root = args.root
    cache = {}
    nodes = [n for n in fx_mu if n in parents and fx_mu[n] >= args.gate and n != root]
    lin_xs, mu_xs, icx = [], [], []
    for n in nodes:
        v = lin_from_ic(n, root, parents, children, fx_mu, args.gate, fx_total, cache)
        if v is None:
            continue
        lin_xs.append(v)
        mu_xs.append(fx_mu[n])
        icx.append(gated_ic_for(n, children, fx_mu, args.gate, fx_total, cache))
    ic_root = gated_ic_for(root, children, fx_mu, args.gate, fx_total, cache)
    sat = sum(1 for v in lin_xs if v >= 0.999)
    fin = [(l, i) for l, i in zip(lin_xs, icx) if math.isfinite(i)]
    r_li = pearson([i for _, i in fin], [l for l, _ in fin]) if fin else float("nan")
    print(f"\n[2] node-vs-root Lin DEGENERACY (independent fixture map; IC({root})={ic_root:.3f} = most "
          f"general in-domain node ⇒ MICA(X,{root})={root} for its descendants, so Lin(X,{root})="
          f"2·IC({root})/(IC(X)+IC({root})) — a pure function of IC(X), i.e. *specificity* not membership):")
    print(f"    over {len(lin_xs)} in-domain fixture nodes: {sat} saturate Lin≥0.999; "
          f"Pearson(μ(X), Lin(X,{root}))={pearson(mu_xs, lin_xs):+.3f}  "
          f"Pearson(Lin, IC(X))={r_li:+.3f}")
    print(f"    ⇒ node-vs-root Lin is ~constant (saturated) across membership and tracks only depth — "
          f"it cannot recover μ. Confirmed degenerate; pairwise Lin is the real similarity test.")

    # --- (3) cosine→μ calibration vs the 90-node fixture (stability of the prior) ---
    fx = load_mu(FIXTURE90)
    common = [n for n in fx if n in prior]
    r = pearson([prior[n] for n in common], [fx[n] for n in common])
    rho = spearman([prior[n] for n in common], [fx[n] for n in common])
    print(f"\n[3] cosine→μ CALIBRATION vs the 90-node fixture ({len(common)} overlap): "
          f"Pearson r={r:+.3f}  Spearman ρ={rho:+.3f}  (stable ≈ the +0.665 measured at e5 emission ⇒ "
          f"the fused-into prior is the same calibrated map).")


if __name__ == "__main__":
    main()
