#!/usr/bin/env python3
"""Validate the symmetric MiniLM μ-encoder (#3287) as the **control baseline** and resolve the three
open theory questions from `DESIGN_directional_attention.md`. Pure analysis on the existing maps +
faithful `gated_ic`/`lin_from_ic` port (no LLM budget). The numbers it prints are the baseline the
directional (e5 multi-operator) track must beat.

Q1  lin_from_ic SATURATION CEILING — what fraction of scored-pair graph-Lin pin at 1.0, *why* (a graph
    property), and the ceiling that puts on the lin-agreement Pearson (the directional model's headroom).
Q2  DECISION-FLIP RATE — of the decision-band categories (control μ ∈ [0.2,0.45]), how many would a
    membership rescore flip across the 0.3 gate? Measured two ways with no new budget: (a) ground-truth
    against the 90-node Haiku fixture, (b) proxy against an independent map (e5-direct).
Q3  COLD-START COVERAGE — does the symmetric map give sane μ for out-of-domain nodes never seen in
    training? Extends the 5-probe gate-leak test to a large graph-distance-based out-of-domain sample,
    and compares trained-vocab vs never-seen parity.

    python3 validate_control_baseline.py --encoder control.pt \
        --control-map dense_mu_control.tsv --proxy-map dense_mu_e5.tsv
"""
from __future__ import annotations

import argparse
import math
import os
from collections import deque
from itertools import combinations

import torch

from validate_lin_agreement import (load_graph, load_mu, lin_from_ic, gated_ic_for,
                                     reflexive_ancestors, pearson, spearman, GRAPH, FIXTURE)
from mu_encoder_torch import Config, MuEncoder, build_minilm_init

ROOT = os.path.dirname(os.path.abspath(__file__))
SCORED = os.path.join(ROOT, "mu_pairs_scored.tsv")
ANCHOR = "Physics"
SAT = 0.999      # Lin ≥ SAT counts as saturated
BAND = (0.2, 0.45)
GATE = 0.3


def bfs_dist_undirected(parents, src):
    adj = {}
    for c, ps in parents.items():
        for p in ps:
            adj.setdefault(c, set()).add(p)
            adj.setdefault(p, set()).add(c)
    dist = {src: 0}
    q = deque([src])
    while q:
        x = q.popleft()
        for y in adj.get(x, ()):
            if y not in dist:
                dist[y] = dist[x] + 1
                q.append(y)
    return dist


def training_vocab():
    names = set()
    with open(SCORED) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                names.update((p[0], p[1]))
    return names


# --------------------------------------------------------------------------------------------------
def unclamped_lin(a, b, parents, children, mu, threshold, total_mu, cache):
    """Lin BEFORE the `.min(1.0)` cap — `2·IC(MICA)/(IC(u)+IC(v))`. None if undefined."""
    au, av = reflexive_ancestors(a, parents), reflexive_ancestors(b, parents)
    mica = None
    for n in au & av:
        ic = gated_ic_for(n, children, mu, threshold, total_mu, cache)
        if math.isfinite(ic) and (mica is None or ic > mica):
            mica = ic
    if mica is None:
        return None
    ica = gated_ic_for(a, children, mu, threshold, total_mu, cache)
    icb = gated_ic_for(b, children, mu, threshold, total_mu, cache)
    if ica + icb <= 0:
        return None
    return 2.0 * mica / (ica + icb)


def q1_saturation(args, parents, children, mu_fix, total_mu):
    print("\n" + "=" * 90)
    print("Q1  lin_from_ic SATURATION CEILING")
    print("=" * 90)
    nodes = [n for n in mu_fix if n in parents and mu_fix[n] >= args.threshold]
    cache = {}
    pairs, lin, unc = [], [], []
    for a, b in combinations(nodes, 2):
        v = lin_from_ic(a, b, parents, children, mu_fix, args.threshold, total_mu, cache)
        if v is not None:
            pairs.append((a, b))
            lin.append(v)
            unc.append(unclamped_lin(a, b, parents, children, mu_fix, args.threshold, total_mu, cache))
    n = len(pairs)
    sat_idx = [i for i, v in enumerate(lin) if v >= SAT]
    p_sat = len(sat_idx) / n
    print(f"{n} scored-node pairs with a finite-IC common ancestor; "
          f"{len(sat_idx)} ({100*p_sat:.1f}%) saturate at Lin ≥ {SAT}")

    # WHY (the graph property): Lin = min(2·IC(MICA)/(IC(u)+IC(v)), 1). It clamps to 1.0 whenever
    # 2·IC(MICA) ≥ IC(u)+IC(v) — and under μ-GATING that is the common case, because gating prunes
    # ancestor cones: a common ancestor reachable only THROUGH low-μ (out-of-domain) connectors has a
    # tiny *gated* descendant cone, hence a HIGH IC — often higher than the nodes it sits above. So the
    # MICA is over-informative and the ratio blows past 1.0 and clamps. (IC is non-monotone up the DAG
    # once the cone is gated — the same in-domain-leak pruning the Rust core does on purpose.)
    clamp_active = sum(1 for i in sat_idx if unc[i] is not None and unc[i] >= 1.0 - 1e-9)
    mica_gt = sum(1 for i in sat_idx if unc[i] is not None and unc[i] > 1.0 + 1e-6)
    med_unc = sorted(unc[i] for i in sat_idx if unc[i] is not None)[len(sat_idx)//2]
    print(f"  WHY: {clamp_active}/{len(sat_idx)} saturated pairs have 2·IC(MICA) ≥ IC(u)+IC(v) (clamp "
          f"active); {mica_gt} have un-clamped Lin strictly > 1 (median un-clamped {med_unc:.2f}). "
          f"μ-gating prunes ancestor cones ⇒ IC(MICA) is often > IC(u),IC(v) (non-monotone up the DAG), "
          f"so the ratio overshoots 1.0. A graph/gating property — independent of the μ map.")

    # semantic cosines for the SAME pairs (control encoder)
    names = sorted({x for p in pairs for x in p})
    init = build_minilm_init(names)
    ckpt = torch.load(args.encoder, map_location="cpu", weights_only=False)
    model = MuEncoder(Config(**ckpt["cfg"]), init_embeddings=init, names=names, freeze_init=True)
    model.blocks.load_state_dict(ckpt["blocks"])
    model.eval()
    with torch.no_grad():
        cos, _ = model.mu(model._ids([a for a, _ in pairs]), model._ids([b for _, b in pairs]))
    cos = cos.clamp(0, 1).tolist()

    r_all, rho_all = pearson(lin, cos), spearman(lin, cos)
    ns = [i for i in range(n) if i not in set(sat_idx)]
    r_ns = pearson([lin[i] for i in ns], [cos[i] for i in ns]) if len(ns) > 2 else float("nan")
    print(f"  control lin-agreement Pearson r={r_all:+.3f}  Spearman ρ={rho_all:+.3f} (all {n})")
    print(f"  on the {len(ns)} NON-saturated pairs (the only ones whose Lin VARIES): Pearson r={r_ns:+.3f}")
    print(f"  CEILING: graph-Lin carries gradable signal on only {len(ns)}/{n} ({100*(1-p_sat):.1f}%) "
          f"of pairs — the other {100*p_sat:.1f}% are pinned at 1.0 and add only label-noise to the "
          f"Pearson. So lin-agreement is a LOW-RESOLUTION metric; the directional model's headroom is "
          f"those {len(ns)} non-saturated pairs (control already +{r_ns:.3f} there), not the global r.")
    return {"pairs": n, "sat_frac": p_sat, "clamp_active": clamp_active / max(1, len(sat_idx)),
            "med_unclamped": med_unc, "r_all": r_all, "rho_all": rho_all, "r_nonsat": r_ns,
            "n_nonsat": len(ns)}


def q2_decision_flip(args, parents, control, proxy, mu_fix):
    print("\n" + "=" * 90)
    print("Q2  DECISION-FLIP RATE (active-learning premise)")
    print("=" * 90)
    lo, hi = BAND
    band = [n for n, m in control.items() if lo <= m <= hi]
    print(f"{len(band)} categories in the control decision band μ∈[{lo},{hi}] (of {len(control)})")

    # (a) ground truth: fixture nodes in the band, do their Haiku μ land on the other side of the gate?
    fix_band = [n for n in band if n in mu_fix]
    flips_gt = sum(1 for n in fix_band if (control[n] < GATE) != (mu_fix[n] < GATE))
    if fix_band:
        print(f"  (a) ground-truth vs 90-node Haiku fixture: {len(fix_band)} band nodes are fixture-"
              f"scored; {flips_gt} ({100*flips_gt/len(fix_band):.0f}%) flip across the {GATE} gate")
    else:
        print("  (a) no band node overlaps the fixture (expected — fixture nodes are high-μ physics)")

    # (b) proxy: an independent map (e5-direct) as a stand-in 'rescore'
    both = [n for n in band if n in proxy]
    flips_px = sum(1 for n in both if (control[n] < GATE) != (proxy[n] < GATE))
    print(f"  (b) proxy vs independent e5-direct map: {flips_px}/{len(both)} "
          f"({100*flips_px/max(1,len(both)):.0f}%) flip across the {GATE} gate")
    # baseline: how often do two maps disagree OUTSIDE the band (confident region)? should be lower.
    conf = [n for n in control if n in proxy and (control[n] < lo or control[n] > hi)]
    flips_conf = sum(1 for n in conf if (control[n] < GATE) != (proxy[n] < GATE))
    print(f"      vs confident region (μ outside the band): {flips_conf}/{len(conf)} "
          f"({100*flips_conf/max(1,len(conf)):.1f}%) flip — the band SHOULD flip more if it is the "
          f"genuinely-uncertain set (validates active learning).")
    return {"band": len(band), "flip_gt": (flips_gt, len(fix_band)),
            "flip_proxy_pct": 100*flips_px/max(1, len(both)),
            "flip_conf_pct": 100*flips_conf/max(1, len(conf))}


def q3_cold_start(args, parents, control, dist):
    print("\n" + "=" * 90)
    print("Q3  COLD-START COVERAGE (out-of-domain, never trained)")
    print("=" * 90)
    train = training_vocab()
    far = args.far_dist
    # out-of-domain proxy: far from Physics in the undirected graph AND never seen in training
    ood = [n for n in control if dist.get(n, 99) >= far and n not in train]
    leak = sum(1 for n in ood if control[n] >= GATE)
    print(f"trained vocab {len(train)} names; {len(ood)} out-of-domain nodes "
          f"(graph dist ≥ {far} from {ANCHOR}, never trained)")
    print(f"  OOD gate-leak: {leak}/{len(ood)} ({100*leak/max(1,len(ood)):.1f}%) pass the {GATE} gate "
          f"(lower = cleaner rejection). μ mean {sum(control[n] for n in ood)/max(1,len(ood)):.3f}, "
          f"max {max(control[n] for n in ood):.2f}")

    # unseen IN-DOMAIN: near Physics (dist ≤ 2) but never trained — cold-start must still score these
    # high (generalisation INTO the domain, not just rejection of far nodes).
    near_unseen = [n for n in control if dist.get(n, 99) <= 2 and n not in train and n != ANCHOR]
    near_hi = sum(1 for n in near_unseen if control[n] >= GATE)
    print(f"  unseen in-domain (dist ≤ 2, never trained): {len(near_unseen)} nodes, "
          f"{100*near_hi/max(1,len(near_unseen)):.0f}% pass the gate, μ̄="
          f"{sum(control[n] for n in near_unseen)/max(1,len(near_unseen)):.3f}")

    # cold-start parity at MATCHED distance (the unbiased test: does being in training change μ?)
    def band_stats(dmin, dmax):
        seen = [control[n] for n in control if dmin <= dist.get(n, 99) <= dmax and n in train]
        uns = [control[n] for n in control if dmin <= dist.get(n, 99) <= dmax and n not in train]
        gp = lambda ms: 100*sum(1 for m in ms if m >= GATE)/max(1, len(ms))
        return (len(seen), gp(seen)), (len(uns), gp(uns))
    for label, (dmin, dmax) in [("near d≤2", (0, 2)), ("far d≥5", (far, 99))]:
        (ns_, gs), (nu, gu) = band_stats(dmin, dmax)
        print(f"  parity {label}: seen ({ns_}) gate-pass {gs:.0f}%  vs  unseen ({nu}) gate-pass {gu:.0f}% "
              f"— matched distance, so a gap would mean memorisation")
    return {"ood": len(ood), "ood_leak_pct": 100*leak/max(1, len(ood)),
            "near_unseen_pass": 100*near_hi/max(1, len(near_unseen)), "n_near_unseen": len(near_unseen)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--control-map", required=True, help="dense μ map from the control encoder")
    ap.add_argument("--proxy-map", required=True, help="independent map (e5-direct) for the flip proxy")
    ap.add_argument("--threshold", type=float, default=GATE)
    ap.add_argument("--far-dist", type=int, default=5, help="graph dist ≥ this = out-of-domain")
    args = ap.parse_args()

    parents, children = load_graph(GRAPH)
    mu_fix = load_mu(FIXTURE)
    total_mu = sum(mu_fix.values())
    control = load_mu(args.control_map)
    proxy = load_mu(args.proxy_map)
    dist = bfs_dist_undirected(parents, ANCHOR)

    r1 = q1_saturation(args, parents, children, mu_fix, total_mu)
    r2 = q2_decision_flip(args, parents, control, proxy, mu_fix)
    r3 = q3_cold_start(args, parents, control, dist)

    print("\n" + "=" * 90)
    print("CONTROL BASELINE SUMMARY (the number the directional track must beat)")
    print("=" * 90)
    print(f"  held-out pairwise-μ corr ............ +0.726  (MSE 0.065)   [from #3287 reproduction]")
    print(f"  lin-agreement Pearson / Spearman .... {r1['r_all']:+.3f} / {r1['rho_all']:+.3f}")
    print(f"  lin saturation fraction ............. {100*r1['sat_frac']:.1f}%  ({r1['n_nonsat']} of "
          f"{r1['pairs']} pairs carry gradable signal)")
    print(f"  lin-agreement on non-saturated ...... {r1['r_nonsat']:+.3f}  (the real headroom)")
    print(f"  decision-flip rate (band, proxy) .... {r2['flip_proxy_pct']:.0f}%  vs {r2['flip_conf_pct']:.1f}% confident region")
    print(f"  cold-start OOD gate-leak ............ {r3['ood_leak_pct']:.1f}%  ({r3['ood']} nodes, dist≥{args.far_dist})")
    print(f"  cold-start unseen in-domain pass .... {r3['near_unseen_pass']:.0f}%  ({r3['n_near_unseen']} near, never trained)")


if __name__ == "__main__":
    main()
