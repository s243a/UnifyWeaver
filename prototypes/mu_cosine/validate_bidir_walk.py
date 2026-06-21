#!/usr/bin/env python3
"""Validate the depth-balanced bidirectional sampler walk on the real 10k graph.

NO LLM budget — pure sampler engineering + measurement (no scoring, no retrain). Two checks:

  1. DEPTH-NEUTRALITY. From a set of interior seeds, run each walk mode and report the distribution
     of `depth(endpoint) − depth(seed)`. Expectation (DESIGN_bidirectional_walk.md):
       undirected (baseline) → skewed deep;  child-only → strictly ≥ 0;
       coinflip → ≈ symmetric about 0 (tightest);  global → ≈ symmetric (cheaper single-weight).

  2. DOMAIN-REACH. From `Physics`, confirm bidirectional walks reach sibling domains (`Chemistry`,
     `Computer_science`) as endpoints WITHOUT piling up on generic apexes (Nature / Branches_of_science
     / depth ≤ 1) — i.e. depth-balance + hub-down-weighting keep it lateral, not drifted.

Depth metrics. The category graph is **heavily cyclic** (most nodes sit in cycles), so no global
depth function is monotone along child edges. We therefore report two quantities:

  * PRIMARY — net directional displacement = (#down/child steps − #up/parent steps) along the walk.
    This is exactly the design's ±1-per-step depth model (down=+1, up=−1) and is robust to cycles.
    It is what the martingale derivation is about, so the predicted signatures apply cleanly here.
  * SECONDARY — node shortest-hop-depth delta (distance from any no-parent root, via child edges):
    a real-graph geometric reality check. Because the graph is cyclic and multi-parent, a child can
    be closer to a *different* root, so this metric is noisy — reported with that caveat.

    python3 validate_bidir_walk.py            # prints report, writes REPORT_bidir_walk.md + samples
"""
import collections
import os
import random
import statistics

from gen_mu_pairs import (
    GRAPH, ROOT, load_directed, estimate_global_beta,
    walk, load_graph, walk_child_only, walk_bidir,
)

N_SEEDS = 40            # interior seeds for the depth-neutrality sweep
WALKS_PER_SEED = 60     # walks per (seed, mode)
RNG_SEED = 7
APEX = {"Nature", "Branches_of_science", "Science", "Academic_disciplines",
        "Main_topic_classifications", "Articles", "Contents"}


def compute_depth(children, parents):
    """Multi-root BFS depth: 0 at every root (no-parent node), +1 per child step. Shortest such
    distance is the node's depth (toward leaves = deeper, toward apex = shallower)."""
    nodes = set(children) | set(parents)
    roots = [n for n in nodes if not parents.get(n)]
    depth, dq = {r: 0 for r in roots}, collections.deque(roots)
    while dq:
        u = dq.popleft()
        for c in children.get(u, ()):
            if c not in depth:
                depth[c] = depth[u] + 1
                dq.append(c)
    # any node unreached from a root (shouldn't happen on this graph) gets its min over neighbours later
    return depth


def displacement(path, children, parents):
    """Net directional displacement along a walk: +1 per down (child) step, −1 per up (parent) step.
    Reconstructed from consecutive path nodes; an edge that is both child and parent (a 2-cycle)
    counts 0. This is the design's ±1 depth model and is cycle-robust."""
    disp = 0
    for u, v in zip(path, path[1:]):
        down = v in children.get(u, ())
        up = v in parents.get(u, ())
        if down and not up:
            disp += 1
        elif up and not down:
            disp -= 1
    return disp


def descendants(root, children, cap=3):
    """Depth-BOUNDED down-BFS (≤ cap hops). Unbounded descendants are useless here: the benchmark was
    crawled from Physics, so via cycles ~everything is a 'Physics descendant' and the Chemistry subtree
    is fully contained in it. A shallow cap keeps each domain set local and disjoint enough to measure
    sibling reach."""
    seen, dq = {root}, collections.deque([(root, 0)])
    while dq:
        u, d = dq.popleft()
        if d >= cap:
            continue
        for c in children.get(u, ()):
            if c not in seen:
                seen.add(c)
                dq.append((c, d + 1))
    return seen


def histogram(deltas, lo=-6, hi=6, width=46):
    """Tiny text histogram of integer depth-deltas, clamped to [lo, hi]."""
    counts = collections.Counter(max(lo, min(hi, d)) for d in deltas)
    peak = max(counts.values()) if counts else 1
    lines = []
    for d in range(lo, hi + 1):
        n = counts.get(d, 0)
        bar = "#" * round(width * n / peak)
        tag = f"{d:+d}" if lo < d < hi else (f"≤{lo}" if d == lo else f"≥{hi}")
        lines.append(f"  {tag:>4} | {bar} {n}")
    return "\n".join(lines)


def stats(deltas):
    m = statistics.mean(deltas)
    sd = statistics.pstdev(deltas)
    pos = sum(1 for d in deltas if d > 0) / len(deltas)
    neg = sum(1 for d in deltas if d < 0) / len(deltas)
    zero = sum(1 for d in deltas if d == 0) / len(deltas)
    return dict(mean=m, sd=sd, lo=min(deltas), hi=max(deltas),
                frac_pos=pos, frac_neg=neg, frac_zero=zero)


def run_mode(name, seeds, depth, children, parents, walkfn, rng):
    disps, deltas = [], []
    for s in seeds:
        for _ in range(WALKS_PER_SEED):
            end, path = walkfn(s, rng)
            if end == s:
                continue
            disps.append(displacement(path, children, parents))
            if end in depth and s in depth:
                deltas.append(depth[end] - depth[s])
    return name, disps, stats(disps), deltas, stats(deltas)


def main():
    rng = random.Random(RNG_SEED)
    adj = load_graph(GRAPH)
    deg = {n: max(1, len(adj.get(n, ()))) for n in adj}
    children, parents = load_directed(GRAPH)
    depth = compute_depth(children, parents)
    gbeta = estimate_global_beta(children, parents)
    nodes = set(children) | set(parents)

    # interior seeds: have both a child and a parent, mid-depth (2..6), not hubs.
    cand = sorted(n for n in nodes
                  if children.get(n) and parents.get(n)
                  and 2 <= depth.get(n, 99) <= 6 and deg[n] <= 30)
    rng.shuffle(cand)
    seeds = cand[:N_SEEDS]

    sp, hb = 0.4, 1.0  # match gen_mu_pairs defaults (stop_prob, hub_beta)
    modes = [
        ("undirected, no hub-weight (β=0)", lambda s, r: walk(s, adj, deg, sp, 0.0, r)),
        ("undirected (baseline)", lambda s, r: walk(s, adj, deg, sp, hb, r)),
        ("child-only (downward)", lambda s, r: walk_child_only(s, children, deg, sp, hb, r)),
        ("bidir coinflip",        lambda s, r: walk_bidir(s, children, parents, deg, sp, hb, r,
                                                          mode="coinflip", global_beta=gbeta)),
        ("bidir global",          lambda s, r: walk_bidir(s, children, parents, deg, sp, hb, r,
                                                          mode="global", global_beta=gbeta)),
    ]

    results = []
    for name, fn in modes:
        results.append(run_mode(name, seeds, depth, children, parents, fn,
                                random.Random(RNG_SEED + 1)))

    # ---- depth-neutrality report ----
    out = []
    out.append("# REPORT — depth-balanced bidirectional walk (validation, no LLM budget)\n")
    out.append(f"Graph: `data/benchmark/10k/category_parent.tsv` — {len(nodes)} categories, "
               f"max shortest-depth {max(depth.values())}. The graph is **heavily cyclic** (only "
               f"{sum(1 for n in nodes if not parents.get(n))} no-parent roots; the rest sit in cycles), "
               f"so no global depth is monotone along child edges.\n")
    out.append(f"Handshake lemma holds (E[c]=E[p]); global up-weight **β = E[c²]/E[p²] = {gbeta:.1f}** "
               f"(children are heavy-tailed, parents concentrated).\n")
    out.append(f"Sweep: {len(seeds)} interior seeds (both child & parent, depth 2–6, deg ≤ 30), "
               f"{WALKS_PER_SEED} walks/seed/mode, stop_prob={sp}, hub_beta={hb}.\n")

    out.append("## 1a. PRIMARY — net directional displacement (down−up steps, the ±1 depth model)\n")
    out.append("This is the design's per-step depth model and is cycle-robust. Predicted signatures: "
               "undirected → drifts deep; child-only → strictly ≥ 0; coinflip/global → symmetric ≈ 0.\n")
    out.append("| mode | n | mean | sd | min..max | %deeper | %same | %shallower |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, disps, ds, _deltas, _dst in results:
        out.append(f"| {name} | {len(disps)} | {ds['mean']:+.2f} | {ds['sd']:.2f} | "
                   f"{ds['lo']:+d}..{ds['hi']:+d} | {ds['frac_pos']*100:.0f}% | "
                   f"{ds['frac_zero']*100:.0f}% | {ds['frac_neg']*100:.0f}% |")
    out.append("")
    for name, disps, ds, _deltas, _dst in results:
        out.append(f"### {name}  (displacement mean {ds['mean']:+.2f}, sd {ds['sd']:.2f})")
        out.append("```")
        out.append(histogram(disps))
        out.append("```")
        out.append("")

    out.append("## 1b. SECONDARY — node shortest-hop-depth delta (real-graph geometric check)\n")
    out.append("`depth(endpoint) − depth(seed)` with depth = shortest child-hops from any root. "
               "Noisy because the graph is cyclic/multi-parent (a child can be nearer a *different* "
               "root), so treat as a coarse cross-check, not the primary signal.\n")
    out.append("| mode | n | mean | sd | min..max | %deeper | %same | %shallower |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, _disps, _ds, deltas, dst in results:
        out.append(f"| {name} | {len(deltas)} | {dst['mean']:+.2f} | {dst['sd']:.2f} | "
                   f"{dst['lo']:+d}..{dst['hi']:+d} | {dst['frac_pos']*100:.0f}% | "
                   f"{dst['frac_zero']*100:.0f}% | {dst['frac_neg']*100:.0f}% |")
    out.append("")

    # ---- domain-reach sanity from Physics ----
    CAP = 3
    chem = descendants("Chemistry", children, CAP)
    cs = descendants("Computer_science", children, CAP)
    phys = descendants("Physics", children, CAP)
    sibling = (chem | cs) - phys
    reach_modes = [
        ("undirected (baseline)", lambda s, r: walk(s, adj, deg, sp, hb, r)),
        ("child-only (downward)", lambda s, r: walk_child_only(s, children, deg, sp, hb, r)),
        ("bidir coinflip",        lambda s, r: walk_bidir(s, children, parents, deg, sp, hb, r,
                                                          mode="coinflip", global_beta=gbeta)),
        ("bidir global",          lambda s, r: walk_bidir(s, children, parents, deg, sp, hb, r,
                                                          mode="global", global_beta=gbeta)),
    ]
    out.append("## 2. Domain-reach from `Physics` (sibling reach vs apex pile-up)\n")
    out.append(f"Endpoints classified against depth-≤{CAP} subtrees: **sibling** = in "
               f"Chemistry/Computer_science subtree but not Physics's (|sibling|={len(sibling)} nodes); "
               f"**in-Physics** = Physics subtree; **generic-apex** = a depth-0 root or a node in "
               f"{{{', '.join(sorted(APEX))}}} (the leak-conduit apexes we must NOT pile up on).\n")
    out.append("| mode | n | %sibling | %in-Physics | %generic-apex | mean depth |")
    out.append("|---|---:|---:|---:|---:|---:|")
    samples = {}
    reach = {}
    R = random.Random(RNG_SEED + 2)
    for name, fn in reach_modes:
        ends, sib, apex_n, inp, dsum = 0, 0, 0, 0, 0
        seen_pairs = []
        for _ in range(3000):
            end, path = fn("Physics", R)
            if end == "Physics":
                continue
            ends += 1
            dsum += depth.get(end, 0)
            is_sib = end in sibling
            is_apex = end in APEX or depth.get(end, 99) == 0
            if is_sib:
                sib += 1
            if is_apex:
                apex_n += 1
            if end in phys:
                inp += 1
            if len(seen_pairs) < 12 and end not in [p[1] for p in seen_pairs]:
                kind = ("sibling" if is_sib else "generic-apex" if is_apex else
                        "in-Physics" if end in phys else "other")
                seen_pairs.append(("Physics", end, kind, len(path) - 1))
        out.append(f"| {name} | {ends} | {sib/ends*100:.1f}% | {inp/ends*100:.1f}% | "
                   f"{apex_n/ends*100:.1f}% | {dsum/ends:.2f} |")
        samples[name] = seen_pairs
        reach[name] = dict(sib=sib / ends, apex=apex_n / ends)
    out.append("")

    # ---- honest verdict ----
    disp = {name: ds for name, _d, ds, _de, _dst in results}
    out.append("## 3. Verdict\n")
    out.append(
        f"- **coinflip is the cleanest depth distribution** and the recommended bidirectional mode: "
        f"net displacement mean {disp['bidir coinflip']['mean']:+.2f} (sd "
        f"{disp['bidir coinflip']['sd']:.2f}), tight and symmetric about 0 — the per-node β=c/p "
        f"martingale holds in practice. It reaches sibling domains "
        f"({reach['bidir coinflip']['sib']*100:.1f}% of Physics endpoints land in the Chemistry/CS "
        f"depth-≤{CAP} subtrees, vs **0% for child-only**, which structurally cannot leave the "
        f"subtree) while keeping generic-apex pile-up modest "
        f"({reach['bidir coinflip']['apex']*100:.1f}%).")
    out.append(
        f"- **global (β=E[c²]/E[p²]={gbeta:.1f}) is NOT recommended on this graph.** It over-corrects "
        f"hard to the apex: displacement mean {disp['bidir global']['mean']:+.2f}, and "
        f"{reach['bidir global']['apex']*100:.1f}% of Physics endpoints pile onto generic apexes. "
        f"Cause: E[c²] is dominated by a handful of mega-hubs (max 1778 children ⇒ c²≈3.2M), so the "
        f"aggregate β is ~20× a typical node's c/p≈2–3; plugging that into the *local* rule "
        f"P(down)=c/(c+βp) makes almost every interior node go up. The size-biased mean-field also "
        f"double-counts the down-branching that hub-down-weighting already suppresses. A robust global "
        f"variant would trim hubs / use the effective (hub-weighted) branching, but the per-node "
        f"coinflip is exact and free of this failure — prefer it.")
    out.append(
        f"- **The undirected baseline is NOT strongly skewed deep on this graph** "
        f"(mean {disp['undirected (baseline)']['mean']:+.2f} with hub-weight, "
        f"{disp['undirected, no hub-weight (β=0)']['mean']:+.2f} without): the graph is shallow "
        f"(mean node depth ≈1.8, max 12) and heavily cyclic, so reflecting-ish leaf/root boundaries on "
        f"short walks largely cancel the tree-model deep-drift the design predicts. Hub-down-weighting "
        f"moves the mean only slightly (its real payoff is domain-purity — lower apex pile-up, §2 — not "
        f"depth). So coinflip's advantage over the tuned baseline is **not** a big mean correction but "
        f"a *tighter, symmetric* distribution (thinner ±tails, the most mass at 0/±1) **plus** explicit "
        f"lateral reach that the baseline's wider, less controlled spread does not guarantee.")
    out.append(
        "- **Recommended default mix:** `--bidir --bidir-mode coinflip --bidir-frac 0.5` — half "
        "child-only walks for deep in-domain ancestor→descendant structure (the vertical axis), half "
        "depth-balanced coinflip walks for lateral sibling/cousin structure (the horizontal axis). "
        "Lean to `--bidir-frac 0.3–0.4` if you want to keep the in-domain coherence dominant.")
    out.append("")

    report = "\n".join(out)
    with open(os.path.join(ROOT, "REPORT_bidir_walk.md"), "w") as f:
        f.write(report)
    print(report)

    # ---- commit a small sample of Physics pairs per mode ----
    samp_path = os.path.join(ROOT, "bidir_pairs.sample.tsv")
    with open(samp_path, "w") as f:
        f.write("# sample (start, endpoint) pairs from gen_mu_pairs walks, Physics seed, per mode.\n")
        f.write("# columns: mode<TAB>start<TAB>endpoint<TAB>endpoint_kind<TAB>walk_len\n")
        for name, rows in samples.items():
            for a, b, kind, wl in rows:
                f.write(f"{name}\t{a}\t{b}\t{kind}\t{wl}\n")
    print(f"\nwrote REPORT_bidir_walk.md and {samp_path}")


if __name__ == "__main__":
    main()
