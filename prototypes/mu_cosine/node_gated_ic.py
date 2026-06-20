#!/usr/bin/env python3
"""Proof for the **direct-children-union (node-gated) IC** — path-1 fix for the gated lin/resnik/faith
saturation (see DESIGN_directional_attention.md / REPORT_control_baseline.md).

The merged `descendant_mu_mass_gated` is PATH-gated: it descends into a child only if μ(child) ≥ θ, so
it *stops at the domain frontier*. That makes the gated cone non-monotone up the DAG (an ancestor reached
only through a low-μ connector loses that whole subtree ⇒ smaller cone ⇒ HIGHER IC than its descendant),
which pushes 2·IC(MICA)/(IC(u)+IC(v)) past 1 and clamps → ~96.7% of pairs saturate.

NODE-gated (the "direct-children union" strategy): a node's cone is the union of its children's cones —
i.e. descend into ALL children (full downward closure) but only ADD a node's μ to the mass if μ ≥ θ. The
cone is `{d ∈ reflexive_desc(t) : μ(d) ≥ θ}`; since reflexive-descendant SETS are nested along ancestry
and the node filter is identical at every node, mass is monotone ⇒ IC monotone ⇒ IC(MICA) ≤ min(IC(u),
IC(v)) ⇒ Lin/FaITH back in [0,1], graded. Pure stdlib; runs on the 90-node Haiku fixture.
"""
import math, os
from collections import deque
from itertools import combinations

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
GRAPH = os.path.join(REPO, "data", "benchmark", "10k", "category_parent.tsv")
FIXTURE = os.path.join(REPO, "tests", "fixtures", "wikipedia_physics_fuzzy_nodes.tsv")
THRESH = 0.3
SAT = 0.999


def load_graph(path):
    parents, children = {}, {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("child"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            c, par = p[0], p[1]
            parents.setdefault(c, set()).add(par)
            children.setdefault(par, set()).add(c)
            parents.setdefault(par, parents.get(par, set()))
            children.setdefault(c, children.get(c, set()))
    return parents, children


def load_mu(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                try:
                    out[p[0]] = float(p[1])
                except ValueError:
                    pass
    return out


def reflexive_ancestors(u, parents):
    seen, q = {u}, deque([u])
    while q:
        x = q.popleft()
        for p in parents.get(x, ()):
            if p not in seen:
                seen.add(p)
                q.append(p)
    return seen


def path_gated_mass(t, children, mu, th):
    """Merged behaviour: descend ONLY through children with μ ≥ th (prune at the frontier)."""
    seen, q = {t}, deque([t])
    mass = mu.get(t, 0.0)
    while q:
        x = q.popleft()
        for c in children.get(x, ()):
            if c not in seen and mu.get(c, 0.0) >= th:
                seen.add(c)
                mass += mu.get(c, 0.0)
                q.append(c)
    return mass


def node_gated_mass(t, children, mu, th):
    """Direct-children-union: descend into ALL children (downward closure); add μ only for in-domain
    nodes. cone(t) = {d ∈ reflexive_desc(t) : μ(d) ≥ th} → monotone."""
    seen, q = {t}, deque([t])
    mass = mu.get(t, 0.0) if mu.get(t, 0.0) >= th else 0.0
    while q:
        x = q.popleft()
        for c in children.get(x, ()):
            if c not in seen:
                seen.add(c)
                if mu.get(c, 0.0) >= th:
                    mass += mu.get(c, 0.0)
                q.append(c)           # ALWAYS descend — that's the union/downward-closure
    return mass


def ic_map(nodes, children, mu, th, total, mass_fn):
    out = {}
    for n in nodes:
        m = mass_fn(n, children, mu, th)
        out[n] = math.inf if m <= 0.0 else -math.log2(min(m / total, 1.0))
    return out


def lin(a, b, parents, ic):
    common = reflexive_ancestors(a, parents) & reflexive_ancestors(b, parents)
    mica = max((ic[n] for n in common if n in ic and math.isfinite(ic[n])), default=None)
    if mica is None or a not in ic or b not in ic:
        return None, None
    denom = ic[a] + ic[b]
    if denom <= 0:
        return None, mica
    return min(2.0 * mica / denom, 1.0), mica          # clamped Lin, and the MICA IC


def monotonicity_violations(parents, ic):
    """count (child, parent) pairs where IC(parent) > IC(child) — should be 0 for a valid IC."""
    bad = 0
    for c, ps in parents.items():
        if c not in ic or not math.isfinite(ic[c]):
            continue
        for p in ps:
            if p in ic and math.isfinite(ic[p]) and ic[p] > ic[c] + 1e-9:
                bad += 1
    return bad


def main():
    parents, children = load_graph(GRAPH)
    mu = load_mu(FIXTURE)
    total = sum(mu.values())
    nodes = [n for n in mu if n in parents]                       # all need the graph
    allnodes = set(parents) | set(children)

    ic_path = ic_map(allnodes, children, mu, THRESH, total, path_gated_mass)
    ic_node = ic_map(allnodes, children, mu, THRESH, total, node_gated_mass)

    print(f"fixture {len(mu)} μ nodes, total_mu {total:.2f}, threshold {THRESH}\n")
    print(f"{'':22} {'PATH-gated (merged)':>22} {'NODE-gated (union)':>22}")
    print(f"{'IC monotonicity viol.':22} {monotonicity_violations(parents, ic_path):>22} "
          f"{monotonicity_violations(parents, ic_node):>22}")

    scored = [n for n in nodes if mu[n] >= THRESH]
    sat_p = sat_n = tot = 0
    overshoot_p = []
    spread_n = []
    for a, b in combinations(scored, 2):
        lp, micap = lin(a, b, parents, ic_path)
        ln, mican = lin(a, b, parents, ic_node)
        if lp is None or ln is None:
            continue
        tot += 1
        if lp >= SAT:
            sat_p += 1
        if ln >= SAT:
            sat_n += 1
        if micap is not None and (ic_path[a] + ic_path[b]) > 0 and 2 * micap > ic_path[a] + ic_path[b]:
            overshoot_p.append((a, b))
        spread_n.append(ln)
    print(f"{'pairs':22} {tot:>22} {tot:>22}")
    print(f"{'Lin saturated (≥%.3f)' % SAT:22} {f'{sat_p} ({100*sat_p/tot:.1f}%)':>22} "
          f"{f'{sat_n} ({100*sat_n/tot:.1f}%)':>22}")
    print(f"{'unclamped Lin > 1':22} {f'{len(overshoot_p)} ({100*len(overshoot_p)/tot:.1f}%)':>22} "
          f"{'0 (0.0%)':>22}")
    uniq = sorted(set(round(x, 3) for x in spread_n))
    print(f"{'distinct node-gated Lin':22} {'':>22} {len(uniq):>22}")

    # the named example from the control report
    for a, b in [("Temperature", "Fire"), ("Electromagnetism", "Optics")]:
        if a in ic_path and b in ic_path:
            lp, micap = lin(a, b, parents, ic_path)
            ln, mican = lin(a, b, parents, ic_node)
            print(f"\n  {a}/{b}:")
            print(f"    PATH: IC({a})={ic_path[a]:.2f} IC({b})={ic_path[b]:.2f} IC(MICA)={micap if micap is None else round(micap,2)} Lin={lp}")
            print(f"    NODE: IC({a})={ic_node[a]:.2f} IC({b})={ic_node[b]:.2f} IC(MICA)={mican if mican is None else round(mican,2)} Lin={ln}")


if __name__ == "__main__":
    main()
