#!/usr/bin/env python3
"""SimpleMind cleanup + honest re-check (user's root-anchoring point). The mindmap lineage has two confounds that
manufacture the 'D↔S co-occurrence rises with depth' pattern: (1) duplicate concept nodes (case/hyphen/underscore
key variants of the SAME title) inserting fake self-hops; (2) chains climbing past the per-map roots into an
ORGANISATIONAL super-layer (Applied mathematics / broad buckets) that isn't a tight taxonomic parent.

Fix: identify concepts by TITLE (merges the key variants), rebuild the parent graph, mark the org super-layer
(= ancestors of the 6 .smmx map roots), and split the scored pairs into content-rooted vs org-rooted to see where
the correlation actually lives. Reliable directional membership = content-node → its MAP root (user).
"""
import os, sys, numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emit_direction_blend import parse_responses

MAPS = {"Chaos theory", "Complex systems theory", "Cybernetics", "Dynamical systems", "Engineering", "LTI System Theory"}
DIR = ["subcategory", "subtopic", "element_of", "super_category"]; SYM = ["see_also", "assoc"]


def main():
    key2title = {}
    for l in open("/tmp/mu_data/mindmap_lin_nodes.tsv", encoding="utf-8"):
        if l.startswith("#"): continue
        c = l.rstrip("\n").split("\t"); key2title[c[0]] = c[3]
    # graph keyed on TITLE (merges chaos-theory/chaos_theory variants); parent = μ=1.0 ancestor's title
    best = defaultdict(lambda: (-1, None))
    for l in open("/tmp/mu_data/mindmap_lin_pairs.tsv", encoding="utf-8"):
        if l.startswith("#"): continue
        c = l.rstrip("\n").split("\t"); n, an, mu = c[0], c[1], float(c[2])
        if mu > best[n][0]: best[n] = (mu, an)
    ptitle = {}
    for n, (mu, an) in best.items():
        if an is None: continue
        t, tp = key2title.get(n, n), key2title.get(an, an)
        if t != tp: ptitle.setdefault(t, tp)                       # dedup: title→parent-title, drop self-loops
    print(f"nodes: {len(key2title)} keys → {len(set(key2title.values()))} distinct titles (merged {len(key2title)-len(set(key2title.values()))} variant duplicates)")

    def anc_titles(t, cap=20):
        out, x, seen = [], t, {t}
        while len(out) < cap:
            p = ptitle.get(x)
            if not p or p in seen: break
            out.append(p); seen.add(p); x = p
        return out
    # ORG super-layer = every title that is an ancestor of a map root
    org = set()
    for mp in MAPS:
        org |= set(anc_titles(mp))
    print(f"organisational super-layer (ancestors of the 6 map roots): {sorted(org)}")

    # split the SCORED pairs: is the ROOT a content concept or an org-layer bucket?
    rows = [ln.rstrip("\n").split("\t") for ln in open("/tmp/mu_data/mm_score_in.tsv", encoding="utf-8") if not ln.startswith("#")]
    byid = parse_responses("/tmp/mu_data/mm_resp.txt")
    def gmu(o, rel): return float((o.get(rel, {}) or {}).get("mu_fwd", 0)) if rel in DIR else float((o.get(rel, {}) or {}).get("mu", 0))
    def corr(D, S):
        D, S = np.array(D), np.array(S)
        return np.corrcoef(D, S)[0, 1] if len(D) > 5 and D.std() > 0 and S.std() > 0 else float("nan")
    grp = {"content-rooted": ([], []), "org-rooted": ([], []), "self-dup (node==root title)": ([], [])}
    byhop = defaultdict(lambda: {"content": ([], []), "org": ([], [])})
    for i, r in enumerate(rows):
        if i not in byid: continue
        node_t, root_t = r[0], r[1]; h = int(r[4][len("mm_h"):]) if r[4].startswith("mm_h") else 0
        D, S = max(gmu(byid[i], x) for x in DIR), max(gmu(byid[i], x) for x in SYM)
        if node_t == root_t: g = "self-dup (node==root title)"
        elif root_t in org: g = "org-rooted"
        else: g = "content-rooted"
        grp[g][0].append(D); grp[g][1].append(S)
        if g in ("content-rooted", "org-rooted"):
            byhop[h]["content" if g == "content-rooted" else "org"][0].append(D)
            byhop[h]["content" if g == "content-rooted" else "org"][1].append(S)
    print("\ncorr(μ_D,μ_S) by root type — where does the co-occurrence live?")
    for g, (D, S) in grp.items():
        print(f"  {g:32s} n={len(D):3d}  corr {corr(D,S):+.2f}")
    print("\nby hop (content-rooted vs org-rooted):")
    print(f"{'h':>2}  {'content corr (n)':>18}  {'org corr (n)':>16}")
    for h in sorted(byhop):
        c, o = byhop[h]["content"], byhop[h]["org"]
        print(f"{h:>2}  {corr(*c):+.2f} (n={len(c[0]):>2})        {corr(*o):+.2f} (n={len(o[0]):>2})")


if __name__ == "__main__":
    main()
