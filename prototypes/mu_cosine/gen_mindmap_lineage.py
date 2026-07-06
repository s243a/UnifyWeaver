#!/usr/bin/env python3
"""LINEAGE from SimpleMind mindmaps (do this BEFORE the 70/30 split — user direction).

For each node, navigate parent links UP to the root and emit the materialized path of TITLES
(root / … / node). This is the mindmap analogue of Pearltrees single-path LINEAGE — a mindmap node
hangs off one structural parent, so lineage is well-defined. Yields data NOT in enwiki.

Privacy: parse_smmx.py already scrubs (privacy.py::is_private_title drops a "private" node + its subtree,
a private root drops the whole map) at parse time — so this builds on already-clean output. Belt-and-
suspenders: we re-check both the RAW and the CLEANED path titles here.

Edges from parse_smmx are src(parent)→dst(child):relation. Parent map:
  subtopic / subcategory : parent[dst] = src   (dst is under src)
  super_category         : parent[src] = dst   (dst is broader than src)  — fallback only, counted+logged
Then materialized_path(node) = titles(root … node). 70/30 directional/superposition sampling comes AFTER.

  python3 gen_mindmap_lineage.py --maps context/*.smmx --out mindmap_lineage.tsv
"""
import argparse, glob, os, re, subprocess, sys


def canon_key(title):
    """Canonical node key from the TITLE, so the same concept appearing in multiple maps (with different .smmx
    slug conventions — case / hyphen / underscore) collapses to ONE node instead of duplicate variants that
    manufacture fake self-hops (SimpleMind cleanup, user 2026-07-06)."""
    return re.sub(r"[^a-z0-9]+", "-", (title or "").strip().lower()).strip("-") or "node"
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    from privacy import is_private_title
except Exception:
    def is_private_title(t): return "private" in (t or "").lower()

SENTINELS = ("root node", "root_node")   # shared top placeholder (folder "root"); dropped from paths
NAV = ("via link",)                       # navigation artifacts


def parse_map(smmx, tmp):
    pref = os.path.join(tmp, os.path.basename(smmx).replace(".smmx", "").replace(" ", "_"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "parse_smmx.py"), smmx, "--out-prefix", pref],
                       capture_output=True, text=True)
    if not os.path.exists(pref + "_nodes.tsv"):
        return {}, {}, set(), r.stderr
    title = {}
    with open(pref + "_nodes.tsv", encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split("\t")
            if c and c[0]:
                title[c[0]] = c[2] if len(c) > 2 and c[2] else c[0]
    struct_parent, broader = {}, {}       # structural (subtopic) vs super_category fallback
    with open(pref + "_edges.tsv", encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split("\t")
            if len(c) < 3:
                continue
            src, dst, rel = c[0], c[1], c[2]
            if rel in ("subtopic", "subcategory"):
                struct_parent.setdefault(dst, src)        # first structural parent wins (principal)
            elif rel == "super_category":
                broader.setdefault(src, dst)
    parent = dict(broader); parent.update(struct_parent)  # structural parent takes precedence
    fallback = set(broader) - set(struct_parent)          # parent came ONLY from super_category
    # BYPASS nav nodes in the STRUCTURE (not just the display path): if a node's parent is a nav node,
    # reconnect it to the nav node's parent, then drop nav nodes entirely. Fixes "via link" appearing as a
    # true parent in the graph judge + reconnects chains the nav node was breaking.
    nav_keys = {k for k, t in title.items() if t.strip().lower() in NAV}
    if nav_keys:
        _orig = dict(parent)                                  # snapshot: real_parent reads the PRE-mutation map
        def real_parent(n):
            p, seen = _orig.get(n), set()
            while p in nav_keys and p not in seen:
                seen.add(p); p = _orig.get(p)
            return p
        parent = {n: real_parent(n) for n in _orig}
        parent = {n: p for n, p in parent.items() if p is not None and p not in nav_keys and n not in nav_keys}
        fallback -= nav_keys
        for k in nav_keys:
            title.pop(k, None)
    return title, parent, fallback, ""


def lineage(node, parent):
    chain, seen = [node], {node}
    while node in parent and parent[node] not in seen:
        node = parent[node]; chain.append(node); seen.add(node)
    return list(reversed(chain))                          # root first


def clean_titles(titles):
    out = []
    for t in titles:
        tl = t.strip().lower()
        if tl in SENTINELS or tl in NAV:
            continue
        if out and out[-1] == t:                          # collapse cross-map join echoes
            continue
        out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=sorted(glob.glob(os.path.join(ROOT, "context", "*.smmx"))))
    ap.add_argument("--tmp", default="/tmp/mu_data/mm_parse")
    ap.add_argument("--out", default=os.path.join(ROOT, "mindmap_lineage.tsv"))
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)

    rows, skipped_private, fb_total, per_map = [], 0, 0, {}
    for smmx in a.maps:
        title, parent, fallback, err = parse_map(smmx, a.tmp)
        mapname = os.path.basename(smmx).replace(".smmx", "")
        if not title:
            print(f"  SKIP {mapname}: {err.strip()[:80] or 'no nodes (private root?)'}"); continue
        fb_total += len(fallback)
        n_map = 0
        for node in title:
            chain = lineage(node, parent)
            if len(chain) < 2:
                continue                                  # roots/orphans — no lineage
            raw = [title.get(k, k) for k in chain]
            clean = clean_titles(raw)
            # belt-and-suspenders: privacy on BOTH raw and cleaned titles (review #9)
            if any(is_private_title(t) for t in raw) or any(is_private_title(t) for t in clean):
                skipped_private += 1; continue
            if len(clean) < 2:
                continue
            # node_key = canonical title-slug (dedups cross-map concept variants) instead of the raw .smmx key
            rows.append((mapname, canon_key(clean[-1]), clean[-1], " / ".join(clean), len(clean)))
            n_map += 1
        per_map[mapname] = n_map

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("map\tnode_key\tnode_title\tmaterialized_path\tdepth\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"\nwrote {len(rows)} lineage chains from {len(a.maps)} maps → {a.out}")
    print(f"  per-map: {per_map}")
    print(f"  depth distribution: {dict(sorted(Counter(r[4] for r in rows).items()))}")
    print(f"  super_category-fallback parents (no structural parent): {fb_total}")
    print(f"  private lineages skipped (belt-and-suspenders): {skipped_private}")


if __name__ == "__main__":
    main()
