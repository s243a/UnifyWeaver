#!/usr/bin/env python3
"""LINEAGE from SimpleMind mindmaps (do this BEFORE the 70/30 split — user direction).

For each node, navigate parent links UP to the root and emit the materialized path of TITLES
(root / … / node). This is the mindmap analogue of Pearltrees single-path LINEAGE — a mindmap node
hangs off one structural parent, so lineage is well-defined. Yields data NOT in enwiki.

Privacy: parse_smmx.py already scrubs (privacy.py::is_private_title drops a "private" node + its subtree,
a private root drops the whole map) at parse time — so this builds on already-clean output. As a belt-and-
suspenders check we ALSO skip any lineage whose path contains a "private"-labelled title.

Edges from parse_smmx are src(parent)→dst(child):relation. Parent map:
  subtopic / subcategory : parent[dst] = src   (dst is under src)
  super_category         : parent[src] = dst   (dst is broader than src)  — only if no structural parent
Then materialized_path(node) = titles(root … node). 70/30 directional/superposition sampling comes AFTER.

  python3 gen_mindmap_lineage.py --maps context/*.smmx --out mindmap_lineage.tsv
"""
import argparse, glob, os, subprocess, sys
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    from privacy import is_private_title
except Exception:
    def is_private_title(t): return "private" in (t or "").lower()


def parse_map(smmx, tmp):
    pref = os.path.join(tmp, os.path.basename(smmx).replace(".smmx", "").replace(" ", "_"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "parse_smmx.py"), smmx, "--out-prefix", pref],
                       capture_output=True, text=True)
    if not os.path.exists(pref + "_nodes.tsv"):
        return {}, {}, r.stderr
    title = {}
    for ln in open(pref + "_nodes.tsv", encoding="utf-8"):
        if ln.startswith("#"):
            continue
        c = ln.rstrip("\n").split("\t")
        if c and c[0]:
            title[c[0]] = c[2] if len(c) > 2 and c[2] else c[0]
    struct_parent, broader = {}, {}      # keep structural (subtopic) separate from super_category fallback
    for ln in open(pref + "_edges.tsv", encoding="utf-8"):
        if ln.startswith("#"):
            continue
        c = ln.rstrip("\n").split("\t")
        if len(c) < 3:
            continue
        src, dst, rel = c[0], c[1], c[2]
        if rel in ("subtopic", "subcategory"):
            struct_parent.setdefault(dst, src)          # first structural parent wins (principal)
        elif rel == "super_category":
            broader.setdefault(src, dst)
    parent = dict(broader); parent.update(struct_parent)  # structural parent takes precedence
    return title, parent, ""


def lineage(node, parent):
    chain, seen = [node], {node}
    while node in parent and parent[node] not in seen:
        node = parent[node]; chain.append(node); seen.add(node)
    return list(reversed(chain))                          # root first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=sorted(glob.glob(os.path.join(ROOT, "context", "*.smmx"))))
    ap.add_argument("--tmp", default="/tmp/mu_data/mm_parse")
    ap.add_argument("--out", default=os.path.join(ROOT, "mindmap_lineage.tsv"))
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)

    rows, skipped_private, per_map = [], 0, {}
    for smmx in a.maps:
        title, parent, err = parse_map(smmx, a.tmp)
        mapname = os.path.basename(smmx).replace(".smmx", "")
        if not title:
            print(f"  SKIP {mapname}: {err.strip()[:80] or 'no nodes (private root?)'}"); continue
        n_map = 0
        for node in title:
            chain = lineage(node, parent)
            if len(chain) < 2:
                continue                                  # roots/orphans — no lineage
            titles = [title.get(k, k) for k in chain]
            if any(is_private_title(t) for t in titles):  # belt-and-suspenders
                skipped_private += 1; continue
            # CLEAN: drop the "Root Node" sentinel (in folder "root", a generic placeholder), drop nav
            # artifacts ("via link"), and collapse consecutive duplicate titles (cross-map join echoes).
            clean = []
            for t in titles:
                tl = t.strip().lower()
                if tl in ("root node", "root_node") or tl == "via link":
                    continue
                if clean and clean[-1] == t:
                    continue
                clean.append(t)
            if len(clean) < 2:
                continue
            rows.append((mapname, node, clean[-1], " / ".join(clean), len(clean)))
            n_map += 1
        per_map[mapname] = n_map

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("map\tnode_key\tnode_title\tmaterialized_path\tdepth\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"\nwrote {len(rows)} lineage chains from {len(a.maps)} maps → {a.out}")
    print(f"  per-map: {per_map}")
    print(f"  depth distribution: {dict(sorted(Counter(r[4] for r in rows).items()))}")
    print(f"  private lineages skipped (belt-and-suspenders): {skipped_private}")


if __name__ == "__main__":
    main()
