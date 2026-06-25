#!/usr/bin/env python3
"""Fuse the three corpora (SimpleMind + Pearltrees + enwiki) into ONE typed graph and take the N-HOP
neighbourhood of a start node — where a "hop" is ANY cross-corpus edge (SM↔SM, SM↔PT, PT↔PT, PT↔enwiki),
not only a path that reaches enwiki. Nodes are corpus-prefixed (`mm:` SimpleMind, `pt:` Pearltrees,
`wiki:` Wikipedia) so the same concept keeps its DISTINCT node-type across corpora, joined by `bridge`
edges — exactly the within-operator type diversity the node-type token needs.

Inputs (ref-based, no harvesting):
  * a SimpleMind .smmx (parsed via parse_smmx.py)
  * the cached Pearltrees harvests for its nodes' tree-ids (.pt_cache/pt_<id>.tsv, from bridge_seeds.py)

    python3 fuse_corpus.py --smmx "System Theory.smmx" --start system-theory --hops 2 --out-prefix systh_fused
"""
import argparse
import collections
import os
import re
import subprocess
import sys

from privacy import is_private_title, propagate as privacy_propagate    # scrub-everywhere (see privacy.py)

ROOT = os.path.dirname(os.path.abspath(__file__))
WIKI = re.compile(r"en\.wikipedia\.org/wiki/(.+)$")
PT_REL = {"element_of": "element_of", "collection_of": "subtopic", "subtopic": "subtopic",
          "shortcut": "assoc", "assoc": "assoc"}


def norm(s):
    return re.sub(r"[^a-z0-9]+", "-", (s or "").replace("Category:", "").lower()).strip("-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smmx", required=True)
    ap.add_argument("--start", default=None, help="start node slug (default: the map filename slug = root)")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--pt-cache", default=os.path.join(ROOT, ".pt_cache"))
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    # 1) parse the SimpleMind map
    sm_pref = "/tmp/_fuse_sm"
    subprocess.run([sys.executable, os.path.join(ROOT, "parse_smmx.py"), args.smmx, "--out-prefix", sm_pref],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    nodes = {}                                          # key → (corpus, type, title)
    edges = []                                          # (a_key, b_key, relation)

    def addn(corpus, slug, ntype, title):
        k = f"{corpus}:{norm(slug)}"
        nodes.setdefault(k, (corpus, ntype, title))
        return k

    sm_seeds = {}                                       # mm slug → pearltrees tree-id
    for ln in open(sm_pref + "_nodes.tsv", encoding="utf-8"):
        if ln.startswith("#"):
            continue
        c = ln.rstrip("\n").split("\t")                 # key, type, title, slug, pt_id, enwiki
        if len(c) < 3:
            continue
        mk = addn("mm", c[0], c[1] if len(c) > 1 else "mindmap_node", c[2] if len(c) > 2 else c[0])
        if len(c) >= 6 and c[5]:                        # direct enwiki anchor on the SM node
            edges.append((mk, addn("wiki", c[5], "category" if c[5].startswith("Category:") else "page", c[5]), "bridge"))
        if len(c) >= 5 and c[4].strip().isdigit():
            sm_seeds[norm(c[3] or c[0])] = c[4].strip()
    for ln in open(sm_pref + "_edges.tsv", encoding="utf-8"):
        if ln.startswith("#"):
            continue
        a, b, rel = (ln.rstrip("\n").split("\t") + ["", "", ""])[:3]
        if a and b:
            edges.append((f"mm:{norm(a)}", f"mm:{norm(b)}", rel))

    # 2) fold in the cached Pearltrees harvest for each SM node, + the SM↔PT and PT↔enwiki bridges.
    # PRIVACY (scrub-everywhere): raw .pt_cache harvests are NOT pre-scrubbed (they come from the private
    # harvester, not parse_pearltrees.py), so apply the SAME rule here — drop private-titled pt nodes and,
    # inherited down pt containment, their whole subtree — before building any fused edge.
    pt_rows, pt_children, pt_priv = [], {}, set()       # rows ; norm(parent)→{norm(child)} ; private norm-keys
    for slug, tid in sm_seeds.items():
        path = os.path.join(args.pt_cache, f"pt_{tid}.tsv")
        if not os.path.exists(path):
            continue
        for hl in open(path, encoding="utf-8"):
            if hl.startswith("#"):
                continue
            f = hl.rstrip("\n").split("\t")             # parent, child, rel, child_type, url, ...
            if len(f) < 3:
                continue
            pt_rows.append(f)
            if is_private_title(f[0]):
                pt_priv.add(norm(f[0]))
            if len(f) > 1 and is_private_title(f[1]):
                pt_priv.add(norm(f[1]))
            if len(f) > 3 and f[3] != "page" and f[1]:  # collection child → a containment edge
                pt_children.setdefault(norm(f[0]), set()).add(norm(f[1]))
    pt_priv = privacy_propagate(pt_priv, pt_children)   # inherit privacy down the subtree
    scrub_pt = 0

    for slug, tid in sm_seeds.items():                  # SM↔PT bridge per seed (skip private seeds)
        if norm(slug) in pt_priv:
            continue
        if os.path.exists(os.path.join(args.pt_cache, f"pt_{tid}.tsv")):
            edges.append((f"mm:{slug}", addn("pt", slug, "pearltrees_collection", slug), "bridge"))

    for f in pt_rows:
        if norm(f[0]) in pt_priv:                       # private parent → scrub the row
            scrub_pt += 1; continue
        pk = addn("pt", f[0], "pearltrees_collection", f[0])
        m = WIKI.search(f[4]) if len(f) > 4 else None
        if m:                                           # PagePearl whose url is enwiki → a cross-corpus link
            title = m.group(1).split("#")[0].replace("%28", "(").replace("%29", ")")
            wk = addn("wiki", title, "category" if title.startswith("Category:") else "page", title)
            # A `bridge` is the SAME concept across corpora (identity). A wiki page in a collection that names
            # a DIFFERENT thing (e.g. "Cybernetics" collection → "Centrifugal governor" page) is not identity
            # — it is the collection's cross-dataset REFERENCE, so use `see_also`, not bridge.
            rel = "bridge" if norm(title) == norm(f[0]) else "see_also"
            edges.append((pk, wk, rel))
        elif len(f) > 3:
            if norm(f[1]) in pt_priv:                   # private child → scrub
                scrub_pt += 1; continue
            ck = addn("pt", f[1], "pearltrees_collection" if f[3] != "page" else "page", f[1])
            edges.append((pk, ck, PT_REL.get(f[2], "assoc")))

    # 3) N-hop BFS from the start over the UNDIRECTED fused graph (a hop = any cross-corpus edge)
    adj = collections.defaultdict(set)
    for a, b, _ in edges:
        adj[a].add(b); adj[b].add(a)
    start = f"mm:{norm(args.start) if args.start else norm(os.path.splitext(os.path.basename(args.smmx))[0])}"
    if start not in nodes:
        start = next((k for k in nodes if k.startswith("mm:")), None)
    seen, frontier = {start}, {start}
    for _ in range(args.hops):
        nxt = set()
        for u in frontier:
            nxt |= adj[u] - seen
        seen |= nxt; frontier = nxt
    sub_edges = [(a, b, r) for a, b, r in edges if a in seen and b in seen]
    seen2, uniq = set(), []
    for e in sub_edges:
        if e[0] != e[1] and e not in seen2:
            seen2.add(e); uniq.append(e)

    from collections import Counter
    ntc = Counter(nodes[k][0] for k in seen if k in nodes)
    print(f"start={start} hops={args.hops}: {len(seen)} nodes {dict(ntc)}, {len(uniq)} edges "
          f"{dict(Counter(r for _, _, r in uniq))}")
    if scrub_pt or pt_priv:
        print(f"  PRIVACY: scrubbed {scrub_pt} pt rows ({len(pt_priv)} private pt nodes, inherited)")
    if args.out_prefix:
        with open(args.out_prefix + "_nodes.tsv", "w", encoding="utf-8") as f:
            f.write("# node_key(corpus:slug)\tcorpus\tnode_type\ttitle\n")
            for k in sorted(seen):
                if k in nodes:
                    co, ty, ti = nodes[k]
                    f.write(f"{k}\t{co}\t{ty}\t{ti}\n")
        with open(args.out_prefix + "_edges.tsv", "w", encoding="utf-8") as f:
            f.write("# a_key\tb_key\trelation\n")
            for a, b, r in uniq:
                f.write(f"{a}\t{b}\t{r}\n")
        print(f"  wrote {args.out_prefix}_nodes.tsv + {args.out_prefix}_edges.tsv")


if __name__ == "__main__":
    main()
