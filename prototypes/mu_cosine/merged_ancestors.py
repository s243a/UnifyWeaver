#!/usr/bin/env python3
"""merged_ancestors.py — build the MULTI-PARENT ancestor DAG and each node's MERGED ancestor list (PATH operator).

The PATH increment-0 representation (DESIGN_path_operator.md, per the design discussion):
  * the graph is a **DAG** — a folder can have several parents (7% of Pearltrees folders do, up to 13). Single-path
    LINEAGE collapses that to one arbitrary chain; PATH scores against the **merged** set of ancestor contexts.
  * a node's passage = its **merged ancestor list**: all reachable ancestors (titles), **ID-keyed merged** (each
    unique ancestor once), **no materialized-id line** (a linear id line can't represent a branching/merged
    structure — the id stays an internal merge key, not embedded text).
  * guards: **cycle detection** (visited set) + **max-depth** cap (deterministic bound so the merged list stays
    precomputable; the stochastic stop-β variant is for later sampling). Optionally **scope** to a root subtree.

Parent edges come from each tree's sub-tree-reference pearls: a pearl's `treeId` is the PARENT, `contentTree.id`
(contentType 2/5/6) is the CHILD it references.
"""
import glob, json, os, collections


def load_multiparent_dag(trees_dir):
    """→ (parents_of: child_id→[parent_id...], title_of: id→title). Multi-parent (the full DAG, not the collapse)."""
    parents_of = collections.defaultdict(list); title_of = {}
    for f in glob.glob(os.path.join(trees_dir, "*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        t = d.get("api_response", {}).get("tree", {}) if isinstance(d, dict) else {}
        if not isinstance(t, dict):
            continue
        pid = t.get("id")
        if pid is None:
            continue
        pid = str(pid)
        if t.get("title"):
            title_of[pid] = t["title"]
        for p in (t.get("pearls") or []):
            if not isinstance(p, dict):
                continue
            ct = p.get("contentTree")
            cid = ct.get("id") if isinstance(ct, dict) else None
            if cid and str(cid) != pid:
                cid = str(cid)
                if pid not in parents_of[cid]:
                    parents_of[cid].append(pid)
                if isinstance(ct, dict) and ct.get("title"):
                    title_of.setdefault(cid, ct["title"])
    return parents_of, title_of


def load_category_graph(tsv):
    """→ (parents_of: child→[parents], title_of: name→title) from a `child<TAB>parent` category graph (enwiki/
    simplewiki — the genuine multi-parent DAG; Pearltrees' multi-parent is just this reflected in, so PATH belongs
    here). Category name IS the node id and the title."""
    from mu_attention import load_dag
    parents, _children, _deg = load_dag(tsv)
    parents_of = {c: list(ps) for c, ps in parents.items()}
    title_of = {n: n.replace("_", " ") for n in parents}
    return parents_of, title_of


def load_assembled_dag(dag_tsv, titles_tsv=None):
    """→ (parents_of: child→[parents], title_of) from an assembled `child<TAB>parent` DAG TSV (e.g. the RDF+API
    Pearltrees union that recovers the parents the truncated RDF export dropped) + an optional `id<TAB>title` map.
    Nodes without a title fall back to the id string."""
    parents_of = collections.defaultdict(list)
    for line in open(dag_tsv, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1] not in parents_of[p[0]]:
            parents_of[p[0]].append(p[1])
    title_of = {}
    if titles_tsv:
        for line in open(titles_tsv, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                title_of[p[0]] = p[1]
    return dict(parents_of), title_of


def subtree_scope(root, parents_of):
    """Set of all descendants of `root` (BFS down) — pass as scope= to bound the ancestor walk to a content subtree
    (e.g. Main_topic_classifications), avoiding admin-category wandering / weird long paths by construction."""
    children_of = collections.defaultdict(list)
    for c, ps in parents_of.items():
        for p in ps:
            children_of[p].append(c)
    scope, frontier = {str(root)}, [str(root)]
    while frontier:
        for ch in children_of.get(frontier.pop(), []):
            if ch not in scope:
                scope.add(ch); frontier.append(ch)
    return scope


def merged_ancestor_list(node, parents_of, title_of, max_depth=15, scope=None, max_ancestors=None):
    """Merged ancestor list for `node`: unique ancestors (ID-keyed) reached by walking up ALL parents.
    Cycle-detected (visited), max-depth bounded. Returns [(id, title, min_hop)] ordered root-ward first.
    `scope` (a set of allowed ids) restricts the walk to a subtree if given. `max_ancestors` keeps only the
    NEAREST-N by hop (protects e5's token limit on dense high-level nodes — deep grab-bag drops first)."""
    node = str(node)
    min_hop = {}                                                   # ancestor id -> min hops from `node`
    # BFS upward over the multi-parent DAG (visited = cycle detection; depth = max_depth bound)
    frontier = [(node, 0)]; seen = {node}
    while frontier:
        cur, h = frontier.pop()
        if h >= max_depth:
            continue
        for par in parents_of.get(cur, []):
            if scope is not None and par not in scope:
                continue                                           # subtree scoping: don't leave the allowed set
            if par == cur:
                continue
            if par not in min_hop or h + 1 < min_hop[par]:
                min_hop[par] = h + 1
            if par not in seen:                                    # cycle detection
                seen.add(par); frontier.append((par, h + 1))
    # merged (ID-keyed → each ancestor once)
    if max_ancestors:                                              # keep NEAREST-N by hop (drop the deep grab-bag)
        keep = sorted(min_hop.items(), key=lambda kv: kv[1])[:max_ancestors]
        min_hop = dict(keep)
    anc = sorted(min_hop.items(), key=lambda kv: -kv[1])           # ordered root-ward first (largest hop = near root)
    return [(aid, title_of.get(aid, aid), hop) for aid, hop in anc]


def render_merged_list(node, parents_of, title_of, max_depth=15, scope=None, max_ancestors=None):
    """Text passage for e5: titles only, root-ward first, node title last. NO id line (merged/branching → no linear
    id anchor); ids stay internal merge keys. Depth shown by indent (informational; the model gets depth via PEs)."""
    anc = merged_ancestor_list(node, parents_of, title_of, max_depth, scope, max_ancestors)
    lines = [f"{'  ' * (len(anc) - i)}- {title}" for i, (_, title, _) in enumerate(anc)]
    lines.append(f"{'  ' * 0}- {title_of.get(str(node), str(node))}")   # the node itself, deepest
    return "\n".join(lines)


if __name__ == "__main__":                                        # demo: category graph (.tsv) or Pearltrees trees dir
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/merged_category_parent.tsv"
    parents_of, title_of = (load_category_graph(src) if src.endswith(".tsv")
                            else load_multiparent_dag(src))
    mp = [(c, ps) for c, ps in parents_of.items() if len(ps) > 1]
    print(f"[DAG] {len(parents_of)} nodes, {len(mp)} multi-parent ({100*len(mp)//max(1,len(parents_of))}%)")
    for cid, ps in mp[:3]:
        print(f"\n=== '{title_of.get(cid, cid)}' — {len(ps)} parents: {[title_of.get(p, p) for p in ps][:6]} ===")
        print("  single-path LINEAGE would pick ONE. Merged PATH list (root-ward first, max-depth 15):")
        print(render_merged_list(cid, parents_of, title_of))
