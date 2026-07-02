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


def merged_ancestor_list(node, parents_of, title_of, max_depth=15, scope=None):
    """Merged ancestor list for `node`: unique ancestors (ID-keyed) reached by walking up ALL parents.
    Cycle-detected (visited), max-depth bounded. Returns [(id, title, min_hop)] ordered root-ward first.
    `scope` (a set of allowed ids) restricts the walk to a subtree if given."""
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
    # merged (ID-keyed → each ancestor once), ordered root-ward first (largest hop = closest to root)
    anc = sorted(min_hop.items(), key=lambda kv: -kv[1])
    return [(aid, title_of.get(aid, aid), hop) for aid, hop in anc]


def render_merged_list(node, parents_of, title_of, max_depth=15, scope=None):
    """Text passage for e5: titles only, root-ward first, node title last. NO id line (merged/branching → no linear
    id anchor); ids stay internal merge keys. Depth shown by indent (informational; the model gets depth via PEs)."""
    anc = merged_ancestor_list(node, parents_of, title_of, max_depth, scope)
    lines = [f"{'  ' * (len(anc) - i)}- {title}" for i, (_, title, _) in enumerate(anc)]
    lines.append(f"{'  ' * 0}- {title_of.get(str(node), str(node))}")   # the node itself, deepest
    return "\n".join(lines)


if __name__ == "__main__":                                        # demo on a known multi-parent node
    import sys
    td = sys.argv[1] if len(sys.argv) > 1 else "../../.local/data/pearltrees_api/trees"
    parents_of, title_of = load_multiparent_dag(td)
    mp = [(c, ps) for c, ps in parents_of.items() if len(ps) > 1]
    print(f"[DAG] {len(parents_of)} nodes with parents, {len(mp)} multi-parent")
    for cid, ps in mp[:3]:
        print(f"\n=== '{title_of.get(cid, cid)}' (id {cid}) — {len(ps)} parents: {[title_of.get(p, p) for p in ps]} ===")
        print("  single-path LINEAGE would pick ONE of those. Merged PATH list:")
        print(render_merged_list(cid, parents_of, title_of))
