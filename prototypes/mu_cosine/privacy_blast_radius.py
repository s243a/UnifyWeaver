#!/usr/bin/env python3
"""B — PRIVACY-LABEL AUDIT SUPPORT: rank non-public trees by BLAST RADIUS (how many
otherwise-public descendants their label removes). The filter stays UNCONDITIONAL — this
supplies relabeling targets, not an escape hatch. Privacy propagates through containment,
so the 687 removed trees are not 687 independent decisions: if a few ancestors carry most
of the radius, relabeling those is the whole corpus-recovery job.

Blast radius of node X = |descendants of X (containment closure) that would be PUBLIC if X
were relabeled public|, computed as: nodes in X's closure that carry their own public claim
and are not blocked by a DIFFERENT private/unknown ancestor outside X's subtree.
Emits a review worklist (title, region, status, direct-vs-propagated, radius) for the owner.
PRIVACY: worklist rows name private trees the OWNER already owns; written to .local (never
committed) with only the top-N by radius echoed to stdout as titles the owner must review."""
import json, os, sys
from collections import defaultdict, deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TREES = "/home/s243a/Projects/UnifyWeaver/.local/data/pearltrees_api/trees"
OUT = "/home/s243a/Projects/UnifyWeaver/.local/privacy_relabel_worklist.json"
TOPN = 20

def run():
    from filing_privacy import build_pearltrees_privacy_index
    from privacy import propagate
    ix = build_pearltrees_privacy_index(TREES)
    status, reasons = dict(ix.status_by_id), dict(ix.reason_by_id)
    # containment edges rebuilt EXACTLY as filing_privacy does: parentTree edges,
    # collection-pearl (contentType 2) contentTree edges, and materialized path chains.
    from filing_privacy import canonical_tree_id, _infer_paths_jsonl
    from pathlib import Path
    children = defaultdict(set)
    for tid, payload in ix.tree_payloads.items():
        parent = payload.get("parentTree")
        if isinstance(parent, dict) and parent.get("id") is not None:
            pid = canonical_tree_id(parent["id"])
            if pid != tid:
                children[pid].add(tid)
        for p in payload.get("pearls", []) if isinstance(payload.get("pearls"), list) else []:
            if isinstance(p, dict) and str(p.get("contentType")) == "2":
                ch = p.get("contentTree")
                if isinstance(ch, dict) and ch.get("id") is not None:
                    cid = canonical_tree_id(ch["id"])
                    if cid != tid:
                        children[tid].add(cid)
    title_of = dict(ix.public_title_by_id)
    for tid, payload in ix.tree_payloads.items():
        if payload.get("title"):
            title_of.setdefault(tid, payload["title"])
        par = payload.get("parentTree")
        if isinstance(par, dict) and par.get("id") is not None and par.get("title"):
            title_of.setdefault(canonical_tree_id(par["id"]), par["title"])
        for p in payload.get("pearls", []) if isinstance(payload.get("pearls"), list) else []:
            if isinstance(p, dict) and str(p.get("contentType")) == "2":
                ch = p.get("contentTree")
                if isinstance(ch, dict) and ch.get("id") is not None:
                    t = ch.get("title") or p.get("title")
                    if t:
                        title_of.setdefault(canonical_tree_id(ch["id"]), t)
    pf = _infer_paths_jsonl(Path(TREES))
    if pf is not None:
        for line in pf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ids = [canonical_tree_id(x) if str(x).isdecimal() else str(x)
                   for x in row.get("path_ids", [])]
            if ids and row.get("title"):
                title_of.setdefault(ids[-1], row["title"])
            for a, b in zip(ids, ids[1:]):
                if a != b:
                    children[a].add(b)
    print(f"containment edges: {sum(len(v) for v in children.values())} "
          f"over {len(children)} parents", flush=True)
    parents = defaultdict(set)
    for a, kids in children.items():
        for k in kids:
            parents[k].add(a)
    nonpublic = [t for t, s in status.items() if s != "public"]
    print(f"trees observed {len(status)}, non-public {len(nonpublic)}", flush=True)

    def closure(root):
        seen, q = set(), deque([root])
        while q:
            x = q.popleft()
            for c in children.get(x, ()):
                if c not in seen:
                    seen.add(c); q.append(c)
        return seen

    def has_own_public_claim(t):
        return any(r.endswith(":public") for r in reasons.get(t, ()))

    rows = []
    for t in nonpublic:
        desc = closure(t)
        # descendants that would become public: own public claim, currently non-public,
        # and every OTHER blocking ancestor lies inside t's own subtree
        gain = 0
        for dnode in desc:
            if status.get(dnode) == "public" or not has_own_public_claim(dnode):
                continue
            blockers = {a for a in parents.get(dnode, ()) if status.get(a) != "public"}
            if all(b == t or b in desc for b in blockers):
                gain += 1
        direct = any(r.endswith(":private") or "private-title" in r for r in reasons.get(t, ()))
        # topmost = no OTHER non-public node is an ancestor: fixing this one is actionable
        # now; non-topmost entries are unblocked only after their ancestor is relabelled.
        blocked_by = {a for a in parents.get(t, ()) if status.get(a) != "public"}
        rows.append({
            "tree_id": t,
            "title": title_of.get(t, "(title not in snapshot)"),
            "topmost": not blocked_by,
            "status": status.get(t),
            "labelled": "direct" if direct else "propagated",
            "reasons": list(reasons.get(t, ()))[:4],
            "n_descendants": len(desc),
            "blast_radius": gain,
        })
    rows.sort(key=lambda r: (-r["blast_radius"], -r["n_descendants"]))
    total_radius = sum(r["blast_radius"] for r in rows)
    top = [r for r in rows if r["topmost"]][:TOPN]      # actionable-now review list
    summary = {
        "stamp": "owner-review-worklist",
        "n_nonpublic": len(nonpublic),
        "n_with_nonzero_radius": sum(1 for r in rows if r["blast_radius"] > 0),
        "total_recoverable_if_all_relabelled": total_radius,
        "top20_share_of_total": round(sum(r["blast_radius"] for r in top) /
                                      max(total_radius, 1), 3),
        "direct_vs_propagated": {
            "direct": sum(1 for r in rows if r["labelled"] == "direct"),
            "propagated": sum(1 for r in rows if r["labelled"] == "propagated")},
        "top20": top,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": summary, "all_rows": rows}, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o600)
    print(json.dumps(summary, indent=1)[:4000], flush=True)
    print(f"\nfull worklist -> {OUT} (0600, .local, never committed)", flush=True)

run()
