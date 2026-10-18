#!/usr/bin/env python3
"""B follow-up — PARTITION THE PRIVACY WORKLIST BY REMEDY (ruling: most top-radius nodes
are `unknown`/uncertified, not `private`, so the fix is harvest coverage, not an owner
privacy judgment). Splits the 155 topmost non-public nodes into:
  (a) OWNER DECISION      — genuinely direct `private` label (explicit private visibility
                            claim or private-title marker on the node itself)
  (b) HARVEST COVERAGE    — `unknown`/uncertified ancestry; sub-split by whether the missing
                            record looks FETCHABLE (numeric tree id absent from the harvest
                            but referenced by parent/path edges) vs non-numeric anchor
  (c) UNRESOLVABLE        — neither: no title, no numeric id, no usable reason
Only (a) needs the owner. PRIVACY: written to .local (0600, uncommitted); stdout prints
counts + the (a) list only."""
import json, os, sys
from collections import defaultdict, deque
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORKLIST = "/home/s243a/Projects/UnifyWeaver/.local/privacy_relabel_worklist.json"
TREES = "/home/s243a/Projects/UnifyWeaver/.local/data/pearltrees_api/trees"
OUT = "/home/s243a/Projects/UnifyWeaver/.local/privacy_remedy_partition.json"

def run():
    from filing_privacy import build_pearltrees_privacy_index
    d = json.load(open(WORKLIST))
    rows = [r for r in d["all_rows"] if r["topmost"]]
    ix = build_pearltrees_privacy_index(TREES)
    harvested = set(ix.tree_payloads)                    # ids with an actual tree JSON
    part = {"owner_decision": [], "harvest_fetchable": [], "harvest_anchor": [],
            "unresolvable": []}
    for r in rows:
        tid, reasons = r["tree_id"], set(r["reasons"])
        direct_private = any(x.endswith(":private") or "private-title" in x for x in reasons)
        if direct_private or r["status"] == "private" and r["labelled"] == "direct":
            part["owner_decision"].append(r); continue
        if r["status"] in ("unknown", "quarantined") or any(
                "uncertified" in x or "unknown-ancestor" in x for x in reasons):
            if str(tid).isdecimal() and tid not in harvested:
                part["harvest_fetchable"].append(r | {"remedy": "fetch tree JSON by id"})
            elif str(tid).isdecimal():
                part["harvest_anchor"].append(r | {
                    "remedy": "harvested but ancestry uncertified — needs parent/path record"})
            else:
                part["harvest_anchor"].append(r | {"remedy": "non-numeric anchor node"})
            continue
        part["unresolvable"].append(r)
    summary = {k: {"n": len(v), "radius_sum": sum(x["blast_radius"] for x in v)}
               for k, v in part.items()}
    summary["total_topmost"] = len(rows)
    json.dump({"summary": summary, "partition": part}, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o600)
    print(json.dumps(summary, indent=1), flush=True)
    print("\n(a) OWNER DECISION — direct private labels, by radius:")
    for r in sorted(part["owner_decision"], key=lambda x: -x["blast_radius"])[:15]:
        print(f"  radius {r['blast_radius']:>4}  desc {r['n_descendants']:>4}  "
              f"{r['title'][:46]}")
    fetch = sorted(part["harvest_fetchable"], key=lambda x: -x["blast_radius"])
    print(f"\n(b) HARVEST COVERAGE — {len(fetch)} fetchable ids, "
          f"radius sum {sum(r['blast_radius'] for r in fetch)}; "
          f"top ids: {[r['tree_id'] for r in fetch[:10]]}")
    print(f"\nfull partition -> {OUT} (0600)", flush=True)

run()
