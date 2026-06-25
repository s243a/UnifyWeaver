#!/usr/bin/env python3
"""Bridge a SimpleMind mindmap to Wikipedia, via Pearltrees — the end-to-end "3 hops from the mindmap"
walk. Ref-based: it reads a parsed mindmap's nodes (parse_smmx.py → *_nodes.tsv), takes each node's
Pearltrees tree-id as a harvest SEED, and reads the harvested Pearltrees subtree for the enwiki links each
collection carries (its "Wiki / Encyclopedia type References"). Output: mindmap-concept → enwiki edges
(`bridge`), the cross-corpus join that connects the mindmap to the Wikipedia category/page graph.

Harvesting is PRIVATE (`.local`): if a seed isn't already harvested under --cache and the private harvester
exists, this drives it (`.local/tools/browser-automation/scripts/fetch_pearltrees_tree.py`, cookie session);
otherwise it skips (graceful on clones without `.local`). See SKILL_understand_pearltrees.md.

    python3 parse_smmx.py "System Theory.smmx" --out-prefix systh
    python3 bridge_seeds.py --nodes systh_nodes.tsv --depth 1 --out systh_bridges.tsv
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
HARVESTER = os.path.join(REPO, ".local", "tools", "browser-automation", "scripts", "fetch_pearltrees_tree.py")
WIKI = re.compile(r"en\.wikipedia\.org/wiki/(.+)$")


def harvested_path(cache, tid):
    return os.path.join(cache, f"pt_{tid}.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="*_nodes.tsv from parse_smmx.py")
    ap.add_argument("--cache", default=os.path.join(ROOT, ".pt_cache"), help="harvested-subtree cache dir")
    ap.add_argument("--depth", type=int, default=1, help="harvest depth per seed (1 = the collection only)")
    ap.add_argument("--account", default="s243a")
    ap.add_argument("--no-harvest", action="store_true", help="only use already-cached harvests")
    ap.add_argument("--out", default=os.path.join(ROOT, "mindmap_bridges.tsv"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    seeds = []                                    # (concept_slug, pearltrees_tree_id)
    for ln in open(args.nodes, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        c = ln.rstrip("\n").split("\t")
        if len(c) >= 5 and c[4].strip().isdigit():   # node_key, node_type, title, slug, pearltrees_id
            seeds.append((c[0], c[4].strip()))
    print(f"{len(seeds)} mindmap concepts carry a Pearltrees tree-id (seeds)")

    can_harvest = (not args.no_harvest) and os.path.exists(HARVESTER)
    if not can_harvest:
        print("  (harvesting disabled — using cached harvests only)" if args.no_harvest
              else "  (private harvester not found in .local — using cached harvests only)")

    rows, harvested, skipped = [], 0, 0
    for slug, tid in seeds:
        path = harvested_path(args.cache, tid)
        if not os.path.exists(path) and can_harvest:
            try:
                subprocess.run([sys.executable, HARVESTER, "--tree-id", tid, "--account", args.account,
                                "--depth", str(args.depth), "--out", path],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            except Exception:
                pass
        if not os.path.exists(path):
            skipped += 1
            continue
        harvested += 1
        seen = set()
        for hl in open(path, encoding="utf-8"):
            if hl.startswith("#"):
                continue
            f = hl.rstrip("\n").split("\t")
            if len(f) >= 5:
                m = WIKI.search(f[4])
                if m:
                    title = m.group(1).split("#")[0]
                    if title not in seen:
                        seen.add(title)
                        rows.append((slug, title))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# mindmap_concept\tenwiki_title\trelation=bridge  (concept → Wikipedia, via Pearltrees)\n")
        for slug, title in rows:
            f.write(f"{slug}\t{title}\tbridge\n")
    cats = sum(1 for _, t in rows if t.startswith("Category:"))
    print(f"harvested {harvested}/{len(seeds)} seeds ({skipped} unavailable); {len(rows)} bridges "
          f"({cats} to enwiki Categories) over {len({s for s, _ in rows})} concepts -> {args.out}")


if __name__ == "__main__":
    main()
