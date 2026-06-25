#!/usr/bin/env python3
"""Ref-based parser for HARVESTED Pearltrees data → a TYPED node/edge listing, the Pearltrees analogue of
parse_smmx.py (corpus=pearltrees, judge=human). It is "ref-based": it reads data the HARVESTER already
pulled (the cookie/API harvester + its skill live in .local — see SKILL_understand_pearltrees.md §Harvest);
it does NOT harvest. Default source is the harvested SQLite DB if present, else pass --db.

Relation semantics (a tree = a Collection of pearls; `contentType` of each pearl):
  * Collection (2)  → `subtopic`   (a child collection — narrower membership)
  * PagePearl  (1)  → `element_of` (a member page; node_type=page). If its url is en.wikipedia.org it ALSO
                      emits a `bridge` edge to the enwiki node (category if Category:…, else page) — the
                      cross-corpus join key. Pearltrees is the bridge-RICH corpus.
  * Shortcut   (5)  → `assoc`      (an alias/cross-reference to another collection)
  * Section (7), Root (4) → skipped (grouping / the tree's own root)

Node identity = the title-derived slug (matching the Pearltrees slugs SimpleMind nodes carry, so the two
corpora join). node_type ∈ pearltrees_collection | page | category.

    python3 parse_pearltrees.py --out-prefix pt        # uses the .local harvested DB if it exists
"""
import argparse
import os
import re
import sqlite3
from collections import Counter
from urllib.parse import unquote

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(os.path.abspath(os.path.join(ROOT, "..", "..")),
                          ".local", "data", "pearltrees_api", "pearltrees_api.db")
WIKI_URL = re.compile(r"en\.wikipedia\.org/wiki/(.+)$")


def slug(title):
    return re.sub(r"[^a-z0-9]+", "_", unquote(title or "").replace("&amp;", "&").lower()).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB, help="harvested pearltrees SQLite DB (default: .local)")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()
    if not os.path.exists(args.db):
        raise SystemExit(f"no harvested DB at {args.db} — harvest first (see the .local harvester skill) "
                         f"or pass --db")

    con = sqlite3.connect(args.db)
    trees = {r[0]: r[1] for r in con.execute("SELECT id, title FROM trees")}
    nodes, edges = {}, []

    def add(key, ntype, title, pid="", enwiki=""):
        nodes.setdefault(key, {"ntype": ntype, "title": title, "pid": pid, "enwiki": enwiki})

    for tid, ctype, title, url, ct_id, ct_title in con.execute(
            "SELECT tree_id, content_type, title, url, content_tree_id, content_tree_title FROM pearls"):
        if tid not in trees:
            continue
        src = slug(trees[tid])
        add(src, "pearltrees_collection", trees[tid], str(tid))
        if ctype == 2 and ct_title:                       # Collection → child collection (subtopic)
            dk = slug(ct_title)
            add(dk, "pearltrees_collection", ct_title, str(ct_id or ""))
            edges.append((src, dk, "subtopic"))
        elif ctype == 1 and title:                        # PagePearl → member page (element_of) + bridge
            dk = slug(title)
            ew = ""
            m = WIKI_URL.search(url or "")
            if m:
                ew = unquote(m.group(1)).split("#")[0]
            add(dk, "page", title, "", ew)
            edges.append((src, dk, "element_of"))
            if ew:                                        # cross-corpus bridge to the enwiki node
                if ew.startswith("Category:"):
                    ek, etype = ew[len("Category:"):], "category"
                else:
                    ek, etype = ew, "page"
                ek = ek.replace(" ", "_").strip("_")
                if ek:
                    add(ek, etype, ek.replace("_", " "), "", ek)
                    edges.append((dk, ek, "bridge"))
        elif ctype == 5 and ct_title:                     # Shortcut → alias/cross-reference (assoc)
            dk = slug(ct_title)
            add(dk, "pearltrees_collection", ct_title, str(ct_id or ""))
            edges.append((src, dk, "assoc"))

    seen, uniq = set(), []
    for a, b, rel in edges:
        if a and b and a != b and (a, b, rel) not in seen:
            seen.add((a, b, rel)); uniq.append((a, b, rel))

    print(f"{os.path.basename(args.db)}: {len(nodes)} nodes {dict(Counter(n['ntype'] for n in nodes.values()))}, "
          f"{len(uniq)} edges {dict(Counter(r for _, _, r in uniq))}")
    print(f"  bridges (pearltrees↔enwiki): {sum(1 for _, _, r in uniq if r == 'bridge')}")

    if args.out_prefix:
        with open(args.out_prefix + "_nodes.tsv", "w", encoding="utf-8") as f:
            f.write("# node_key\tnode_type\ttitle\tpearltrees_id\tenwiki_alias\n")
            for k, n in sorted(nodes.items()):
                f.write(f"{k}\t{n['ntype']}\t{n['title']}\t{n['pid']}\t{n['enwiki']}\n")
        with open(args.out_prefix + "_edges.tsv", "w", encoding="utf-8") as f:
            f.write("# src_key\tdst_key\trelation  (subtopic|element_of|bridge|assoc)\n")
            for a, b, rel in uniq:
                f.write(f"{a}\t{b}\t{rel}\n")
        print(f"  wrote {args.out_prefix}_nodes.tsv + {args.out_prefix}_edges.tsv")


if __name__ == "__main__":
    main()
