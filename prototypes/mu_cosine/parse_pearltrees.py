#!/usr/bin/env python3
"""Ref-based parser for HARVESTED Pearltrees data → a TYPED node/edge listing, the Pearltrees analogue of
parse_smmx.py (corpus=pearltrees, judge=human). It is "ref-based": it reads data the HARVESTER already
pulled (the cookie/API harvester + its skill live in .local — see SKILL_understand_pearltrees.md §Harvest);
it does NOT harvest. Default source is the harvested SQLite DB if present, else pass --db.

SECTION headers (contentType 7) retype the pearls that FOLLOW them (until the next section), just like
SimpleMind's structural containers — read in `left_index` order:
  * "Subcategories"                        → `subcategory`    (narrower category)
  * "Subtopics" / "More Subtopics"         → `element_of`     (element relations)
  * "Super Categories"                     → `super_category` (parent — super-cat, or a page's parent)
  * "Navigate Up"                          → `super_category` too, but REDUNDANT: it just re-points to the
       principal parent (tree containment) as a navigate-from-the-bottom convenience; dedup-safe.
  * "See Also"                             → `see_also`       (associative)
  * "Wiki / Encyclopedia type References"  → the wiki links bridge the whole TREE to enwiki
  * topical/junk headers (Algebra, Meta, To sort, …) → no retype; contentType default

Default relation (no section, or an unrecognised one), by `contentType`:
  * Collection (2) → `subtopic`   ;  PagePearl (1) → `element_of`  ;  Shortcut (5) → `assoc`
ANY PagePearl whose url is en.wikipedia.org ALSO emits a `bridge` to the enwiki node (category if
Category:…, else page) — Pearltrees is the bridge-RICH corpus. Root (4) is skipped.

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

    def enwiki_node(ew):                                  # register the enwiki node, return (key, type)
        if ew.startswith("Category:"):
            ek, etype = ew[len("Category:"):], "category"
        else:
            ek, etype = ew, "page"
        ek = ek.replace(" ", "_").strip("_")
        if ek:
            add(ek, etype, ek.replace("_", " "), "", ek)
        return ek, etype

    def section_mode(text):
        """A SECTION header retypes the pearls that FOLLOW it (until the next section), like a SimpleMind
        container. Recognised headers and the relation they impose:"""
        t = (text or "").lower()
        if "subcategor" in t:                       return "subcategory"      # → subcategory (narrower cat)
        if "subtopic" in t:                         return "element_of"       # → element relations
        if ("super" in t and "categor" in t) or "navigate up" in t:
            return "super_category"                 # → parent (super-cat, or a page's parent)
        if "see also" in t:                         return "see_also"         # → associative
        if "wiki" in t or "encyclopedia" in t or "reference" in t:
            return "reference"                      # → links that bridge the TREE to enwiki/encyclopedia
        return None                                 # topical/junk header (Algebra, Meta, …) ⇒ default

    # walk each tree's pearls IN ORDER (left_index) so section headers scope the pearls after them
    rows = con.execute("SELECT tree_id, content_type, title, url, content_tree_id, content_tree_title, "
                       "section_text, left_index FROM pearls ORDER BY tree_id, left_index")
    cur_tree, mode = None, None
    for tid, ctype, title, url, ct_id, ct_title, sec_text, _li in rows:
        if tid not in trees:
            continue
        if tid != cur_tree:
            cur_tree, mode = tid, None                    # reset section scope at each tree
        src = slug(trees[tid])
        add(src, "pearltrees_collection", trees[tid], str(tid))
        if ctype == 7:                                    # Section header → set the mode for what follows
            mode = section_mode(sec_text or title)
            continue
        if ctype == 4:                                    # Root pearl
            continue
        # target node + base relation by contentType
        ew = ""
        if ctype == 2 and ct_title:                       # Collection (child tree)
            dk, base = slug(ct_title), "subtopic"
            add(dk, "pearltrees_collection", ct_title, str(ct_id or ""))
        elif ctype == 1 and title:                        # PagePearl (a member page / url)
            dk, base = slug(title), "element_of"
            m = WIKI_URL.search(url or "")
            if m:
                ew = unquote(m.group(1)).split("#")[0]
            add(dk, "page", title, "", ew)
        elif ctype == 5 and ct_title:                     # Shortcut (alias)
            dk, base = slug(ct_title), "assoc"
            add(dk, "pearltrees_collection", ct_title, str(ct_id or ""))
        else:
            continue
        # relation = the section mode if it names one, else the contentType default
        rel = mode if mode in ("subcategory", "element_of", "super_category", "see_also") else base
        edges.append((src, dk, rel))
        if ctype == 1 and ew:                             # any wiki PagePearl bridges to its enwiki node
            ek, _ = enwiki_node(ew)
            if ek:
                edges.append((dk, ek, "bridge"))
                if mode == "reference":                   # a wiki/encyclopedia-REFERENCE section also
                    edges.append((src, ek, "bridge"))     # bridges the whole TREE to enwiki

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
            f.write("# src_key\tdst_key\trelation  "
                    "(subtopic|subcategory|element_of|super_category|see_also|bridge|assoc)\n")
            for a, b, rel in uniq:
                f.write(f"{a}\t{b}\t{rel}\n")
        print(f"  wrote {args.out_prefix}_nodes.tsv + {args.out_prefix}_edges.tsv")


if __name__ == "__main__":
    main()
