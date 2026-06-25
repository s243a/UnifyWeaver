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
corpora join). node_type ∈ pearltrees_collection | page | category. Each pearltrees node also carries its
`account` (provenance: e.g. `s243a` vs the `s243a_groups`/teams account) from `trees.account`; enwiki-side
bridge nodes carry none. See DESIGN_provenance_and_representation.md (account = maskable provenance token;
groups/teams = a transform of e5, or a "Team <name> <id>" e5-text prefix).

PRIVACY (scrub-everywhere): a private tree (visibility≠public) or a collection/pearl titled "private" is
dropped at parse time WITH its whole subtree (privacy propagates down tree-containment) — private data never
reaches the public dataset. See DESIGN_provenance_and_representation.md §Privacy.

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
# PRIVACY (scrub-everywhere — DESIGN_provenance_and_representation.md §Privacy): private data must NEVER reach
# the public dataset, so it is dropped at PARSE time, inherited down the subtree, with no include-private
# escape hatch. Two markers: (1) the title contains the word "private" (the user's "*private*" marker node /
# a root named "… private …"); (2) the Pearltrees `visibility` is set to anything other than public (0).
# We err toward dropping (a topical "Private equity" would be scrubbed too) and LOG every scrub — a false
# positive only loses public data, a false negative would leak private data.
PRIVATE_RE = re.compile(r"(?i)\bprivate\b")


def is_private_title(t):
    return bool(t) and bool(PRIVATE_RE.search(t))


def vis_private(v):                                       # Pearltrees visibility: 0 = public; else restricted
    return v is not None and str(v).strip() not in ("", "0")


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
    trees, tree_acct = {}, {}                              # tree_id → title ; tree_id → account
    priv_trees = set()                                     # tree_ids that are private (seed + inherited)
    for tid_, title_, acct_, vis_ in con.execute("SELECT id, title, account, visibility FROM trees"):
        trees[tid_] = title_
        tree_acct[tid_] = acct_ or ""
        if vis_private(vis_) or is_private_title(title_):  # seed: this tree itself is private
            priv_trees.add(tid_)
    nodes, edges = {}, []
    scrub = Counter()                                      # what privacy dropped, by kind (logged, not silent)

    def add(key, ntype, title, pid="", enwiki="", account=""):
        n = nodes.get(key)
        if n is None:
            nodes[key] = {"ntype": ntype, "title": title, "pid": pid, "enwiki": enwiki, "account": account}
        elif account and not n.get("account"):            # backfill account once we learn it
            n["account"] = account

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

    rows = list(con.execute("SELECT tree_id, content_type, title, url, content_tree_id, content_tree_title, "
                            "section_text, left_index FROM pearls ORDER BY tree_id, left_index"))

    # PRE-PASS (privacy): a collection NAMED private seeds privacy on its child tree; build tree containment
    # (parent → child collection) and propagate privacy DOWN it so a private collection takes its whole
    # harvested subtree with it (children inherit, exactly like the RDF `private` field).
    contain = {}
    for tid, ctype, title, _u, ct_id, ct_title, _s, _li in rows:
        if tid not in trees or ctype != 2 or not ct_id:
            continue
        contain.setdefault(tid, set()).add(ct_id)
        if is_private_title(ct_title):
            priv_trees.add(ct_id)
    frontier = list(priv_trees)                           # BFS the private mark down the containment graph
    while frontier:
        t = frontier.pop()
        for c in contain.get(t, ()):
            if c not in priv_trees:
                priv_trees.add(c); frontier.append(c)

    # walk each tree's pearls IN ORDER (left_index) so section headers scope the pearls after them
    cur_tree, mode = None, None
    for tid, ctype, title, url, ct_id, ct_title, sec_text, _li in rows:
        if tid not in trees:
            continue
        if tid in priv_trees:                             # PRIVATE tree → scrub it and everything under it
            scrub["tree"] += 1
            continue
        if tid != cur_tree:
            cur_tree, mode = tid, None                    # reset section scope at each tree
        acct = tree_acct.get(tid, "")                     # this tree's Pearltrees account (provenance)
        src = slug(trees[tid])
        add(src, "pearltrees_collection", trees[tid], str(tid), account=acct)
        if ctype == 7:                                    # Section header → set the mode for what follows
            mode = section_mode(sec_text or title)
            continue
        if ctype == 4:                                    # Root pearl
            continue
        # target node + base relation by contentType — but SCRUB any private target first
        ew = ""
        if ctype == 2 and ct_title:                       # Collection (child tree)
            if ct_id in priv_trees or is_private_title(ct_title):
                scrub["collection"] += 1; continue
            dk, base = slug(ct_title), "subtopic"
            add(dk, "pearltrees_collection", ct_title, str(ct_id or ""), account=tree_acct.get(ct_id) or acct)
        elif ctype == 1 and title:                        # PagePearl (a member page / url)
            if is_private_title(title):
                scrub["page"] += 1; continue
            dk, base = slug(title), "element_of"
            m = WIKI_URL.search(url or "")
            if m:
                ew = unquote(m.group(1)).split("#")[0]
            add(dk, "page", title, "", ew, account=acct)
        elif ctype == 5 and ct_title:                     # Shortcut (alias)
            if ct_id in priv_trees or is_private_title(ct_title):
                scrub["shortcut"] += 1; continue
            dk, base = slug(ct_title), "assoc"
            add(dk, "pearltrees_collection", ct_title, str(ct_id or ""), account=tree_acct.get(ct_id) or acct)
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
    print(f"  accounts: {dict(Counter(n['account'] for n in nodes.values() if n['account']))}")
    print(f"  PRIVACY scrubbed (dropped, never emitted): {dict(scrub) or 'none'}"
          f"  [{len(priv_trees)} private tree-ids]")

    if args.out_prefix:
        with open(args.out_prefix + "_nodes.tsv", "w", encoding="utf-8") as f:
            f.write("# node_key\tnode_type\ttitle\tpearltrees_id\tenwiki_alias\taccount\n")
            for k, n in sorted(nodes.items()):
                f.write(f"{k}\t{n['ntype']}\t{n['title']}\t{n['pid']}\t{n['enwiki']}\t{n['account']}\n")
        with open(args.out_prefix + "_edges.tsv", "w", encoding="utf-8") as f:
            f.write("# src_key\tdst_key\trelation  "
                    "(subtopic|subcategory|element_of|super_category|see_also|bridge|assoc)\n")
            for a, b, rel in uniq:
                f.write(f"{a}\t{b}\t{rel}\n")
        print(f"  wrote {args.out_prefix}_nodes.tsv + {args.out_prefix}_edges.tsv")


if __name__ == "__main__":
    main()
