#!/usr/bin/env python3
"""Harvest a Pearltrees tree (and, recursively, its child collections) via the getTreeAndPearls API, for
nodes important to our region that enwiki has no category for (e.g. "Circuit Theory"). Cookie-based: reuses
the saved session in .local/tools/browser-automation/pearltrees_cookies.txt (the API needs session cookies
even for public trees — see PEARLTREES_API_NOTES.md). Emits membership edges tagged for the pearltrees
corpus, distinguishing collection-membership (subtree) from page-membership (a PagePearl, which carries its
external URL — usually the Wikipedia anchor = the join key back into enwiki).

    python3 fetch_pearltrees_tree.py --tree-id 13580844 --account s243a --depth 2 --out pt_circuit.tsv
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import http.cookiejar

ROOT = os.path.dirname(os.path.abspath(__file__))
COOKIES = os.path.join(os.path.abspath(os.path.join(ROOT, "..", "..")),
                       ".local", "tools", "browser-automation", "pearltrees_cookies.txt")
API = "https://www.pearltrees.com/s/treeandpearlsapi/getTreeAndPearls"
UA = "Mozilla/5.0 (X11; Linux x86_64) UnifyWeaver-pt-harvest/0.1"
CT = {1: "page", 2: "collection", 4: "root", 5: "shortcut", 7: "section"}


def opener(cookie_path):
    cj = http.cookiejar.MozillaCookieJar(cookie_path)
    cj.load(ignore_discard=True, ignore_expires=True)
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", UA), ("X-Requested-With", "XMLHttpRequest")]
    return op


def get_tree(op, tree_id, retries=4):
    url = API + "?" + urllib.parse.urlencode({"treeId": tree_id})
    for a in range(retries):
        try:
            with op.open(url, timeout=30) as r:
                d = json.load(r)
            return d.get("tree", d)
        except Exception as e:
            if a < retries - 1:
                time.sleep(2 ** a)
                continue
            print(f"  ! tree {tree_id} failed: {e}")
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-id", type=int, required=True)
    ap.add_argument("--account", default="s243a")
    ap.add_argument("--depth", type=int, default=2, help="recurse into child collections this deep")
    ap.add_argument("--cookies", default=COOKIES)
    ap.add_argument("--delay", type=float, default=0.6)
    ap.add_argument("--out", default=os.path.join(ROOT, "pt_tree.tsv"))
    args = ap.parse_args()
    op = opener(args.cookies)

    rows, seen = [], set()
    # BFS over collections up to depth
    frontier = [(args.tree_id, args.depth)]
    while frontier:
        tid, d = frontier.pop(0)
        if tid in seen:
            continue
        seen.add(tid)
        tree = get_tree(op, tid)
        time.sleep(args.delay)
        if not tree:
            continue
        ptitle = (tree.get("title") or "").replace(" ", "_")
        npe = 0
        for p in tree.get("pearls", []):
            ct = CT.get(p.get("contentType"))
            if ct in (None, "root", "section"):
                continue
            title = (p.get("title") or "").replace(" ", "_")
            url = (p.get("url") or {}).get("url", "")
            ctree = p.get("contentTree") or {}
            child_id = ctree.get("id", "")
            rel = {"page": "element_of", "collection": "collection_of", "shortcut": "shortcut"}[ct]
            rows.append((ptitle, title, rel, ct, url, str(tid), str(child_id)))
            npe += 1
            if ct == "collection" and d > 1 and child_id:
                frontier.append((child_id, d - 1))
        print(f"  tree {tid} '{ptitle}': {npe} edges")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# pearltrees harvest (fetch_pearltrees_tree.py; corpus=pearltrees). "
                "parent=collection child=page|collection|shortcut. url=external anchor (often Wikipedia). "
                "cols: parent\tchild\trelation\tchild_type\turl\tparent_tree_id\tchild_tree_id\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    from collections import Counter
    c = Counter(r[3] for r in rows)
    print(f"wrote {len(rows)} edges over {len(seen)} trees -> {args.out}: {dict(c)}")


if __name__ == "__main__":
    main()
