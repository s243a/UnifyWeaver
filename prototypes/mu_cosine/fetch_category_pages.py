#!/usr/bin/env python3
"""Pull MEMBER PAGES (not subcategories) of named Wikipedia categories via the MediaWiki API, for
categories that are important to our target region but category-thin (few/no subcategories) — so the
page-level membership is the only rich signal. Emits page-membership edges distinct from the category
subcategory edges: a row is `page<TAB>category<TAB>element_of`, so downstream pairing/tagging can route
them to the element-of operator + node-type=page (see DESIGN_calibrated_judges.md §7).

Uses list=categorymembers with cmtype split so pages and subcats never get confused. Polite: descriptive
User-Agent, continuation handled, small delay between requests.

    python3 fetch_category_pages.py --cat Bifurcation_theory --cat Nonlinear_systems \
        --out page_members.tsv
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

API = "https://en.wikipedia.org/w/api.php"
UA = "UnifyWeaver-mu-cosine/0.1 (research prototype; github.com/s243a/UnifyWeaver)"
ROOT = os.path.dirname(os.path.abspath(__file__))


def api_get(params, retries=6):
    params = {**params, "format": "json", "maxlag": "5"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries - 1:
                wait = int(e.headers.get("Retry-After", 0)) or (2 ** attempt)
                print(f"    [{e.code}] backoff {wait}s")
                time.sleep(wait)
                continue
            raise


def members(cat, cmtype, delay=1.0):
    """Yield member titles (underscored) of Category:cat with the given cmtype (page|subcat)."""
    cont = {}
    while True:
        data = api_get({"action": "query", "list": "categorymembers",
                        "cmtitle": f"Category:{cat}", "cmtype": cmtype,
                        "cmlimit": "500", **cont})
        for m in data.get("query", {}).get("categorymembers", []):
            title = m["title"]
            if cmtype == "subcat" and title.startswith("Category:"):
                title = title[len("Category:"):]
            yield title.replace(" ", "_")
        if "continue" in data:
            cont = data["continue"]
            time.sleep(delay)
        else:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", action="append", default=[], required=True)
    ap.add_argument("--recurse-subcats", type=int, default=0, help="AUGMENT: also pull pages from "
                    "subcategories down to this depth (attributed to the parent --cat), to push a thin "
                    "category over a sampling threshold (select_svd_coverage.py --min-pages)")
    ap.add_argument("--out", default=os.path.join(ROOT, "page_members.tsv"))
    args = ap.parse_args()

    rows, summary = [], []
    for cat in args.cat:
        if args.recurse_subcats > 0:
            # BFS the subcategory subtree; pool member pages of cat + all subcats, attributed to `cat`
            seen_sub, frontier = {cat}, [cat]
            pages = set(members(cat, "page"))
            for _ in range(args.recurse_subcats):
                nxt = []
                for sc in frontier:
                    for sub in members(sc, "subcat"):
                        if sub not in seen_sub:
                            seen_sub.add(sub); nxt.append(sub)
                            pages.update(members(sub, "page"))
                            time.sleep(0.3)
                frontier = nxt
            pages = sorted(pages)
            nsub = len(seen_sub) - 1
        else:
            pages = sorted(set(members(cat, "page")))
            nsub = sum(1 for _ in members(cat, "subcat"))
        for p in pages:
            rows.append((p, cat, "element_of"))
        summary.append((cat, len(pages), nsub))
        print(f"  {cat:26} pages={len(pages):4d}  subcats={nsub}"
              + (f"  (incl. subtree depth {args.recurse_subcats})" if args.recurse_subcats else ""))
        time.sleep(1.0)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Wikipedia category MEMBER PAGES via MediaWiki API (cmtype=page). "
                "relation=element_of (page in category); node-type page->category. "
                "cols: page\tcategory\trelation\n")
        for p, c, rel in rows:
            f.write(f"{p}\t{c}\t{rel}\n")
    print(f"wrote {len(rows)} page-membership edges over {len(args.cat)} categories -> {args.out}")
    print("thin-check (pages >> subcats ⇒ page data is the rich signal):")
    for c, npg, nsub in summary:
        print(f"  {c:26} {npg} pages / {nsub} subcats")


if __name__ == "__main__":
    main()
