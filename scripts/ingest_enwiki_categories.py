#!/usr/bin/env python3
"""Correct enwiki category-graph ingest for the **2024+ MediaWiki schema**.

The old `fetch_wikipedia_categories.py` assumed the pre-2024 `categorylinks` layout where the parent
category name lived in a `cl_to` text column. As of the 2024 schema there is NO `cl_to`:

    categorylinks(cl_from, cl_sortkey, cl_timestamp, cl_sortkey_prefix, cl_type, cl_collation_id,
                  cl_target_id)

The parent is now `cl_target_id`, a bigint into the `linktarget(lt_id, lt_namespace, lt_title)` table;
and the child is `cl_from`, a page_id into `page(page_id, page_namespace, page_title, ...)`. The old
parser also mis-tokenised the binary `cl_sortkey` varbinary (its naive `\\(([^)]+)\\)` regex splits on
bytes inside the sortkey), which is why the broken DB ended up with sortkey junk (`*Astrophysics`,
`Karpov, Anatoly`) where category names should be.

This script does the correct 3-dump join, streaming, with a quote-aware tokenizer:
  child_title  = page[cl_from]        (namespace 14 = Category)
  parent_title = linktarget[cl_target_id]   (namespace 14)
emitting `child<TAB>parent` category edges (subcat links only).

    python3 scripts/ingest_enwiki_categories.py \
        --page      context/gemini/UnifyWeaver/data/enwiki/enwiki-latest-page.sql.gz \
        --linktarget context/gemini/UnifyWeaver/data/enwiki/enwiki-latest-linktarget.sql.gz \
        --categorylinks context/gemini/UnifyWeaver/data/enwiki/enwiki-latest-categorylinks.sql.gz \
        --out data/benchmark/enwiki_named/category_parent.tsv
    # --sample N  : only read the first N INSERT statements per dump (fast validation)
"""
import argparse
import gzip
import os
import sys

CATEGORY_NS = 14


def insert_tuples(path, sample=0):
    """Stream tuples from a mysqldump .sql.gz. Yields each VALUES row as a list of raw field strings
    (unquoted numbers as-is; quoted strings with surrounding quotes stripped and \\-escapes decoded).
    Quote-aware: commas/parens inside quoted (incl. binary) fields don't split."""
    n_ins = 0
    op = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    with op as f:
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            n_ins += 1
            if sample and n_ins > sample:
                break
            i = line.find("VALUES")
            if i < 0:
                continue
            s = line[i + 6:]
            yield from _parse_values(s)


def _parse_values(s):
    """Parse `(a,'b',...),(...);` → list-of-fields per tuple, respecting single-quoted strings."""
    i, n = 0, len(s)
    while i < n:
        if s[i] != "(":
            i += 1
            continue
        # parse one tuple
        i += 1
        fields, cur, in_str = [], [], False
        while i < n:
            ch = s[i]
            if in_str:
                if ch == "\\":                      # backslash escape — keep next char literally
                    cur.append(s[i + 1] if i + 1 < n else "")
                    i += 2
                    continue
                if ch == "'":
                    if i + 1 < n and s[i + 1] == "'":   # '' = literal quote
                        cur.append("'")
                        i += 2
                        continue
                    in_str = False
                    i += 1
                    continue
                cur.append(ch)
                i += 1
            else:
                if ch == "'":
                    in_str = True
                    i += 1
                elif ch == ",":
                    fields.append("".join(cur))
                    cur = []
                    i += 1
                elif ch == ")":
                    fields.append("".join(cur))
                    i += 1
                    break
                else:
                    cur.append(ch)
                    i += 1
        yield fields
        # skip to next "(" (past the comma/semicolon)
        while i < n and s[i] != "(":
            i += 1


def load_titles(path, label, sample):
    """page / linktarget dumps: cols (id, namespace, title, ...). Keep namespace-14 id→title."""
    d = {}
    for t in insert_tuples(path, sample):
        if len(t) < 3:
            continue
        try:
            ns = int(t[1])
        except ValueError:
            continue
        if ns != CATEGORY_NS:
            continue
        try:
            d[int(t[0])] = t[2]
        except ValueError:
            continue
    print(f"  {label}: {len(d):,} category-namespace titles", file=sys.stderr)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True)
    ap.add_argument("--linktarget", required=True)
    ap.add_argument("--categorylinks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0, help="first N INSERTs per dump (0 = full)")
    args = ap.parse_args()

    print("[1/3] page dump → page_id→title (ns14)…", file=sys.stderr)
    page = load_titles(args.page, "page", args.sample)
    print("[2/3] linktarget dump → lt_id→title (ns14)…", file=sys.stderr)
    lt = load_titles(args.linktarget, "linktarget", args.sample)

    print("[3/3] categorylinks (subcat) → child<TAB>parent…", file=sys.stderr)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_edges = n_seen = n_unres = 0
    with open(args.out, "w") as out:
        out.write("child\tparent\n")
        for t in insert_tuples(args.categorylinks, args.sample):
            # cl_from=0, cl_sortkey=1, cl_timestamp=2, cl_sortkey_prefix=3, cl_type=4, coll=5, cl_target_id=6
            if len(t) < 7 or t[4] != "subcat":
                continue
            n_seen += 1
            try:
                child = page.get(int(t[0]))
                parent = lt.get(int(t[6]))
            except ValueError:
                continue
            if child is None or parent is None:
                n_unres += 1
                continue
            out.write(f"{child}\t{parent}\n")
            n_edges += 1
    print(f"  subcat links seen {n_seen:,}; emitted {n_edges:,}; unresolved {n_unres:,}", file=sys.stderr)
    print(f"wrote {n_edges:,} category edges → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
