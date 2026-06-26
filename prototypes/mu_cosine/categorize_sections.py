#!/usr/bin/env python3
"""SECOND STEP (separate from the harvest): read the RAW section table that the harvester captured
(`#SECTION` lines in the .pt_cache harvests) and emit the categorisation — `section_pos_id → relation,
category, method, confidence, text`. This is the "feed the text output into a db and map categories to
columns" step: it is re-runnable over cached harvests (NO re-harvest) and the `--method` is upgradeable
(exact_phrase → fuzzy → llm_template). The method+confidence are PROVENANCE for each relation.

    python3 categorize_sections.py --cache .pt_cache --out section_categories.tsv
"""
import argparse
import glob
import os
from collections import Counter

from pt_sections import categorize, CATEGORY_RELATION

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, ".pt_cache"), help="dir of raw pt_*.tsv harvests")
    ap.add_argument("--method", default="exact_phrase", help="exact_phrase (fuzzy / llm_template later)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seen, rows = set(), []
    for path in sorted(glob.glob(os.path.join(args.cache, "pt_*.tsv"))):
        for ln in open(path, encoding="utf-8"):
            if not ln.startswith("#SECTION\t"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3 or p[1] in seen:
                continue
            seen.add(p[1])
            cat, method, conf = categorize(p[2], args.method)
            rows.append((p[1], CATEGORY_RELATION.get(cat) or "", cat or "", method, f"{conf:.2f}", p[2]))

    matched = [r for r in rows if r[1]]
    print(f"{len(rows)} sections → {len(matched)} categorised ({args.method})")
    print(f"  by relation: {dict(Counter(r[1] for r in matched))}")
    uncategorised = [r[5] for r in rows if not r[1]][:12]
    if uncategorised:
        print(f"  no rule fired (→ structural fallback downstream): {uncategorised}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# section_pos_id\trelation\tcategory\tmethod\tconfidence\ttext\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
