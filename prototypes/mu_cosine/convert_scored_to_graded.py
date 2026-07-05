#!/usr/bin/env python3
"""Convert a §14 scored TSV (score_with_codex.py / score_inferred_tail.ingest output) into the graded
`_pairs.tsv` schema mu_posterior.py consumes. The classification TARGET is the LLM's argmax relation over the
applies distribution — an independent label (not from 1/d), so the JointPosterior + 1/d ablation is a fair test.

  <out>: node  root  mu  op  relation  node_type  root_type  corpus  judge  conf

  python3 convert_scored_to_graded.py --scored /tmp/mu_data/wiki_rel_scored.tsv --out /tmp/mu_data/wiki_rel_graded_pairs.tsv
"""
import argparse
from collections import Counter

NAMED = ["element_of", "subcategory", "subtopic", "super_category", "see_also", "assoc", "unknown", "none"]
OP_OF = {"element_of": "ELEM", "subcategory": "HIER", "subtopic": "HIER", "super_category": "HIER",
         "see_also": "SYM", "assoc": "SYM", "unknown": "SYM", "none": "SYM"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="enwiki")
    ap.add_argument("--judge", default="gpt-5.5-low")
    a = ap.parse_args()

    hdr = open(a.scored, encoding="utf-8").readline().lstrip("#").strip().split("\t")
    pcol = {c: hdr.index(f"P[{c}]") for c in NAMED}
    fi = hdr.index("E_mu_fwd")
    rows, tally = [], Counter()
    for ln in open(a.scored, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        c = ln.rstrip("\n").split("\t")
        if len(c) <= fi:
            continue
        node, root = c[0], c[1]
        P = [(float(c[pcol[r]]), r) for r in NAMED]
        _, rel = max(P)                                    # LLM argmax relation = the label
        mu = float(c[fi])
        rows.append((node, root, mu, OP_OF[rel], rel))
        tally[rel] += 1

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# node\troot\tmu\top\trelation\tnode_type\troot_type\tcorpus\tjudge\tconf\n")
        for node, root, mu, op, rel in rows:
            f.write(f"{node}\t{root}\t{mu:.3f}\t{op}\t{rel}\tcategory\tcategory\t{a.corpus}\t{a.judge}\t1.0\n")
    print(f"wrote {len(rows)} graded pairs → {a.out}")
    print("relation distribution (the classifier's classes):", dict(tally))


if __name__ == "__main__":
    main()
