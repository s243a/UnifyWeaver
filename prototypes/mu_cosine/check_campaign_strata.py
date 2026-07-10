#!/usr/bin/env python3
"""Verify the campaign's design hypothesis: S VARIES on lateral strata (siblings/cousins), unlike the
transitive-ancestor pairs that all prior S labels came from (B1 failure analysis).

Reports mean/sd of the D and S labels per stratum. Success = sd(S) on sib/cous clearly exceeds sd(S) on
transitive strata; random pairs give a low-D low-S negative mass.

  python3 check_campaign_strata.py --scored /tmp/mu_data/campaign_scored.tsv
"""
import argparse
from collections import defaultdict

import numpy as np

DIR = ["subcategory", "subtopic", "element_of", "super_category"]
SYM = ["see_also", "assoc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="/tmp/mu_data/campaign_scored.tsv")
    a = ap.parse_args()
    by = defaultdict(lambda: ([], []))
    with open(a.scored, encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < len(header):
                continue
            tag = c[col["neighborhood"]]
            D = max(float(c[col[f"mu[{r}]"]]) for r in DIR)
            S = max(float(c[col[f"mu[{r}]"]]) for r in SYM)
            by[tag][0].append(D); by[tag][1].append(S)

    print(f"{'stratum':16s} {'n':>5s} {'D mean±sd':>14s} {'S mean±sd':>14s}")
    order = sorted(by, key=lambda t: (not t.startswith("campaign_h"), t))
    for tag in order:
        D, S = np.array(by[tag][0]), np.array(by[tag][1])
        print(f"{tag:16s} {len(D):>5d} {D.mean():>7.3f}±{D.std():.3f} {S.mean():>7.3f}±{S.std():.3f}")
    trans = np.concatenate([by[t][1] for t in by if t.startswith("campaign_h")])
    lat = np.concatenate([by[t][1] for t in by if t in ("campaign_sib", "campaign_cous")])
    print(f"\nHYPOTHESIS CHECK — sd(S): transitive {np.std(trans):.3f} vs lateral (sib+cous) {np.std(lat):.3f}"
          f"  {'← CONFIRMED (lateral varies more)' if np.std(lat) > np.std(trans) else '← NOT confirmed'}")


if __name__ == "__main__":
    main()
