#!/usr/bin/env python3
"""Pair-matched judge comparison on the stratified campaign — the §7 luna verdict, re-measured where it
matters. §7 rejected luna on the fresh 250, but those are ALL-TRANSITIVE pairs: S barely varies there, so
luna's S weakness (0.35-0.44 vs the 0.766 ceiling) was measured on the S-starved stratum. The campaign's
sib/cous rows are where S varies (the whole point of the stratified sample) — this is the fair test.

Per stratum × channel: corr(luna, 5.5), MAE, bias (luna − 5.5), and both sds. Bias should reproduce the
measured tilt (+D/−S); the decision-relevant number is S corr on sib/cous.

  python3 compare_judges_campaign.py --a /tmp/mu_data/campaign_scored.tsv --b /tmp/mu_data/campaign_scored_luna.tsv
"""
import argparse
from collections import defaultdict

import numpy as np

DIRR = ["subcategory", "subtopic", "element_of", "super_category"]
SYMM = ["see_also", "assoc"]


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < len(header):
                continue
            D = max(float(c[col[f"mu[{r}]"]]) for r in DIRR)
            S = max(float(c[col[f"mu[{r}]"]]) for r in SYMM)
            tag = c[col["neighborhood"]]
            out[(c[col["node"]], c[col["root"]])] = (tag, D, S)
    return out


def group(tag):
    return "trans" if tag.startswith("campaign_h") else tag.replace("campaign_", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="/tmp/mu_data/campaign_scored.tsv", help="reference judge (gpt-5.5-low)")
    ap.add_argument("--b", default="/tmp/mu_data/campaign_scored_luna.tsv", help="candidate judge (luna)")
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    common = sorted(set(A) & set(B))
    print(f"pair-matched: {len(common)} (A {len(A)}, B {len(B)})")

    by = defaultdict(list)
    for p in common:
        tag, Da, Sa = A[p]
        _, Db, Sb = B[p]
        by[group(tag)].append((Da, Sa, Db, Sb))
        by["ALL"].append((Da, Sa, Db, Sb))

    r = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else float("nan")
    print(f"\n{'stratum':8s} {'n':>5s} | {'D corr':>7s} {'D MAE':>6s} {'D bias':>7s} {'sd A/B':>11s} | "
          f"{'S corr':>7s} {'S MAE':>6s} {'S bias':>7s} {'sd A/B':>11s}")
    for g in ["trans", "sib", "cous", "rand", "ALL"]:
        if g not in by:
            continue
        v = np.array(by[g])
        Da, Sa, Db, Sb = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        print(f"{g:8s} {len(v):>5d} | {r(Da, Db):+7.3f} {np.abs(Db - Da).mean():6.3f} "
              f"{(Db - Da).mean():+7.3f} {Da.std():5.3f}/{Db.std():5.3f} | "
              f"{r(Sa, Sb):+7.3f} {np.abs(Sb - Sa).mean():6.3f} "
              f"{(Sb - Sa).mean():+7.3f} {Sa.std():5.3f}/{Sb.std():5.3f}")
    print("\n(reference ceilings from the fresh-250 self-consistency: D 0.954, S 0.766 — same-judge repeat)")


if __name__ == "__main__":
    main()
