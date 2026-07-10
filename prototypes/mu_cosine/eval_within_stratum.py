#!/usr/bin/env python3
"""Within/between-stratum decomposition of the channel-head correlations on the campaign held-out pairs
(REPORT_channel_campaign §8) — the honest eval for B2: pooled correlations conflate stratum ORDERING
(partly inferable from graph position) with within-stratum SEMANTIC discrimination.

Per (corpus, channel): pooled r; between-strata r (correlation of stratum means, strata = transitive/
sib/cous/rand, campaign_h* pooled); within-stratum r (both series demeaned per stratum, then pooled);
per-stratum r. Comparisons between checkpoints should use THIS script for both sides so method details
cancel.

  python3 eval_within_stratum.py --ckpt model_channel_heads_campaign.pt model_channel_heads_namecond.pt
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fine_tune_channel_heads import load_campaign_datasets, load_expanded, mu_batch
from mu_attention import CORPORA, JUDGES, OPS

ROOT = os.path.dirname(os.path.abspath(__file__))


def group(tag):
    return "trans" if tag.startswith("campaign_h") else tag.replace("campaign_", "")


def decompose(mu, tgt, groups):
    by = defaultdict(list)
    for m, t, g in zip(mu, tgt, groups):
        by[g].append((m, t))
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else float("nan")
    pooled = r(mu, tgt)
    gm = {g: (np.mean([m for m, _ in v]), np.mean([t for _, t in v])) for g, v in by.items()}
    between = r([gm[g][0] for g in sorted(by)], [gm[g][1] for g in sorted(by)])
    dm = np.array([m - gm[g][0] for m, g in zip(mu, groups)])
    dt = np.array([t - gm[g][1] for t, g in zip(tgt, groups)])
    within = r(dm, dt)
    per = {g: r([m for m, _ in v], [t for _, t in v]) for g, v in sorted(by.items())}
    return pooled, between, within, per


def eval_ckpt(ckpt, dss, dev="cpu"):
    model, _ = load_expanded(ckpt, dev=dev)
    model.eval()
    print(f"\n=== {os.path.basename(ckpt)} ({'name-cond' if model.judge_name is not None else 'indexed'}) ===")
    for name, ds in dss.items():
        he = ds["he"]
        groups = [group(ds["tags"][i]) for i in he]
        for cname, judge, op, tgt in [("llm-D", "gpt-5.5-low", "HIER", ds["D"]),
                                      ("llm-S", "gpt-5.5-low", "SYM", ds["S"])]:
            items = [(ds["pairs"][i][0], ds["pairs"][i][1], OPS[op], CORPORA["enwiki"], JUDGES[judge])
                     for i in he]
            with torch.no_grad():
                mu = np.array(mu_batch(model, ds["tok"], items, dev).cpu())
            pooled, between, within, per = decompose(mu, tgt[he], groups)
            per_s = " ".join(f"{g} {v:+.3f}" for g, v in per.items())
            print(f"{name:22s} {cname}: pooled {pooled:+.3f}  between {between:+.3f}  "
                  f"WITHIN {within:+.3f}   [{per_s}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=[os.path.join(ROOT, "model_channel_heads_campaign.pt")])
    a = ap.parse_args()
    dss = load_campaign_datasets()
    for c in a.ckpt:
        eval_ckpt(c, dss)


if __name__ == "__main__":
    main()
