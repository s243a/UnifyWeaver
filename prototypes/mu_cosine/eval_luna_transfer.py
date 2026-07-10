#!/usr/bin/env python3
"""The §6.6 transfer test — the acceptance criterion the name-function migration exists for: a HELD-OUT
judge's zero-residual readout (pure name prior, cond = W·e5(card), r = 0) should beat the old scheme's
zero-init-row baseline (cond = 0, which the probe showed barely routes).

Held-out judge: gpt-5.6-luna (not in JUDGES; never trained). Labels: its validation run on the fresh
Behavior 250 (sigma_hop_fresh_scored_luna.tsv) — pairs the campaign sampler EXCLUDED from training, so
this is clean held-out data. Three conditionings against luna's own D/S labels:
  zero-row   — the old onboarding story (resid forced to −W·e so cond = 0 exactly)
  name-prior — luna at r=0 (what the migration buys: family-graded transfer, luna↔5.5 card cosine 0.971)
  5.5-row    — conditioning on gpt-5.5-low's own row (the borrow-the-family-row upper reference)

  python3 eval_luna_transfer.py --ckpt model_channel_heads_namecond.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fine_tune_channel_heads import mu_batch
from judge_cards import judge_card_e5
from mu_attention import CORPORA, JUDGES, OPS, MuAttention, Tokenizer
from run_product_kalman_realdata import DATASETS
from sigma_hop_confirmatory import FeatureGraphConfig, load_e5_cache_and_filter, load_feature_graph

ROOT = os.path.dirname(os.path.abspath(__file__))
LUNA_TSV = "/tmp/mu_data/sigma_hop_fresh_scored_luna.tsv"
DIRR = ["subcategory", "subtopic", "element_of", "super_category"]
SYMM = ["see_also", "assoc"]


def load_luna(path):
    pairs, D, S = [], [], []
    with open(path, encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < len(header):
                continue
            pairs.append((c[col["node"]], c[col["root"]]))
            D.append(max(float(c[col[f"mu[{r}]"]]) for r in DIRR))
            S.append(max(float(c[col[f"mu[{r}]"]]) for r in SYMM))
    return pairs, np.array(D), np.array(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_channel_heads_namecond.pt"))
    a = ap.parse_args()
    dev = "cpu"

    names = sorted(JUDGES, key=JUDGES.get) + ["gpt-5.6-luna"]
    E, _ = judge_card_e5(names, cache_path="/tmp/mu_data/judge_cards_e5_all.pt")
    LUNA = len(names) - 1

    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    sd = dict(ck["state"]); cfg = ck["cfg"]
    assert torch.allclose(sd["judge_name.name_e5"], E[:len(JUDGES)], atol=1e-6), "card table drifted vs ckpt"
    sd["judge_name.name_e5"] = E
    r = sd["judge_name.resid.weight"]
    sd["judge_name.resid.weight"] = torch.cat([r, torch.zeros(len(names) - r.shape[0], r.shape[1])], 0)
    y = sd["judge_emb.weight"]
    sd["judge_emb.weight"] = torch.cat([y, torch.zeros(len(names) - y.shape[0], y.shape[1])], 0)
    model = MuAttention(d_model=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
                        n_ops=sd["op_emb.weight"].shape[0], n_corpus=sd["corpus_emb.weight"].shape[0],
                        n_judge=len(names), n_nodetype=sd["nodetype_emb.weight"].shape[0],
                        judge_name_e5=E).to(dev)
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert not unexp, f"unexpected: {unexp}"
    model.eval()

    pairs, D, S = load_luna(LUNA_TSV)
    fcfg = DATASETS["fresh"]
    cache, idx, pairs, _, D, S = load_e5_cache_and_filter(pairs, np.zeros(len(pairs)), D, S, fcfg["e5_cache"])
    parents, _, deg, _ = load_feature_graph(FeatureGraphConfig(**fcfg["graph"]))
    tok = Tokenizer(cache["query"], cache["passage"], idx, parents, deg)
    print(f"luna-labelled held-out pairs: {len(pairs)}")

    def corr(judge_idx):
        out = {}
        for cname, op, tgt in [("D", "HIER", D), ("S", "SYM", S)]:
            items = [(x, y, OPS[op], CORPORA["enwiki"], judge_idx) for x, y in pairs]
            with torch.no_grad():
                mu = np.array(mu_batch(model, tok, items, dev).cpu())
            out[cname] = float(np.corrcoef(mu, tgt)[0, 1]) if mu.std() > 1e-9 else 0.0
        return out

    name_prior = corr(LUNA)                                   # r=0: pure name prior
    with torch.no_grad():                                     # zero-row baseline: force cond = 0 exactly
        saved = model.judge_name.resid.weight[LUNA].clone()
        model.judge_name.resid.weight[LUNA] = -model.judge_name.W(model.judge_name.name_e5[LUNA])
    zero_row = corr(LUNA)
    with torch.no_grad():
        model.judge_name.resid.weight[LUNA] = saved
    row55 = corr(JUDGES["gpt-5.5-low"])                       # borrow-the-family-row upper reference

    print(f"{'conditioning':22s} {'D corr':>8s} {'S corr':>8s}   (vs luna's own labels)")
    for label, c in [("zero-row (old onboard)", zero_row), ("name-prior (r=0)", name_prior),
                     ("gpt-5.5-low row", row55)]:
        print(f"{label:22s} {c['D']:+8.3f} {c['S']:+8.3f}")


if __name__ == "__main__":
    main()
