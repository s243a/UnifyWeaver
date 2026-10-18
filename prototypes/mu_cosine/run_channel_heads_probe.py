#!/usr/bin/env python3
"""Lever B step 2 probe: does the EXISTING checkpoint already carry per-channel heads?

`DESIGN_amortized_fusion_heads.md` step 2 calls for per-channel heads (mu_graph, mu_LLM). The architecture
already has the machinery — JUDGES contains `graph` and `gpt-5.5-low` as learned provenance tokens, and the
tokenizer accepts (node, root, op, corpus, judge) — so the training history may have partially amortized the
channels already. Before training anything: MEASURE.

Probe: query model_prod.pt on both corpora under three conditionings —
  agnostic (3-tuple, provenance masked) | judge=graph | judge=gpt-5.5-low  (corpus=enwiki for both datasets)
and correlate each readout (mu_HIER max-dir, mu_SYM) against each channel's ground truth
(walk hit_prob d; LLM labels D, S). Success criterion for "step 2 already amortized": the graph-conditioned
readout predicts the walk better than the other conditionings do, and the LLM-conditioned readout predicts the
labels better — i.e., the judge tokens genuinely ROUTE, not just shift.

  python3 run_channel_heads_probe.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_relatedness import build_model
from mu_attention import CORPORA, JUDGES, OPS, Tokenizer
from run_product_kalman_realdata import DATASETS
from sigma_hop_confirmatory import FeatureGraphConfig, load_e5_cache_and_filter, load_feature_graph, load_scored_pairs
from emit_transitive_hops import hit_prob

ROOT = os.path.dirname(os.path.abspath(__file__))


def readouts(model, tokenizer, pairs, device, conds):
    """mu readouts per conditioning: {cond_name: {op_name: array}}. HIER is max over both directions."""
    out = {}
    for cname, cj in conds.items():
        per_op = {}
        for op in ("HIER", "SYM"):
            def mu(batch_pairs):
                if cj is None:
                    rows = [(x, y, OPS[op]) for x, y in batch_pairs]
                else:
                    rows = [(x, y, OPS[op], CORPORA["enwiki"], JUDGES[cj]) for x, y in batch_pairs]
                vals = []
                for i in range(0, len(rows), 512):
                    b = tokenizer.build(rows[i:i + 512], train=False)
                    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                    with torch.no_grad():
                        vals += model(**b).cpu().tolist()
                return np.array(vals)
            if op == "HIER":
                per_op["HIER"] = np.maximum(mu(pairs), mu([(y, x) for x, y in pairs]))
            else:
                per_op["SYM"] = mu(pairs)
        out[cname] = per_op
    return out


def run(name):
    cfg = DATASETS[name]
    pairs, hop, D, S = load_scored_pairs(cfg["score_in"], cfg["responses"], prefix="transitive_h")
    cache, idx, pairs, hop, D, S = load_e5_cache_and_filter(pairs, hop, D, S, cfg["e5_cache"])
    parents, _, deg, _ = load_feature_graph(FeatureGraphConfig(**cfg["graph"]))
    tokenizer = Tokenizer(cache["query"], cache["passage"], idx, parents, deg)
    model = build_model(os.path.join(ROOT, "model_prod.pt"), "cpu")
    d = np.array([hit_prob(parents, x, y) for x, y in pairs])
    targets = {"walk d": d, "LLM D": np.array(D), "LLM S": np.array(S)}

    # model_prod.pt carries 5 judge rows (haiku/graph/human/sonnet/opus) — it predates the gpt-5.5-low row,
    # so the checkpoint's trained LLM judge is HAIKU. Probe what exists.
    conds = {"agnostic": None, "judge=graph": "graph", "judge=llm": "haiku"}
    R = readouts(model, tokenizer, list(pairs), "cpu", conds)

    print(f"\n=== {name}: n={len(pairs)} — corr(readout, channel ground truth) ===")
    print(f"    {'conditioning':14s} {'readout':6s} | " + " ".join(f"{t:>8s}" for t in targets))
    for cname in conds:
        for op in ("HIER", "SYM"):
            r = R[cname][op]
            cs = [np.corrcoef(r, t)[0, 1] for t in targets.values()]
            print(f"    {cname:14s} {op:6s} | " + " ".join(f"{c:+8.3f}" for c in cs))
    # routing check: how different ARE the conditioned readouts?
    for op in ("HIER", "SYM"):
        dg = R["judge=graph"][op] - R["agnostic"][op]
        dl = R["judge=llm"][op] - R["agnostic"][op]
        print(f"    Δ{op}: |judge=graph − agnostic| mean {np.abs(dg).mean():.4f}; "
              f"|judge=llm − agnostic| mean {np.abs(dl).mean():.4f}; corr(Δg, Δl) "
              f"{np.corrcoef(dg, dl)[0, 1] if dg.std() > 0 and dl.std() > 0 else float('nan'):+.3f}")


def main():
    for n in ("exploratory", "fresh"):
        run(n)


if __name__ == "__main__":
    main()
