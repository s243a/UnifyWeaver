#!/usr/bin/env python3
"""Generalize the behavior-preserving name migration to OPS and CORPORA — the §6.7 'one mechanism' step.

Same procedure as migrate_judge_names.py, per table: ridge-lstsq W on the informative rows of the indexed
embedding, r = old − W·e (exact reproduction at init), then the model conditions through
NameFunctionCond(cards). Three tables:

  judges  → judge_name.*   (already migrated for the working checkpoints; redone here idempotently)
  ops     → op_name.*      (the operator TOKEN embedding; the per-operator READOUT stays indexed — scope)
  corpora → corpus_name.*

The op table has an extra consumer: the BLENDED-operator path (`op_weights @ op_emb.weight`,
DESIGN_inferred_operator_superposition) — NameFunctionCond.table() supplies the full matrix, and the
verification below exercises BOTH the indexed and blended paths.

  python3 migrate_name_tables.py --ckpt model_channel_heads_namecond_r0.pt --out model_namecond_full.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_cards import corpus_card_e5, judge_card_e5, op_card_e5
from migrate_judge_names import fit_name_function, synthetic_prov_batch
from mu_attention import CORPORA, JUDGES, OPS, MuAttention

ROOT = os.path.dirname(os.path.abspath(__file__))
TABLES = {  # name → (emb key in state dict, cond prefix, card builder, index dict)
    "judges": ("judge_emb.weight", "judge_name", judge_card_e5, JUDGES),
    "ops": ("op_emb.weight", "op_name", op_card_e5, OPS),
    "corpora": ("corpus_emb.weight", "corpus_name", corpus_card_e5, CORPORA),
}


def build(sd, cfg, name_tables, dev="cpu"):
    sz = lambda k, d: sd[k].shape[0] if k in sd else d
    m = MuAttention(d_model=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
                    n_ops=sz("op_emb.weight", len(OPS)), n_corpus=sz("corpus_emb.weight", len(CORPORA)),
                    n_judge=sz("judge_emb.weight", len(JUDGES)),
                    n_nodetype=sz("nodetype_emb.weight", 4), **name_tables).to(dev)
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not unexp, f"unexpected keys: {unexp}"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_channel_heads_namecond_r0.pt"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--tables", default="ops,corpora", help="comma list of judges,ops,corpora")
    a = ap.parse_args()
    out = a.out or a.ckpt.rsplit(".pt", 1)[0] + "_full.pt"

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = dict(ck["state"]); cfg = ck.get("cfg", {"d_model": 384, "heads": 4, "layers": 3})
    old = build(sd, cfg, {"judge_name_e5": sd.get("judge_name.name_e5")})

    name_tables = {"judge_name_e5": sd.get("judge_name.name_e5")}
    for t in a.tables.split(","):
        emb_key, prefix, card_fn, idx = TABLES[t]
        Y = sd[emb_key]
        E, names = card_fn()
        assert E.shape[0] >= Y.shape[0], f"{t}: {E.shape[0]} cards < {Y.shape[0]} rows"
        E = E[:Y.shape[0]]
        W, resid, keepm = fit_name_function(Y, E, ridge=a.ridge)
        rec = (E @ W.t() + resid - Y).abs().max().item()
        print(f"{t}: fit on {int(keepm.sum())}/{len(Y)} rows "
              f"({[n for n, k in zip(names, keepm) if k]}); reconstruction {rec:.2e}")
        sd[f"{prefix}.name_e5"], sd[f"{prefix}.W.weight"], sd[f"{prefix}.resid.weight"] = E, W, resid
        name_tables[f"{prefix.replace('_name', '')}_name_e5"] = E

    new = build(sd, cfg, name_tables)
    old.eval(); new.eval()
    # forward equivalence: (1) provenance batch over all judges; (2) op/corpus sweep; (3) BLENDED op path
    b = synthetic_prov_batch(cfg["d_model"], sd["judge_emb.weight"].shape[0])
    n_ops, n_corp = sd["op_emb.weight"].shape[0], sd["corpus_emb.weight"].shape[0]
    for i in range(b["op_pos"].shape[0]):
        b["op_pos"][i, 0] = i % n_ops; b["op_of"][i] = i % n_ops
        b["corpus_of"][i, 3] = i % n_corp
    with torch.no_grad():
        e1 = (old(**b) - new(**b)).abs().max().item()
        w = torch.rand(b["op_of"].shape[0], n_ops); w = w / w.sum(1, keepdim=True)
        b2 = {k: v for k, v in b.items()}; b2["op_weights"] = w
        e2 = (old(**b2) - new(**b2)).abs().max().item()
    print(f"forward max|Δ|: indexed {e1:.2e}  blended-op {e2:.2e}")
    assert e1 < 1e-5 and e2 < 1e-5, "behavior-preserving migration FAILED"

    torch.save({"state": new.state_dict(),
                "cfg": {**cfg, "judge_name": True, "op_name": "ops" in a.tables,
                        "corpus_name": "corpora" in a.tables}}, out)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
