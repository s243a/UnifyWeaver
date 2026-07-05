#!/usr/bin/env python3
"""Emit SYM training targets under the constructed BLEND judge (user 2026-07-05): the model fits
    blend = (1−λ)·μ_e5_sym  ⊕  λ·P(SYM | 1/d, asym-ops)
where P(SYM|1/d,asym-ops) is a confidence-weighted superposition of the distance proxy 1/d and the asymmetric
membership readouts μ_HIER/μ_ELEM (max over fwd/bwd). All components come from the current model (model_prod)
+ the struct embedding; the target is a *constructed judge*, tagged judge=blend so its own judge_emb row
calibrates it (distinct from the LLM judge). λ default 0.5; --random draws λ~U(0,1) per pair (blend regulariser).

  <out>: node root mu op=SYM relation node_type root_type corpus judge=blend conf
  python3 emit_blend_judge.py --pairs /tmp/mu_data/wiki_rel_pairs.tsv --e5-cache /tmp/mu_data/wiki_rel_e5.pt \
      --model model_prod.pt --struct-emb /tmp/mu_data/struct_emb_recip.pt --lam 0.5 --out /tmp/mu_data/blend_judge_pairs.tsv
"""
import argparse, os, random, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import OPS, membership_readouts, Tokenizer, build_e5_tables, load_dag
from eval_relatedness import build_model, score_pairs
from mu_posterior import struct_dist_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="8-col wiki_rel_pairs.tsv (node_title root_title ...)")
    ap.add_argument("--e5-cache", required=True)
    ap.add_argument("--model", default="model_prod.pt")
    ap.add_argument("--struct-emb", required=True)
    ap.add_argument("--lam", type=float, default=0.5, help="graph-judge portion / λ mean; blend = (1−λ)·e5 ⊕ λ·graph")
    ap.add_argument("--lam-dist", choices=["fixed", "uniform", "truncated"], default="fixed",
                    help="per-pair λ: fixed(=--lam) | uniform U(0,1) | truncated-normal(mean=--lam, std=--lam-std) "
                    "on [0,1] (resampled, NOT clamped — no boundary mass pile-up; user 2026-07-05). The blend "
                    "regulariser — some randomness spreads the family.")
    ap.add_argument("--lam-std", type=float, default=0.15, help="std for --lam-dist truncated-normal")
    ap.add_argument("--random", action="store_true", help="[deprecated alias for --lam-dist uniform]")
    # measured reliabilities from measure_membership_confidence.py (corr with SYM judge; leakage-inflated first
    # pass — see REPORT_mu_posterior_dist.md). Used INSIDE the blend-judge CONSTRUCTION (not the final fusion
    # combiner, so not subject to the playbook's "don't hand-set combiners over correlated sources" rule).
    ap.add_argument("--c-dist", type=float, default=0.35)
    ap.add_argument("--c-subcat", type=float, default=0.72)
    ap.add_argument("--c-elem", type=float, default=0.82)
    ap.add_argument("--corpus", default="enwiki")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device); rng = random.Random(0)

    pairs = []
    for ln in open(a.pairs, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        c = ln.rstrip("\n").split("\t")
        if len(c) >= 2:
            pairs.append((c[0], c[1]))

    d = torch.load(a.e5_cache, weights_only=False)
    idx = {n: i for i, n in enumerate(d["names"])}
    parents, children, deg = load_dag()
    tok = Tokenizer(d["query"], d["passage"], idx, parents, deg)
    pairs = [(x, y) for x, y in pairs if x in idx and y in idx]
    model = build_model(a.model, dev)
    n_ops = model.op_emb.num_embeddings

    ow = torch.zeros(1, n_ops); ow[0, OPS["SYM"]] = 1.0
    mu_e5_sym = np.array(score_pairs(model, tok, idx, pairs, ow, dev))         # e5 judge (SYM readout)
    ms, me = membership_readouts(model, tok, pairs, dev)                       # asym-ops (max fwd/bwd), detached
    ms, me = ms.cpu().numpy(), me.cpu().numpy()
    sdist = struct_dist_fn(a.struct_emb)
    _sv = [sdist(x, y) for x, y in pairs]                  # call sdist once per pair (review nit)
    dist01 = np.array([min(1.0, v / 3.0) if v == v else 0.0 for v in _sv])

    # P(SYM | 1/d, asym-ops): confidence-weighted superposition (weights = measured reliabilities), ∈[0,1]
    wsum = a.c_dist + a.c_subcat + a.c_elem
    mu_graph_sym = (a.c_dist * dist01 + a.c_subcat * ms + a.c_elem * me) / wsum

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# node\troot\tmu\top\trelation\tnode_type\troot_type\tcorpus\tjudge\tconf\n")
        dist = "uniform" if a.random else a.lam_dist
        for i, (x, y) in enumerate(pairs):
            if dist == "uniform":
                lam = rng.random()
            elif dist == "truncated":
                lam = rng.gauss(a.lam, a.lam_std)
                while lam < 0.0 or lam > 1.0:            # TRUNCATED normal: resample (no boundary pile-up)
                    lam = rng.gauss(a.lam, a.lam_std)
            else:
                lam = a.lam
            blend = float((1 - lam) * mu_e5_sym[i] + lam * mu_graph_sym[i])
            blend = max(0.0, min(1.0, blend))
            # relation=see_also is a PLACEHOLDER: these are SYM-operator targets (op=SYM is what trains); the
            # relation field is unused for SYM rows. conf=1.0 is a placeholder too — the target is a construction,
            # not a graded label (a |blend−0.5|-based conf is a documented future option; review #3491 F).
            f.write(f"{x}\t{y}\t{blend:.3f}\tSYM\tsee_also\tcategory\tcategory\t{a.corpus}\tblend\t1.0\n")
    print(f"wrote {len(pairs)} blend-judge SYM rows → {a.out}  "
          f"(λ={'random' if a.random else a.lam}; mean e5-sym {mu_e5_sym.mean():.3f}, graph-sym {mu_graph_sym.mean():.3f})")


if __name__ == "__main__":
    main()
