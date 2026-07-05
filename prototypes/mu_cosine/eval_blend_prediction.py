#!/usr/bin/env python3
"""The relevant measure (user 2026-07-05): how well does a model predict the JUDGE SUPERPOSITION on HELD-OUT
data? Sidesteps the LLM-alignment confound — the target's graph half is NOT LLM.

Target  T = (1−λ)·e5_ref  +  λ·graph_ref     on held-out pairs the model never trained on:
  e5_ref    = (cos(e5)+1)/2           — raw frozen-e5 SYM proxy, model-independent
  graph_ref = conf-weighted( 1/d , μ_HIER, μ_ELEM )   — from the REFERENCE model (model_prod), fixed
Then corr( each model's SYM readout , T ). A model that learned the superposition (not just the LLM judge)
predicts T better on held-out.

  python3 eval_blend_prediction.py --pairs /tmp/mu_data/wiki_rel_heldout.tsv --e5-cache /tmp/mu_data/wiki_rel_heldout_e5.pt \
      --struct-emb /tmp/mu_data/struct_emb_recip.pt --ref model_prod.pt \
      --models prod=model_prod.pt A=/tmp/mu_data/model_A_ft.pt B=/tmp/mu_data/model_blend_ft.pt --lam 0.5
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import OPS, CORPORA, JUDGES, membership_readouts, Tokenizer, load_dag
from eval_relatedness import build_model
from mu_posterior import struct_dist_fn


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def score_sym(model, tok, pairs, dev, judge=None, corpus="enwiki"):
    items = [((a, b, OPS["SYM"]) if judge is None else (a, b, OPS["SYM"], CORPORA[corpus], JUDGES[judge]))
             for a, b in pairs]
    out = []
    for i in range(0, len(items), 512):
        bd = tok.build(items[i:i + 512], train=False)
        bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bd.items()}
        with torch.no_grad():
            out += model(**bd).cpu().tolist()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--e5-cache", required=True)
    ap.add_argument("--struct-emb", required=True)
    ap.add_argument("--ref", default="model_prod.pt", help="reference model for graph_ref memberships")
    ap.add_argument("--models", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--lam", type=float, default=0.5, help="[legacy single λ; --lam-eval overrides]")
    ap.add_argument("--lam-eval", default=None, help="comma-sep λ values to build T at, e.g. 0.3,0.5,0.7 — "
                    "eval a model trained at one λ against T at OTHER λ tests generality vs recipe-memorisation "
                    "(review #3491 III.L; user's 'try other blends').")
    ap.add_argument("--c-dist", type=float, default=0.35)
    ap.add_argument("--c-subcat", type=float, default=0.72)
    ap.add_argument("--c-elem", type=float, default=0.82)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    d = torch.load(a.e5_cache, weights_only=False)
    idx = {n: i for i, n in enumerate(d["names"])}
    q, p = d["query"].numpy(), d["passage"].numpy()
    parents, children, deg = load_dag()
    tok = Tokenizer(d["query"], d["passage"], idx, parents, deg)
    pairs = []
    for ln in open(a.pairs, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        c = ln.rstrip("\n").split("\t")
        if len(c) >= 2 and c[0] in idx and c[1] in idx:
            pairs.append((c[0], c[1]))
    print(f"held-out pairs scored: {len(pairs)}")

    # e5_ref (model-independent): raw e5 cosine → [0,1]. Direction = passage(node)·query(root), i.e. the same
    # e5-prefix regime the SYM readout consumes (μ(node|root)); |cos| is symmetric enough for this reference.
    e5_ref = np.array([(float(p[idx[x]] @ q[idx[y]]) + 1.0) / 2.0 for x, y in pairs])
    # graph_ref from the REFERENCE model (fixed): conf-weighted(1/d, μ_HIER, μ_ELEM)
    ref = build_model(a.ref, dev)
    ms, me = membership_readouts(ref, tok, pairs, dev); ms, me = ms.cpu().numpy(), me.cpu().numpy()
    sd = struct_dist_fn(a.struct_emb)
    _sv = [sd(x, y) for x, y in pairs]                      # call sd once per pair (nit: review #3491)
    dist01 = np.array([min(1.0, v / 3.0) if v == v else 0.0 for v in _sv])
    ws = a.c_dist + a.c_subcat + a.c_elem
    graph_ref = (a.c_dist * dist01 + a.c_subcat * ms + a.c_elem * me) / ws
    lams = [float(x) for x in a.lam_eval.split(",")] if a.lam_eval else [a.lam]
    print(f"mean e5_ref {e5_ref.mean():.3f} (std {e5_ref.std():.3f}), graph_ref {graph_ref.mean():.3f} (std {graph_ref.std():.3f})")
    print(f"target(s) T(λ) = (1−λ)·e5_ref ⊕ λ·graph_ref  for λ ∈ {lams}\n")

    hdr = "  ".join(f"corr@λ={l:g}" for l in lams)
    print(f"{'model':10s} {'judge input':12s}  {hdr}   corr(e5_ref) corr(graph_ref)")
    for spec in a.models:
        name, path = spec.split("=", 1)
        if os.path.abspath(path) == os.path.abspath(a.ref):
            print(f"# note: {name} is the --ref model (graph_ref uses its memberships) — not independent")
        m = build_model(path, dev)
        for judge in (None, "blend"):
            if judge == "blend" and m.judge_emb.num_embeddings <= JUDGES["blend"]:
                continue                                   # model has no blend judge row (pre-PR checkpoint)
            sym = score_sym(m, tok, pairs, dev, judge=judge)
            cols = "  ".join(f"{corr(sym, (1-l)*e5_ref + l*graph_ref):+.3f}   " for l in lams)
            print(f"{name:10s} {str(judge or 'agnostic'):12s}  {cols} {corr(sym,e5_ref):+.3f}       {corr(sym,graph_ref):+.3f}")


if __name__ == "__main__":
    main()
