#!/usr/bin/env python3
"""Direction eval for the cross-judge direction superposition (DESIGN_cross_judge_direction.md). On held-out
pairs, correlate each model's HIER directional asymmetry μ(a|b)−μ(b|a) with the consensus direction target
d_true = mean(d_graph, d_element, d_subcat), read WITH the `dir-blend` judge input and AGNOSTICALLY. The
dir-blend-trained model should carry the direction in the trunk (agnostic) → judge-independent (the truncated-λ
story, for direction).

  python3 eval_direction.py --pairs /tmp/mu_data/wiki_rel_pairs.tsv --ids /tmp/mu_data/dir_held_ids.txt \
      --responses /tmp/mu_data/wiki_rel_resp.txt --graph .../100k_cats/category_parent.tsv \
      --e5-cache /tmp/mu_data/wiki_rel_e5.pt --models prod=model_prod.pt eq=... dir=...
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import OPS, CORPORA, JUDGES, Tokenizer, load_dag
from eval_relatedness import build_model
from emit_direction_blend import up_hops, parse_responses


def corr(a, b): return float(np.corrcoef(a, b)[0, 1])


def hier_dir(model, tok, pairs, dev, judge=None, corpus="enwiki"):
    """μ_HIER(a|b) − μ_HIER(b|a) per pair (the directional asymmetry)."""
    def score(ps):
        items = [((x, y, OPS["HIER"]) if judge is None else (x, y, OPS["HIER"], CORPORA[corpus], JUDGES[judge]))
                 for x, y in ps]
        out = []
        for i in range(0, len(items), 512):
            b = tok.build(items[i:i + 512], train=False)
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.no_grad():
                out += model(**b).cpu().tolist()
        return np.array(out)
    return score(pairs) - score([(y, x) for x, y in pairs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ids", required=True, help="held pair-index file")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--e5-cache", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    allpairs = [ln.rstrip("\n").split("\t")[:2] for ln in open(a.pairs, encoding="utf-8") if not ln.startswith("#")]
    held = [int(x) for x in open(a.ids).read().split()]
    parents, _, _ = load_dag(a.graph)
    byid = parse_responses(a.responses)

    def dgraph(x, y):
        uf, ur = up_hops(parents, x, y), up_hops(parents, y, x)
        return ((3/(1+uf) if uf is not None else 0.0) - (3/(1+ur) if ur is not None else 0.0)) / 3.0

    def dllm(o, rel): e = o.get(rel, {}); return float(e.get("mu_fwd", 0)) - float(e.get("mu_rev", 0))
    pairs, d_true = [], []
    for i in held:
        if i not in byid: continue
        x, y = allpairs[i]
        dg, de, ds = dgraph(x, y), dllm(byid[i], "element_of"), dllm(byid[i], "subcategory")
        if abs(dg) < 1e-6 and abs(de) < 1e-6 and abs(ds) < 1e-6: continue      # no direction — skip
        pairs.append((x, y)); d_true.append((dg + de + ds) / 3.0)
    d_true = np.array(d_true)
    d = torch.load(a.e5_cache, weights_only=False); idx = {n: i for i, n in enumerate(d["names"])}
    pairs = [(x, y) for x, y in pairs if x in idx and y in idx]
    d_true = d_true[:len(pairs)]
    tok = Tokenizer(d["query"], d["passage"], idx, parents, load_dag(a.graph)[2])
    print(f"held directional pairs: {len(pairs)}  (target = mean(d_graph,d_element,d_subcat), sign-consensus)\n")

    print(f"{'model':8s} {'judge input':11s}  corr(HIER-asym, direction)   sign-acc")
    for spec in a.models:
        name, path = spec.split("=", 1)
        m = build_model(path, dev)
        for judge in (None, "dir-blend"):
            if judge == "dir-blend" and m.judge_emb.num_embeddings <= JUDGES["dir-blend"]:
                continue
            asym = hier_dir(m, tok, pairs, dev, judge=judge)
            sacc = (np.sign(asym) == np.sign(d_true))[np.abs(d_true) > 1e-6].mean() * 100
            print(f"{name:8s} {str(judge or 'agnostic'):11s}  {corr(asym, d_true):+.3f}                      {sacc:.0f}%")


if __name__ == "__main__":
    main()
