#!/usr/bin/env python3
"""SHADOW: multi-vector candidates (owner's composition-in-embedding-space point).
Each directory = a SET of in-distribution e5 vectors: its leaf name + member-map titles
(leave-one-out: a query's own vector never represents its destination, train or eval).
score(q, dir) = max over the set (and a top2-mean variant). No OOD list-strings."""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from mu_attention import E5_MODEL, E5_REVISION
from sentence_transformers import SentenceTransformer

def main():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer(E5_MODEL, revision=E5_REVISION, device=dev)
    man, pairs, _ = pl.load_pairs_verified()
    catalog = sorted({p["candidate"] for p in pairs})
    ci = {c: j for j, c in enumerate(catalog)}
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    titles = sh._titles()
    by_dest = defaultdict(list)
    for q, d in dest_of.items():
        by_dest[d].append(q)
    # one in-distribution embedding per atomic text
    leaf_texts = {c: c.rsplit("/", 1)[-1] for c in catalog}
    qs_all = sorted(dest_of)
    uniq = sorted({t for t in leaf_texts.values()} | {titles[q] for q in qs_all})
    pv = st.encode(["passage: " + t for t in uniq], batch_size=128,
                   normalize_embeddings=True)
    qv = st.encode(["query: " + titles[q] for q in qs_all], batch_size=128,
                   normalize_embeddings=True)
    P = {t: pv[i] for i, t in enumerate(uniq)}
    Qv = {q: qv[i] for i, q in enumerate(qs_all)}
    res = defaultdict(list)
    for qi, q in enumerate(qs_all):
        v = Qv[q]
        s_max, s_t2, s_leaf = [], [], []
        for c in catalog:
            vecs = [P[leaf_texts[c]]] + [P[titles[m]] for m in by_dest.get(c, []) if m != q]
            cs = sorted((float(v @ x) for x in vecs), reverse=True)
            s_leaf.append(float(v @ P[leaf_texts[c]]))
            s_max.append(cs[0])
            s_t2.append(float(np.mean(cs[:2])) if len(cs) >= 2 else cs[0])
        di = ci[dest_of[q]]
        for name, s in (("leaf_only", s_leaf), ("multivec_max", s_max),
                        ("multivec_top2", s_t2)):
            res[name].append(1.0 / pl.recompute_rank(s, catalog, di))
    print(json.dumps({k: float(np.mean(v)) for k, v in res.items()} |
                     {"note": "leaf_only == the 0.573 baseline recomputed; "
                              "leave-one-out throughout"}, indent=1), flush=True)

main()
