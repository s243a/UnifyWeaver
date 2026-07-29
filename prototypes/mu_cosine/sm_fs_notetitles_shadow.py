#!/usr/bin/env python3
"""SHADOW: title-source fix (owner's correction) — candidate text = leaf name + titles of
TRAIN-side maps filed there (fold-wise, leakage-safe; held titles never describe their own
destination). Re-baseline frozen e5, then the anchored CE-only head, on the fixed inputs."""
import json, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from mu_attention import E5_MODEL, E5_REVISION
from sentence_transformers import SentenceTransformer

SEED, STEPS, SLATE, D, R = 3997001, 2400, 32, 384, 16
MAXT = 12

class Anchored(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.zeros(D, R))
        self.B = torch.nn.Parameter(torch.zeros(D, R))
        torch.nn.init.normal_(self.B, std=0.01)
    def forward(self, qv, cv):
        return (qv * cv).sum(-1) + ((qv @ self.A) * (cv @ self.B)).sum(-1)

def main():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer(E5_MODEL, revision=E5_REVISION, device=dev)
    man, pairs, _ = pl.load_pairs_verified()
    catalog = sorted({p["candidate"] for p in pairs})
    ci = {c: j for j, c in enumerate(catalog)}
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    titles = sh._titles()                       # map_path -> map title (root-node name)
    fold_of = {p["query"]: p["fold"] for p in pairs}
    by_dest = defaultdict(list)                 # dest dir -> [(query_path, title)]
    for q, d in dest_of.items():
        by_dest[d].append(q)
    held_q = defaultdict(list)
    for q, f in fold_of.items():
        held_q[f].append(q)
    res = defaultdict(dict)
    for f in range(5):
        held = set(held_q[f])
        def dir_text(d):
            leaf = d.rsplit("/", 1)[-1]
            names = sorted({titles[q] for q in by_dest.get(d, []) if q not in held})[:MAXT]
            return leaf + (": " + ", ".join(names) if names else "")
        cand_texts = ["passage: " + dir_text(c) for c in catalog]
        cv = torch.tensor(st.encode(cand_texts, batch_size=128,
                                    normalize_embeddings=True), device=dev)
        qs = sorted(held)
        qv = torch.tensor(st.encode(["query: " + titles[q] for q in qs], batch_size=128,
                                    normalize_embeddings=True), device=dev)
        # frozen e5 re-baseline on fixed inputs
        rrs = []
        for i, q in enumerate(qs):
            s = [float(v) for v in (qv[i] @ cv.T).cpu()]
            rrs.append(1.0 / pl.recompute_rank(s, catalog, ci[dest_of[q]]))
        res["e5_notetitles"][f] = float(np.mean(rrs))
        # anchored CE-only on fixed inputs (train queries embed with the same fold texts)
        tq = sorted(q for q in dest_of if q not in held)
        tqv = torch.tensor(st.encode(["query: " + titles[q] for q in tq], batch_size=128,
                                     normalize_embeddings=True), device=dev)
        tqi = {q: i for i, q in enumerate(tq)}
        rng = np.random.default_rng(SEED)
        torch.manual_seed(SEED)
        model = Anchored().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=5e-5)
        for step in range(STEPS):
            q = tq[rng.integers(len(tq))]
            di = ci[dest_of[q]]
            neg = rng.integers(0, len(catalog), SLATE - 1)
            slate = np.concatenate([[di], neg])
            s = model(tqv[tqi[q]].expand(len(slate), D), cv[torch.tensor(slate, device=dev)])
            loss = torch.nn.functional.cross_entropy(
                (s * 8).unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        rrs2 = []
        with torch.no_grad():
            for i, q in enumerate(qs):
                s = [float(v) for v in model(qv[i].expand(len(catalog), D), cv).cpu()]
                rrs2.append(1.0 / pl.recompute_rank(s, catalog, ci[dest_of[q]]))
        res["anchored_ce_notetitles"][f] = float(np.mean(rrs2))
        print(f"[nt] fold {f}: e5 {res['e5_notetitles'][f]:.4f}  "
              f"anchored {res['anchored_ce_notetitles'][f]:.4f}", flush=True)
    print(json.dumps({k: float(np.mean(list(v.values()))) for k, v in res.items()} |
                     {"old_inputs": {"e5": 0.573, "anchored_ce": 0.552}}, indent=1), flush=True)

main()
