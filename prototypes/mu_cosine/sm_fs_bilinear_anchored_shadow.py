#!/usr/bin/env python3
"""SHADOW: bilinear identity-floor head — score = qhat^T W chat, W init=I (starts AT e5's
0.573 by construction), conditioning as zero-init low-rank modulation (starts as no-op).
Trained with the proven two-function mix (rank-CE + graded MSE). Owner's question made
architecture: no tokenizer/attention stack between the e5 vectors and the score."""
import json, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from mu_attention import E5_REVISION, build_e5_tables

SEED, STEPS, SLATE, D, R = 3997001, 2400, 32, 384, 16

class BilinearHead(torch.nn.Module):
    """ANCHORED: W = I frozen; only a zero-init low-rank delta trains (~12k params).
    The head cannot leave the e5 floor except through the small learned delta."""
    def __init__(self, n_cond=2):
        super().__init__()
        self.register_buffer("I", torch.eye(D))
        self.A = torch.nn.Parameter(torch.zeros(D, R))
        self.B = torch.nn.Parameter(torch.zeros(D, R) )
        torch.nn.init.normal_(self.B, std=0.01)   # A zero-init keeps delta exactly 0 at start
        self.U = torch.nn.Parameter(torch.zeros(D, R))
        self.V = torch.nn.Parameter(torch.zeros(D, R))
        self.g = torch.nn.Embedding(n_cond, R)
        torch.nn.init.zeros_(self.g.weight)
    def forward(self, qv, cv, cond):
        base = (qv * cv).sum(-1)                         # exact cosine (unit vectors)
        delta = ((qv @ self.A) * (cv @ self.B)).sum(-1)  # low-rank learned correction
        mod = ((qv @ self.U) * self.g(cond) * (cv @ self.V)).sum(-1)
        return base + delta + mod

def main():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    titles = sh._titles()
    qt, pt, ix = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                 model_revision=E5_REVISION)
    Q = torch.tensor(qt.numpy(), device=dev)
    P = torch.tensor(pt.numpy(), device=dev)
    man, pairs, _ = pl.load_pairs_verified()
    catalog = sorted({p["candidate"] for p in pairs})
    ci = {c: j for j, c in enumerate(catalog)}
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    held = defaultdict(list)
    for p in pairs:
        held[p["fold"]].append(p)
    cat_ids = torch.tensor([ix[c] for c in catalog], device=dev)
    res = defaultdict(dict)
    for f in range(5):
        rows = sh._projection(f)
        pos, buckets = defaultdict(list), defaultdict(lambda: defaultdict(list))
        for i, p in enumerate(rows):
            (pos[p["query"]].append(i) if p["class"].startswith("positive")
             else buckets[p["query"]][p["hardness"]].append(i))
        dest_idx = {}
        for q, idxs in pos.items():
            for i in idxs:
                if rows[i]["candidate"] == dest_of[q]:
                    dest_idx[q] = i
        train_q = sorted(q for q in pos if q in dest_idx)
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        model = BilinearHead().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=5e-5)
        t0 = time.time()
        for step in range(STEPS):
            if step % 2 == 0:            # rank-CE step
                q = train_q[rng.integers(len(train_q))]
                negs = []
                for b, n in (("hard", 12), ("medium", 10), ("easy", 9)):
                    mem = buckets[q].get(b, [])
                    if mem:
                        negs += [mem[i] for i in rng.integers(0, len(mem), min(n, len(mem)))]
                slate = [dest_idx[q]] + negs[:SLATE-1]
                qv = Q[ix[q]].expand(len(slate), D)
                cv = P[torch.tensor([ix[rows[i]["candidate"]] for i in slate], device=dev)]
                cond = torch.zeros(len(slate), dtype=torch.long, device=dev)
                s = model(qv, cv, cond)
                loss = torch.nn.functional.cross_entropy(
                    (s*8).unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
            else:                        # graded MSE step (cond id 1)
                c_sel, k_sel = pl.sample_step(f, SEED, step, train_q, pos, buckets,
                                              "graded_negative")
                sel = c_sel + k_sel
                qv = Q[torch.tensor([ix[rows[i]["query"]] for i in sel], device=dev)]
                cv = P[torch.tensor([ix[rows[i]["candidate"]] for i in sel], device=dev)]
                cond = torch.ones(len(sel), dtype=torch.long, device=dev)
                tgt = torch.tensor([rows[i]["target"] for i in sel], dtype=torch.float32,
                                   device=dev)
                s = model(qv, cv, cond)
                loss = torch.mean((torch.sigmoid(s*4) - tgt) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        rrs = []
        with torch.no_grad():
            hq = sorted({p["query"] for p in held[f]})
            cv_all = P[cat_ids]
            for q in hq:
                s = model(Q[ix[q]].expand(len(catalog), D), cv_all,
                          torch.zeros(len(catalog), dtype=torch.long, device=dev))
                s = [float(v) for v in s.cpu()]
                rrs.append(1.0 / pl.recompute_rank(s, catalog, ci[dest_of[q]]))
        res["bilinear_anchored"][f] = float(np.mean(rrs))
        print(f"[bilin] fold {f}: MRR {res['bilinear_anchored'][f]:.4f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(json.dumps({k: float(np.mean(list(v.values()))) for k, v in res.items()} |
                     {"comparators": {"attention_mix": 0.347, "e5_frozen": 0.573}}, indent=1),
          flush=True)

main()
