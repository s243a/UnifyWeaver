#!/usr/bin/env python3
"""SHADOW: continuous function superposition (owner's design).

Per step draw lambda ~ Beta(0.5, 0.5); condition on the lambda-blended op embedding
(scratch row = lam*T[LINEAGE_RANK] + (1-lam)*T[LINEAGE], differentiable to both) and train
loss = lam*CE(slate) + (1-lam)*MSE(graded rows) under that SAME conditioning. The sampling
density over lambda is the declared knob. Eval: MRR(lambda) curve — does the model
interpolate the function family, and is there an interior optimum beating both endpoints?"""
import copy, io, json, os, sys, time, tempfile
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from fine_tune_channel_heads import load_expanded, mu_batch
from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, build_e5_tables

SEED, STEPS, SLATE = 3997001, 2400, 32
CM, J, MM = CORPORA["mindmap"], JUDGES["graph"], NODETYPE["mindmap_node"]
SCRATCH = len(OPS)                      # grown scratch op index
LAM = [0.5]                             # mutable current lambda

def main():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    titles = sh._titles()
    qt, pt, ix = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                 model_revision=E5_REVISION)
    tok = Tokenizer(qt, pt, ix, {}, {})
    man, pairs, _ = pl.load_pairs_verified()
    catalog = sorted({p["candidate"] for p in pairs})
    ci = {c: j for j, c in enumerate(catalog)}
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    held = defaultdict(lambda: defaultdict(dict))
    for p in pairs:
        held[p["fold"]][p["query"]][p["candidate"]] = p
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
        torch.manual_seed(SEED); aug = np.random.default_rng(SEED + 1)
        rng = np.random.default_rng(SEED)
        blob = torch.load(io.BytesIO(pl.read_bound(
            os.path.join(pl.RANK_DIR, f"init_seed{SEED}.pt"),
            expect_sha=pl.INIT_SHA[SEED], private=True)), map_location="cpu",
            weights_only=False)
        sd = blob["state"]
        for key in ("op_name.name_e5", "op_name.resid.weight", "op_emb.weight"):
            if key in sd:
                sd[key] = torch.cat([sd[key], torch.zeros(1, sd[key].shape[1])])
        fd, tmp = tempfile.mkstemp(suffix=".pt"); os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as fh:
            torch.save(blob, fh)
        model, cfg = load_expanded(tmp, dev="cpu"); os.unlink(tmp)
        model = model.to(dev)
        # differentiable scratch-row mix
        target_mod = model.op_name if model.op_name is not None else model.op_emb
        if model.op_name is not None:
            orig = model.op_name.table
            def mixed_table():
                T = orig()
                mix = LAM[0]*T[OPS["LINEAGE_RANK"]] + (1-LAM[0])*T[OPS["LINEAGE"]]
                return torch.cat([T[:SCRATCH], mix.unsqueeze(0)], 0)
            model.op_name.table = mixed_table
        ref = copy.deepcopy(model); ref.eval()
        for p in ref.parameters(): p.requires_grad = False
        for p in model.parameters(): p.requires_grad = True
        tensors = list(model.parameters())
        opt = torch.optim.Adam(tensors, lr=2e-4)
        model.train(); t0 = time.time()
        for step in range(STEPS):
            LAM[0] = float(rng.beta(0.5, 0.5))
            # CE component: slate under scratch op
            q = train_q[rng.integers(len(train_q))]
            negs = []
            for b, n in (("hard", 12), ("medium", 10), ("easy", 9)):
                mem = buckets[q].get(b, [])
                if mem:
                    negs += [mem[i] for i in rng.integers(0, len(mem), min(n, len(mem)))]
            slate = [dest_idx[q]] + negs[:SLATE-1]
            items = [(rows[i]["query"], rows[i]["candidate"], SCRATCH, CM, J, MM, MM)
                     for i in slate]
            mu = mu_batch(model, tok, items, dev, train=True, rng=aug)
            ce = torch.nn.functional.cross_entropy(
                (mu*8.0).unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
            # MSE component: graded rows under the SAME scratch conditioning
            c_sel, k_sel = pl.sample_step(f, SEED, step, train_q, pos, buckets,
                                          "graded_negative")
            sel = (c_sel + k_sel)[:SLATE]
            items2 = [(rows[i]["query"], rows[i]["candidate"], SCRATCH, CM, J, MM, MM)
                      for i in sel]
            tgt = torch.tensor([rows[i]["target"] for i in sel], dtype=torch.float32,
                               device=dev)
            mu2 = mu_batch(model, tok, items2, dev, train=True, rng=aug)
            mse = torch.mean((mu2 - tgt) ** 2)
            loss = LAM[0]*ce + (1-LAM[0])*mse
            ag = [(it[0], it[1], OPS["LINEAGE"]) for it in items2[:8]]
            mu_ag = mu_batch(model, tok, ag, dev)
            with torch.no_grad():
                mu_ref = mu_batch(ref, tok, ag, dev)
            loss = loss + torch.mean((mu_ag - mu_ref) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(tensors, 1.0); opt.step()
        print(f"[lam] fold {f} trained ({(time.time()-t0)/60:.1f} min)", flush=True)
        model.eval()
        for lam_eval in (0.0, 0.25, 0.5, 0.75, 1.0):
            LAM[0] = lam_eval
            rrs = []
            with torch.no_grad():
                for q in sorted(held[f]):
                    items = [(q, c, SCRATCH, CM, J, MM, MM) for c in catalog]
                    mu = [float(v) for v in mu_batch(model, tok, items, dev).cpu()]
                    rrs.append(1.0 / pl.recompute_rank(mu, catalog, ci[dest_of[q]]))
            res[f"lam={lam_eval}"][f] = float(np.mean(rrs))
        print(f"[lam] fold {f}: " + " ".join(
            f"{k}={res[k][f]:.3f}" for k in sorted(res)), flush=True)
    print(json.dumps({k: float(np.mean(list(v.values()))) for k, v in sorted(res.items())} |
                     {"comparators": {"full_mix_discrete": 0.347, "e5_frozen": 0.573}},
                     indent=1), flush=True)

main()
