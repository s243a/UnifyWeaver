#!/usr/bin/env python3
"""SHADOW: THREE-PARAMETER SUPERPOSITION BLEND (owner: 'if you solved collapse, you can
train on some three parameter blends') — per step draw lam ~ Dirichlet(0.5,0.5,0.5) over
three functions and train loss = lam1*MSE(graded) + lam2*rankCE(truth, logit space) +
lam3*KL(slate softmax || e5 soft ranking), with the operator token = the lam-blend of
LINEAGE and LINEAGE_RANK rows via the native op_weights path (one-hot reproduces the
indexed path exactly; NO op-table growth — sidesteps the load_expanded bug that killed the
first lambda runner). The old trimix destabilized WITHOUT these lessons; this retry carries
all of them: logit-space CE, graded channel present from step 0 (in expectation every step,
weight lam1), anchored init. Eval sweeps MRR(lam) over simplex corners/edges/center per
fold — the owner's interpolation surface, now 3-parameter. 5 folds, seed 3997001."""
import copy, json, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, build_e5_tables

SEED, STEPS, SLATE = 3997001, 1600, 32
CM, J, MM = CORPORA["mindmap"], JUDGES["graph"], NODETYPE["mindmap_node"]
EVAL_LAMS = {"graded_corner": (1, 0, 0), "rank_corner": (0, 1, 0), "e5_corner": (0, 0, 1),
             "center": (1/3, 1/3, 1/3), "rank_e5_edge": (0, .5, .5),
             "graded_rank_edge": (.5, .5, 0)}

def blend_batch(model, tok, items, dev, w_ops, train=False, rng=None):
    b = tok.build(items, train=train, rng=rng, p_mask_prov=0.0)
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    b["op_weights"] = w_ops.expand(len(items), -1)
    return model(**b)

def w_of(lam, n_ops, dev):
    w = torch.zeros(n_ops, device=dev)
    w[OPS["LINEAGE"]] = lam[0]
    w[OPS["LINEAGE_RANK"]] = lam[1] + lam[2]
    return w.unsqueeze(0)

def run():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    titles = sh._titles()
    qt, pt, ix = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                 model_revision=E5_REVISION)
    tok = Tokenizer(qt, pt, ix, {}, {})
    Qn, Pn = qt.numpy(), pt.numpy()
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
        ck = pl.read_bound(os.path.join(pl.RANK_DIR, f"init_seed{SEED}.pt"),
                           expect_sha=pl.INIT_SHA[SEED], private=True)
        model, cfg = pl.load_checkpoint_bytes(ck, dev)
        n_ops = model.readout_w.shape[0]
        ref = copy.deepcopy(model); ref.eval()
        for p in ref.parameters(): p.requires_grad = False
        for p in model.parameters(): p.requires_grad = True
        tensors = list(model.parameters())
        opt = torch.optim.Adam(tensors, lr=2e-4)
        model.train(); t0 = time.time()
        for step in range(STEPS):
            lam = rng.dirichlet((0.5, 0.5, 0.5))
            w = w_of(lam, n_ops, dev)
            loss = 0.0
            # graded-MSE component on a graded minibatch (stabilizer, weight lam1)
            if lam[0] > 1e-3:
                c, k = pl.sample_step(f, SEED, step, train_q, pos, buckets, "graded_negative")
                sel = c + k
                g_items = [(rows[i]["query"], rows[i]["candidate"], OPS["LINEAGE"],
                            CM, J, MM, MM) for i in sel]
                tgt = torch.tensor([rows[i]["target"] for i in sel], dtype=torch.float32,
                                   device=dev)
                mu_g = blend_batch(model, tok, g_items, dev, w, train=True, rng=aug)
                loss = loss + lam[0] * torch.mean((mu_g - tgt) ** 2)
            # slate components (truth CE + e5 KL) on one slate forward
            if lam[1] + lam[2] > 1e-3:
                q = train_q[rng.integers(len(train_q))]
                negs = []
                for b, n in (("hard", 12), ("medium", 10), ("easy", 9)):
                    mem = buckets[q].get(b, [])
                    if mem:
                        negs += [mem[i] for i in rng.integers(0, len(mem), min(n, len(mem)))]
                slate = [dest_idx[q]] + negs[:SLATE - 1]
                s_items = [(rows[i]["query"], rows[i]["candidate"], OPS["LINEAGE_RANK"],
                            CM, J, MM, MM) for i in slate]
                mu_s = blend_batch(model, tok, s_items, dev, w, train=True, rng=aug)
                logits = torch.logit(mu_s.clamp(1e-6, 1 - 1e-6))
                if lam[1] > 1e-3:
                    loss = loss + lam[1] * torch.nn.functional.cross_entropy(
                        logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
                if lam[2] > 1e-3:
                    cos = torch.tensor([float(Qn[ix[it[0]]] @ Pn[ix[it[1]]])
                                        for it in s_items], device=dev)
                    soft = torch.softmax(8.0 * cos, dim=0)
                    loss = loss + lam[2] * torch.nn.functional.kl_div(
                        torch.log_softmax(logits, dim=0), soft, reduction="sum")
            ag = [(q2, c2, OPS["LINEAGE_RANK"]) for q2, c2 in
                  [(rows[i]["query"], rows[i]["candidate"]) for i in
                   [dest_idx[qq] for qq in train_q[:8]]]]
            mu_ag = blend_batch(model, tok, ag, dev, w_of((0, 1, 0), n_ops, dev))
            with torch.no_grad():
                mu_ref = blend_batch(ref, tok, ag, dev, w_of((0, 1, 0), n_ops, dev))
            loss = loss + torch.mean((mu_ag - mu_ref) ** 2) * 0.1
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(tensors, 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            for name, lam in EVAL_LAMS.items():
                w = w_of(lam, n_ops, dev)
                rrs = []
                for q in sorted(held[f]):
                    items = [(q, c, OPS["LINEAGE_RANK"], CM, J, MM, MM) for c in catalog]
                    mu = blend_batch(model, tok, items, dev, w)
                    rrs.append(1.0 / pl.recompute_rank([float(v) for v in mu.cpu()],
                                                       catalog, ci[dest_of[q]]))
                res[name][f] = float(np.mean(rrs))
        print(f"[tri3] fold {f}: " +
              " ".join(f"{k} {res[k][f]:.3f}" for k in EVAL_LAMS) +
              f" ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(json.dumps({k: float(np.mean(list(v.values()))) for k, v in res.items()} |
                     {"comparators": {"two_mix_sigmoid": 0.347, "trimix_old_destab": "day-3",
                                      "e5_frozen": 0.573}}, indent=1), flush=True)

run()
