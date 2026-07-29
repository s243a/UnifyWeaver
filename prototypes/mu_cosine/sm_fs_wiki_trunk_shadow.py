#!/usr/bin/env python3
"""SHADOW: TRUNK AT WIKI SCALE (cross-corpus lever, post-round-6) — the wiki crossover null
killed the bilinear head at scale (global linear correction extracts nothing; failures are
per-node/contextual); the trunk is the surviving vehicle. Train the full attention trunk on
the 27k wiki direct-parent decisions with the round-6 winning recipe: three-channel
Dirichlet superposition (graded lineage MSE from the decayed targets.tsv rows / rank-CE
truth in logit space / e5-KL), enwiki corpus conditioning. Eval: (a) wiki held sampled-999
MRR (300 queries, same distractor sets for model and e5 floor — paired), (b) zero-shot
SM-FS full-catalog transfer under mindmap conditioning (prior sigmoid-era zero-shot was
0.072). Seed 3997001, 12k steps."""
import copy, json, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from fine_tune_channel_heads import mu_batch
from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, build_e5_tables

OUT = os.path.expanduser("~/mu_data/wiki_lineage_v1")
SEED, STEPS, SLATE = 3997001, int(os.environ.get("TRUNK_STEPS", "12000")), 32
CE, CM = CORPORA["enwiki"], CORPORA["mindmap"]
J, MM, CAT = JUDGES["graph"], NODETYPE["mindmap_node"], NODETYPE["category"]
N_EVAL_Q, N_DIST = 300, 999

def run():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pos_rows, graded = [], []
    for ln in open(os.path.join(OUT, "targets.tsv"), encoding="utf-8"):
        if ln.startswith("#"):
            continue
        n, a, t, kind = ln.rstrip("\n").split("\t")
        if kind == "pos" and t == "1.0":
            pos_rows.append((n, a))
        if kind in ("pos", "easy"):
            graded.append((n, a, float(t)))
    names = sorted({x for r in graded for x in r[:2]})
    qtbl, ptbl, ix = build_e5_tables(names, cache_path=os.path.join(OUT, "wiki_e5.pt"),
                                     batch_size=512, model_revision=E5_REVISION)
    tok = Tokenizer(qtbl, ptbl, ix, {}, {})
    Q = qtbl.to(dev); P = ptbl.to(dev)
    catalog = sorted({a for _, a in pos_rows})
    ci = {c: j for j, c in enumerate(catalog)}
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pos_rows))
    held = [pos_rows[i] for i in order[:3000]]
    train = [pos_rows[i] for i in order[3000:]]
    train_set = {n for n, _ in train}
    graded_tr = [g for g in graded if g[0] in train_set]
    cat_ids = torch.tensor([ix[c] for c in catalog], device=dev)
    cv = P[cat_ids]
    # hard negatives for train slates
    tq_ids = torch.tensor([ix[n] for n, _ in train], device=dev)
    td_col = torch.tensor([ci[a] for _, a in train], device=dev)
    hard = torch.empty(len(train), 32, dtype=torch.long)
    with torch.no_grad():
        for s in range(0, len(train), 1024):
            sc = Q[tq_ids[s:s + 1024]] @ cv.T
            sc.scatter_(1, td_col[s:s + 1024].unsqueeze(1), -1e9)
            hard[s:s + 1024] = sc.topk(32, dim=1).indices.cpu()
    hard = hard.numpy()
    # frozen eval slates: 300 held queries x (dest + 999 sampled distractors)
    ev = []
    for n, a in held[:N_EVAL_Q]:
        dist = [catalog[i] for i in rng.integers(0, len(catalog), N_DIST) if catalog[i] != a]
        ev.append((n, a, dist[:N_DIST - 1]))
    def wiki_eval(score_fn):
        rrs = []
        for n, a, dist in ev:
            cands = [a] + dist
            s = score_fn(n, cands)
            rrs.append(1.0 / (1 + sum(1 for v in s[1:] if v > s[0])))
        return float(np.mean(rrs))
    e5_floor = wiki_eval(lambda n, cands: [float(Q[ix[n]] @ P[ix[c]]) for c in cands])
    print(f"[trunk] wiki e5 floor (sampled-{N_DIST}): {e5_floor:.4f}", flush=True)
    torch.manual_seed(SEED); aug = np.random.default_rng(SEED + 1)
    ck = pl.read_bound(os.path.join(pl.RANK_DIR, f"init_seed{SEED}.pt"),
                       expect_sha=pl.INIT_SHA[SEED], private=True)
    model, cfg = pl.load_checkpoint_bytes(ck, dev)
    ref = copy.deepcopy(model); ref.eval()
    for p in ref.parameters(): p.requires_grad = False
    for p in model.parameters(): p.requires_grad = True
    tensors = list(model.parameters())
    opt = torch.optim.Adam(tensors, lr=2e-4)
    model.train(); t0 = time.time()
    Qn, Pn = qtbl.numpy(), ptbl.numpy()
    for step in range(STEPS):
        lam = rng.dirichlet((0.5, 0.5, 0.5))
        loss = 0.0
        if lam[0] > 1e-3:                      # graded lineage MSE (stabilizer)
            sel = rng.integers(0, len(graded_tr), 48)
            g_items = [(graded_tr[i][0], graded_tr[i][1], OPS["LINEAGE"], CE, J, MM, CAT)
                       for i in sel]
            tgt = torch.tensor([graded_tr[i][2] for i in sel], dtype=torch.float32,
                               device=dev)
            mu_g = mu_batch(model, tok, g_items, dev, train=True, rng=aug)
            loss = loss + lam[0] * torch.mean((mu_g - tgt) ** 2)
        if lam[1] + lam[2] > 1e-3:             # slate channels
            i = int(rng.integers(len(train)))
            negs = list(hard[i][rng.integers(0, 32, 16)]) + \
                   list(rng.integers(0, len(catalog), SLATE - 17))
            cands = [train[i][1]] + [catalog[int(x)] for x in negs]
            s_items = [(train[i][0], c, OPS["LINEAGE_RANK"], CE, J, MM, CAT) for c in cands]
            mu_s = mu_batch(model, tok, s_items, dev, train=True, rng=aug)
            logits = torch.logit(mu_s.clamp(1e-6, 1 - 1e-6))
            if lam[1] > 1e-3:
                loss = loss + lam[1] * torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
            if lam[2] > 1e-3:
                cos = torch.tensor([float(Qn[ix[it[0]]] @ Pn[ix[it[1]]]) for it in s_items],
                                   device=dev)
                loss = loss + lam[2] * torch.nn.functional.kl_div(
                    torch.log_softmax(logits, dim=0), torch.softmax(8.0 * cos, dim=0),
                    reduction="sum")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(tensors, 1.0); opt.step()
        if (step + 1) % 4000 == 0:
            model.eval()
            with torch.no_grad():
                m = wiki_eval(lambda n, cands: [float(v) for v in mu_batch(
                    model, tok, [(n, c, OPS["LINEAGE_RANK"], CE, J, MM, CAT)
                                 for c in cands], dev).cpu()])
            model.train()
            print(f"[trunk] step {step+1}: wiki held {m:.4f} (floor {e5_floor:.4f}, "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)
    torch.save(model.state_dict(), os.path.join(OUT, f"trunk_wiki_{STEPS}.pt"))
    # zero-shot SM-FS transfer (mindmap conditioning, full catalog)
    model.eval()
    titles = sh._titles()
    sqt, spt, six = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                    model_revision=E5_REVISION)
    stok = Tokenizer(sqt, spt, six, {}, {})
    man, pairs, _ = pl.load_pairs_verified()
    scat = sorted({p["candidate"] for p in pairs})
    sci = {c: j for j, c in enumerate(scat)}
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    rrs = []
    with torch.no_grad():
        for q in sorted(dest_of):
            items = [(q, c, OPS["LINEAGE_RANK"], CM, J, MM, MM) for c in scat]
            mu = [float(v) for v in mu_batch(model, stok, items, dev).cpu()]
            rrs.append(1.0 / pl.recompute_rank(mu, scat, sci[dest_of[q]]))
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing",
           "wiki_e5_floor_sampled": e5_floor,
           "smfs_zeroshot_fullcatalog": float(np.mean(rrs)),
           "comparators": {"sigmoid_era_zeroshot": 0.072, "smfs_scratch_best": 0.385,
                           "smfs_e5": 0.573}}
    print(json.dumps(res, indent=1), flush=True)

run()
