#!/usr/bin/env python3
"""SHADOW: STEM MULTI-OP TRAINING (owner's relation taxonomy at enwiki scale).
Data: ~/mu_data/enwiki_transitive_v1/targets_clean.tsv — ELEM (terminating, pos+easy),
LINEAGE (asymmetric transitive, decay^h), SYM (associative siblings, IDF-weighted);
graph-judged; STEM subtree (region with the largest e5-over-mu gap, +0.566).
Recipe: warm-start from the v2 crossover trunk (trunk_wiki_40000.pt, the first model to
beat its e5 floor); per-step lam ~ Dirichlet(0.5^4) over {ELEM MSE, LINEAGE MSE, SYM MSE,
rank-CE (logit space, slate = true direct cat + hard/random cats)}; graded channels present
from step 0 (fold-2 lesson); validation early stopping on held-article membership ranking
(mid-tie ranks; e5 floor on identical slates). Saves best checkpoint for the region
before/after readout (does the stem_computing gap close; does the gate shift toward mu)."""
import os, sys, json, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
from fine_tune_channel_heads import mu_batch, load_expanded
from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, build_e5_tables

DATA = os.path.expanduser("~/mu_data/enwiki_transitive_v1/targets_clean.tsv")
WARM = os.path.expanduser("~/mu_data/wiki_lineage_v2/trunk_wiki_40000.pt")
OUTCK = os.path.expanduser("~/mu_data/enwiki_transitive_v1/trunk_stem_multiop.pt")
SEED, STEPS, SLATE, EVERY = 3997001, 24000, 32, 4000
N_HELD, N_CAT_EVAL = 4000, 2000
CE, J, MM, CAT = CORPORA["enwiki"], JUDGES["graph"], NODETYPE["mindmap_node"], NODETYPE["category"]

def run():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = {"ELEM": [], "LINEAGE": [], "SYM": []}
    elem_pos = defaultdict(list)
    for ln in open(DATA, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        n, o, t, op, kind = ln.rstrip("\n").split("\t")
        rows[op].append((n, o, float(t)))
        if op == "ELEM" and kind == "pos":
            elem_pos[n].append(o)
    names = sorted({x for v in rows.values() for n, o, _ in v for x in (n, o)})
    print(f"rows: { {k: len(v) for k, v in rows.items()} }, names {len(names)}", flush=True)
    qt, pt, ix = build_e5_tables(names, cache_path=os.path.expanduser(
        "~/mu_data/enwiki_transitive_v1/e5.pt"), batch_size=256, model_revision=E5_REVISION)
    tok = Tokenizer(qt, pt, ix, {}, {})
    arts = sorted(elem_pos)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(arts))
    held = [arts[i] for i in order[:N_HELD]]
    train_arts = set(arts[i] for i in order[N_HELD:])
    tr = {k: [r for r in v if r[0] in train_arts] for k, v in rows.items()}
    cats = sorted({o for n, o, _ in rows["ELEM"]} | {o for n, o, _ in rows["LINEAGE"]})
    eval_cats = [cats[i] for i in rng.integers(0, len(cats), N_CAT_EVAL)]
    ev = []
    for a in held[:1000]:
        dest = elem_pos[a][0]
        ev.append((a, dest, [c for c in eval_cats if c != dest][:N_CAT_EVAL - 1]))
    def evaluate(score_fn):
        rrs = []
        for a, dest, dist in ev:
            s = score_fn(a, [dest] + dist)
            gt = sum(1 for v in s[1:] if v > s[0])
            eq = sum(1 for v in s[1:] if v == s[0])
            rrs.append(1.0 / (1 + gt + eq / 2.0))
        return float(np.mean(rrs))
    Qn, Pn = qt.numpy(), pt.numpy()
    floor = evaluate(lambda a, cs: [float(Qn[ix[a]] @ Pn[ix[c]]) for c in cs])
    print(f"[stem] e5 floor (sampled-{N_CAT_EVAL}): {floor:.4f}", flush=True)
    ck = pl.read_bound(os.path.join(pl.RANK_DIR, f"init_seed{SEED}.pt"),
                       expect_sha=pl.INIT_SHA[SEED], private=True)
    model, cfg = pl.load_checkpoint_bytes(ck, dev)
    model.load_state_dict(torch.load(WARM, map_location=dev))   # bare sd from the wiki trunk
    for p in model.parameters():
        p.requires_grad = True
    tensors = list(model.parameters())
    opt = torch.optim.Adam(tensors, lr=2e-4)
    model.train()
    best, best_sd, t0 = -1.0, None, time.time()
    OPMAP = {"ELEM": OPS["ELEM"], "LINEAGE": OPS["LINEAGE"], "SYM": OPS["SYM"]}
    for step in range(1, STEPS + 1):
        lam = rng.dirichlet((0.5, 0.5, 0.5, 0.5))
        loss = 0.0
        for li, op in enumerate(("ELEM", "LINEAGE", "SYM")):
            if lam[li] < 1e-3 or not tr[op]:
                continue
            sel = rng.integers(0, len(tr[op]), 48)
            items = [(tr[op][i][0], tr[op][i][1], OPMAP[op], CE, J, MM,
                      CAT if op != "SYM" else MM) for i in sel]
            tgt = torch.tensor([tr[op][i][2] for i in sel], dtype=torch.float32, device=dev)
            mu = mu_batch(model, tok, items, dev, train=True, rng=np.random)
            loss = loss + lam[li] * torch.mean((mu - tgt) ** 2)
        if lam[3] > 1e-3:
            a = None
            while a is None or a not in train_arts:
                a = arts[rng.integers(len(arts))]
            dest = elem_pos[a][0]
            negs = [cats[i] for i in rng.integers(0, len(cats), SLATE - 1) if cats[i] != dest]
            slate = [dest] + negs[:SLATE - 1]
            items = [(a, c, OPS["LINEAGE_RANK"], CE, J, MM, CAT) for c in slate]
            mu = mu_batch(model, tok, items, dev, train=True, rng=np.random)
            logits = torch.logit(mu.clamp(1e-6, 1 - 1e-6))
            loss = loss + lam[3] * torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(tensors, 1.0); opt.step()
        if step % EVERY == 0:
            model.eval()
            with torch.no_grad():
                m = evaluate(lambda a, cs: [float(v) for v in mu_batch(
                    model, tok, [(a, c, OPS["LINEAGE_RANK"], CE, J, MM, CAT)
                                 for c in cs], dev).cpu()])
            model.train()
            tag = ""
            if m > best:
                best, best_sd = m, {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()}
                tag = " *best*"
            print(f"[stem] step {step}: held {m:.4f} (floor {floor:.4f}, "
                  f"{(time.time()-t0)/60:.1f} min){tag}", flush=True)
    if best_sd is not None:
        torch.save(best_sd, OUTCK)
    print(json.dumps({"stamp": "shadow-exploratory-tier1-not-decision-bearing",
                      "e5_floor": round(floor, 4), "best_held": round(best, 4),
                      "ratio_to_floor": round(best / max(floor, 1e-9), 3),
                      "checkpoint": OUTCK}), flush=True)

run()
