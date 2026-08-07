#!/usr/bin/env python3
"""SHADOW: STEM multi-op fine-tune WITH REPLAY (the prescribed discipline, applied this
time: 40% replay from general wiki_lineage_v2 decisions, lr = base/3, multi-seed, plus a
replay-only PLACEBO arm so churn is separable from the STEM effect). Owner frame: gate
softmax makes off-target regression tolerable, but with enough data and capacity all
regions should improve — this measures whether replay delivers that.
Env: ARM=replay|placebo  SEED=...  OUT=...  (defaults below)."""
import os, sys, json, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
from fine_tune_channel_heads import mu_batch
from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, build_e5_tables

DATA = os.path.expanduser("~/mu_data/enwiki_transitive_v1/targets_clean.tsv")
REPLAY = os.path.expanduser("~/mu_data/wiki_lineage_v2/targets.tsv")
WARM = os.path.expanduser("~/mu_data/wiki_lineage_v2/trunk_wiki_40000.pt")
ARM = os.environ.get("ARM", "replay")
SEED = int(os.environ.get("SEED", "3997001"))
OUT = os.environ.get("OUT", os.path.expanduser(
    f"~/mu_data/enwiki_transitive_v1/trunk_stem_{ARM}_s{SEED}.pt"))
STEPS, SLATE, EVERY, REPLAY_FRAC, LR = 24000, 32, 4000, 0.4, 2e-4 / 3
N_HELD, N_CAT_EVAL = 4000, 2000
CE, J, MM, CAT = CORPORA["enwiki"], JUDGES["graph"], NODETYPE["mindmap_node"], NODETYPE["category"]

def load_stem():
    rows = {"ELEM": [], "LINEAGE": [], "SYM": []}
    elem_pos = defaultdict(list)
    for ln in open(DATA, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        n, o, t, op, kind = ln.rstrip("\n").split("\t")
        rows[op].append((n, o, float(t)))
        if op == "ELEM" and kind == "pos":
            elem_pos[n].append(o)
    return rows, elem_pos

def load_replay():
    rows, dest = [], {}
    for ln in open(REPLAY, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        n, a, t, kind = ln.rstrip("\n").split("\t")
        rows.append((n, a, float(t)))
        if kind == "pos" and t == "1.0":
            dest.setdefault(n, a)
    return rows, dest

def run():
    pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, elem_pos = load_stem()
    rrows, rdest = load_replay()
    names = sorted({x for v in rows.values() for n, o, _ in v for x in (n, o)})
    qt, pt, ix = build_e5_tables(names, cache_path=os.path.expanduser(
        "~/mu_data/enwiki_transitive_v1/e5.pt"), batch_size=256, model_revision=E5_REVISION)
    tok = Tokenizer(qt, pt, ix, {}, {})
    rnames = sorted({x for r in rrows for x in r[:2]})
    rqt, rpt, rix = build_e5_tables(rnames, cache_path=os.path.expanduser(
        "~/mu_data/wiki_lineage_v2/wiki_e5.pt"), batch_size=256, model_revision=E5_REVISION)
    rtok = Tokenizer(rqt, rpt, rix, {}, {})
    rcat = sorted({a for _, a in rdest.items()})
    rkeys = sorted(rdest)
    arts = sorted(elem_pos)
    rng = np.random.default_rng(SEED)
    order = np.random.default_rng(3997001).permutation(len(arts))  # held split FIXED across arms
    held = [arts[i] for i in order[:N_HELD]]
    train_arts = set(arts[i] for i in order[N_HELD:])
    tr = {k: [r for r in v if r[0] in train_arts] for k, v in rows.items()}
    cats = sorted({o for n, o, _ in rows["ELEM"]} | {o for n, o, _ in rows["LINEAGE"]})
    ev_rng = np.random.default_rng(3997001)
    eval_cats = [cats[i] for i in ev_rng.integers(0, len(cats), N_CAT_EVAL)]
    ev = []
    for a in held[:1000]:
        dest = elem_pos[a][0]
        ev.append((a, dest, [c for c in eval_cats if c != dest][:N_CAT_EVAL - 1]))
    def evaluate(model):
        model.eval()
        rrs = []
        with torch.no_grad():
            for a, dest, dist in ev:
                cs = [dest] + dist
                s = [float(v) for v in mu_batch(
                    model, tok, [(a, c, OPS["LINEAGE_RANK"], CE, J, MM, CAT)
                                 for c in cs], dev).cpu()]
                gt = sum(1 for v in s[1:] if v > s[0])
                eq = sum(1 for v in s[1:] if v == s[0])
                rrs.append(1.0 / (1 + gt + eq / 2.0))
        model.train()
        return float(np.mean(rrs))
    ck = pl.read_bound(os.path.join(pl.RANK_DIR, "init_seed3997001.pt"),
                       expect_sha=pl.INIT_SHA[3997001], private=True)
    model, cfg = pl.load_checkpoint_bytes(ck, dev)
    model.load_state_dict(torch.load(WARM, map_location=dev))
    for p in model.parameters():
        p.requires_grad = True
    tensors = list(model.parameters())
    opt = torch.optim.Adam(tensors, lr=LR)
    torch.manual_seed(SEED)
    model.train()
    best, best_sd, t0 = -1.0, None, time.time()
    OPMAP = {"ELEM": OPS["ELEM"], "LINEAGE": OPS["LINEAGE"], "SYM": OPS["SYM"]}
    for step in range(1, STEPS + 1):
        use_replay = (ARM == "placebo") or (rng.random() < REPLAY_FRAC)
        loss = 0.0
        if use_replay:
            lam = rng.dirichlet((0.5, 0.5))
            if lam[0] > 1e-3:
                sel = rng.integers(0, len(rrows), 48)
                items = [(rrows[i][0], rrows[i][1], OPS["LINEAGE"], CE, J, MM, CAT)
                         for i in sel]
                tgt = torch.tensor([rrows[i][2] for i in sel], dtype=torch.float32,
                                   device=dev)
                mu = mu_batch(model, rtok, items, dev, train=True, rng=np.random)
                loss = loss + lam[0] * torch.mean((mu - tgt) ** 2)
            if lam[1] > 1e-3:
                n = rkeys[rng.integers(len(rkeys))]
                dest = rdest[n]
                negs = [rcat[i] for i in rng.integers(0, len(rcat), SLATE - 1)
                        if rcat[i] != dest]
                slate = [dest] + negs[:SLATE - 1]
                items = [(n, c, OPS["LINEAGE_RANK"], CE, J, MM, CAT) for c in slate]
                mu = mu_batch(model, rtok, items, dev, train=True, rng=np.random)
                logits = torch.logit(mu.clamp(1e-6, 1 - 1e-6))
                loss = loss + lam[1] * torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
        else:
            lam = rng.dirichlet((0.5, 0.5, 0.5, 0.5))
            for li, op in enumerate(("ELEM", "LINEAGE", "SYM")):
                if lam[li] < 1e-3 or not tr[op]:
                    continue
                sel = rng.integers(0, len(tr[op]), 48)
                items = [(tr[op][i][0], tr[op][i][1], OPMAP[op], CE, J, MM,
                          CAT if op != "SYM" else MM) for i in sel]
                tgt = torch.tensor([tr[op][i][2] for i in sel], dtype=torch.float32,
                                   device=dev)
                mu = mu_batch(model, tok, items, dev, train=True, rng=np.random)
                loss = loss + lam[li] * torch.mean((mu - tgt) ** 2)
            if lam[3] > 1e-3:
                a = None
                while a is None or a not in train_arts:
                    a = arts[rng.integers(len(arts))]
                dest = elem_pos[a][0]
                negs = [cats[i] for i in rng.integers(0, len(cats), SLATE - 1)
                        if cats[i] != dest]
                slate = [dest] + negs[:SLATE - 1]
                items = [(a, c, OPS["LINEAGE_RANK"], CE, J, MM, CAT) for c in slate]
                mu = mu_batch(model, tok, items, dev, train=True, rng=np.random)
                logits = torch.logit(mu.clamp(1e-6, 1 - 1e-6))
                loss = loss + lam[3] * torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(tensors, 1.0); opt.step()
        if step % EVERY == 0:
            m = evaluate(model)
            tag = ""
            if m > best:
                best, best_sd = m, {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()}
                tag = " *best*"
            print(f"[{ARM} s{SEED}] step {step}: held {m:.4f} "
                  f"({(time.time()-t0)/60:.1f} min){tag}", flush=True)
    if best_sd is not None:
        torch.save(best_sd, OUT)
    print(json.dumps({"arm": ARM, "seed": SEED, "best_held_stem": round(best, 4),
                      "checkpoint": OUT,
                      "stamp": "shadow-exploratory-tier1-not-decision-bearing"}), flush=True)

run()
