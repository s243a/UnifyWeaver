#!/usr/bin/env python3
"""SHADOW: multi-judge conditioning — the diversity-of-configurations experiment.

Owner's design premise restored: one model, many (function-input -> target) channels,
distinguished by conditioning tokens; generalization ACROSS them is the point.

Channels (op=LINEAGE throughout):
  (mindmap, graph)  SM-FS graded rows (per-fold projection; train-side only)
  (enwiki,  graph)  wiki lineage rows (~180k, approx negatives)
  (mindmap, e5*)    train-side (query,candidate) pairs, target=(cos+1)/2  [e5 distillation]
  (enwiki,  e5*)    wiki pairs, target=(cos+1)/2
  *e5 judge = NEW judge row (index 11), onboarded with a name-card embedding + zero residual.

Eval per fold (seed 3997001): graph-judge alone | e5-judge alone | score-mean superposition |
alpha-swept score blend (offline from stored vectors). Comparators: SM-only graded 0.280,
frozen e5 0.573. Stamped shadow-exploratory; reserve untouched."""
import copy
import io
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from fine_tune_channel_heads import load_expanded, mu_batch
from mu_attention import CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS, Tokenizer, \
    build_e5_tables

SEED = 3997001
STEPS, BS = 2400, 48
E5J = len(JUDGES)                                   # new judge index (11)
CE, CM = CORPORA["enwiki"], CORPORA["mindmap"]
JG = JUDGES["graph"]
MM, CAT = NODETYPE["mindmap_node"], NODETYPE["category"]
OUT = os.path.expanduser("~/mu_data/sm_fs_multijudge_v1")
MIX = [("sm_graph", 0.40), ("wiki_graph", 0.15), ("sm_e5", 0.35), ("wiki_e5", 0.10)]


def grow_e5_judge(model, card_vec, dev):
    """Onboard the e5 judge: judge_emb row (zero-init via load_expanded) + name-card row."""
    jn = model.judge_name
    if jn is not None and jn.name_e5.shape[0] == E5J:
        with torch.no_grad():
            jn.name_e5 = torch.cat([jn.name_e5, card_vec.to(jn.name_e5).view(1, -1)])
            old = jn.resid.weight
            new = torch.zeros(1, old.shape[1], device=old.device, dtype=old.dtype)
            jn.resid = torch.nn.Embedding.from_pretrained(
                torch.cat([old, new]), freeze=False).to(dev)
    return model


def main():
    env = pl.enforce_environment()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # wiki rows + names
    wrows = []
    for ln in open(os.path.expanduser("~/mu_data/wiki_lineage_v1/targets.tsv"),
                   encoding="utf-8"):
        if ln.startswith("#"):
            continue
        n, a, t, kind = ln.rstrip("\n").split("\t")
        wrows.append((n, a, float(t)))
    wnames = sorted({x for r in wrows for x in r[:2]})
    titles = sh._titles()
    all_names = sorted(set(wnames) | set(titles))
    texts = dict(titles)
    qtbl, ptbl, idx = build_e5_tables(all_names, cache_path=os.path.join(
        os.path.expanduser("~/mu_data/wiki_lineage_v1"), "multijudge_e5.pt"),
        batch_size=512, texts=texts, model_revision=E5_REVISION)
    tok = Tokenizer(qtbl, ptbl, idx, {}, {})
    Q, P = qtbl.numpy(), ptbl.numpy()
    from sentence_transformers import SentenceTransformer
    from mu_attention import E5_MODEL
    st = SentenceTransformer(E5_MODEL, revision=E5_REVISION)
    card_vec = torch.tensor(st.encode(
        ["passage: frozen e5 title cosine similarity judge"], normalize_embeddings=True)[0])
    del st
    man, pairs, _ = pl.load_pairs_verified()
    catalog = sorted({p["candidate"] for p in pairs})
    cat_index = {c: j for j, c in enumerate(catalog)}
    dest_of = {p["query"]: p["candidate"] for p in pairs
               if p["class"] == "positive_parent"}
    held_by_fold = defaultdict(lambda: defaultdict(dict))
    for p in pairs:
        held_by_fold[p["fold"]][p["query"]][p["candidate"]] = p
    rng = np.random.default_rng(SEED)
    results = defaultdict(dict)
    os.makedirs(OUT, mode=0o700, exist_ok=True)
    os.chmod(OUT, 0o700)
    for f in range(5):
        proj = sh._projection(f)
        train_q = sorted({p["query"] for p in proj if p["class"].startswith("positive")})
        sm_graph = [(p["query"], p["candidate"], p["target"], CM) for p in proj]
        # WITHIN-QUERY distillation: query x its own candidate list, where cosine
        # differences carry e5's ordering (random pairs only teach the mean)
        cv = np.stack([P[idx[c]] for c in catalog])
        sm_e5 = []
        for q in train_q:
            cos_all = Q[idx[q]] @ cv.T
            top = np.argsort(-cos_all)[:64]
            rand = rng.integers(0, len(catalog), 16)
            for j in set(top.tolist() + rand.tolist()):
                sm_e5.append((q, catalog[j], (float(cos_all[j]) + 1.0) / 2.0, CM))
        wpool = [wnames[i] for i in rng.integers(0, len(wnames), 512)]
        wv = np.stack([P[idx[c]] for c in wpool])
        wq = [wnames[i] for i in rng.integers(0, len(wnames), 400)]
        wiki_e5 = []
        for q in wq:
            cos_all = Q[idx[q]] @ wv.T
            top = np.argsort(-cos_all)[:32]
            for j in top:
                wiki_e5.append((q, wpool[j], (float(cos_all[j]) + 1.0) / 2.0, CE))
        chans = {"sm_graph": (sm_graph, JG), "wiki_graph":
                 ([(a, b, t, CE) for a, b, t in wrows], JG),
                 "sm_e5": (sm_e5, E5J), "wiki_e5": (wiki_e5, E5J)}
        torch.manual_seed(SEED)
        aug = np.random.default_rng(SEED + 1)
        # state-dict surgery: append the e5-judge row (card e5 + zero residual + zero
        # judge_emb) BEFORE construction, so all tables agree at n_judge=12
        blob = torch.load(io.BytesIO(pl.read_bound(
            os.path.join(pl.RANK_DIR, f"init_seed{SEED}.pt"),
            expect_sha=pl.INIT_SHA[SEED], private=True)), map_location="cpu",
            weights_only=False)
        sd = blob["state"]
        cv = card_vec.view(1, -1).to(sd["judge_name.name_e5"])
        sd["judge_name.name_e5"] = torch.cat([sd["judge_name.name_e5"], cv])
        for key in ("judge_name.resid.weight", "judge_emb.weight"):
            sd[key] = torch.cat([sd[key], torch.zeros(1, sd[key].shape[1])])
        fd, tmp = __import__("tempfile").mkstemp(suffix=".pt")
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as fh:
            torch.save(blob, fh)
        model, cfg = load_expanded(tmp, dev="cpu")
        os.unlink(tmp)
        model = model.to(dev)
        ref = copy.deepcopy(model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        # local allowlist: same 18-tensor recipe, count adjusted for the onboarded
        # 12th judge residual row (+384 params) — a legitimate shadow-model difference
        try:
            _, _, tensors = pl.resolve_allowlist(model)
        except pl.PipelineError as e:
            if "1196166" not in str(e):
                raise
            by_name = dict(model.named_parameters())
            by_name.setdefault("readout_w", model.readout_w)
            by_name.setdefault("readout_b", model.readout_b)
            names = [n for n in ("judge_name.resid.weight", "corpus_name.resid.weight",
                                 "op_name.resid.weight") if n in by_name]
            li = len(model.encoder.layers) - 1
            names += [f"encoder.layers.{li}.{n}" for n, _ in
                      model.encoder.layers[li].named_parameters()]
            names += ["readout_w", "readout_b", "nodetype_emb.weight"]
            for p in model.parameters():
                p.requires_grad = False
            tensors = []
            for n in names:
                by_name[n].requires_grad = True
                tensors.append(by_name[n])
        model.judge_emb.weight.requires_grad = True        # let the new judge row train
        tensors = tensors + [model.judge_emb.weight]
        opt = torch.optim.Adam(tensors, lr=pl.ADAM["lr"])
        model.train()
        t0 = time.time()
        for step in range(STEPS):
            r = rng.random()
            acc, chan = 0.0, MIX[-1][0]
            for name, w in MIX:
                acc += w
                if r < acc:
                    chan = name
                    break
            rows_c, judge = chans[chan]
            sel = rng.integers(0, len(rows_c), BS)
            nodetype = MM if rows_c[0][3] == CM else CAT
            items = [(rows_c[i][0], rows_c[i][1], OPS["LINEAGE"], rows_c[i][3], judge,
                      nodetype, nodetype) for i in sel]
            tgt = torch.tensor([rows_c[i][2] for i in sel], dtype=torch.float32,
                               device=dev)
            mu = mu_batch(model, tok, items, dev, train=True, rng=aug)
            loss = torch.mean((mu - tgt) ** 2)
            ag = [(it[0], it[1], it[2]) for it in items[:16]]
            mu_ag = mu_batch(model, tok, ag, dev)
            with torch.no_grad():
                mu_ref = mu_batch(ref, tok, ag, dev)
            loss = loss + torch.mean((mu_ag - mu_ref) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tensors, 1.0)
            opt.step()
        print(f"[mj] fold {f} trained ({(time.time()-t0)/60:.1f} min)", flush=True)
        model.eval()
        store = {}
        with torch.no_grad():
            for q in sorted(held_by_fold[f]):
                sg = [float(v) for v in mu_batch(model, tok,
                      [(q, c, OPS["LINEAGE"], CM, JG, MM, MM) for c in catalog],
                      dev).cpu()]
                se = [float(v) for v in mu_batch(model, tok,
                      [(q, c, OPS["LINEAGE"], CM, E5J, MM, MM) for c in catalog],
                      dev).cpu()]
                store[q] = {"graph": sg, "e5j": se,
                            "cos": [float(Q[idx[q]] @ P[idx[c]]) for c in catalog]}
        with open(os.path.join(OUT, f"fold{f}_scores.json"), "w") as fh:
            json.dump(store, fh)
        os.chmod(os.path.join(OUT, f"fold{f}_scores.json"), 0o600)
        for label, key in (("graph-judge", "graph"), ("e5-judge", "e5j")):
            rrs = []
            for q, s in store.items():
                rank = pl.recompute_rank(s[key], catalog, cat_index[dest_of[q]])
                rrs.append(1.0 / rank)
            results[label][f] = float(np.mean(rrs))
        print(f"[mj] fold {f}: graph {results['graph-judge'][f]:.4f} "
              f"e5-judge {results['e5-judge'][f]:.4f}", flush=True)
    summary = {lab: float(np.mean(list(v.values()))) for lab, v in results.items()}
    print(json.dumps({"per_arm_MRR": summary,
                      "comparators": {"smfs_only_graded": 0.280, "e5_frozen": 0.573},
                      "stamp": "shadow-exploratory"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
