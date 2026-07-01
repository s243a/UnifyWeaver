#!/usr/bin/env python3
"""train_lineage.py — add a LINEAGE operator that scores μ(node | hierarchical-list-PATH), fine-tune with replay.

The new operator (DESIGN_model_applications.md, "B"): LINEAGE is the *undifferentiated* generalization of ELEM
(element-of) + WIKI (subcategory) — it scores how well a bookmark fits an ORDERED FOLDER PATH (the materialized
`/id/id` line + indented root→leaf titles, the same `target_text` the old filer embedded). Per the design decision:
  * NOT hand-initialised from ELEM/WIKI — a FRESH operator row; the model LEARNS its relation to the others by
    training with REPLAY (the generalization emerges, it isn't assumed).
  * warm-start everything else from `--ckpt` (row-copy so op_emb/readout grow 4→5, old rows preserved).
  * masking = ID-dropout (sample a `/*`-wildcarded path variant) + path-PREFIX dropout (train on ancestor prefixes
    ⇒ free depth-placement at inference). Variants are PRECOMPUTED into the e5 cache (pay the re-embed once, not
    per step) — the materialised-id line gives unique anchors, ID-dropout stops it becoming a memorisation crutch.
  * replay = interleave ELEM(bookmark|folder-title) batches so filing-elem stays sharp and elem↔lineage is learned.

  python3 train_lineage.py --ckpt model_filing.pt --trees ../../.local/data/pearltrees_api/trees \
      --paths ../../.local/data/api_tree_paths_v8.jsonl --save model_lineage.pt --steps 500
"""
import argparse, collections, json, random, os, torch
import torch.nn.functional as F
from mu_attention import build_e5_tables, Tokenizer, MuAttention, OPS
from eval_filing import load_filing, metrics

LINEAGE = len(OPS)                                                  # new op index in the grown (n_ops+1) model


def load_grow(ckpt, dev, extra=1):
    """Load a checkpoint into a model with `extra` MORE operators; copy overlapping rows, leave new rows fresh."""
    ck = torch.load(ckpt, weights_only=False); sd = ck["state"]
    cfg = ck.get("cfg", {"d_model": 384, "heads": 4, "layers": 3})
    g = lambda k, d: sd[k].shape[0] if k in sd else d
    model = MuAttention(d_model=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
                        n_ops=g("op_emb.weight", len(OPS)) + extra, n_corpus=g("corpus_emb.weight", 2),
                        n_judge=g("judge_emb.weight", 2), n_nodetype=g("nodetype_emb.weight", 4)).to(dev)
    new = model.state_dict()
    for k, v in sd.items():
        if k not in new:
            continue
        if new[k].shape == v.shape:
            new[k] = v
        else:                                                       # n_ops-sized (op_emb.weight, readout_w, readout_b): copy old rows
            new[k][:v.shape[0]] = v
    model.load_state_dict(new, strict=False)
    return model, cfg


def path_variants(target_text, max_drop=2):
    """target_text = '/id..\\n- t0\\n  - t1\\n ...' → list of (idmode, drop, text) variants: {full,wild id}×{prefix}."""
    lines = target_text.split("\n")
    id_line = lines[0] if lines and lines[0].startswith("/") else None
    titles = lines[1:] if id_line is not None else lines            # indented '- title' lines, root→leaf
    out = []
    for drop in range(min(max_drop, max(0, len(titles) - 1)) + 1):  # drop 0..max_drop deepest levels (keep ≥1)
        kept = titles[:len(titles) - drop]
        for idmode, idl in (("full", id_line), ("wild", "/*")):
            body = "\n".join(([idl] if idl is not None else []) + kept)
            out.append((idmode, drop, body))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--trees", required=True); ap.add_argument("--paths", required=True)
    ap.add_argument("--save", default=None); ap.add_argument("--min-bm", type=int, default=3)
    ap.add_argument("--eval-frac", type=float, default=0.3); ap.add_argument("--max-eval", type=int, default=400)
    ap.add_argument("--steps", type=int, default=500); ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--replay", type=float, default=1.0,
                    help="weight on the ELEM(bookmark|folder-title) replay loss")
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", default="/tmp/lineage_e5.pt")
    ap.add_argument("--eval-only", action="store_true", help="load --save model and eval only (no training)")
    a = ap.parse_args(); dev = torch.device(a.device); rng = random.Random(a.seed)

    queries, cand = load_filing(a.trees, a.min_bm)                  # [(bm_text, folder_tid)], {folder_tid: title}
    path_of, pids_of = {}, {}                                       # folder_tid -> target_text ; -> path_ids (root→leaf)
    for line in open(a.paths, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("tree_id") and r.get("target_text"):
            path_of[str(r["tree_id"])] = r["target_text"]
            pids_of[str(r["tree_id"])] = [str(x) for x in (r.get("path_ids") or [])]
    queries = [(b, str(f)) for b, f in queries if str(f) in path_of]   # keep bookmarks whose folder has a path
    cand = {str(f): t for f, t in cand.items() if str(f) in path_of}   # normalise folder ids to str (match path_of)
    print(f"[DATA] {len(cand)} folders w/ paths, {len(queries)} bookmarks (of harvested set)")

    # per-folder bookmark-holdout split (fixed seed) — folders are a stable taxonomy, held-out BOOKMARKS never trained
    byf = collections.defaultdict(list)
    for q in queries:
        byf[q[1]].append(q)
    eval_q, pool = [], []
    for fid, qs in byf.items():
        qs = qs[:]; rng.shuffle(qs); k = max(1, int(a.eval_frac * len(qs))) if len(qs) >= 2 else 0
        eval_q += qs[:k]; pool += qs[k:]
    rng.shuffle(eval_q); eval_q = eval_q[:a.max_eval]
    print(f"[SPLIT] {len(pool)} train bookmarks, {len(eval_q)} held-out eval")

    # ── build e5 tables: bookmarks (query) + folder titles + PRECOMPUTED path variants (passage). One embed pass. ──
    bm_list = eval_q + pool
    bm_key = [f"B:{i}" for i in range(len(bm_list))]
    fold_ids = list(cand)
    variants = {f: path_variants(path_of[f]) for f in fold_ids}     # folder -> [(idmode,drop,text)]
    texts, names = {}, []
    for i, (bt, _) in enumerate(bm_list):
        texts[bm_key[i]] = bt; names.append(bm_key[i])
    for f in fold_ids:
        texts[f"F:{f}"] = cand[f]; names.append(f"F:{f}")           # folder title (ELEM replay target)
        for vi, (_, _, vtext) in enumerate(variants[f]):
            k = f"P:{f}:{vi}"; texts[k] = vtext; names.append(k)     # path variant (LINEAGE target)
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=a.cache, texts=texts, device=a.device)
    tok = Tokenizer(qtbl, ptbl, idx, parents={}, deg={})
    full_variant = {f: next(vi for vi, (im, dr, _) in enumerate(variants[f]) if im == "full" and dr == 0) for f in fold_ids}

    model, cfg = load_grow(a.save if a.eval_only else a.ckpt, dev, extra=0 if a.eval_only else 1)
    n_ops = model.op_emb.weight.shape[0]
    print(f"[MODEL] {'eval-only ' + os.path.basename(a.save) if a.eval_only else f'grew 4→{n_ops} ops, warm-start ' + os.path.basename(a.ckpt)} (LINEAGE={LINEAGE}) {cfg}")
    OW = lambda op: torch.zeros(1, n_ops, device=dev).index_fill_(1, torch.tensor([op], device=dev), 1.0)
    ow_lin, ow_elem = OW(LINEAGE), OW(OPS["ELEM"])

    bm_folder = [f for _, f in bm_list]
    train_idx = list(range(len(eval_q), len(bm_list)))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    trng = random.Random(a.seed + 1)

    def contrastive(bkidx, passages, ow):                           # B×B μ(bm_i | passage_j), same-folder = positive
        bk = [bm_key[i] for i in bkidx]
        items = [(bk[i], passages[j], 0) for i in range(len(bk)) for j in range(len(passages))]
        b = tok.build(items, train=False); b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        mu = model(**b, op_weights=ow.expand(len(items), n_ops)).view(len(bk), len(passages))
        fi = torch.tensor([hash(bm_folder[i]) for i in bkidx], device=dev)
        target = (fi[:, None] == fi[None, :]).float()
        bce = F.binary_cross_entropy(mu.clamp(1e-6, 1 - 1e-6), target, reduction="none")
        return bce[target > 0.5].mean() + bce[target <= 0.5].mean()

    model.train()
    for step in ([] if a.eval_only else range(a.steps)):
        bkidx = trng.sample(train_idx, min(a.bs, len(train_idx)))
        # LINEAGE: passage = a SAMPLED path variant (id-dropout + prefix-dropout) of each bookmark's folder
        lin_pass = [f"P:{bm_folder[i]}:{trng.randrange(len(variants[bm_folder[i]]))}" for i in bkidx]
        loss = contrastive(bkidx, lin_pass, ow_lin)
        if a.replay > 0:                                            # ELEM replay: passage = folder TITLE
            loss = loss + a.replay * contrastive(bkidx, [f"F:{bm_folder[i]}" for i in bkidx], ow_elem)
        opt.zero_grad(); loss.backward(); opt.step()

    # ── eval on held-out bookmarks: rank candidate folders; LINEAGE uses each folder's FULL path ──
    model.eval()
    ev_idx = list(range(len(eval_q)))
    truepos = [fold_ids.index(bm_folder[i]) for i in ev_idx]

    @torch.no_grad()
    def score(passkeys, ow):
        S = []
        bk = [bm_key[i] for i in ev_idx]
        for i in range(len(bk)):
            items = [(bk[i], pk, 0) for pk in passkeys]
            b = tok.build(items, train=False); b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            S.append(model(**b, op_weights=ow.expand(len(passkeys), n_ops)).cpu().tolist())
        return S
    def ranks(S):
        return [1 + sum(1 for j, s in enumerate(row) if s > row[truepos[r]] or (s == row[truepos[r]] and j < truepos[r]))
                for r, row in enumerate(S)]
    def lcp(a_ids, b_ids):                                          # shared ancestor depth from root
        n = 0
        for x, y in zip(a_ids, b_ids):
            if x != y:
                break
            n += 1
        return n
    def top1(S, r): return max(range(len(S[r])), key=lambda j: S[r][j])
    def path_overlap(S, subset=None):                              # mean fraction of TRUE ancestor lineage recovered by top-1
        vals = []
        for r in (range(len(S)) if subset is None else subset):
            pred = fold_ids[top1(S, r)]; true = bm_folder[ev_idx[r]]
            tp = pids_of.get(true, []);  pp = pids_of.get(pred, [])
            if tp:
                vals.append(lcp(pp, tp) / len(tp))
        return sum(vals) / len(vals) if vals else 0.0
    lin_keys = [f"P:{f}:{full_variant[f]}" for f in fold_ids]       # full path per folder
    fol_keys = [f"F:{f}" for f in fold_ids]
    qn = qtbl[[idx[bm_key[i]] for i in ev_idx]]; fn = ptbl[[idx[k] for k in fol_keys]]
    C = (qn @ fn.T).cpu()
    e5_ranks = [1 + int((C[r] > C[r][truepos[r]]).sum().item()) for r in range(C.shape[0])]
    S_elem, S_lin = score(fol_keys, ow_elem), score(lin_keys, ow_lin)
    S_e5 = C.tolist()
    elem_miss = [r for r in range(len(S_elem)) if top1(S_elem, r) != truepos[r]]   # FIXED hard subset (paired)
    print(f"\n  [HELD-OUT n={len(ev_idx)}]  {'ranker':12} {'recall@1':>9} {'MRR':>7} {'overlap(all)':>13} "
          f"{'overlap|elem-miss':>18}   [paired hard subset n={len(elem_miss)}]")
    for nm, R, S in (("e5-cos", e5_ranks, S_e5), ("mu-elem", ranks(S_elem), S_elem), ("mu-lineage", ranks(S_lin), S_lin)):
        m = metrics(R)
        print(f"  {'':13}{nm:12} {m['recall@1']:9.3f} {m['MRR']:7.3f} {path_overlap(S):13.3f} {path_overlap(S, elem_miss):18.3f}")
    print("  (overlap|elem-miss = ancestor-lineage recovered on the SAME hard cases where the leaf-specialist mu-elem")
    print("   fails — the paired graceful-degradation test: when the leaf can't be nailed, who stays in the right branch?)")

    if a.save:
        torch.save({"state": model.state_dict(), "cfg": cfg, "ops_extra": ["LINEAGE"]}, a.save)
        print(f"\n[SAVE] lineage model (LINEAGE op @ {LINEAGE}) → {a.save}")


if __name__ == "__main__":
    main()
