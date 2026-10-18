#!/usr/bin/env python3
"""SM-FS LINEAGE onboarding fine-tune — POSITIVE-ONLY DIAGNOSTIC, not ranking evidence.

Trains μ's LINEAGE head on the certified public-only SM-FS bundle (sm_fs_freeze.py v3:
content-bound ledger/targets/manifest, owner exclusions sealed, strict lineage-family isolation).
Row contract per gpt-5.6-sol (2026-07-25):

    ((map_path, ancestor_path, LINEAGE, mindmap, graph, mindmap_node, mindmap_node), 0.85^(hop-1))

- Exact PATHS are identities; TITLES are embedding text only (build_e5_tables texts= override) —
  duplicate titles stay distinct.
- Frozen reference = deepcopy of the exact warm-started model (never a second load).
- Trainable: per-identity name residuals + last encoder layer + readouts + nodetype embedding;
  shared name-transform W matrices stay frozen.
- Agnostic three-field readouts anchored to the frozen reference.
- Per-row weight 1/ancestors_for_map so deep paths don't dominate.

RIGOR STATUS (sol's two corrections, accepted):
  1. The bundle contains POSITIVE ancestor targets only. Without separately frozen non-ancestor
     negatives this run is an onboarding diagnostic — it must not be cited as ranking gain.
  2. Warm-starting from model_pt_filing_lin.pt and later scoring Pearltrees measures RETENTION,
     not transfer. No Pearltrees scoring happens here; the retention-vs-transfer protocol is to
     be frozen first (paired arms from a pre-Pearltrees base for genuine transfer).
The 1,481 reserved rows are untouched and never drive checkpoint or hyperparameter selection;
the held slice below is a within-explore, map-level validation cut (descriptive only).

  python3 fine_tune_sm_fs.py --out model_sm_fs_lin.pt
"""
import argparse
import copy
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fine_tune_channel_heads import mu_batch
from fine_tune_pearltrees_filing import load_with_lineage_ops
from mu_attention import (CORPORA, E5_REVISION, JUDGES, NODETYPE, OPS,
                          Tokenizer, build_e5_tables)
from sm_fs_freeze import verify_private_bundle

ROOT = os.path.dirname(os.path.abspath(__file__))
MM = NODETYPE["mindmap_node"]


def load_targets(bundle):
    path = os.path.join(bundle, "lineage_fs_targets.tsv")
    header, rows = {}, []
    cols = None
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("#"):
            parts = ln[1:].strip().split("\t")
            if parts[0] == "map_path":
                cols = parts
            elif len(parts) == 2:
                header[parts[0]] = parts[1]
            continue
        v = dict(zip(cols, ln.rstrip("\n").split("\t")))
        rows.append(v)
    assert header.get("process_expression") == "lineage(fs,decay=0.85)", header
    assert header.get("training_privacy_policy") == "public-only", header
    assert header.get("e5_revision") == E5_REVISION, "bundle e5 revision mismatch"
    return header, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.expanduser("~/mu_data/sm_fs_v3"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_pt_filing_lin.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "model_sm_fs_lin.pt"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--e5-cache", default=os.path.expanduser("~/mu_data/sm_fs_train_e5.pt"))
    a = ap.parse_args(argv)

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    augment_rng = np.random.default_rng(a.seed + 1)

    verify_private_bundle(a.bundle)                       # integrity: hashes + bindings
    header, t_rows = load_targets(a.bundle)
    man_sha = hashlib.sha256(open(os.path.join(a.bundle, "manifest.json"), "rb").read()
                             ).hexdigest()[:16]
    print(f"bundle verified ({a.bundle}, manifest {man_sha}); {len(t_rows)} target rows; dev {dev}")
    print("STATUS: positive-only onboarding diagnostic — not ranking evidence (see docstring)")

    # identities = exact paths; titles = embedding text
    texts = {}
    for r in t_rows:
        texts[r["map_path"]] = r["map_title"]
        texts[r["ancestor_path"]] = r["ancestor_title"]
    names = sorted(texts)
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=a.e5_cache, batch_size=128,
                                      texts=texts, model_revision=E5_REVISION)
    tok = Tokenizer(qtbl, ptbl, idx, {}, {})

    # map-level within-explore validation cut (descriptive)
    maps = sorted({r["map_path"] for r in t_rows})
    val_maps = set(rng.choice(maps, size=max(1, int(len(maps) * a.val_frac)), replace=False))
    n_anc = {}
    for r in t_rows:
        n_anc[r["map_path"]] = n_anc.get(r["map_path"], 0) + 1
    C, J = CORPORA["mindmap"], JUDGES["graph"]
    train_rows, val_rows = [], []
    for r in t_rows:
        item = (r["map_path"], r["ancestor_path"], OPS["LINEAGE"], C, J, MM, MM)
        rec = (item, float(r["target"]), 1.0 / n_anc[r["map_path"]])
        (val_rows if r["map_path"] in val_maps else train_rows).append(rec)
    print(f"maps {len(maps)} (val {len(val_maps)}); rows train {len(train_rows)} / "
          f"val {len(val_rows)}; weights 1/ancestors_for_map")

    ckpt_sha = hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest()
    model, cfg = load_with_lineage_ops(a.ckpt, dev=dev)
    assert model.judge_name is not None, "checkpoint must be name-migrated"
    # step-0 baseline of the exact initialization (the gap sol's #4005 audit exposed:
    # the pilot receipt neither hashed nor evaluated its warm start)
    model.eval()
    with torch.no_grad():
        v_items = [r_[0] for r_ in val_rows]
        v_tg = np.array([r_[1] for r_ in val_rows])
        mu0 = np.array(mu_batch(model, tok, v_items, dev).cpu())
    step0_corr = float(np.corrcoef(mu0, v_tg)[0, 1])
    print(f"init {os.path.basename(a.ckpt)} sha {ckpt_sha[:16]}; "
          f"step-0 val corr {step0_corr:+.3f}")
    ref = copy.deepcopy(model)                            # exact warm-started model, per sol
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for mod in (model.judge_name, getattr(model, "corpus_name", None),
                getattr(model, "op_name", None)):
        if mod is not None:
            mod.resid.weight.requires_grad = True
            trainable += [mod.resid.weight]
    last = model.encoder.layers[-1]
    for p in last.parameters():
        p.requires_grad = True
    model.readout_w.requires_grad = True
    model.readout_b.requires_grad = True
    model.nodetype_emb.weight.requires_grad = True
    trainable += [model.readout_w, model.readout_b,
                  model.nodetype_emb.weight] + list(last.parameters())
    opt = torch.optim.Adam(trainable, lr=a.lr)
    print(f"trainable tensors {len(trainable)} "
          f"({sum(p.numel() for p in trainable)}/{sum(p.numel() for p in model.parameters())} params)")

    model.train()
    for step in range(1, a.steps + 1):
        sel = rng.choice(len(train_rows), size=min(a.bs, len(train_rows)), replace=False)
        items = [train_rows[j][0] for j in sel]
        tgt = torch.tensor([train_rows[j][1] for j in sel], dtype=torch.float32, device=dev)
        w = torch.tensor([train_rows[j][2] for j in sel], dtype=torch.float32, device=dev)
        mu = mu_batch(model, tok, items, dev, train=True, rng=augment_rng)
        loss = torch.sum(w * (mu - tgt) ** 2) / torch.sum(w)
        ag_items = [(it[0], it[1], it[2]) for it in items]
        mu_ag = mu_batch(model, tok, ag_items, dev)
        with torch.no_grad():
            mu_ref = mu_batch(ref, tok, ag_items, dev)
        loss = loss + a.anchor_weight * torch.mean((mu_ag - mu_ref) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step:4d} loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        for name_, rows_ in (("train", train_rows), ("val(map-disjoint)", val_rows)):
            items = [r[0] for r in rows_]
            tg = np.array([r[1] for r in rows_])
            mu = np.array(mu_batch(model, tok, items, dev).cpu())
            print(f"{name_:18s}: corr {np.corrcoef(mu, tg)[0, 1]:+.3f}  "
                  f"MAE {np.abs(mu - tg).mean():.3f}  n={len(rows_)}")

    torch.save({"state": model.state_dict(), "cfg": cfg}, a.out)
    run = {"bundle": a.bundle, "manifest_sha16": man_sha,
           "init_ckpt_sha256": ckpt_sha, "step0_val_corr": step0_corr,
           "process_expression": header["process_expression"],
           "e5_revision": E5_REVISION, "ckpt": os.path.basename(a.ckpt),
           "steps": a.steps, "bs": a.bs, "lr": a.lr, "anchor_weight": a.anchor_weight,
           "seed": a.seed, "val_frac": a.val_frac,
           "status": "positive-only onboarding diagnostic; no Pearltrees scoring; "
                     "retention-vs-transfer protocol to be frozen before PT eval"}
    json.dump(run, open(a.out + ".run.json", "w"), indent=1)
    print(f"saved -> {a.out} (+ .run.json provenance)")


if __name__ == "__main__":
    main()
