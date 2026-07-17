#!/usr/bin/env python3
"""Filing v1 fine-tune: onboard the Pearltrees corpus on the fused cheap-judge targets.

The champion recipe (fine_tune_fused_head_luna.py) deployed on the first live Pearltrees campaign:
warm-start the campaign-independent name-conditioned base (AFTER migrate_name_tables.py --tables
ops,corpora so the corpus onboards BY CARD at zero residual), 800 steps, lr 5e-4, last encoder
layer + readout + judge/corpus name residuals + nodetype embedding trainable, agnostic-anchor loss,
grad clip 1.0.

Row types (7-tuples — positions 5/6 carry NODETYPE[pearltrees_collection] so the zero-init nodetype
embedding finally trains on real collection rows):
  (x, y, HIER/SYM, pearltrees, gpt-5.5-low)   → 5.5 labels        (overlap + routed rows only)
  (x, y, HIER/SYM, pearltrees, gpt-5.6-luna)  → raw luna labels   (channel heads keep raw supervision)
  (x, y, HIER,     pearltrees, graph)         → hit_prob walk d
  (x, y, HIER/SYM, pearltrees, kalman-fused)  → fused posterior   (the distillation)
  (x, y, LINEAGE,  pearltrees, graph)         → 0.85^(hop−1) on principal rows; 0.0 on pt_rand rows
LINEAGE targets are the graded single-path decay (DESIGN_mindmap_lineage §3b) WITHOUT the
lca_depth_frac factor — the partial multi-record DAG has no canonical depth; documented deviation.
sib/cous rows are excluded from LINEAGE (ambiguous filing distance), retained in HIER/SYM channels.

Split: node-disjoint over campaign pairs (seed 0). Train rows come only from the train side; the
held overlap rows give the honest head-vs-5.5 readout (within-stratum decomposition).

  python3 fine_tune_pearltrees_filing.py --out model_pt_filing.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_within_stratum import decompose
from fine_tune_channel_heads import load_expanded, mu_batch
from mu_attention import CORPORA, JUDGES, NODETYPE, OPS
from node_disjoint_eval import node_disjoint_pair_split
from run_pearltrees_fusion import load_pearltrees_campaign

ROOT = os.path.dirname(os.path.abspath(__file__))
FUSED_TARGETS = "/tmp/mu_data/pt_fused_targets.tsv"
ROUTED_55 = "/tmp/mu_data/pt_conflict_scored_55.tsv"
PT = NODETYPE["pearltrees_collection"]
LINEAGE_DECAY = 0.85


def load_fused_targets(path=FUSED_TARGETS):
    out = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < len(header):
                continue
            out[(c[col["node"]], c[col["root"]])] = {
                "post": (float(c[col["post_D"]]), float(c[col["post_S"]])),
                "tag": c[col["tag"]],
                "hop": int(c[col["hop"]]),
                "y55": (float(c[col["y55_D"]]), float(c[col["y55_S"]])) if c[col["y55_D"]] else None,
                "routed": c[col["routed"]] == "1",
            }
    return out


def load_routed_labels(path=ROUTED_55):
    if not os.path.exists(path):
        return {}
    from eval_luna_transfer import load_luna
    p, d, s = load_luna(path)
    return {pair: (d[i], s[i]) for i, pair in enumerate(p)}


def group_of(tag):
    return "principal" if tag.startswith("principal") else tag


def load_with_lineage_ops(ckpt, dev="cpu"):
    """load_expanded, then grow the op tables 4→len(OPS) so LINEAGE/LINEAGE_RANK are addressable.

    New op_emb rows are zero-init fresh rows (the train_lineage.py precedent); new op_name card
    rows get their e5 card (name prior) with a ZERO residual — the onboarding story."""
    import math

    m, cfg_ = load_expanded(ckpt, dev=dev)
    n_old = m.op_emb.weight.shape[0]
    if n_old < len(OPS):
        grow = len(OPS) - n_old
        d_model = m.op_emb.weight.shape[1]
        with torch.no_grad():
            m.op_emb.weight.data = torch.cat(
                [m.op_emb.weight.data, torch.zeros(grow, d_model)], 0)
            # per-operator readout heads grow with FRESH rows (constructor init — train_lineage's
            # "copy old rows, leave new rows fresh"); deterministic under the caller's torch seed
            m.readout_w.data = torch.cat(
                [m.readout_w.data, torch.randn(grow, d_model) * (1.0 / math.sqrt(d_model))], 0)
            m.readout_b.data = torch.cat([m.readout_b.data, torch.zeros(grow)], 0)
        if getattr(m, "op_name", None) is not None:
            from judge_cards import op_card_e5
            E, _ = op_card_e5()
            with torch.no_grad():
                m.op_name.name_e5 = E[: len(OPS)].to(m.op_name.name_e5.dtype)
                m.op_name.resid.weight.data = torch.cat(
                    [m.op_name.resid.weight.data,
                     torch.zeros(grow, m.op_name.resid.weight.shape[1])], 0)
    return m, cfg_


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod_namecond_full.pt"),
                    help="warm start; must carry judge+op+corpus name tables (migrate_name_tables.py)")
    ap.add_argument("--out", default=os.path.join(ROOT, "model_pt_filing.pt"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args(argv)
    dev = "cpu"
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    augment_rng = np.random.default_rng(a.seed + 1)

    ds = load_pearltrees_campaign()
    fused = load_fused_targets()
    routed = load_routed_labels()
    pairs, tags, luna, d = ds["pairs"], ds["tags"], ds["luna"], ds["d"]
    ov_set = {tuple(pairs[i]) for i in ds["overlap_idx"]}
    y55_by = {tuple(pairs[k]): ds["y55"][j] for j, k in enumerate(ds["overlap_idx"])}
    y55_by.update(routed)                       # routed labels train the 5.5 channel, never the blocks
    print(f"campaign rows {len(pairs)}; 5.5-labeled rows {len(y55_by)} "
          f"(overlap {len(ov_set)}, routed {len(routed)})")

    strata = [group_of(t) for t in tags]
    split = node_disjoint_pair_split(pairs, a.split_seed, strata=strata)
    tr, he = split.train, split.held
    print(f"node-disjoint split (seed {a.split_seed}): train {len(tr)} / held {len(he)} / "
          f"cross {len(split.cross)} (cross discarded)")

    model, cfg = load_with_lineage_ops(a.ckpt, dev=dev)
    assert model.judge_name is not None, "checkpoint must be name-migrated"
    ref, _ = load_with_lineage_ops(a.ckpt, dev=dev)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for mod in (model.judge_name, getattr(model, "corpus_name", None), getattr(model, "op_name", None)):
        if mod is not None:
            mod.W.weight.requires_grad = True
            mod.resid.weight.requires_grad = True
            trainable += [mod.W.weight, mod.resid.weight]
    last = model.encoder.layers[-1]
    for p in last.parameters():
        p.requires_grad = True
    model.readout_w.requires_grad = True
    model.readout_b.requires_grad = True
    model.nodetype_emb.weight.requires_grad = True
    trainable += [model.readout_w, model.readout_b, model.nodetype_emb.weight] + list(last.parameters())
    opt = torch.optim.Adam(trainable, lr=a.lr)
    print(f"trainable tensors: {len(trainable)} "
          f"({sum(p.numel() for p in trainable)} params of {sum(p.numel() for p in model.parameters())})")

    C, rows = CORPORA["pearltrees"], []
    for i in tr:
        x, y = pairs[i]
        info = fused.get((x, y))
        if info is None:
            continue
        base = (x, y)
        y55 = y55_by.get((x, y))
        if y55 is not None:
            rows.append(((*base, OPS["HIER"], C, JUDGES["gpt-5.5-low"], PT, PT), y55[0]))
            rows.append(((*base, OPS["SYM"], C, JUDGES["gpt-5.5-low"], PT, PT), y55[1]))
        rows.append(((*base, OPS["HIER"], C, JUDGES["gpt-5.6-luna"], PT, PT), luna[i, 0]))
        rows.append(((*base, OPS["SYM"], C, JUDGES["gpt-5.6-luna"], PT, PT), luna[i, 1]))
        rows.append(((*base, OPS["HIER"], C, JUDGES["graph"], PT, PT), d[i]))
        rows.append(((*base, OPS["HIER"], C, JUDGES["kalman-fused"], PT, PT), info["post"][0]))
        rows.append(((*base, OPS["SYM"], C, JUDGES["kalman-fused"], PT, PT), info["post"][1]))
        if tags[i].startswith("principal"):
            rows.append(((*base, OPS["LINEAGE"], C, JUDGES["graph"], PT, PT),
                         LINEAGE_DECAY ** (info["hop"] - 1)))
        elif tags[i] == "pt_rand":
            rows.append(((*base, OPS["LINEAGE"], C, JUDGES["graph"], PT, PT), 0.0))
    print(f"{len(rows)} train rows (5.5 + luna + graph + fused + LINEAGE channels)")

    model.train()
    for step in range(1, a.steps + 1):
        sel = rng.choice(len(rows), size=min(a.bs, len(rows)), replace=False)
        items = [rows[j][0] for j in sel]
        tgt = torch.tensor([rows[j][1] for j in sel], dtype=torch.float32, device=dev)
        mu = mu_batch(model, ds["tok"], items, dev, train=True, rng=augment_rng)
        loss = torch.mean((mu - tgt) ** 2)
        ag_items = [(it[0], it[1], it[2]) for it in items]
        mu_ag = mu_batch(model, ds["tok"], ag_items, dev)
        with torch.no_grad():
            mu_ref = mu_batch(ref, ds["tok"], ag_items, dev)
        loss = loss + a.anchor_weight * torch.mean((mu_ag - mu_ref) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"step {step:4d} loss {loss.item():.4f}")

    model.eval()
    he_ov = [i for i in he if tuple(pairs[i]) in y55_by and tuple(pairs[i]) in ov_set]
    print(f"\nheld overlap rows for the honest readout: {len(he_ov)}")
    if he_ov:
        groups = [group_of(tags[i]) for i in he_ov]
        y55h = np.array([y55_by[tuple(pairs[i])] for i in he_ov])
        mus = {}
        with torch.no_grad():
            for key, judge, op in [("fused_D", "kalman-fused", "HIER"), ("fused_S", "kalman-fused", "SYM"),
                                   ("luna_D", "gpt-5.6-luna", "HIER"), ("luna_S", "gpt-5.6-luna", "SYM"),
                                   ("llm_D", "gpt-5.5-low", "HIER"), ("llm_S", "gpt-5.5-low", "SYM")]:
                items = [(pairs[i][0], pairs[i][1], OPS[op], C, JUDGES[judge], PT, PT) for i in he_ov]
                mus[key] = np.array(mu_batch(model, ds["tok"], items, dev).cpu())
        print("held-overlap heads vs the 5.5 labels — pooled / between / WITHIN:")
        for label, mu, tgt in [
            ("fused vs 5.5 D", mus["fused_D"], y55h[:, 0]),
            ("luna  vs 5.5 D", mus["luna_D"], y55h[:, 0]),
            ("5.5h  vs 5.5 D", mus["llm_D"], y55h[:, 0]),
            ("fused vs 5.5 S", mus["fused_S"], y55h[:, 1]),
            ("luna  vs 5.5 S", mus["luna_S"], y55h[:, 1]),
            ("5.5h  vs 5.5 S", mus["llm_S"], y55h[:, 1]),
        ]:
            pooled, between, within, _ = decompose(mu, tgt, groups)
            print(f"  {label:16s}: {pooled:+.3f} / {between:+.3f} / {within:+.3f}")

    he_lin = [i for i in he if tags[i].startswith("principal")]
    if he_lin:
        with torch.no_grad():
            items = [(pairs[i][0], pairs[i][1], OPS["LINEAGE"], C, JUDGES["graph"], PT, PT) for i in he_lin]
            mu_lin = np.array(mu_batch(model, ds["tok"], items, dev).cpu())
        tgt_lin = np.array([LINEAGE_DECAY ** (fused[tuple(pairs[i])]["hop"] - 1) for i in he_lin])
        print(f"held LINEAGE (principal rows, n={len(he_lin)}): "
              f"corr {np.corrcoef(mu_lin, tgt_lin)[0, 1]:+.3f}, MAE {np.abs(mu_lin - tgt_lin).mean():.3f}")

    torch.save({"state": model.state_dict(), "cfg": cfg}, a.out)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
