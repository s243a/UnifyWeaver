#!/usr/bin/env python3
"""B1: train per-channel heads via the judge-embedding rows ONLY (frozen trunk).

The probe (`run_channel_heads_probe.py`) showed the judge tokens don't route and the S channel is missing.
This fine-tune asks the cleanest version of the question: **can the provenance token alone carry channel
routing?** — by freezing EVERYTHING except `judge_emb` and training channel-tagged rows:

  (node, root, HIER, enwiki, judge=gpt-5.5-low) → LLM D label
  (node, root, SYM,  enwiki, judge=gpt-5.5-low) → LLM S label      (the MISSING channel)
  (node, root, HIER, enwiki, judge=graph)       → walk hit_prob

Trunk non-degradation is guaranteed BY CONSTRUCTION: channel rows are built with p_mask_prov=0 (the agnostic
path never sees channel targets) and only judge_emb has gradients (3,456 trainable params of ~14M). The
checkpoint's 5-row judge table is expanded to 9 (rows 5-8 zero-init; JUDGES["gpt-5.5-low"]=6).

Acceptance test: re-run the probe on the saved checkpoint — routing must APPEAR (graph-conditioned HIER tracks
the walk; llm-conditioned SYM tracks S) while the agnostic readouts stay bit-identical.

  python3 fine_tune_channel_heads.py --steps 800 --out model_channel_heads.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emit_transitive_hops import hit_prob
from mu_attention import CORPORA, JUDGES, OPS, MuAttention, Tokenizer
from run_product_kalman_realdata import DATASETS
from sigma_hop_confirmatory import (
    FeatureGraphConfig,
    descendant_disjoint_split,
    load_e5_cache_and_filter,
    load_feature_graph,
    load_scored_pairs,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
NEW_KEYS = ("account_emb.weight", "prefix_emb.weight", "sym_struct_w", "struct_lambda", "struct_g", "struct_h",
            "prec_g", "prec_h", "c_dist", "c_mem_ceiling", "c_subcat", "c_elem")


def load_expanded(ckpt, n_judge=None, dev="cpu"):
    """build_model, but with judge_emb expanded to n_judge rows (old rows copied, new zero-init).
    n_judge=None → max(len(JUDGES), checkpoint rows), so old callers survive JUDGES growing and newer
    checkpoints never get truncated (size-mismatched keys crash even strict=False loads).
    Post-migration checkpoints (migrate_judge_names.py) carry judge_name.* keys → the model is built with
    the name-function pathway; NEW judge rows there get their card's e5 (name prior) + a ZERO residual —
    the onboarding story the migration exists for."""
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = dict(ck["state"]); cfg = ck.get("cfg", {"d_model": 384, "heads": 4, "layers": 3})
    old = sd["judge_emb.weight"]
    if n_judge is None:
        n_judge = max(len(JUDGES), old.shape[0])
    if old.shape[0] < n_judge:
        sd["judge_emb.weight"] = torch.cat([old, torch.zeros(n_judge - old.shape[0], old.shape[1])], 0)
    jn_e5 = sd.get("judge_name.name_e5")
    if jn_e5 is not None and jn_e5.shape[0] < n_judge:
        from judge_cards import judge_card_e5
        E, _ = judge_card_e5()                                # full JUDGES-order card table
        sd["judge_name.name_e5"] = jn_e5 = E[:n_judge]
        r = sd["judge_name.resid.weight"]
        sd["judge_name.resid.weight"] = torch.cat([r, torch.zeros(n_judge - r.shape[0], r.shape[1])], 0)
    sz = lambda k, d: sd[k].shape[0] if k in sd else d
    m = MuAttention(d_model=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
                    n_ops=sz("op_emb.weight", len(OPS)), n_corpus=sz("corpus_emb.weight", 2),
                    n_judge=n_judge, n_nodetype=sz("nodetype_emb.weight", 4),
                    judge_name_e5=jn_e5).to(dev)
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not unexp, f"unexpected keys: {unexp}"
    bad = [k for k in miss if not any(k.endswith(n) for n in NEW_KEYS)]
    assert not bad, f"missing keys: {bad}"
    return m, cfg


def load_dataset(name):
    cfg = DATASETS[name]
    pairs, hop, D, S = load_scored_pairs(cfg["score_in"], cfg["responses"], prefix="transitive_h")
    cache, idx, pairs, hop, D, S = load_e5_cache_and_filter(pairs, hop, D, S, cfg["e5_cache"])
    parents, _, deg, _ = load_feature_graph(FeatureGraphConfig(**cfg["graph"]))
    tok = Tokenizer(cache["query"], cache["passage"], idx, parents, deg)
    d = np.array([hit_prob(parents, x, y) for x, y in pairs])
    tr, he = descendant_disjoint_split(list(pairs), 0, held_frac=0.30)
    return dict(pairs=list(pairs), D=np.array(D), S=np.array(S), d=d, tok=tok, tr=tr, he=he)


CAMPAIGN_SCORED = "/tmp/mu_data/campaign_scored.tsv"
CAMPAIGN_E5_100K = "/tmp/mu_data/campaign_100k_e5.pt"


def load_campaign_datasets(campaign_scored=None):
    """The stratified campaign, split by corpus membership. Same ds-dict shape as load_dataset; strata tags
    kept so eval can slice by pair type (the S channel should now have VARIANCE on sib/cous rows).

    ``campaign_scored`` is explicit for provenance-sensitive runners; existing callers retain the canonical
    campaign default.
    """
    campaign_scored = CAMPAIGN_SCORED if campaign_scored is None else os.path.abspath(campaign_scored)
    DIRR = ["subcategory", "subtopic", "element_of", "super_category"]; SYMM = ["see_also", "assoc"]
    rows = []
    with open(campaign_scored, encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < len(header):
                continue
            D = max(float(c[col[f"mu[{r}]"]]) for r in DIRR)
            S = max(float(c[col[f"mu[{r}]"]]) for r in SYMM)
            rows.append((c[col["node"]], c[col["root"]], c[col["neighborhood"]], D, S))
    out = {}
    for name, e5 in (("exploratory", CAMPAIGN_E5_100K), ("fresh", DATASETS["fresh"]["e5_cache"])):
        parents, children, deg, _ = load_feature_graph(FeatureGraphConfig(**DATASETS[name]["graph"]))
        in_graph = set(parents) | {c for kids in children.values() for c in kids}
        sub = [r for r in rows if r[0] in in_graph and r[1] in in_graph]
        pairs = [(r[0], r[1]) for r in sub]
        hop = np.zeros(len(pairs))                            # dummy — campaign rows carry strata, not hops
        D = np.array([r[3] for r in sub]); S = np.array([r[4] for r in sub])
        cache, idx, pairs, hop, D, S = load_e5_cache_and_filter(pairs, hop, D, S, e5)
        kept = set(pairs)
        tags = [r[2] for r in sub if (r[0], r[1]) in kept]
        tok = Tokenizer(cache["query"], cache["passage"], idx, parents, deg)
        d = np.array([hit_prob(parents, x, y) for x, y in pairs])
        tr, he = descendant_disjoint_split(list(pairs), 0, held_frac=0.30)
        out[f"{name}-campaign"] = dict(pairs=list(pairs), D=D, S=S, d=d, tok=tok, tr=tr, he=he, tags=tags)
        print(f"{name}-campaign: {len(pairs)} scored pairs kept")
    return out


def channel_rows(ds, idxs):
    """(item5, target) rows for the three channels."""
    rows = []
    for i in idxs:
        x, y = ds["pairs"][i]
        rows.append(((x, y, OPS["HIER"], CORPORA["enwiki"], JUDGES["gpt-5.5-low"]), ds["D"][i]))
        rows.append(((x, y, OPS["SYM"], CORPORA["enwiki"], JUDGES["gpt-5.5-low"]), ds["S"][i]))
        rows.append(((x, y, OPS["HIER"], CORPORA["enwiki"], JUDGES["graph"]), ds["d"][i]))
    return rows


def mu_batch(model, tok, items, dev, train=False, rng=None):
    b = tok.build(items, train=train, rng=rng, p_mask_prov=0.0)
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    return model(**b)


def eval_channels(model, ds, dev):
    """held-out corr per (conditioning, channel) — the routing acceptance metric."""
    model.eval()
    out = {}
    with torch.no_grad():
        for cname, judge, op, tgt in [("llm-D", "gpt-5.5-low", "HIER", ds["D"]),
                                      ("llm-S", "gpt-5.5-low", "SYM", ds["S"]),
                                      ("graph-d", "graph", "HIER", ds["d"])]:
            items = [(ds["pairs"][i][0], ds["pairs"][i][1], OPS[op], CORPORA["enwiki"], JUDGES[judge])
                     for i in ds["he"]]
            mu = np.array(mu_batch(model, ds["tok"], items, dev).cpu())
            t = tgt[ds["he"]]
            out[cname] = float(np.corrcoef(mu, t)[0, 1]) if mu.std() > 1e-9 else 0.0
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "model_channel_heads.pt"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--unfreeze-last", action="store_true",
                    help="B1b: also train the LAST encoder layer + readout (capacity for the missing S channel); "
                         "trunk honesty then enforced by the agnostic-anchor loss instead of by construction")
    ap.add_argument("--data", choices=["multihop", "campaign"], default="multihop",
                    help="campaign = the 2,000-pair stratified set (B1-retry: the S channel gets variance)")
    ap.add_argument("--anchor-weight", type=float, default=1.0,
                    help="B1b: weight of the agnostic-anchor distillation loss (readouts on 3-tuple rows must "
                         "match the frozen reference)")
    ap.add_argument("--resid-weight", type=float, default=0.0,
                    help="name-cond checkpoints only: L2 pull of the per-judge residuals toward 0 (§6.5). "
                         "Default 0: measured at 1e-2 it cost fresh-S within-stratum (-0.08) with no "
                         "transfer gain (REPORT_judge_name_migration.md)")
    a = ap.parse_args()
    dev = "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    model, cfg = load_expanded(a.ckpt, dev=dev)
    ref = None
    for p in model.parameters():
        p.requires_grad = False
    if model.judge_name is not None:
        # post-migration: judge condition = W·e5(card) + r — train the translation + residuals
        # (‖r‖ regularized toward 0 below, so the name prior stays the default; §6.5)
        model.judge_name.W.weight.requires_grad = True
        model.judge_name.resid.weight.requires_grad = True
        trainable = [model.judge_name.W.weight, model.judge_name.resid.weight]
    else:
        model.judge_emb.weight.requires_grad = True
        trainable = [model.judge_emb.weight]
    if a.unfreeze_last:
        ref, _ = load_expanded(a.ckpt, dev=dev)   # frozen reference for the anchor loss
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        last = model.encoder.layers[-1]
        for p in last.parameters():
            p.requires_grad = True
        model.readout_w.requires_grad = True
        model.readout_b.requires_grad = True
        trainable += list(last.parameters()) + [model.readout_w, model.readout_b]
    opt = torch.optim.Adam(trainable, lr=a.lr)
    n_tr = sum(p.numel() for p in trainable)
    cond = "judge_name (W + resid)" if model.judge_name is not None else "judge_emb"
    print(f"trainable params: {n_tr} ({cond} {'+ last layer + readout (B1b, anchored)' if a.unfreeze_last else 'only; trunk FROZEN'})")

    if a.data == "campaign":
        dss = load_campaign_datasets()
    else:
        dss = {n: load_dataset(n) for n in ("exploratory", "fresh")}
    train_rows = {n: channel_rows(ds, ds["tr"]) for n, ds in dss.items()}
    for n, r in train_rows.items():
        print(f"{n}: {len(r)} channel rows (train), {len(dss[n]['he'])} held pairs")

    # agnostic reference (must stay bit-identical — the by-construction guarantee, verified)
    ag_ref = {}
    with torch.no_grad():
        for n, ds in dss.items():
            items = [(ds["pairs"][i][0], ds["pairs"][i][1], OPS["HIER"]) for i in ds["he"][:32]]
            ag_ref[n] = np.array(mu_batch(model, ds["tok"], items, dev).cpu())

    names = list(train_rows)
    model.train()
    for step in range(1, a.steps + 1):
        n = names[step % len(names)]
        rows = train_rows[n]
        sel = rng.choice(len(rows), size=min(a.bs, len(rows)), replace=False)
        items = [rows[j][0] for j in sel]
        tgt = torch.tensor([rows[j][1] for j in sel], dtype=torch.float32, device=dev)
        mu = mu_batch(model, dss[n]["tok"], items, dev, train=True, rng=np.random)
        loss = torch.mean((mu - tgt) ** 2)
        if model.judge_name is not None:
            loss = loss + a.resid_weight * model.judge_name.resid_penalty()
        if ref is not None:
            # agnostic-anchor: on 3-tuple (provenance-masked) versions of the same pairs, the readouts must
            # match the frozen reference — the explicit replacement for the frozen-trunk guarantee
            ag_items = [(it[0], it[1], it[2]) for it in items]
            mu_ag = mu_batch(model, dss[n]["tok"], ag_items, dev)
            with torch.no_grad():
                mu_ref = mu_batch(ref, dss[n]["tok"], ag_items, dev)
            loss = loss + a.anchor_weight * torch.mean((mu_ag - mu_ref) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)      # B1b stability: unclipped 5e-3 collapsed the readout
        opt.step()
        if step % a.eval_every == 0 or step == 1:
            ev = {n2: eval_channels(model, ds, dev) for n2, ds in dss.items()}
            line = " | ".join(f"{n2}: " + " ".join(f"{k} {v:+.3f}" for k, v in e.items()) for n2, e in ev.items())
            print(f"step {step:4d} loss {loss.item():.4f} | held corr: {line}")

    # verify the agnostic path is untouched
    with torch.no_grad():
        for n, ds in dss.items():
            items = [(ds["pairs"][i][0], ds["pairs"][i][1], OPS["HIER"]) for i in ds["he"][:32]]
            ag = np.array(mu_batch(model, ds["tok"], items, dev).cpu())
            drift = float(np.abs(ag - ag_ref[n]).max())
            note = "anchored — should be SMALL" if a.unfreeze_last else "frozen — must be ~0"
            print(f"agnostic-path max drift ({n}): {drift:.2e} ({note})")

    torch.save({"state": model.state_dict(), "cfg": cfg}, a.out)
    print(f"saved → {a.out}")


if __name__ == "__main__":
    main()
