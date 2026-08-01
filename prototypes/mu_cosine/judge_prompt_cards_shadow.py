#!/usr/bin/env python3
"""SHADOW: R10 judge-channel refinement bundle (encoder-lane handoff; P1 gate blocked, see
P1_GATE_REPORT.md). Arms on the standard channel-heads recipe (campaign data, W+resid
trainable, trunk frozen, MSE to channel targets):
  A descriptive   baseline JUDGE_CARDS
  B prompt-text   prompt_text_cards() — actual harness prompt as the card (R10)
  C prompt+drop30 B + judge-channel dropout p=0.3 during training (asymmetric: op/graph
  D prompt+drop60 B + p=0.6                                        channels never masked)
Eval per arm: held-out corr per channel (llm-D, llm-S = gpt-5.5-low conditioned; graph-d =
control, card unchanged) — plus the same eval with the judge channel FULLY masked
(degradation curve = the point of the dropout experiment). Slot-compat is asserted in
test_judge_channel_slot.py. Cards cache content-addressed (build_e5_tables revalidates)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from fine_tune_channel_heads import (ROOT, load_expanded, load_campaign_datasets,
                                     channel_rows, mu_batch)
from judge_cards import JUDGE_CARDS, prompt_text_cards, _card_e5
from mu_attention import JUDGES, OPS

STEPS, BS, LR, SEED = 800, 64, 1e-2, 0

def mu_batch_p(model, tok, items, dev, train=False, rng=None, p_mask_prov=0.0):
    b = tok.build(items, train=train, rng=rng, p_mask_prov=p_mask_prov)
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    return model(**b)

def eval_channels_p(model, ds, dev, ablate_judge):
    """ablate_judge=True evaluates on 3-tuple (provenance-masked) items — the codebase's
    agnostic pathway — because p_mask_prov only masks during training builds."""
    model.eval()
    out = {}
    with torch.no_grad():
        for cname, judge, op, tgt in [("llm-D", "gpt-5.5-low", "HIER", ds["D"]),
                                      ("llm-S", "gpt-5.5-low", "SYM", ds["S"]),
                                      ("graph-d", "graph", "HIER", ds["d"])]:
            from mu_attention import CORPORA
            if ablate_judge:
                items = [(ds["pairs"][i][0], ds["pairs"][i][1], OPS[op])
                         for i in ds["he"]]
            else:
                items = [(ds["pairs"][i][0], ds["pairs"][i][1], OPS[op], CORPORA["enwiki"],
                          JUDGES[judge]) for i in ds["he"]]
            mu = np.array(mu_batch_p(model, ds["tok"], items, dev).cpu())
            t = tgt[ds["he"]]
            out[cname] = float(np.corrcoef(mu, t)[0, 1]) if mu.std() > 1e-9 else 0.0
    model.train()
    return out

def run_arm(tag, cards, p_drop, dss, dev="cpu"):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model, cfg = load_expanded(os.path.join(ROOT, "model_prod_namecond_full.pt"), dev=dev)
    for p in model.parameters(): p.requires_grad = False
    tbl, names = _card_e5(cards, JUDGES,
                          cache_path=f"/tmp/mu_data/judge_cards_e5_{tag}.pt")
    assert model.judge_name is not None, "R10 bundle requires NameFunctionCond checkpoint"
    assert tbl.shape == model.judge_name.name_e5.shape, "slot-compat: card table shape"
    model.judge_name.name_e5.copy_(tbl)
    model.judge_name.W.weight.requires_grad = True
    model.judge_name.resid.weight.requires_grad = True
    trainable = [model.judge_name.W.weight, model.judge_name.resid.weight]
    opt = torch.optim.Adam(trainable, lr=LR)
    train_rows = {n: channel_rows(ds, ds["tr"]) for n, ds in dss.items()}
    names_ds = list(train_rows)
    model.train()
    for step in range(1, STEPS + 1):
        n = names_ds[step % len(names_ds)]
        rows = train_rows[n]
        sel = rng.choice(len(rows), size=min(BS, len(rows)), replace=False)
        items = [rows[j][0] for j in sel]
        tgt = torch.tensor([rows[j][1] for j in sel], dtype=torch.float32)
        mu = mu_batch_p(model, dss[n]["tok"], items, dev, train=True, rng=np.random,
                        p_mask_prov=p_drop)
        loss = torch.mean((mu - tgt) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    res = {}
    for n, ds in dss.items():
        res[n] = eval_channels_p(model, ds, dev, False)
        res[n + "_judge_ablated"] = eval_channels_p(model, ds, dev, True)
    print(f"[{tag}] " + json.dumps(res), flush=True)
    return res

def main():
    dss = load_campaign_datasets()
    out = {}
    for tag, cards, p in (("descriptive", JUDGE_CARDS, 0.0),
                          ("prompt_text", prompt_text_cards(), 0.0),
                          ("prompt_drop30", prompt_text_cards(), 0.3),
                          ("prompt_drop60", prompt_text_cards(), 0.6)):
        out[tag] = run_arm(tag, cards, p, dss)
    print(json.dumps({"stamp": "shadow-exploratory-tier1-not-decision-bearing",
                      "arms": out}, indent=1), flush=True)

main()
