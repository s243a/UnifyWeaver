#!/usr/bin/env python3
"""Two-timescale CE-calibration meta-judge (DESIGN_meta_judge_calibration.md).

Per batch: a FAST inner step optimizes the Kalman μ estimate (distill the fused posterior into the
μ heads — the existing Filing v1 fine-tune step), and every K batches a SLOW outer step calibrates
the judge μ by candidate-ranking cross-entropy (true parent should rank first among sampled
candidate folders) plus a held-out SONNET magnitude anchor. Sonnet-5 is NOT a Kalman channel — it is
the independent reference the fusion cannot supply (the meta hook). At inference folders are ranked
by the calibrated Kalman μ (eval_pearltrees_filing), no judge in the loop.

  python3 train_meta_judge.py --out model_pt_meta_judge.pt
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_luna_transfer import load_luna
from fine_tune_channel_heads import mu_batch
from fine_tune_pearltrees_filing import (
    PT, group_of, load_fused_targets, load_routed_labels, load_with_lineage_ops)
from mu_attention import CORPORA, JUDGES, OPS
from node_disjoint_eval import node_disjoint_pair_split
from run_pearltrees_fusion import load_pearltrees_campaign

ROOT = os.path.dirname(os.path.abspath(__file__))
SONNET_TSV = "/tmp/mu_data/pt_campaign_scored_sonnet.tsv"
RANK_JUDGE = "kalman-fused"     # the head the filing eval ranks with — calibrate ITS μ


def sonnet_D(path=SONNET_TSV):
    """Held-out sonnet directional D per overlap pair (title, title) → D. Empty if not yet scored."""
    if not os.path.exists(path):
        return {}
    p, d, s = load_luna(path)          # same mu[] schema; DIRR-max = D
    return {pair: float(d[i]) for i, pair in enumerate(p)}


def candidate_scores(model, tok, node, cands, dev):
    """μ(node | c, HIER, pearltrees, RANK_JUDGE) for each candidate folder c (7-tuples w/ nodetypes)."""
    items = [(node, c, OPS["HIER"], CORPORA["pearltrees"], JUDGES[RANK_JUDGE], PT, PT) for c in cands]
    return mu_batch(model, tok, items, dev)          # differentiable [len(cands)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod_namecond_full.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "model_pt_meta_judge.pt"))
    ap.add_argument("--targets", default="/tmp/mu_data/pt_fused_targets.tsv")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr-fast", type=float, default=5e-4, help="Adam lr for the fast Kalman-μ heads")
    ap.add_argument("--lr-slow", type=float, default=0.5,
                    help="SGD lr for the judge calibration. SGD (not Adam) so the two-timescale EMERGES "
                         "from information content: quantization-noisy CE gradients average out, mean "
                         "drift ∝ signal (Adam would unit-normalize the step and chase the noise).")
    ap.add_argument("--slow-every", type=int, default=1,
                    help="cadence of the CE-calibration step; default 1 = every batch, so the slow "
                         "timescale is emergent (from the SGD/SNR), not imposed by skipping batches")
    ap.add_argument("--n-cand", type=int, default=16, help="candidates per ranking example (1 true + negs)")
    ap.add_argument("--rank-bs", type=int, default=16, help="ranking examples per slow step")
    ap.add_argument("--tau", type=float, default=0.1, help="candidate-softmax temperature")
    ap.add_argument("--lam-cal", type=float, default=1.0, help="sonnet magnitude-anchor weight")
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--no-sonnet", action="store_true", help="ablation: CE only, no held-out anchor")
    ap.add_argument("--neg-alpha", type=float, default=1.0,
                    help="SCALE-FREE negative sampling over e5-distance rank: P(rank r) ∝ (r+1)^-alpha. "
                         "alpha=1 (Zipf) = mostly close/confusable candidates + a heavy tail of a few "
                         "distant ones (the information lives in the close choices; the far ones anchor "
                         "the easy regime). alpha=0 = uniform (ablation: dilutes the CE information).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args(argv)
    dev = "cpu"
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    aug = np.random.default_rng(a.seed + 1)

    ds = load_pearltrees_campaign()
    fused = load_fused_targets(a.targets)
    routed = load_routed_labels()
    pairs, tags, luna, d = ds["pairs"], ds["tags"], ds["luna"], ds["d"]
    y55_by = {tuple(pairs[k]): ds["y55"][j] for j, k in enumerate(ds["overlap_idx"])}
    y55_by.update(routed)
    son = {} if a.no_sonnet else sonnet_D()
    print(f"campaign {len(pairs)} rows; 5.5-labeled {len(y55_by)}; sonnet-scored {len(son)} "
          f"{'(ablated)' if a.no_sonnet else ''}")

    split = node_disjoint_pair_split(pairs, a.split_seed, strata=[group_of(t) for t in tags])
    tr = split.train
    tr_set = set(tr.tolist())

    # ranking examples: principal-path descendants with their true parent folder (train side)
    folder_pool = sorted({pairs[i][1] for i in range(len(pairs)) if tags[i].startswith("principal")})
    rank_ex = [(pairs[i][0], pairs[i][1]) for i in tr if tags[i].startswith("principal")]
    # per-descendant true-ancestor set (to exclude from negatives)
    true_anc = {}
    for i in range(len(pairs)):
        if tags[i].startswith("principal"):
            true_anc.setdefault(pairs[i][0], set()).add(pairs[i][1])
    print(f"ranking examples (train principal): {len(rank_ex)}; folder pool {len(folder_pool)}")

    # HARD-NEGATIVE ranking (user's point): the CE signal carries information where folders are
    # semantically CLOSE — uniform negatives are mostly trivial (far) and dilute the gradient. For
    # each true parent, rank the pool by e5 cosine (passage embeddings; unit-normed) so negatives are
    # the confusable NEAR folders — raising CE information and focusing calibration on the hard stratum.
    cache = ds["cache"]
    cidx = {n: i for i, n in enumerate(cache["names"])}
    Pmat = cache["passage"]
    pool_in = [c for c in folder_pool if c in cidx]
    pool_vecs = torch.stack([Pmat[cidx[c]] for c in pool_in]) if pool_in else None
    hard_order = {}
    if a.neg_alpha > 0 and pool_vecs is not None:
        for f in {ft for _, ft in rank_ex if ft in cidx}:
            sims = (pool_vecs @ Pmat[cidx[f]]).tolist()
            hard_order[f] = [pool_in[k] for k in np.argsort([-s for s in sims])]  # closest first
        print(f"scale-free negatives: P(rank)∝(r+1)^-{a.neg_alpha} (mostly close + heavy tail)")
    else:
        print("uniform negatives (neg-alpha=0, ablation)")

    # fast (distillation) training rows — the existing Filing v1 channel rows
    C = CORPORA["pearltrees"]
    rows = []
    for i in tr:
        x, y = pairs[i]
        info = fused.get((x, y))
        if info is None:
            continue
        y55 = y55_by.get((x, y))
        if y55 is not None:
            rows.append(((x, y, OPS["HIER"], C, JUDGES["gpt-5.5-low"], PT, PT), y55[0]))
            rows.append(((x, y, OPS["SYM"], C, JUDGES["gpt-5.5-low"], PT, PT), y55[1]))
        rows.append(((x, y, OPS["HIER"], C, JUDGES["gpt-5.6-luna"], PT, PT), luna[i, 0]))
        rows.append(((x, y, OPS["SYM"], C, JUDGES["gpt-5.6-luna"], PT, PT), luna[i, 1]))
        rows.append(((x, y, OPS["HIER"], C, JUDGES["graph"], PT, PT), d[i]))
        rows.append(((x, y, OPS["HIER"], C, JUDGES[RANK_JUDGE], PT, PT), info["post"][0]))
        rows.append(((x, y, OPS["SYM"], C, JUDGES[RANK_JUDGE], PT, PT), info["post"][1]))
    print(f"fast distillation rows: {len(rows)}")

    model, cfg = load_with_lineage_ops(a.ckpt, dev=dev)
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    for p in model.parameters():
        p.requires_grad = False
    # Clean two-timescale split (no shared tensor across optimizers):
    #   FAST inner estimator = shared trunk (last layer) + readout + nodetype + corpus/op residuals.
    #   SLOW meta judge      = judge_name.resid (the judge calibration). μ(·,RANK_JUDGE) = trunk(fast
    #                          representation) + judge_resid[RANK_JUDGE](slow CE+sonnet calibration).
    last = model.encoder.layers[-1]
    fast_params = list(last.parameters()) + [model.readout_w, model.readout_b, model.nodetype_emb.weight]
    for mod in (getattr(model, "corpus_name", None), getattr(model, "op_name", None)):
        if mod is not None:
            mod.resid.weight.requires_grad = True
            fast_params.append(mod.resid.weight)
    for p in fast_params:
        p.requires_grad = True
    slow_params = [model.judge_name.resid.weight]
    model.judge_name.resid.weight.requires_grad = True
    opt_fast = torch.optim.Adam(fast_params, lr=a.lr_fast)
    rank_row = JUDGES[RANK_JUDGE]
    # The judge (meta) parameter uses a PRECISION/SNR-GATED update, not SGD or Adam: the CE gradient
    # is a noisy observation of the calibration, and the step is scaled by its signal-to-noise so a
    # low-information (quantization-limited) batch moves the judge little. This is what makes the
    # two-timescale EMERGE from information content — the user's point — rather than being imposed.
    # EMA mean m and second-moment s of the judge-row gradient → gain = ‖m‖²/(s+eps) ∈ [0,1]
    # (≈1 coherent signal, ≈0 pure noise); step = lr · gain · m̂. Adam would set gain≡1 (unit RMS),
    # destroying exactly this scaling.
    ema_beta = 0.9
    g_m = torch.zeros_like(model.judge_name.resid.weight[rank_row])
    g_s = torch.zeros(())
    print(f"fast params {sum(p.numel() for p in fast_params)} (trunk/readout/corpus/op, Adam {a.lr_fast}); "
          f"slow judge params {sum(p.numel() for p in slow_params)} (SNR-gated, base {a.lr_slow}); "
          f"slow every {a.slow_every}")

    # emergent-timescale instrumentation: per-step drift of the judge row vs a fast reference (readout),
    # and CE-signal noise — to TEST whether the judge naturally moves slower (not assume it)
    fast_drift, slow_drift, ce_hist, ce_grad = [], [], [], []
    model.train()
    for step in range(1, a.steps + 1):
        # ---- FAST inner: distill the Kalman fused μ into the heads ----
        sel = rng.choice(len(rows), size=min(a.bs, len(rows)), replace=False)
        items = [rows[j][0] for j in sel]
        tgt = torch.tensor([rows[j][1] for j in sel], dtype=torch.float32, device=dev)
        mu = mu_batch(model, ds["tok"], items, dev, train=True, rng=aug)
        loss = torch.mean((mu - tgt) ** 2)
        ag = [(it[0], it[1], it[2]) for it in items]
        mu_ag = mu_batch(model, ds["tok"], ag, dev)
        with torch.no_grad():
            mu_ref = mu_batch(ref, ds["tok"], ag, dev)
        loss = loss + a.anchor_weight * torch.mean((mu_ag - mu_ref) ** 2)
        readout_before = model.readout_w.detach().clone()
        opt_fast.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fast_params, 1.0)
        opt_fast.step()
        fast_drift.append(float((model.readout_w.detach() - readout_before).norm()))

        # ---- SLOW outer: candidate-ranking CE + held-out sonnet anchor calibrates the judge μ ----
        if step % a.slow_every == 0 and rank_ex:
            idxs = rng.choice(len(rank_ex), size=min(a.rank_bs, len(rank_ex)), replace=False)
            ce, cal, n_cal = 0.0, 0.0, 0
            for j in idxs:
                node, f_true = rank_ex[j]
                excl = true_anc.get(node, set()) | {f_true}
                if a.neg_alpha > 0 and f_true in hard_order:
                    ranked = [c for c in hard_order[f_true] if c not in excl]     # closest-first
                    k = min(a.n_cand - 1, len(ranked))
                    w = np.array([(r + 1.0) ** (-a.neg_alpha) for r in range(len(ranked))])
                    pick = rng.choice(len(ranked), size=k, replace=False, p=w / w.sum())
                    negs = [ranked[t] for t in pick]
                else:
                    pool = [c for c in folder_pool if c not in excl]
                    negs = [pool[t] for t in rng.choice(len(pool), size=min(a.n_cand - 1, len(pool)),
                                                         replace=False)]
                cands = [f_true] + negs
                scores = candidate_scores(model, ds["tok"], node, cands, dev)   # [K]
                ce = ce + torch.nn.functional.cross_entropy(
                    (scores / a.tau).unsqueeze(0), torch.zeros(1, dtype=torch.long, device=dev))
                sd = son.get((node, f_true))
                if sd is not None:
                    cal = cal + (scores[0] - float(sd)) ** 2
                    n_cal += 1
            meta = ce / len(idxs) + (a.lam_cal * cal / n_cal if n_cal else 0.0)
            opt_fast.zero_grad()
            if model.judge_name.resid.weight.grad is not None:
                model.judge_name.resid.weight.grad.zero_()
            meta.backward()
            g = model.judge_name.resid.weight.grad[rank_row].detach()
            # precision/SNR-gated update (manual — no optimizer normalization)
            g_m.mul_(ema_beta).add_(g, alpha=1 - ema_beta)
            g_s.mul_(ema_beta).add_((g * g).sum(), alpha=1 - ema_beta)
            gain = float((g_m @ g_m) / (g_s + 1e-12))        # ‖signal‖²/power ∈ [0,1]
            row_before = model.judge_name.resid.weight[rank_row].detach().clone()
            with torch.no_grad():
                step_vec = a.lr_slow * gain * g_m / (g_m.norm() + 1e-9)
                model.judge_name.resid.weight[rank_row] -= step_vec
            ce_grad.append(gain)                              # record the emergent gain (effective rate)
            slow_drift.append(float((model.judge_name.resid.weight[rank_row].detach() - row_before).norm()))
            ce_hist.append(float(ce / len(idxs)))

        if step % 200 == 0 or step == 1:
            msg = f"step {step:4d} fast_loss {loss.item():.4f}"
            if ce_hist:
                msg += f"  CE {ce_hist[-1]:.4f}"
            print(msg)

    # did the two-timescale EMERGE? report the drift ratio + CE-signal noise
    if slow_drift:
        fd, sd_ = np.array(fast_drift), np.array(slow_drift)
        gains = np.array(ce_grad)
        print(f"\nEMERGENT TIMESCALE (SNR-gated judge; measured, not imposed):")
        print(f"  fast μ-head drift/step (‖Δreadout‖):   mean {fd.mean():.4e}")
        print(f"  slow judge drift/step (‖Δjudge_row‖):  mean {sd_.mean():.4e}")
        print(f"  emergent judge gain ‖m‖²/power:        mean {gains.mean():.3f}  "
              f"(≈0 ⇒ noise-dominated ⇒ self-limited slow; ≈1 ⇒ coherent signal)")
        print(f"  → the gain, not a hand-set lr, is what makes the calibration slow: with a "
              f"quantization-noisy CE signal it stays low, so the judge self-limits.")

    torch.save({"state": model.state_dict(), "cfg": cfg}, a.out)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
