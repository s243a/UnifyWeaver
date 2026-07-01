#!/usr/bin/env python3
"""eval_self_anneal.py — does μ's confidence (and thus its adaptive blend weight) RISE with training?

The self-annealing prediction (DESIGN_model_applications.md, confidence-adaptive blend, property #2): under the
adaptive rule α rises with μ's top-score, so as μ trains on more data its confident regions expand ⇒ the effective
mean α climbs on its own and e5 recedes. Test it directly: run the SAME queries + SAME e5 shortlists (frozen e5 ⇒
checkpoint-independent) through checkpoints of increasing training maturity, and watch the confidence distribution.

For each checkpoint we report, over the e5 top-N shortlist per query:
  mean top1-μ    — average confidence (top μ-max candidate) — the raw signal
  high-conf %    — fraction of queries with top1-μ ≥ τ (cleared the confidence bar)
  eff. mean α    — average per-query blend weight under α_q = 0.3 + 0.6·top1-μ (⇒ how much μ is trusted on avg)
  MRR            — retrieval quality of μ-max alone (sanity: capability should track confidence)

Expect all four to rise along nodetype → dir → dir_disc → prod. NOTE: these checkpoints differ in OBJECTIVE, not
purely data volume — so this is a capability-progression proxy for the data-curve, not a data-controlled ablation.

  python3 eval_self_anneal.py --graph /tmp/merged_category_parent.tsv --n-queries 1000 --topn 20 \
      --ckpts model_nodetype.pt:nodetype model_dir.pt:+dir model_dir_disc.pt:+disc model_prod.pt:prod
"""
import argparse, random, torch
from mu_attention import build_e5_tables, Tokenizer, OPS, load_dag
from eval_arch_control import build_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True, help="path[:label] in training order")
    ap.add_argument("--n-queries", type=int, default=1000); ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.5, help="high-confidence threshold on top1-μ")
    ap.add_argument("--min-children", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device); rng = random.Random(a.seed)
    parents, children, deg = load_dag(a.graph)
    cands = [p for p, k in children.items() if len(k) >= a.min_children]; cset = set(cands)
    queries = [(c, p) for p in cands for c in children[p] if p in cset]
    rng.shuffle(queries); queries = queries[:a.n_queries]
    names = sorted(set(cands) | {c for c, p in queries})
    qt, pt, idx = build_e5_tables(names, cache_path="/tmp/anneal_e5.pt", texts={n: n.replace('_', ' ') for n in names}, device=a.device)
    tok = Tokenizer(qt, pt, idx, parents={}, deg={})
    C = pt[[idx[c] for c in cands]]; cand_pos = {c: i for i, c in enumerate(cands)}

    # e5 shortlists (frozen ⇒ identical for every checkpoint) + the true-parent position within each
    shortlists = []
    for c, tp in queries:
        e5s = (qt[idx[c]] @ C.T).cpu()
        top = torch.argsort(-e5s)[:a.topn].tolist()
        shortlists.append((c, cand_pos[tp], top))

    def rr(ranks, k): return sum(x <= k for x in ranks) / len(ranks)
    print(f"[DATA] {len(queries)} queries · e5 top-{a.topn} shortlists (frozen, shared) · τ={a.tau}\n")
    print(f"  {'checkpoint':14} {'mean top1-μ':>12} {'mean margin':>12} {'MRR':>7}   {'ΔMRR':>7}")
    base_mrr = None
    for spec in a.ckpts:
        path, _, label = spec.partition(":"); label = label or path
        model = build_model(path, dev); n_ops = model.op_emb.weight.shape[0]
        OPW = {k: torch.zeros(1, n_ops).index_fill_(1, torch.tensor([OPS[k.upper()]]), 1.0) for k in ("elem", "wiki", "sym")}

        @torch.no_grad()
        def mu(prs, ow):
            out = []
            for i in range(0, len(prs), 512):
                ch = prs[i:i+512]; b = tok.build([(x, y, 0) for x, y in ch], train=False)
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                out += model(**b, op_weights=ow.to(dev).expand(len(ch), n_ops)).cpu().tolist()
            return out

        tops, margins, ranks = [], [], []
        for c, tp_i, top in shortlists:
            mvs = {k: mu([(c, cands[j]) for j in top], OPW[k]) for k in ("elem", "wiki", "sym")}
            mm = [max(mvs["elem"][i], mvs["wiki"][i], mvs["sym"][i]) for i in range(len(top))]   # μ-max (raw [0,1])
            sm = sorted(mm, reverse=True)
            tops.append(sm[0]); margins.append(sm[0] - (sm[1] if len(sm) > 1 else 0.0))          # level vs margin
            order = sorted(range(len(top)), key=lambda i: -mm[i])
            ranks.append(1 + order.index(top.index(tp_i)) if tp_i in top else a.topn + 1)
        mean_top = sum(tops) / len(tops); mean_mrg = sum(margins) / len(margins)
        mrr = sum(1.0 / x for x in ranks) / len(ranks)
        if base_mrr is None: base_mrr = mrr
        print(f"  {label:14} {mean_top:12.3f} {mean_mrg:12.3f} {mrr:7.3f}   {mrr-base_mrr:+7.3f}")
    print("\n  LEVEL (mean top1-μ) vs MARGIN (top1−top2). If margin tracks MRR and flags the confident-but-wrong")
    print("  checkpoint (high level, low margin, low MRR), margin is the calibration-invariant confidence signal.")


if __name__ == "__main__":
    main()
