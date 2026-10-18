#!/usr/bin/env python3
"""eval_hybrid.py — the application: hybrid retrieval (e5 coarse rank → μ re-rank) vs e5-alone.

The payoff of the whole μ-vs-e5 arc. Task = "find my container": given a node, recover its true parent category
from the full candidate pool. Stage 1 (e5): rank candidates by cosine — cheap, but confuses the true parent with
SIBLINGS and with the reverse direction (all topically similar). Stage 2 (μ): re-rank e5's top-N by directional
membership, which knows a sibling is NOT a container and which way membership points.

This script prints three things:
  1. TUNING GRID — μ-part (which operators, combined how) × α (e5 weight). The winning score is the non-linear OR
     `max(μ-elem, μ-wiki, μ-sym)` (a container is relevant by membership OR relatedness; `max` keeps a strong
     single-operator hit a superposition/mean averages away) — NOT pure ELEM, NOT the internal μ-super. Best:
     `max(elem,wiki,sym)` at α≈0.9 → ~+25% MRR over e5-cos.
  2. CONTAINER-VS-SIBLING — μ's decisive close-neg regime: parent ranked above ALL the child's siblings.
  3. CONFIDENCE-ADAPTIVE α — α per-query from μ's own top-score (calibrated [0,1] degree = free confidence signal).
     Diagnostic shows μ earns its weight where confident (+0.037 MRR) vs not (+0.003); adaptive α∈[0.3,0.9] beats
     fixed α=0.9 by +0.006 in-dist (μ is neutral-not-harmful where unconfident, so its real payoff is OOD).

  python3 eval_hybrid.py --ckpt model_prod.pt --graph /tmp/merged_category_parent.tsv --n-queries 1000 --topn 20
"""
import argparse, random, torch
from mu_attention import build_e5_tables, Tokenizer, OPS, load_dag
from eval_arch_control import build_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--graph", required=True)
    ap.add_argument("--n-queries", type=int, default=500); ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--min-children", type=int, default=5, help="candidate parents must have >= this many children")
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", default="/tmp/hybrid_e5.pt")
    a = ap.parse_args(); dev = torch.device(a.device); rng = random.Random(a.seed)
    parents, children, deg = load_dag(a.graph)

    # candidate containers = categories with >= min_children; queries = their children (recover the true parent)
    cands = [p for p, k in children.items() if len(k) >= a.min_children]
    cset = set(cands)
    queries = [(c, p) for p in cands for c in children[p] if p in cset]     # (child, true parent)
    rng.shuffle(queries); queries = queries[:a.n_queries]
    names = sorted(set(cands) | {c for c, p in queries})
    qt, pt, idx = build_e5_tables(names, cache_path=a.cache, texts={n: n.replace('_', ' ') for n in names}, device=a.device)
    tok = Tokenizer(qt, pt, idx, parents={}, deg={})
    model = build_model(a.ckpt, dev); n_ops = model.op_emb.weight.shape[0]
    OPW = {"elem": torch.zeros(1, n_ops).index_fill_(1, torch.tensor([OPS["ELEM"]]), 1.0),
           "hier": torch.zeros(1, n_ops).index_fill_(1, torch.tensor([OPS["HIER"]]), 1.0),
           "sym":  torch.zeros(1, n_ops).index_fill_(1, torch.tensor([OPS["SYM"]]), 1.0),
           "super": torch.full((1, n_ops), 1.0 / n_ops)}                    # blend = operator superposition
    C = pt[[idx[c] for c in cands]]                                          # candidate containers as passage

    @torch.no_grad()
    def mu(prs, ow):
        out = []
        for i in range(0, len(prs), 512):
            ch = prs[i:i+512]; b = tok.build([(x, y, 0) for x, y in ch], train=False)
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            out += model(**b, op_weights=ow.to(dev).expand(len(ch), n_ops)).cpu().tolist()
        return out

    def rr(ranks, k):
        return sum(x <= k for x in ranks) / len(ranks)
    nz = lambda v: [(x - min(v)) / (max(v) - min(v) + 1e-9) for x in v]
    cand_pos = {c: i for i, c in enumerate(cands)}
    e5_ranks = []
    store = []                                                              # per query: (tp_i, top, RAW e5, {op:RAW μ})
    for c, tp in queries:
        qv = qt[idx[c]]
        e5s = (qv @ C.T).cpu()
        order = torch.argsort(-e5s).tolist(); tp_i = cand_pos[tp]
        e5_ranks.append(1 + order.index(tp_i))
        top = order[:a.topn]
        muvs = {k: mu([(c, cands[j]) for j in top], OPW[k]) for k in ("elem", "hier", "sym", "super")}
        store.append((tp_i, top, [e5s[j].item() for j in top], muvs))

    def blend_ranks(mufn, alpha):
        """mufn(muvs)→raw μ-part vec; score = (1−α)·nz(e5) + α·nz(μ-part). alpha=1 ⇒ μ-only, 0 ⇒ e5-only."""
        R = []
        for tp_i, top, e5v, muvs in store:
            e, m = nz(e5v), nz(mufn(muvs))
            sc = [(1 - alpha) * e[i] + alpha * m[i] for i in range(len(e))]
            ro = [top[j] for j in sorted(range(len(top)), key=lambda j: -sc[j])]
            R.append(1 + ro.index(tp_i) if tp_i in ro else a.topn + 1)
        return R
    def report(nm, R):
        print(f"  {nm:26} {rr(R,1):9.3f} {rr(R,5):9.3f} {sum(1.0/x for x in R)/len(R):7.3f}")
    mx = lambda *ks: (lambda m: [max(m[k][i] for k in ks) for i in range(len(m[ks[0]]))])   # OR over operators

    e5_mrr = sum(1.0/x for x in e5_ranks)/len(e5_ranks)
    print(f"[DATA] {len(queries)} queries, {len(cands)} candidate containers, e5 top-{a.topn} → re-rank (prefixed e5)")
    print(f"  e5-cos alone: recall@1 {rr(e5_ranks,1):.3f}  recall@5 {rr(e5_ranks,5):.3f}  MRR {e5_mrr:.3f}")
    # GRID: μ-part (which operators, combined how) × α (e5 weight; α=1 ⇒ μ-only, 0 ⇒ e5-only)
    muparts = {"super": lambda m: m["super"], "elem": lambda m: m["elem"],
               "mean(e,w,s)": lambda m: [(m["elem"][i]+m["hier"][i]+m["sym"][i])/3 for i in range(len(m["elem"]))],
               "max(e,s)": mx("elem", "sym"), "max(e,w)": mx("elem", "hier"),
               "max(e,w,s)": mx("elem", "hier", "sym"), "max(sup,e,w,s)": mx("super", "elem", "hier", "sym")}
    alphas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    print(f"\n  MRR grid — μ-part × α (e5 weight):   [e5-cos={e5_mrr:.3f}]")
    print(f"  {'μ-part':14} " + "".join(f"α={a_:<5}" for a_ in alphas))
    best = (None, -1)
    for nm, f in muparts.items():
        row, cells = [], []
        for al in alphas:
            R = blend_ranks(f, al); m = sum(1.0/x for x in R)/len(R); cells.append(m)
            if m > best[1]: best = (f"μ-{nm} @ α={al}", m, R)
        star = lambda m: ("*" if m == max(cells) else " ")
        print(f"  {nm:14} " + "".join(f"{c:.3f}{star(c)} " for c in cells))
    print(f"\n  BEST: {best[0]}  MRR {best[1]:.3f}  (recall@1 {rr(best[2],1):.3f}  recall@5 {rr(best[2],5):.3f})  vs e5-cos {e5_mrr:.3f}")
    hy_ranks = blend_ranks(mx("elem", "hier", "sym"), 1.0)                  # μ-max for the sibling metric below
    # how often μ promotes the true parent that e5 buried below rank 1 but within the shortlist
    fixed = sum(1 for e, h in zip(e5_ranks, hy_ranks) if e > 1 and h == 1)
    broke = sum(1 for e, h in zip(e5_ranks, hy_ranks) if e == 1 and h > 1)
    print(f"\n  μ promoted true-parent to #1 that e5 had lower: {fixed}; μ demoted an e5 #1: {broke}  (net {fixed-broke})")

    # μ's DECISIVE regime: true-parent vs the child's own SIBLINGS (where e5 confuses — the close-neg win in a
    # retrieval context). For queries whose e5 shortlist contains ≥1 sibling, does the scorer rank the true parent
    # ABOVE all those siblings? (a container-vs-sibling discrimination, not a which-level question)
    e5_win, mu_win, n_sib = 0, 0, 0
    for c, tp in queries:
        sibs = [s for s in children[tp] if s != c and s in cand_pos]        # child's siblings that are candidates
        if not sibs:
            continue
        qv = qt[idx[c]]
        e5s = (qv @ C.T).cpu()
        top = torch.argsort(-e5s)[:a.topn].tolist()
        sib_in = [cand_pos[s] for s in sibs if cand_pos[s] in top]
        if cand_pos[tp] not in top or not sib_in:
            continue
        n_sib += 1
        cand_i = [cand_pos[tp]] + sib_in
        muv = mu([(c, cands[j]) for j in cand_i], OPW["super"])
        e5v = [e5s[j].item() for j in cand_i]
        e5_win += int(e5v[0] == max(e5v))                                   # e5 ranks true parent above all siblings?
        mu_win += int(muv[0] == max(muv))                                   # μ ranks true parent above all siblings?
    if n_sib:
        print(f"\n  container-vs-sibling ({n_sib} queries w/ siblings in shortlist): parent ranked above ALL siblings — "
              f"e5-cos {e5_win/n_sib:.1%}  |  μ {mu_win/n_sib:.1%}  (μ's close-neg win, in the pipeline)")

    # ══ confidence-adaptive α (per-query blend weight from μ's own confidence) ══
    # Coverage-insurance made per-query: trust μ where it is confident (sharp / high top-μ over the shortlist), fall
    # back to e5 where μ is flat/low (untrained/OOD). μ is a calibrated [0,1] membership degree ⇒ the top candidate's
    # μ-max IS a confidence signal, computed for free from the scores we already have.
    mm_all = [[max(mv["elem"][i], mv["hier"][i], mv["sym"][i]) for i in range(len(mv["elem"]))] for _, _, _, mv in store]
    t1 = [max(mm) for mm in mm_all]; med = sorted(t1)[len(t1) // 2]; alli = list(range(len(store)))
    def mrr_on(idxs, alpha_of):                                             # α_of(i)→per-query blend weight
        rs = []
        for i in idxs:
            tp_i, top, e5v, _ = store[i]; e, m = nz(e5v), nz(mm_all[i]); al = alpha_of(i)
            sc = [(1 - al) * e[j] + al * m[j] for j in range(len(e))]
            ro = [top[j] for j in sorted(range(len(top)), key=lambda j: -sc[j])]
            rs.append(1 + ro.index(tp_i) if tp_i in ro else a.topn + 1)
        return sum(1.0 / x for x in rs) / len(rs)
    print("\n  ══ confidence-adaptive α ══   (confidence = top1 of μ-max over the shortlist)")
    # 1) diagnostic — does confidence predict the better fixed α? (the crossover that justifies adapting)
    for lab, idxs in (("high-conf (top1-μ ≥ med)", [i for i in alli if t1[i] >= med]),
                      ("low-conf  (top1-μ < med)",  [i for i in alli if t1[i] <  med])):
        m03, m09 = mrr_on(idxs, lambda i: 0.3), mrr_on(idxs, lambda i: 0.9)
        print(f"    {lab:26} n={len(idxs):4}  MRR@α0.3 {m03:.3f}   @α0.9 {m09:.3f}   Δ(0.9−0.3) {m09-m03:+.3f}")
    # 2) adaptive vs best fixed — α_q linear in top1-μ across the observed range
    lo, hi = min(t1), max(t1)
    amap = lambda i, alo, ahi: alo + (ahi - alo) * ((t1[i] - lo) / (hi - lo + 1e-9))
    mfix = mrr_on(alli, lambda i: 0.9)
    print(f"    {'fixed α=0.9':30}     MRR {mfix:.3f}")
    for alo, ahi in ((0.3, 1.0), (0.5, 1.0), (0.3, 0.9)):
        ma = mrr_on(alli, lambda i, alo=alo, ahi=ahi: amap(i, alo, ahi))
        print(f"    adaptive α∈[{alo},{ahi}] (top1-μ)     MRR {ma:.3f}   Δ vs fixed {ma-mfix:+.3f}")


if __name__ == "__main__":
    main()
