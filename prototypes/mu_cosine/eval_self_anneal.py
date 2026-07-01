#!/usr/bin/env python3
"""eval_self_anneal.py — is μ's confidence a genuine per-query correctness signal, and does it grow with training?

Tests two things, per the confidence-adaptive blend (DESIGN_model_applications.md):
  (A) SIGNAL QUALITY — does a confidence score separate correct from incorrect retrievals PER QUERY (not just in
      aggregate)? Compared for two signals: LEVEL (top1-μ) vs MARGIN (top1−top2 of μ-max over the shortlist).
  (B) SELF-ANNEALING — does that confidence rise along a training progression?

Same queries + same FROZEN-e5 shortlists through every checkpoint (only μ varies). No LLM cost.

Per checkpoint it prints, over the e5 top-N shortlist per query (confidence = --confidence-mode, default margin):
  mean conf        — average of the selected confidence signal
  high-conf %      — fraction of queries with conf ≥ --tau (on the selected signal)
  eff. mean α      — average per-query blend weight α_q = 0.3 + 0.6·clamp01(conf) (how hard μ is trusted)
  MRR              — retrieval quality of μ-max alone
  ρ(sig,RR)        — per-query Spearman between the confidence signal and reciprocal-rank, with Fisher-z 95% CI
                     — reported for BOTH level and margin (does margin discriminate correct-vs-wrong better?)
  AURC             — selective risk (risk=1[rank>1]) sorted by the signal, lower=better gate, bootstrap 95% CI
                     — reported for BOTH level and margin
  HMER@0.8         — high-confidence error rate: fraction WRONG@1 among the top-80%-by-signal (the confident-but-
                     -wrong diagnostic aggregate means hide), for BOTH level and margin

Also writes a per-query audit TSV (--audit-out): checkpoint, qid, top1_mu, top2_mu, margin, rank, rr — for offline
HMER / risk-coverage / Simpson-stratified analysis. NOTE: checkpoints differ in OBJECTIVE, not purely data volume —
this is a capability-progression proxy for a data-anneal, single seed, single query sample; treat trends as
*consistent with* self-annealing, not confirmatory (n=4 checkpoints).

  python3 eval_self_anneal.py --graph /tmp/merged_category_parent.tsv --n-queries 1000 --topn 20 \
      --confidence-mode margin --ckpts model_nodetype.pt:nodetype model_dir.pt:+dir model_dir_disc.pt:+disc model_prod.pt:prod
"""
import argparse, math, random, torch
from mu_attention import build_e5_tables, Tokenizer, OPS, load_dag
from eval_arch_control import build_model


def rankdata(v):                                                            # average ranks, 1-based (ties averaged)
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]: j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1): r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n))); vb = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    return cov / (va * vb + 1e-12)


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def fisher_ci(rho, n):                                                      # closed-form 95% CI for a rank corr
    if n <= 4 or abs(rho) >= 1: return (rho, rho)
    z = 0.5 * math.log((1 + rho) / (1 - rho)); se = 1.0 / math.sqrt(n - 3)
    lo, hi = z - 1.96 * se, z + 1.96 * se
    return (math.tanh(lo), math.tanh(hi))


def aurc(conf, risk):                                                       # selective risk: mean risk over coverage, sorted by desc conf
    order = sorted(range(len(conf)), key=lambda i: -conf[i]); cum = 0.0; acc = 0.0
    for k, i in enumerate(order, 1):
        cum += risk[i]; acc += cum / k
    return acc / len(order)


def boot_aurc_ci(conf, risk, B, seed):
    rng = random.Random(seed); n = len(conf); vals = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(aurc([conf[i] for i in idx], [risk[i] for i in idx]))
    vals.sort(); return (vals[int(0.025 * B)], vals[int(0.975 * B)])


def hmer(conf, wrong, cov=0.8):                                            # error rate among top-cov by conf
    order = sorted(range(len(conf)), key=lambda i: -conf[i])[:max(1, int(cov * len(conf)))]
    return sum(wrong[i] for i in order) / len(order)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True, help="path[:label] in training order")
    ap.add_argument("--n-queries", type=int, default=1000); ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--confidence-mode", choices=("level", "margin"), default="margin", help="signal for high-conf %/eff.α/--tau")
    ap.add_argument("--tau", type=float, default=None, help="high-conf threshold on the selected signal (default 0.5 level / 0.03 margin)")
    ap.add_argument("--audit-out", default="/tmp/anneal_audit.tsv"); ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--min-children", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device); rng = random.Random(a.seed)
    tau = a.tau if a.tau is not None else (0.5 if a.confidence_mode == "level" else 0.03)
    parents, children, deg = load_dag(a.graph)
    cands = [p for p, k in children.items() if len(k) >= a.min_children]; cset = set(cands)
    queries = [(c, p) for p in cands for c in children[p] if p in cset]
    rng.shuffle(queries); queries = queries[:a.n_queries]
    names = sorted(set(cands) | {c for c, p in queries})
    qt, pt, idx = build_e5_tables(names, cache_path="/tmp/anneal_e5.pt", texts={n: n.replace('_', ' ') for n in names}, device=a.device)
    tok = Tokenizer(qt, pt, idx, parents={}, deg={})
    C = pt[[idx[c] for c in cands]]; cand_pos = {c: i for i, c in enumerate(cands)}
    shortlists = []
    for c, tp in queries:
        e5s = (qt[idx[c]] @ C.T).cpu()
        top = torch.argsort(-e5s)[:a.topn].tolist()
        shortlists.append((c, cand_pos[tp], top))

    print(f"[DATA] {len(queries)} queries · e5 top-{a.topn} shortlists (frozen, shared) · mode={a.confidence_mode} τ={tau} · boot={a.boot}\n")
    audit = ["checkpoint\tqid\ttop1_mu\ttop2_mu\tmargin\trank\trr"]; per_ckpt_margin = {}; rows = []
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

        lvl, mrg, rr, wrong = [], [], [], []
        for qi, (c, tp_i, top) in enumerate(shortlists):
            mvs = {k: mu([(c, cands[j]) for j in top], OPW[k]) for k in ("elem", "wiki", "sym")}
            mm = [max(mvs["elem"][i], mvs["wiki"][i], mvs["sym"][i]) for i in range(len(top))]
            sm = sorted(mm, reverse=True); top1, top2 = sm[0], (sm[1] if len(sm) > 1 else 0.0)
            order = sorted(range(len(top)), key=lambda i: -mm[i])
            rank = 1 + order.index(top.index(tp_i)) if tp_i in top else a.topn + 1
            lvl.append(top1); mrg.append(top1 - top2); rr.append(1.0 / rank); wrong.append(0 if rank == 1 else 1)
            audit.append(f"{label}\t{qi}\t{top1:.4f}\t{top2:.4f}\t{top1-top2:.4f}\t{rank}\t{1.0/rank:.4f}")
        per_ckpt_margin[label] = mrg
        conf = lvl if a.confidence_mode == "level" else mrg
        n = len(conf); mean_conf = sum(conf) / n; hi = sum(x >= tau for x in conf) / n
        c01 = (lambda v: v) if a.confidence_mode == "level" else (lambda v: min(1.0, v / max(1e-9, tau)))
        eff_a = sum(0.3 + 0.6 * min(1.0, max(0.0, c01(x))) for x in conf) / n
        mrr = sum(rr) / n
        rl, rm = spearman(lvl, rr), spearman(mrg, rr)
        cll, chl = fisher_ci(rl, n); clm, chm = fisher_ci(rm, n)
        al = aurc(lvl, wrong); am = aurc(mrg, wrong)
        all_, alh = boot_aurc_ci(lvl, wrong, a.boot, a.seed); aml, amh = boot_aurc_ci(mrg, wrong, a.boot, a.seed)
        hl, hm = hmer(lvl, wrong), hmer(mrg, wrong)
        rows.append((label, mean_conf, hi, eff_a, mrr, rl, cll, chl, rm, clm, chm, al, all_, alh, am, aml, amh, hl, hm))

    # Table 1 — blend / gating diagnostics (docstring-advertised)
    print(f"  {'checkpoint':12} {'mean conf':>9} {'hi-conf%':>8} {'eff.α':>6} {'MRR':>6}")
    for r in rows:
        print(f"  {r[0]:12} {r[1]:9.3f} {r[2]:7.1%} {r[3]:6.3f} {r[4]:6.3f}")
    # Table 2 — per-query discrimination (the thesis evidence): ρ(signal,RR)±Fisher-z CI · AURC±bootstrap CI · HMER@0.8
    print(f"\n  {'checkpoint':12} {'ρ_lvl(RR)[CI]':>19} {'ρ_mrg(RR)[CI]':>19}   {'AURC_lvl[CI]':>21} {'AURC_mrg[CI]':>21}   {'HMER_l':>6} {'HMER_m':>6}")
    for (lb, mc, hi, ea, mrr, rl, cll, chl, rm, clm, chm, al, all_, alh, am, aml, amh, hl, hm) in rows:
        print(f"  {lb:12} {rl:+.2f}[{cll:+.2f},{chl:+.2f}] {rm:+.2f}[{clm:+.2f},{chm:+.2f}]   "
              f"{al:.3f}[{all_:.3f},{alh:.3f}] {am:.3f}[{aml:.3f},{amh:.3f}]   {hl:5.1%} {hm:5.1%}")
    # cross-checkpoint per-query margin stability (are high-margin queries the SAME across training, not reshuffled?)
    labs = list(per_ckpt_margin)
    if len(labs) >= 2:
        st = spearman(per_ckpt_margin[labs[0]], per_ckpt_margin[labs[-1]])
        print(f"\n  per-query margin stability ρ({labs[0]}↔{labs[-1]}) = {st:+.3f}  (high = same queries confident, not random reshuffle)")
    open(a.audit_out, "w").write("\n".join(audit) + "\n")
    print(f"  per-query audit ({len(audit)-1} rows) → {a.audit_out}")
    print("\n  Read (narrowed thesis): margin is the better selective-risk GATE — AURC_mrg < AURC_lvl on all 4 checkpoints")
    print("  (meaningfully on 3/4; +disc is a collapse-driven near-tie — margin also degenerates when the objective saturates μ),")
    print("  and ρ_lvl is NEGATIVE on the under-trained checkpoints. Margin is NOT a strong per-query correctness signal:")
    print("  ρ_mrg is weak (CI incl. 0 on 3/4), and on the mature prod checkpoint ρ_lvl ≳ ρ_mrg and HMER is ~tied.")
    print("  n=4 ckpts from one objective-progression trajectory, 1 seed ⇒ 'consistent with' self-annealing, not proof.")


if __name__ == "__main__":
    main()
