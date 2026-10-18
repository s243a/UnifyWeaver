#!/usr/bin/env python3
"""eval_relatedness.py — the ALIGNED eval: within-vs-cross subdomain pair discrimination.

eval_filing measures μ(member|exact-parent) ranked against ALL folders — a classification task relative to
roots that rewards exact-title-match (favours e5-cos, understates μ; a member is related to its whole subdomain,
not just its one parent). This eval matches what the SAMPLER trains on (within-subdomain pairs = related, noise =
not) and isolates the regime where e5 should be WEAK and μ strong: **same coarse domain, different fine
subdomain** (e5 sees "all physics-ish"; μ trained on subdomain walks should separate fine structure).

Strata (pairs (x,y) from the merged category graph):
  POS       — x,y descend from the SAME fine subdomain (a direct child of a coarse root)  → should be HIGH μ
  HARD-NEG  — x,y in DIFFERENT fine subdomains of the SAME coarse root                     → the e5-weak regime
  EASY-NEG  — x,y in DIFFERENT coarse roots                                                → both should reject

Reports AUC(POS vs HARD-NEG) and AUC(POS vs EASY-NEG) for e5-cos, mu-sym, mu-super. The headline is
AUC(POS vs HARD-NEG): does μ beat e5-cos at FINE subdomain discrimination?

  python3 eval_relatedness.py --ckpt model_nodetype.pt --graph /tmp/merged_category_parent.tsv \
      --coarse Physics,Mathematics,Chemistry,Linguistics,Political_science
"""
import argparse, collections, random
import torch
from mu_attention import build_e5_tables, Tokenizer, MuAttention, OPS, load_dag


def build_model(ckpt, dev):
    ck = torch.load(ckpt, weights_only=False); sd = ck["state"]; cfg = ck.get("cfg", {"d_model":384,"heads":4,"layers":3})
    sz = lambda k, d: sd[k].shape[0] if k in sd else d
    m = MuAttention(d_model=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
                    n_ops=sz("op_emb.weight", len(OPS)), n_corpus=sz("corpus_emb.weight", 2),
                    n_judge=sz("judge_emb.weight", 2), n_nodetype=sz("nodetype_emb.weight", 4)).to(dev)
    # Older checkpoints (e.g. model_prod.pt) predate the account/prefix tags and the SYM dual-judge
    # struct/precision/confidence params+buffers; all are zero-init / no-op, so loading strict=False and ignoring
    # EXACTLY these is safe. Explicit leaf-name allow-list (not broad substrings) so a genuinely-missing/renamed
    # key in a future refactor still trips the assert (PR #3488 review, finding #9).
    _NEW = ("account_emb.weight", "prefix_emb.weight", "sym_struct_w", "struct_lambda", "struct_g", "struct_h",
            "prec_g", "prec_h", "c_dist", "c_mem_ceiling", "c_subcat", "c_elem")
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not unexp, f"unexpected keys loading {ckpt}: {unexp}"
    bad = [k for k in miss if not any(k.endswith(n) for n in _NEW)]
    assert not bad, f"unexpected MISSING keys loading {ckpt}: {bad}"
    m.eval(); return m


def descendants(children, root, cap=400, max_depth=4):
    seen, frontier = set(), [root]
    for _ in range(max_depth):
        nxt = []
        for n in frontier:
            for c in children.get(n, ()):
                if c not in seen:
                    seen.add(c); nxt.append(c)
                    if len(seen) >= cap:
                        return seen
        frontier = nxt
    return seen


def auc(pos, neg, rng, n=200000):
    """P(pos_score > neg_score) via random sampling — Mann-Whitney AUC."""
    if not pos or not neg:
        return float("nan")
    w = 0
    for _ in range(n):
        a, b = rng.choice(pos), rng.choice(neg)
        w += 1 if a > b else (0.5 if a == b else 0)
    return w / n


@torch.no_grad()
def score_pairs(model, tok, idx, pairs, ow, dev, batch=512):
    items = [(a, b, 0) for a, b in pairs]
    out = []
    for i in range(0, len(items), batch):
        ch = items[i:i+batch]
        bd = tok.build(ch, train=False); bd = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in bd.items()}
        out += model(**bd, op_weights=ow.expand(len(ch), ow.shape[1]).to(dev)).cpu().tolist()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--coarse", required=True, help="comma-sep coarse-domain roots (underscored)")
    ap.add_argument("--n-pairs", type=int, default=800, help="pairs per stratum")
    ap.add_argument("--cap", type=int, default=300, help="descendants kept per fine subdomain")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", default="/tmp/relatedness_e5.pt")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    dev = torch.device(a.device); rng = random.Random(a.seed)
    parents, children, deg = load_dag(a.graph)
    coarse = [c.strip() for c in a.coarse.split(",") if c.strip()]

    # fine subdomains = direct children of each coarse root that actually have descendants
    fine = {}   # coarse -> {fine_subdomain: [descendant nodes incl. itself]}
    for cz in coarse:
        fs = {}
        for f in children.get(cz, ()):
            d = descendants(children, f, a.cap); d.add(f)
            if len(d) >= 3:
                fs[f] = sorted(d)
        if len(fs) >= 2:
            fine[cz] = fs
    coarse = [c for c in coarse if c in fine]
    print(f"[DATA] coarse domains usable: {coarse}")
    for cz in coarse:
        print(f"  {cz}: {len(fine[cz])} fine subdomains, {sum(len(v) for v in fine[cz].values())} nodes")

    def pick(pool):
        return rng.choice(pool)

    pos, hardneg, easyneg = [], [], []
    for _ in range(a.n_pairs):
        cz = pick(coarse); fs = fine[cz]; fkeys = list(fs)
        f = pick(fkeys); pool = fs[f]
        if len(pool) >= 2:
            x, y = rng.sample(pool, 2); pos.append((x, y))
        f1, f2 = rng.sample(fkeys, 2) if len(fkeys) >= 2 else (fkeys[0], fkeys[0])
        hardneg.append((pick(fs[f1]), pick(fs[f2])))
        if len(coarse) >= 2:
            c1, c2 = rng.sample(coarse, 2)
            easyneg.append((pick(fine[c1][pick(list(fine[c1]))]), pick(fine[c2][pick(list(fine[c2]))])))
    strata = {"POS": pos, "HARD-NEG": hardneg, "EASY-NEG": easyneg}

    # e5 + model over all involved nodes
    names = sorted({n for s in strata.values() for p in s for n in p})
    disp = {n: n.replace("_", " ") for n in names}
    qt, pt, idx = build_e5_tables(names, cache_path=a.cache, texts=disp, device=a.device)
    tok = Tokenizer(qt, pt, idx, parents={}, deg={})
    model = build_model(a.ckpt, dev)
    n_ops = model.op_emb.weight.shape[0]
    sym = torch.zeros(1, n_ops).index_fill_(1, torch.tensor([OPS["SYM"]]), 1.0)
    sup = torch.full((1, n_ops), 1.0 / n_ops)

    # scores per stratum: e5-cos (symmetric), mu-sym, mu-super (μ(x|y))
    def e5cos(prs):
        qn = qt[[idx[x] for x, _ in prs]]; pn = pt[[idx[y] for _, y in prs]]
        return (qn * pn).sum(-1).tolist()
    sc = {}
    for name, prs in strata.items():
        sc[name] = {"e5-cos": e5cos(prs),
                    "mu-sym": score_pairs(model, tok, idx, prs, sym, dev),
                    "mu-super": score_pairs(model, tok, idx, prs, sup, dev)}

    mean = lambda L: sum(L) / len(L)
    print(f"\n{'scorer':9} {'AUC(P vs HARD)':>14} {'AUC(P vs EASY)':>14}   {'mean P/HARD/EASY':>20}")
    for scorer in ("e5-cos", "mu-sym", "mu-super"):
        ah = auc(sc["POS"][scorer], sc["HARD-NEG"][scorer], rng)
        ae = auc(sc["POS"][scorer], sc["EASY-NEG"][scorer], rng)
        print(f"{scorer:9} {ah:14.3f} {ae:14.3f}   "
              f"{mean(sc['POS'][scorer]):.3f}/{mean(sc['HARD-NEG'][scorer]):.3f}/{mean(sc['EASY-NEG'][scorer]):.3f}")

    # NEGATIVE-REJECTION at a fixed operating point: threshold = 10th-percentile of POS (≈90% TPR); report the
    # fraction of negatives that still pass (FPR). e5's compressed band ⇒ can't set a cutoff that rejects negs.
    print(f"\n  Negative rejection — FPR at ~90% positive-recall (lower = better; tests THRESHOLDING, not rank):")
    print(f"  {'scorer':9} {'threshold':>10} {'FPR HARD-NEG':>13} {'FPR EASY-NEG':>13}")
    for scorer in ("e5-cos", "mu-sym", "mu-super"):
        ps = sorted(sc["POS"][scorer]); thr = ps[max(0, int(0.10 * len(ps)))]   # admit ~90% of positives
        fpr_h = sum(s >= thr for s in sc["HARD-NEG"][scorer]) / len(sc["HARD-NEG"][scorer])
        fpr_e = sum(s >= thr for s in sc["EASY-NEG"][scorer]) / len(sc["EASY-NEG"][scorer])
        print(f"  {scorer:9} {thr:10.3f} {fpr_h:13.2%} {fpr_e:13.2%}")
    print("\n  headline: AUC = rank (e5's turf); FPR@90%TPR = can you THRESHOLD to reject negatives (μ's turf —"
          " e5's compressed cosine band can't).")


if __name__ == "__main__":
    main()
