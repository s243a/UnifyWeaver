#!/usr/bin/env python3
"""eval_rerank.py — two-stage retrieval: (μ-blend | e5) top-K → LLM re-rank → precision@1.

Completes the pipeline: the tuned blend `max(μ-elem,μ-wiki,μ-sym)+0.1·e5` gets the answer into a tight top-K
(recall@5 ~0.50), then an LLM picks precision@1. Reports end-to-end precision@1 for blend→LLM vs e5→LLM, AND —
the "re-rank doubles as an eval" point — **LLM↔ranker agreement**: the less the LLM has to move things, the
better the ranker's shortlist already was.

  stage gen:   python3 eval_rerank.py gen --ckpt model_prod.pt --graph /tmp/merged_category_parent.tsv --n 60
  [Haiku scores /tmp/rerank_prompt.txt → /tmp/rerank_picks.json]
  stage score: python3 eval_rerank.py score --picks /tmp/rerank_picks.json
"""
import argparse, json, random, torch
from mu_attention import build_e5_tables, Tokenizer, OPS, load_dag
from eval_arch_control import build_model

SHORT = "/tmp/rerank_shortlists.json"
PROMPT = "/tmp/rerank_prompt.txt"


def gen(a):
    dev = torch.device(a.device); rng = random.Random(a.seed)
    parents, children, deg = load_dag(a.graph)
    cands = [p for p, k in children.items() if len(k) >= a.min_children]; cset = set(cands)
    queries = [(c, p) for p in cands for c in children[p] if p in cset]
    rng.shuffle(queries); queries = queries[:a.n]
    names = sorted(set(cands) | {c for c, p in queries})
    qt, pt, idx = build_e5_tables(names, cache_path="/tmp/rerank_e5.pt",
                                  texts={n: n.replace('_', ' ') for n in names}, device=a.device)
    tok = Tokenizer(qt, pt, idx, parents={}, deg={})
    model = build_model(a.ckpt, dev); n_ops = model.op_emb.weight.shape[0]
    OPW = {k: (torch.zeros(1, n_ops).index_fill_(1, torch.tensor([OPS[k.upper()]]), 1.0)) for k in ("elem", "hier", "sym")}
    C = pt[[idx[c] for c in cands]]; cand_pos = {c: i for i, c in enumerate(cands)}
    nz = lambda v: [(x - min(v)) / (max(v) - min(v) + 1e-9) for x in v]

    @torch.no_grad()
    def mu(prs, ow):
        b = tok.build([(x, y, 0) for x, y in prs], train=False)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        return model(**b, op_weights=ow.to(dev).expand(len(prs), n_ops)).cpu().tolist()

    disp = lambda s: s.replace('_', ' ')
    out = []
    for c, tp in queries:
        qv = qt[idx[c]]; e5s = (qv @ C.T)
        e5_top = torch.argsort(-e5s)[:a.topk].tolist()
        # blend over e5's top-N pool then take top-K
        pool = torch.argsort(-e5s)[:a.pool].tolist()
        muv = {k: nz(mu([(c, cands[j]) for j in pool], OPW[k])) for k in ("elem", "hier", "sym")}
        e5p = nz([e5s[j].item() for j in pool])
        blend = [0.9 * max(muv["elem"][i], muv["hier"][i], muv["sym"][i]) + 0.1 * e5p[i] for i in range(len(pool))]
        bl_top = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -blend[i])[:a.topk]]
        out.append({"child": c, "true": tp,
                    "e5": [cands[j] for j in e5_top],           # e5 shortlist (ranked)
                    "blend": [cands[j] for j in bl_top]})        # blend shortlist (ranked)
    json.dump(out, open(SHORT, "w"))
    # build one LLM prompt per shortlist source (blend, e5) — SAME queries, so the pick quality is comparable
    def write_prompt(key):
        P = ("For each query, a Wikipedia CHILD category and %d candidate PARENT categories are given. Pick the ONE "
             "candidate that the child is most directly a MEMBER/subcategory of (its true parent). Return ONLY a JSON "
             "array of objects {\"i\": int, \"pick\": int} where pick is the 0-based index into that query's candidate "
             "list. Cover all i=0..%d.\n\n" % (a.topk, len(out) - 1))
        for i, r in enumerate(out):
            P += f"{i}: child='{disp(r['child'])}' candidates=[" + ", ".join(f"{j}:{disp(x)}" for j, x in enumerate(r[key])) + "]\n"
        path = PROMPT.replace(".txt", f"_{key}.txt"); open(path, "w").write(P)
        return path, len(P)
    pb, lb = write_prompt("blend"); pe, le = write_prompt("e5")
    inpar = sum(r["true"] in r["blend"] for r in out); ine5 = sum(r["true"] in r["e5"] for r in out)
    print(f"[GEN] {len(out)} queries, top-{a.topk}. true-parent in shortlist: blend {inpar}/{len(out)}  e5 {ine5}/{len(out)}")
    print(f"  shortlists → {SHORT}")
    print(f"  blend prompt → {pb} ({lb} chars)")
    print(f"  e5    prompt → {pe} ({le} chars)")


def score(a):
    out = json.load(open(SHORT)); n = len(out)
    def one(key, picks_path):
        picks = {int(p["i"]): int(p["pick"]) for p in json.load(open(picks_path))}
        top1 = sum(r[key][0] == r["true"] for r in out) / n                 # ranker's own #1 (no LLM)
        llm = sum(0 <= picks.get(i, -1) < len(out[i][key]) and out[i][key][picks[i]] == out[i]["true"] for i in range(n)) / n
        agree = sum(picks.get(i, -1) == 0 for i in range(n)) / n            # LLM kept ranker's #1
        moved = [picks[i] for i in range(n) if i in picks and picks[i] != 0]
        ub = sum(r["true"] in r[key] for r in out) / n                      # ceiling: true parent in shortlist
        return top1, llm, agree, moved, ub
    print(f"[SCORE] {n} queries, top-{len(out[0]['blend'])} → LLM re-rank\n")
    print(f"  {'source':7} {'#1(no LLM)':>11} {'→LLM p@1':>9} {'ceiling':>8} {'LLM kept #1':>12} {'moved-to rank':>14}")
    res = {}
    for key, pk in (("blend", a.picks_blend), ("e5", a.picks_e5)):
        if not pk:
            continue
        t1, llm, ag, mv, ub = one(key, pk); res[key] = llm
        print(f"  {key:7} {t1:11.3f} {llm:9.3f} {ub:8.3f} {ag:11.1%} {(sum(mv)/max(1,len(mv))):13.1f}")
    if "blend" in res and "e5" in res:
        d = res["blend"] - res["e5"]
        print(f"\n  head-to-head: blend→LLM {res['blend']:.3f} vs e5→LLM {res['e5']:.3f}  (Δ {d:+.3f})"
              f"  — {'blend wins' if d > 0 else 'e5 wins' if d < 0 else 'tie'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--ckpt", required=True); g.add_argument("--graph", required=True)
    g.add_argument("--n", type=int, default=60); g.add_argument("--topk", type=int, default=5)
    g.add_argument("--pool", type=int, default=20); g.add_argument("--min-children", type=int, default=5)
    g.add_argument("--seed", type=int, default=7); g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s = sub.add_parser("score"); s.add_argument("--picks-blend"); s.add_argument("--picks-e5")
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else score(a)
