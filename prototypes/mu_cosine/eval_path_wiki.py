#!/usr/bin/env python3
"""eval_path_wiki.py — does a MERGED shallow multi-ancestor passage beat a single-path / bare-title passage for
directional membership on a DENSE multi-parent DAG (Wikipedia categories)?

Reuses eval_hybrid's "find my container" task (query = child category, target = true parent) but varies the
CANDIDATE PASSAGE that μ scores against, holding the operator at HIER to isolate the PASSAGE effect:
  HIER     = bare candidate title (baseline)
  LINEAGE  = candidate's single-path ancestor chain (first parent each hop)
  PATH     = candidate's MERGED shallow ancestor context (ALL parents, max_depth 2) — the multi-path passage

Reports recall@1/@5/MRR on ALL queries and on the MULTI-PARENT subset (children with >1 parent — where PATH can
differ). e5 shortlist is by title (same for all three), so only the μ re-rank passage changes.

  python3 eval_path_wiki.py --ckpt model_prod.pt --graph ../../data/benchmark/wide_enwiki_math/category_parent.tsv
"""
import argparse, random, torch
from mu_attention import build_e5_tables, Tokenizer, OPS
from eval_arch_control import build_model
from merged_ancestors import load_category_graph, render_merged_list


def single_path_text(node, parents_of, title_of, max_depth=6):
    """One lineage chain: walk up following the first unseen parent each hop (the LINEAGE-style single path)."""
    chain, cur, seen = [node], node, {node}
    for _ in range(max_depth):
        nxt = next((p for p in parents_of.get(cur, []) if p not in seen), None)
        if not nxt:
            break
        chain.append(nxt); seen.add(nxt); cur = nxt
    return "\n".join(f"{'  ' * i}- {title_of.get(n, n)}" for i, n in enumerate(reversed(chain)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--graph", required=True)
    ap.add_argument("--n-queries", type=int, default=800); ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--min-children", type=int, default=5)
    ap.add_argument("--max-depth", type=int, default=2, help="PATH merged-context depth (shallow for dense DAGs)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device); rng = random.Random(a.seed)

    parents_of, title_of = load_category_graph(a.graph)
    children = {}
    for c, ps in parents_of.items():
        for p in ps:
            children.setdefault(p, []).append(c)
    cands = [p for p, k in children.items() if len(k) >= a.min_children]; cset = set(cands)
    queries = [(c, p) for p in cands for c in children[p] if p in cset]
    rng.shuffle(queries); queries = queries[:a.n_queries]
    names = sorted(set(cands) | {c for c, p in queries})
    print(f"[PATH-WIKI] {len(queries)} queries, {len(cands)} candidate containers, {len(names)} names; embedding 3 passage variants (CPU-slow, cached)...", flush=True)

    texts_title = {n: title_of.get(n, n) for n in names}
    texts_lin = {n: single_path_text(n, parents_of, title_of) for n in names}
    texts_path = {n: render_merged_list(n, parents_of, title_of, max_depth=a.max_depth) for n in names}
    qt, pt_title, idx = build_e5_tables(names, cache_path="/tmp/pathwiki_title.pt", texts=texts_title, device=a.device)
    _, pt_lin, _ = build_e5_tables(names, cache_path="/tmp/pathwiki_lin.pt", texts=texts_lin, device=a.device)
    _, pt_path, _ = build_e5_tables(names, cache_path="/tmp/pathwiki_path.pt", texts=texts_path, device=a.device)

    model = build_model(a.ckpt, dev); n_ops = model.op_emb.weight.shape[0]
    OW = torch.zeros(1, n_ops).index_fill_(1, torch.tensor([OPS["HIER"]]), 1.0)
    toks = {"HIER(title)": Tokenizer(qt, pt_title, idx, parents={}, deg={}),
            "LINEAGE(1-path)": Tokenizer(qt, pt_lin, idx, parents={}, deg={}),
            "PATH(merged)": Tokenizer(qt, pt_path, idx, parents={}, deg={})}

    @torch.no_grad()
    def mu(tok, prs):
        out = []
        for i in range(0, len(prs), 512):
            ch = prs[i:i+512]; b = tok.build([(x, y, 0) for x, y in ch], train=False)
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            out += model(**b, op_weights=OW.to(dev).expand(len(ch), n_ops)).cpu().tolist()
        return out

    cand_pos = {c: i for i, c in enumerate(cands)}
    C_e5 = pt_title[[idx[c] for c in cands]]                       # e5 shortlist by title (same for all variants)
    R = {k: [] for k in toks}; Rmp = {k: [] for k in toks}
    for c, tp in queries:
        qv = qt[idx[c]]; e5s = (qv @ C_e5.T).cpu()
        order = torch.argsort(-e5s).tolist(); tp_i = cand_pos[tp]; top = order[:a.topn]
        mp = len(parents_of.get(c, [])) > 1
        for k, tok in toks.items():
            m = mu(tok, [(c, cands[j]) for j in top])
            ro = [top[j] for j in sorted(range(len(top)), key=lambda j: -m[j])]
            r = 1 + ro.index(tp_i) if tp_i in ro else a.topn + 1
            R[k].append(r)
            if mp:
                Rmp[k].append(r)

    def rep(nm, ranks):
        r1 = sum(x <= 1 for x in ranks) / len(ranks); r5 = sum(x <= 5 for x in ranks) / len(ranks)
        mrr = sum(1.0 / x for x in ranks) / len(ranks)
        print(f"  {nm:16} recall@1 {r1:.3f}  recall@5 {r5:.3f}  MRR {mrr:.3f}  (n={len(ranks)})")
    print(f"\n== ALL queries (op=HIER; passage varies) ==")
    for k in toks:
        rep(k, R[k])
    print(f"== MULTI-PARENT children only (where PATH can differ; n={len(Rmp['HIER(title)'])}) ==")
    for k in toks:
        rep(k, Rmp[k])


if __name__ == "__main__":
    main()
