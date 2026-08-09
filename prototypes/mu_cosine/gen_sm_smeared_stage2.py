#!/usr/bin/env python3
"""SimpleMind OOD arc — STAGE 2: smear the SM-FS gap field onto enwiki categories and
generate gap-targeted training data (the cross-corpus targeting loop, aimed at where the
model fails on SimpleMind filing — gap 0.235, 56% nonzero).

Cross-corpus smears (SM-FS queries are NOT enwiki nodes, so graph diffusion needs a bridge):
  semantic  weakness(c) = sum_q k(sim(c, q)) gap_q / sum_q k(...)   [e5 kernel, tau]
  graph     BRIDGE each SM-FS query to its nearest enwiki category (semantic), seed the gap
            there, diffuse over category_parent edges (deg-normalized, alpha, T)
  hybrid    geomean(semantic, graph) — weak only where both agree
w(c) = (stem_affinity(c) * weakness_hybrid(c))^0.5, tempered, no region buckets (v2 lesson).
Reuses the enwiki category pool + e5 cache from gen_gap_smeared_v3. Emits ELEM/LINEAGE/SYM
(v0.5 headers). Stage 3 = replay fine-tune (separate, stops before the SM-FS reserve)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

S = os.path.expanduser("~/mu_data/sm_gap_v1")
POOL_E5 = os.path.expanduser("~/mu_data/enwiki_gap_v3/pool_e5.pt")
LMDB = "/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_correct_v2/lmdb"
EDGES = "/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_correct_v2/edges_child_parent.tsv"
TITLES = ("/tmp/claude-1000/-home-s243a-Projects-UnifyWeaver/"
          "be81a5e2-7bea-4a18-9241-de457531548d/scratchpad/ns14_id_title.tsv")
SEED, TAU, ALPHA, T_DIFF = 3997001, 0.10, 0.5, 3
TOP_CATS, SAMPLE_CATS, MAX_ARTICLES = 60000, 25000, 400000
MAX_UP_HOPS, DECAY, SIBS, EASY = 4, 0.85, 2, 2
STEM_ANCHORS = ["physics", "mathematics", "computer science", "statistics", "engineering",
                "chemistry", "machine learning", "algorithms", "software", "electronics",
                "systems theory", "differential equations", "dynamical systems"]

def run():
    import torch, lmdb
    from mu_attention import build_e5_tables, E5_REVISION
    from eval_filing import is_admin
    g = torch.load(f"{S}/gapfield.pt", weights_only=False)
    gap, qvec = g["gap"], torch.tensor(g["qvec"])
    print(f"[s2] SM-FS gap field: {len(gap)} queries, mean {gap.mean():.4f}", flush=True)
    rng = np.random.default_rng(SEED)
    title = {}
    for ln in open(TITLES, encoding="utf-8", errors="replace"):
        i, _, t = ln.rstrip("\n").partition("\t")
        title[int(i)] = t
    # enwiki category pool (top by membership, non-admin) + e5
    env = lmdb.open(LMDB, max_dbs=4, readonly=True, lock=False)
    ac = env.open_db(b"article_category", dupsort=True)
    counts = {}
    with env.begin(db=ac) as txn:
        for _, v in txn.cursor():
            counts[v] = counts.get(v, 0) + 1
    top = sorted(counts, key=counts.get, reverse=True)
    top = [c for c in top if int(c) in title and not is_admin(title[int(c)])][:TOP_CATS]
    texts = {f"C:{c.decode()}": title[int(c)].replace("_", " ") for c in top}
    texts |= {f"A:{a}": a for a in STEM_ANCHORS}
    _, pt, ix = build_e5_tables(sorted(texts), cache_path=f"{S}/pool_e5.pt",
                                batch_size=256, model_revision=E5_REVISION)
    CV = torch.stack([pt[ix[f"C:{c.decode()}"]] for c in top])
    AV = torch.stack([pt[ix[f"A:{a}"]] for a in STEM_ANCHORS])
    aff = (CV @ AV.T).max(1).values.numpy()
    # semantic smear
    SIM = (CV @ qvec.T).numpy()
    K = np.exp((SIM - 1.0) / TAU)
    weak_sem = (K @ gap) / (K.sum(1) + 1e-9)
    # graph smear via semantic bridge: each SM-FS query -> nearest enwiki cat, seed gap, diffuse
    nearest = SIM.argmax(0)                       # for each query, index into `top`
    seed = {}
    for qi, ci in enumerate(nearest):
        tid = int(top[ci])
        seed[tid] = seed.get(tid, 0.0) + float(gap[qi])
    adj = {}
    with open(EDGES) as fh:
        next(fh)
        for ln in fh:
            c, _, p = ln.rstrip("\n").partition("\t")
            c, p = int(c), int(p)
            adj.setdefault(c, []).append(p); adj.setdefault(p, []).append(c)
    field, cur = dict(seed), dict(seed)
    for _ in range(T_DIFF):
        nxt = {}
        for node, v in cur.items():
            nb = adj.get(node, [])
            if not nb:
                continue
            sh = ALPHA * v / len(nb)
            for x in nb:
                nxt[x] = nxt.get(x, 0.0) + sh
        for node, v in nxt.items():
            field[node] = field.get(node, 0.0) + v
        cur = nxt
    weak_gr = np.array([field.get(int(c), 0.0) for c in top])
    nz = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    ws, wg = nz(weak_sem), nz(weak_gr)
    weak_hy = np.sqrt((ws + 1e-6) * (wg + 1e-6))
    affn = nz(aff)
    for tag, wk in (("semantic", ws), ("graph", wg), ("hybrid", weak_hy)):
        w = (affn * wk + 1e-8) ** 0.5
        tix = np.argsort(-w)[:10]
        print(f"[{tag}] top: " + ", ".join(title[int(top[i])][:28] for i in tix), flush=True)
    w = (affn * weak_hy + 1e-8) ** 0.5
    p = w / w.sum()
    sel = rng.choice(len(top), size=min(SAMPLE_CATS, len(top)), replace=False, p=p)
    chosen = {top[i] for i in sel}
    # generation (same emission as v3)
    art_cats = {}
    with env.begin(db=ac) as txn:
        for k, v in txn.cursor():
            if v in chosen:
                art_cats.setdefault(k, []).append(int(v))
    keys = sorted(art_cats)
    if len(keys) > MAX_ARTICLES:
        pick = rng.choice(len(keys), MAX_ARTICLES, replace=False)
        keys = [keys[i] for i in sorted(pick)]
    meta = env.open_db(b"article_meta")
    atitle = {}
    with env.begin(db=meta) as txn:
        for k in keys:
            t = txn.get(k, db=meta)
            if t is not None:
                atitle[k] = t.decode("utf-8", errors="replace")
    keys = [k for k in keys if k in atitle]
    print(f"articles with metadata: {len(keys)}", flush=True)
    child, parent = [], []
    with open(EDGES) as fh:
        next(fh)
        for ln in fh:
            c, _, pp = ln.rstrip("\n").partition("\t")
            child.append(int(c)); parent.append(int(pp))
    child, parent = np.array(child), np.array(parent)
    o2 = np.argsort(child, kind="stable"); cs2, ps2 = child[o2], parent[o2]
    ust = np.searchsorted(cs2, np.unique(cs2))
    pars_of = {int(c): (int(a), int(b)) for c, a, b in
               zip(np.unique(cs2), ust, np.append(ust[1:], len(cs2)))}
    members = {}
    for k in keys:
        for c in art_cats[k]:
            members.setdefault(c, []).append(k)
    cat_list = sorted({c for v in art_cats.values() for c in v})
    n_rows = {"ELEM": 0, "LINEAGE": 0, "SYM": 0, "easy": 0}
    out_path = f"{S}/targets.tsv"
    with open(out_path, "w", encoding="utf-8") as out:
        out.write('# process_expression\tSM-FS-targeted (gap smear onto enwiki, hybrid); '
                  'graph-judged\n')
        out.write('#   ELEM    = lineage(enwiki,mu=graph,estimand="element_of")\n')
        out.write('#   LINEAGE = lineage(enwiki,decay=0.85,mu=graph,estimand="ancestry")\n')
        out.write('#   SYM     = cowalk(enwiki,walk="sibling",weight="idf_node_size",'
                  'mu=graph,estimand="path")\n')
        out.write("# node\tother\ttarget\top\tkind\n")
        for k in keys:
            at = atitle[k].replace("\t", " ").replace("_", " ")
            direct = art_cats[k]
            for c in direct:
                if not is_admin(title[c]):
                    out.write(f"{at}\t{title[c].replace('_',' ')}\t1.0\tELEM\tpos\n")
                    n_rows["ELEM"] += 1
            c = int(direct[rng.integers(len(direct))])
            node, seen = c, {c}
            for h in range(1, MAX_UP_HOPS + 1):
                if node not in pars_of:
                    break
                a, b = pars_of[node]
                cand = [int(x) for x in ps2[a:b] if int(x) not in seen]
                if not cand:
                    break
                node = cand[rng.integers(len(cand))]; seen.add(node)
                if node in title and not is_admin(title[node]):
                    out.write(f"{at}\t{title[node].replace('_',' ')}\t"
                              f"{round(DECAY**h, 6)}\tLINEAGE\tpos\n")
                    n_rows["LINEAGE"] += 1
            mem = members.get(c, [])
            if len(mem) > 1:
                wgt = round(min(1.0, 1.0 / max(np.log2(len(mem)), 1.0)), 6)
                for _ in range(SIBS):
                    s2 = mem[rng.integers(len(mem))]
                    if s2 != k and s2 in atitle:
                        out.write(f"{at}\t{atitle[s2].replace(chr(9),' ').replace('_',' ')}"
                                  f"\t{wgt}\tSYM\tsib\n")
                        n_rows["SYM"] += 1
            for _ in range(EASY):
                e = cat_list[rng.integers(len(cat_list))]
                if e not in direct and e in title and not is_admin(title[e]):
                    out.write(f"{at}\t{title[e].replace('_',' ')}\t0.0\tELEM\teasy\n")
                    n_rows["easy"] += 1
    print(json.dumps({"rows": n_rows, "articles": len(keys), "sampled_cats": len(chosen),
                      "out": out_path}), flush=True)

run()
