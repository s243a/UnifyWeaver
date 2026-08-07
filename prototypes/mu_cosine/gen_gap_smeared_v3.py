#!/usr/bin/env python3
"""Gap-weighted sampler v3 — SMEARED WEAKNESS (owner: interpolate/smear weakness over
semantic space or the graph, aka filtering; try both + hybrid).

Two degenerate v2 attempts established: discrete region buckets fail (7-keyword classifier
too crude for enwiki's people-heavy head; unmeasured defaults poison the axis). v3 removes
buckets entirely:
  STAGE 1  measure a DENSE per-query gap field: MAXQ=2000 fresh enwiki queries x 2000-cat
           catalog, scored with the CURRENT best trunk (replay s3997003) and e5;
           gap_q = clip(RR_e5(q) - RR_bestmu(q), 0).  [GPU, cached]
  STAGE 2  smear the field onto the top-60k category pool three ways:
           semantic  weakness(c) = sum_q k(sim) gap_q / sum_q k(sim), k = exp((sim-1)/tau)
           graph     seed gap_q at the query's true category, diffuse along
                     category_parent edges (both directions, degree-normalized,
                     alpha-decay, T iterations)
           hybrid    geometric mean of the two (a category is weak only if both views
                     agree; either alone can be fooled — semantic by vocabulary,
                     graph by disconnection)
  STAGE 3  w = stem_affinity x weakness_hybrid, TEMPERED sampling ~ w^0.5, no buckets;
           region mix printed as a DIAGNOSTIC lens only. Generate ELEM/LINEAGE/SYM rows
           (v0.5 headers).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

S = os.path.expanduser("~/mu_data/enwiki_gap_v3")
CK = os.path.expanduser("~/mu_data/enwiki_transitive_v1/trunk_stem_replay_s3997003.pt")
LMDB = "/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_correct_v2/lmdb"
EDGES = "/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_correct_v2/edges_child_parent.tsv"
TITLES = ("/tmp/claude-1000/-home-s243a-Projects-UnifyWeaver/"
          "be81a5e2-7bea-4a18-9241-de457531548d/scratchpad/ns14_id_title.tsv")
ENWIKI_NAMED = "/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_named/category_parent.tsv"
SEED, MAXQ, NCAT = 3997001, 2000, 2000
TAU, ALPHA, T_DIFF = 0.10, 0.5, 3
TOP_CATS, SAMPLE_CATS, MAX_ARTICLES = 60000, 25000, 400000
MAX_UP_HOPS, DECAY, SIBS, EASY = 4, 0.85, 2, 2
STEM_ANCHORS = ["physics", "mathematics", "computer science", "statistics", "engineering",
                "chemistry", "machine learning", "algorithms", "software", "electronics"]

def stage1(dev):
    import torch, random
    import sm_fs_ranking_pipeline as pl
    from eval_filing import score_mu
    from mu_attention import OPS, Tokenizer, build_e5_tables, E5_REVISION
    cache = f"{S}/gapfield_2000.pt"
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    from collections import defaultdict
    nchild = defaultdict(int)
    with open(ENWIKI_NAMED, encoding="utf-8", errors="replace") as fh:
        next(fh)
        for ln in fh:
            i = ln.find("\t")
            if i > 0:
                nchild[ln[i+1:].rstrip("\n")] += 1
    eligible = [p for p, c in nchild.items() if c >= 3]
    rng = random.Random(SEED)
    catalog = sorted(rng.sample(eligible, NCAT))
    catset = set(catalog)
    kids = defaultdict(list)
    with open(ENWIKI_NAMED, encoding="utf-8", errors="replace") as fh:
        next(fh)
        for ln in fh:
            i = ln.find("\t")
            if i > 0:
                p = ln[i+1:].rstrip("\n")
                if p in catset and len(kids[p]) < 40:
                    kids[p].append(ln[:i])
    pairs = [(c, p) for p, cs in kids.items() for c in cs]
    rng.shuffle(pairs)
    queries = [(c.replace("_", " "), p) for c, p in pairs[:MAXQ]]
    print(f"[s1] queries {len(queries)} catalog {len(catalog)}", flush=True)
    texts = {f"F:{t}": t.replace("_", " ") for t in catalog} | \
            {f"B:{i}": q for i, (q, _) in enumerate(queries)}
    qt, pt, ix = build_e5_tables(sorted(texts), cache_path=f"{S}/eval_e5.pt",
                                 batch_size=256, model_revision=E5_REVISION)
    tok = Tokenizer(qt, pt, ix, {}, {})
    ckb = pl.read_bound(os.path.join(pl.RANK_DIR, "init_seed3997001.pt"),
                        expect_sha=pl.INIT_SHA[3997001], private=True)
    model, _ = pl.load_checkpoint_bytes(ckb, dev)
    model.load_state_dict(__import__("torch").load(CK, map_location=dev))
    model.eval()
    q_keys = [f"B:{i}" for i in range(len(queries))]
    f_keys = [f"F:{t}" for t in catalog]
    ow = lambda op: torch.zeros(1, model.op_emb.weight.shape[0]).index_fill_(
        1, torch.tensor([OPS[op]]), 1.0)
    sm = lambda o: torch.tensor(score_mu(model, tok, ix, q_keys, f_keys, o, dev))
    E, A, Sy = sm(ow("ELEM")), sm(ow("HIER")), sm(ow("SYM"))
    C = (qt[[ix[k] for k in q_keys]] @ pt[[ix[k] for k in f_keys]].T).cpu()
    ci = {c: j for j, c in enumerate(catalog)}
    tp = torch.tensor([ci[p] for _, p in queries])
    def rr(M):
        out = []
        for r in range(M.shape[0]):
            s = M[r]
            gt = (s > s[tp[r]]).sum().item()
            eq = (s == s[tp[r]]).sum().item() - 1
            out.append(1.0 / (1 + gt + eq / 2.0))
        return np.array(out)
    gap = np.clip(rr(C) - np.maximum.reduce([rr(E), rr(A), rr(Sy)]), 0, None)
    d = dict(queries=queries, catalog=catalog, gap=gap,
             qvec=qt[[ix[k] for k in q_keys]].numpy())
    __import__("torch").save(d, cache)
    print(f"[s1] gap field: mean {gap.mean():.4f}, nonzero {(gap>0).mean():.2%}", flush=True)
    return d

def run():
    import torch, lmdb
    from mu_attention import build_e5_tables, E5_REVISION
    from eval_filing import is_admin
    from sa_phase1_shadow import REGIONS
    os.makedirs(S, mode=0o700, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = stage1(dev)
    rng = np.random.default_rng(SEED)
    title = {}
    for ln in open(TITLES, encoding="utf-8", errors="replace"):
        i, _, t = ln.rstrip("\n").partition("\t")
        title[int(i)] = t
    env = lmdb.open(LMDB, max_dbs=4, readonly=True, lock=False)
    ac = env.open_db(b"article_category", dupsort=True)
    counts = {}
    with env.begin(db=ac) as txn:
        for _, v in txn.cursor():
            counts[v] = counts.get(v, 0) + 1
    top = sorted(counts, key=counts.get, reverse=True)
    top = [c for c in top if int(c) in title and not is_admin(title[int(c)])][:TOP_CATS]
    texts = {f"C:{c.decode()}": title[int(c)].replace("_", " ") for c in top}
    texts |= {f"A:{a}": a for a in STEM_ANCHORS} | {f"R:{r}": v for r, v in REGIONS.items()}
    _, pt, ix = build_e5_tables(sorted(texts), cache_path=f"{S}/pool_e5.pt",
                                batch_size=256, model_revision=E5_REVISION)
    CV = torch.stack([pt[ix[f"C:{c.decode()}"]] for c in top])
    AV = torch.stack([pt[ix[f"A:{a}"]] for a in STEM_ANCHORS])
    aff = (CV @ AV.T).max(1).values.numpy()
    # --- semantic smear
    QV = torch.tensor(g["qvec"])
    SIM = (CV @ QV.T).numpy()
    K = np.exp((SIM - 1.0) / TAU)
    weak_sem = (K @ g["gap"]) / (K.sum(1) + 1e-9)
    # --- graph smear: seed true cats with gap, diffuse (deg-normalized, both directions)
    seed_w = {}
    for (qt_, p), gp in zip(g["queries"], g["gap"]):
        seed_w[p] = seed_w.get(p, 0.0) + float(gp)
    name2id = {t: i for i, t in title.items()}
    cur = {name2id[p]: v for p, v in seed_w.items() if p in name2id}
    adj = {}
    with open(EDGES) as fh:
        next(fh)
        for ln in fh:
            c, _, p = ln.rstrip("\n").partition("\t")
            c, p = int(c), int(p)
            adj.setdefault(c, []).append(p)
            adj.setdefault(p, []).append(c)
    field = dict(cur)
    for _ in range(T_DIFF):
        nxt = {}
        for node, v in cur.items():
            nbrs = adj.get(node, [])
            if not nbrs:
                continue
            share = ALPHA * v / len(nbrs)
            for nb in nbrs:
                nxt[nb] = nxt.get(nb, 0.0) + share
        for node, v in nxt.items():
            field[node] = field.get(node, 0.0) + v
        cur = nxt
    weak_gr = np.array([field.get(int(c), 0.0) for c in top])
    # normalize both to [0,1]; hybrid = geometric mean
    nz = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    ws, wg = nz(weak_sem), nz(weak_gr)
    weak_hy = np.sqrt((ws + 1e-6) * (wg + 1e-6))
    affn = nz(aff)
    for tag, wk in (("semantic", ws), ("graph", wg), ("hybrid", weak_hy)):
        w = (affn * wk + 1e-8) ** 0.5          # product objective, tempered
        p = w / w.sum()
        top_ix = np.argsort(-p)[:10]
        print(f"[{tag}] top cats: " +
              ", ".join(title[int(top[i])][:32] for i in top_ix), flush=True)
    w = (affn * weak_hy + 1e-8) ** 0.5
    p = w / w.sum()
    sel = rng.choice(len(top), size=min(SAMPLE_CATS, len(top)), replace=False, p=p)
    chosen = {top[i] for i in sel}
    RV = torch.stack([pt[ix[f"R:{r}"]] for r in list(REGIONS)])
    regs = (CV @ RV.T).argmax(1)
    mix = {}
    for i in sel:
        r = list(REGIONS)[int(regs[i])]
        mix[r] = mix.get(r, 0) + 1
    print(f"[diagnostic] region mix of sample: {json.dumps(mix)}", flush=True)
    # --- generation (same emission as v2)
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
    child = np.array(child); parent = np.array(parent)
    o2 = np.argsort(child, kind="stable")
    cs2, ps2 = child[o2], parent[o2]
    ust = np.searchsorted(cs2, np.unique(cs2))
    pars_of = {int(c): (int(a), int(b)) for c, a, b in
               zip(np.unique(cs2), ust, np.append(ust[1:], len(cs2)))}
    members = {}
    for k in keys:
        for c in art_cats[k]:
            members.setdefault(c, []).append(k)
    cat_list = sorted({c for v in art_cats.values() for c in v})
    n_rows = {"ELEM": 0, "LINEAGE": 0, "SYM": 0, "easy": 0}
    with open(f"{S}/targets.tsv", "w", encoding="utf-8") as out:
        out.write('# process_expression\tgap-smeared v3: w=(stem_affinity*hybrid_weakness)'
                  '^0.5; weakness = geomean(semantic kernel tau=0.1, graph diffusion '
                  'alpha=0.5 T=3) over 2000-query gap field vs replay-s3 trunk\n')
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
                node = cand[rng.integers(len(cand))]
                seen.add(node)
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
    print(json.dumps({"rows": n_rows, "articles": len(keys),
                      "sampled_cats": len(chosen)}), flush=True)

run()
