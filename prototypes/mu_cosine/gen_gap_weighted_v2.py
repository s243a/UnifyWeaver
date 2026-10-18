#!/usr/bin/env python3
"""Gap-weighted scale-up sampler (owner objective, 2026-08-07): sample training categories
with weight w(c) = stem_affinity(c) x relative_weakness(region(c)) — the PRODUCT of how
STEM-adjacent a category is (e5 cosine to the anchor set) and how weak mu still is vs e5
in its region (measured per-region gaps from the 3-seed replay readout). The objective is
self-annealing: as training closes a region's gap, its weight falls and budget flows to
the next-weakest front.
Output: ~/mu_data/enwiki_gap_v2/targets.tsv (same schema + v0.5 headers as v1)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

S = os.path.expanduser("~/mu_data/enwiki_gap_v2")
TITLES = ("/tmp/claude-1000/-home-s243a-Projects-UnifyWeaver/"
          "be81a5e2-7bea-4a18-9241-de457531548d/scratchpad/ns14_id_title.tsv")
LMDB = "/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_correct_v2/lmdb"
EDGES = "/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_correct_v2/edges_child_parent.tsv"
# replay-arm mean gaps (PHASE6_REPLAY_S1-3) — the weakness axis, updated per round
REGION_GAPS = {"stem_physical": 0.10, "stem_computing": 0.085, "society": 0.061,
               "culture": 0.144, "geography": 0.037, "everyday": 0.119,
               "life_health": 0.10}   # unmeasured regions get the mean gap ~0.10
STEM_ANCHORS = ["physics", "mathematics", "computer science", "statistics", "engineering",
                "chemistry", "machine learning", "algorithms", "software", "electronics"]
TOP_CATS, SAMPLE_CATS, MAX_ARTICLES, SEED = 60000, 25000, 400000, 3997001
MAX_UP_HOPS, DECAY, SIBS, EASY = 4, 0.85, 2, 2

def run():
    from sa_phase1_shadow import REGIONS
    from mu_attention import build_e5_tables, E5_REVISION
    from eval_filing import is_admin
    import torch, lmdb
    os.makedirs(S, mode=0o700, exist_ok=True)
    rng = np.random.default_rng(SEED)
    title = {}
    for ln in open(TITLES, encoding="utf-8", errors="replace"):
        i, _, t = ln.rstrip("\n").partition("\t")
        title[int(i)] = t
    # membership counts from LMDB (stream)
    env = lmdb.open(LMDB, max_dbs=4, readonly=True, lock=False)
    ac = env.open_db(b"article_category", dupsort=True)
    counts = {}
    with env.begin(db=ac) as txn:
        for _, v in txn.cursor():
            counts[v] = counts.get(v, 0) + 1
    print(f"categories with members: {len(counts)}", flush=True)
    top = sorted(counts, key=counts.get, reverse=True)
    top = [c for c in top if int(c) in title and not is_admin(title[int(c)])][:TOP_CATS]
    print(f"top non-admin candidate cats: {len(top)}", flush=True)
    # e5: stem affinity + region assignment
    texts = {f"C:{c.decode()}": title[int(c)].replace("_", " ") for c in top}
    texts |= {f"A:{a}": a for a in STEM_ANCHORS}
    texts |= {f"R:{r}": v for r, v in REGIONS.items()}
    _, pt, ix = build_e5_tables(sorted(texts), cache_path=os.path.join(S, "cat_e5.pt"),
                                batch_size=256, model_revision=E5_REVISION)
    AV = torch.stack([pt[ix[f"A:{a}"]] for a in STEM_ANCHORS])
    rnames = list(REGIONS)
    RV = torch.stack([pt[ix[f"R:{r}"]] for r in rnames])
    CV = torch.stack([pt[ix[f"C:{c.decode()}"]] for c in top])
    aff = (CV @ AV.T).max(1).values                      # stem affinity
    reg = (CV @ RV.T).argmax(1)                          # region
    gaps = torch.tensor([REGION_GAPS.get(rnames[int(r)], 0.10) for r in reg])
    affn = (aff - aff.min()) / (aff.max() - aff.min() + 1e-9)
    gapn = (gaps - gaps.min()) / (gaps.max() - gaps.min() + 1e-9)
    w = (affn * gapn).numpy() + 1e-6                     # THE PRODUCT OBJECTIVE
    # STRATIFIED allocation (first run collapsed: 99% life_health — one region won both
    # axes and unstratified sampling amplified it). Region budget ~ mean product weight,
    # capped at 35% of the total; sample ~ w only WITHIN each region.
    import collections
    by_reg = collections.defaultdict(list)
    for i in range(len(top)):
        by_reg[rnames[int(reg[i])]].append(i)
    mw = {r: float(np.mean([w[i] for i in ixs])) for r, ixs in by_reg.items()}
    tot = sum(mw.values())
    budget = {r: min(int(SAMPLE_CATS * mw[r] / tot), int(0.35 * SAMPLE_CATS))
              for r in mw}
    scale = SAMPLE_CATS / max(sum(budget.values()), 1)
    budget = {r: min(int(b * scale), len(by_reg[r])) for r, b in budget.items()}
    print(f"region budgets: {json.dumps(budget)}", flush=True)
    sel = []
    for r, ixs in by_reg.items():
        if budget.get(r, 0) < 1:
            continue
        ws = np.array([w[i] for i in ixs]); ps = ws / ws.sum()
        pick = rng.choice(len(ixs), size=min(budget[r], len(ixs)), replace=False, p=ps)
        sel.extend(ixs[j] for j in pick)
    chosen = {top[i] for i in sel}
    reg_share = {}
    for i in sel:
        r = rnames[int(reg[i])]
        reg_share[r] = reg_share.get(r, 0) + 1
    print(f"sampled {len(chosen)} cats; region mix: {json.dumps(reg_share)}", flush=True)
    # collect articles + generate (v1 machinery, inline)
    art_cats = {}
    with env.begin(db=ac) as txn:
        for k, v in txn.cursor():
            if v in chosen:
                art_cats.setdefault(k, []).append(int(v))
    keys = sorted(art_cats)
    if len(keys) > MAX_ARTICLES:
        pick = rng.choice(len(keys), MAX_ARTICLES, replace=False)
        keys = [keys[i] for i in sorted(pick)]
        print(f"CAP: MAX_ARTICLES={MAX_ARTICLES} (sampled, logged)", flush=True)
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
        out.write('# process_expression\tgap-weighted v2: w(c)=stem_affinity*region_gap '
                  '(product objective, self-annealing); graph-judged\n')
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
                      "sampled_cats": len(chosen), "region_mix": reg_share}), flush=True)

run()
