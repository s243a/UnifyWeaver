#!/usr/bin/env python3
"""Generate STEM-targeted transitive-relation training data from enwiki_correct_v2 LMDB.

Owner's relation taxonomy (2026-08-06):
  ELEM     article -> category: the TERMINATING asymmetric relation (pairwise membership).
  LINEAGE  asymmetric TRANSITIVE: elem followed by up-walks over subcat parents —
           article -> cat -> cat^h; graph-judged target decay^h (the lineage_op family).
  SYM      transitive but NOT asymmetric (mixed up+down walks) = ASSOCIATIVE:
           siblings (share a category; up-then-down), graph-judged by parent specificity
           (1/log2(|parent|) — an IDF-style weight: a shared 500-article category is a
           weak claim, a shared 8-article category a strong one).
Targets are GRAPH-judged here; model judges can relabel later (the judge axis is separate).
Region targeting: weakest e5-vs-mu regions on real enwiki (stem_computing +0.566 worst,
per the region map), seeded from STEM + stem-adjacent anchor categories, BFS down the
subcat graph. All caps logged — no silent truncation.
Output: ~/mu_data/enwiki_transitive_v1/targets.tsv
  columns: node_title, other_title, target, op(ELEM|LINEAGE|SYM), kind(pos|sib|easy)
"""
import os, sys, json
import numpy as np

REPO = "/home/s243a/Projects/UnifyWeaver"
S = os.path.expanduser("~/mu_data/enwiki_transitive_v1")
EDGES = f"{REPO}/data/benchmark/enwiki_correct_v2/edges_child_parent.tsv"
LMDB = f"{REPO}/data/benchmark/enwiki_correct_v2/lmdb"
TITLES = "/tmp/claude-1000/-home-s243a-Projects-UnifyWeaver/be81a5e2-7bea-4a18-9241-de457531548d/scratchpad/ns14_id_title.tsv"
ANCHORS = ["Computer_science", "Mathematics", "Physics", "Statistics", "Engineering",
           "Chemistry", "Software", "Algorithms", "Computer_programming", "Data_structures",
           "Machine_learning", "Computer_security", "Computer_networks", "Electronics"]
MAX_DOWN_HOPS, MAX_CATS = 4, 40000
MAX_ARTICLES, MAX_UP_HOPS, DECAY = 120000, 4, 0.85
SIBS_PER_ART, EASY_PER_ART, SEED = 2, 2, 3997001

def run():
    os.makedirs(S, mode=0o700, exist_ok=True)
    rng = np.random.default_rng(SEED)
    title = {}
    for ln in open(TITLES, encoding="utf-8", errors="replace"):
        i, _, t = ln.rstrip("\n").partition("\t")
        title[int(i)] = t
    tid = {t: i for i, t in title.items()}
    print(f"cat titles: {len(title)}", flush=True)
    child, parent = [], []
    with open(EDGES) as fh:
        next(fh)
        for ln in fh:
            c, _, p = ln.rstrip("\n").partition("\t")
            child.append(int(c)); parent.append(int(p))
    child = np.array(child); parent = np.array(parent)
    # parent -> children index (numpy sort-based)
    order = np.argsort(parent, kind="stable")
    ps, cs = parent[order], child[order]
    starts = np.searchsorted(ps, np.unique(ps))
    upar = np.unique(ps)
    kids_of = {int(p): (int(a), int(b)) for p, a, b in
               zip(upar, starts, np.append(starts[1:], len(ps)))}
    # child -> parents index
    order2 = np.argsort(child, kind="stable")
    cs2, ps2 = child[order2], parent[order2]
    ustart = np.searchsorted(cs2, np.unique(cs2))
    uch = np.unique(cs2)
    pars_of = {int(c): (int(a), int(b)) for c, a, b in
               zip(uch, ustart, np.append(ustart[1:], len(cs2)))}
    # BFS DOWN from anchors
    seeds = [tid[a] for a in ANCHORS if a in tid]
    print(f"anchors resolved: {len(seeds)}/{len(ANCHORS)}", flush=True)
    sub, frontier = set(seeds), list(seeds)
    for hop in range(MAX_DOWN_HOPS):
        nxt = []
        for x in frontier:
            if x in kids_of:
                a, b = kids_of[x]
                for k in cs[a:b]:
                    k = int(k)
                    if k not in sub:
                        sub.add(k); nxt.append(k)
            if len(sub) >= MAX_CATS:
                break
        frontier = nxt
        print(f"  down hop {hop+1}: subtree {len(sub)} cats", flush=True)
        if len(sub) >= MAX_CATS:
            print(f"  CAP HIT: MAX_CATS={MAX_CATS} (subtree truncated — logged, not silent)",
                  flush=True)
            break
    # articles in subtree via LMDB article_category + article_meta
    import lmdb
    env = lmdb.open(LMDB, max_dbs=4, readonly=True, lock=False)
    ac = env.open_db(b"article_category", dupsort=True)
    meta = env.open_db(b"article_meta")
    subs = {str(x).encode() for x in sub}
    art_cats = {}
    with env.begin(db=ac) as txn:
        cur = txn.cursor()
        for k, v in cur:
            if v in subs:
                art_cats.setdefault(k, []).append(int(v))
    print(f"articles touching subtree: {len(art_cats)}", flush=True)
    keys = sorted(art_cats)
    if len(keys) > MAX_ARTICLES:
        sel = rng.choice(len(keys), MAX_ARTICLES, replace=False)
        keys = [keys[i] for i in sorted(sel)]
        print(f"  CAP HIT: MAX_ARTICLES={MAX_ARTICLES} (sampled — logged)", flush=True)
    atitle = {}
    with env.begin(db=meta) as txn:
        for k in keys:
            t = txn.get(k, db=meta)
            if t is not None:
                atitle[k] = t.decode("utf-8", errors="replace")
    keys = [k for k in keys if k in atitle]
    print(f"articles with metadata (non-admin-categorized): {len(keys)}", flush=True)
    cat_list = sorted(sub)
    all_arts = keys
    n_rows = {"ELEM": 0, "LINEAGE": 0, "SYM": 0, "easy": 0}
    # members per cat (for sibling sampling + specificity weight)
    members = {}
    for k in keys:
        for c in art_cats[k]:
            members.setdefault(c, []).append(k)
    with open(f"{S}/targets.tsv", "w", encoding="utf-8") as out:
        out.write("# process_expression\tenwiki_transitive_v1: elem(terminating) + "
                  "lineage(asym transitive, decay=0.85) + assoc(sym, idf-weighted siblings); "
                  "graph-judged; STEM-targeted subtree\n")
        out.write("# node\tother\ttarget\top\tkind\n")
        for k in keys:
            at = atitle[k].replace("\t", " ").replace("_", " ")
            direct = art_cats[k]
            for c in direct:
                out.write(f"{at}\t{title[c].replace('_',' ')}\t1.0\tELEM\tpos\n")
                n_rows["ELEM"] += 1
            # LINEAGE: up-walk from one random direct cat
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
                if node in title:
                    out.write(f"{at}\t{title[node].replace('_',' ')}\t"
                              f"{round(DECAY**h, 6)}\tLINEAGE\tpos\n")
                    n_rows["LINEAGE"] += 1
            # SYM siblings: share cat c — target = idf-style specificity of the shared cat
            mem = members.get(c, [])
            if len(mem) > 1:
                w = round(min(1.0, 1.0 / max(np.log2(len(mem)), 1.0)), 6)
                for _ in range(SIBS_PER_ART):
                    s = mem[rng.integers(len(mem))]
                    if s != k:
                        out.write(f"{at}\t{atitle[s].replace(chr(9),' ').replace('_',' ')}\t"
                                  f"{w}\tSYM\tsib\n")
                        n_rows["SYM"] += 1
            for _ in range(EASY_PER_ART):
                e = cat_list[rng.integers(len(cat_list))]
                if e not in direct and e in title:
                    out.write(f"{at}\t{title[e].replace('_',' ')}\t0.0\tELEM\teasy\n")
                    n_rows["easy"] += 1
    print(json.dumps({"rows": n_rows, "articles": len(keys), "subtree_cats": len(sub),
                      "out": f"{S}/targets.tsv"}), flush=True)

run()
