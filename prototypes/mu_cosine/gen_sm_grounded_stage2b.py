#!/usr/bin/env python3
"""SimpleMind OOD arc — STAGE 2b: GROUNDED targeting (owner's method; replaces the failed
popularity-pool smear wholesale).

Chain: SM-FS query -> root-node pearltrees link -> harvested tree -> the Wikipedia page/
category pearls the OWNER filed there -> enwiki node -> its category neighborhood. The pool
IS the failure region by construction; no popularity ranking, no region classifier, no
fuzzy title matching (typos irrelevant — the link is the identity). stem_affinity dropped
(grounding makes it redundant; its distortion broke the previous sampler).

Weakness seeding: each grounded category receives the SUM of gaps of the SM-FS queries
grounding into it. Field spread by graph diffusion (deg-normalized, alpha, T) and reported;
generation samples ~ field^0.5 over the diffused support. Coverage + failures logged."""
import os, sys, json, re, glob
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

S = os.path.expanduser("~/mu_data/sm_gap_v1")
GROUND = "/home/s243a/Projects/UnifyWeaver/.local/data/pearltrees_api/smfs_ground"
LMDB = "/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_correct_v2/lmdb"
EDGES = "/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_correct_v2/edges_child_parent.tsv"
TITLES = ("/tmp/claude-1000/-home-s243a-Projects-UnifyWeaver/"
          "be81a5e2-7bea-4a18-9241-de457531548d/scratchpad/ns14_id_title.tsv")
ROOTMAP = os.path.expanduser("~/mu_data/sm_gap_v1/query_tree_map.json")
SEED, ALPHA, T_DIFF = 3997001, 0.5, 3
SAMPLE_CATS, MAX_ARTICLES = 20000, 350000
MAX_UP_HOPS, DECAY, SIBS, EASY = 4, 0.85, 2, 2

def build_query_tree_map():
    """query path -> pearltrees tree id (from the .smmx root links, cached)."""
    import zipfile
    import xml.etree.ElementTree as ET
    import torch
    if os.path.exists(ROOTMAP):
        return json.load(open(ROOTMAP))
    g = torch.load(f"{S}/gapfield.pt", weights_only=False)
    ROOT = "/mnt/c/Users/johnc/Dropbox/root"
    out = {}
    for path in g["queries"]:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            continue
        try:
            with zipfile.ZipFile(p) as z:
                data = z.read(next(n for n in z.namelist() if n.endswith("mindmap.xml")))
            rt = next((el for el in ET.fromstring(data).iter("topic")
                       if el.get("parent") == "-1"), None)
            link = rt.find("link") if rt is not None else None
            url = link.get("urllink") if link is not None else None
            m = re.search(r"/id(\d+)", url or "")
            if m:
                out[path] = m.group(1)
        except Exception:
            pass
    json.dump(out, open(ROOTMAP, "w"))
    return out

def run():
    import torch, lmdb
    from eval_filing import is_admin
    g = torch.load(f"{S}/gapfield.pt", weights_only=False)
    gap_of = {q: float(x) for q, x in zip(g["queries"], g["gap"])}
    qtree = build_query_tree_map()
    # wikipedia pearls per harvested tree
    wiki_of_tree = defaultdict(list)
    for f in glob.glob(f"{GROUND}/pt_*.tsv"):
        tid = os.path.basename(f)[3:-4]
        for ln in open(f, encoding="utf-8", errors="replace"):
            if ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 5 and "wikipedia.org/wiki/" in parts[4]:
                m = re.search(r"wikipedia\.org/wiki/([^?#]+)", parts[4])
                if m:
                    wiki_of_tree[tid].append(m.group(1))
    # ground: query -> wiki targets -> seed categories with gap
    title = {}
    for ln in open(TITLES, encoding="utf-8", errors="replace"):
        i, _, t = ln.rstrip("\n").partition("\t")
        title[int(i)] = t
    cat_id = {t: i for i, t in title.items()}
    env = lmdb.open(LMDB, max_dbs=4, readonly=True, lock=False)
    meta = env.open_db(b"article_meta")
    ac = env.open_db(b"article_category", dupsort=True)
    art_id = {}
    with env.begin(db=meta) as txn:
        for k, v in txn.cursor():
            art_id[v.decode("utf-8", errors="replace").replace(" ", "_")] = k
    seed = defaultdict(float)
    n_grounded = n_cat_direct = n_via_article = n_miss = 0
    for q, tid in qtree.items():
        targets = wiki_of_tree.get(tid, [])
        if not targets:
            continue
        gp = gap_of.get(q, 0.0)
        hit = False
        for t in targets:
            t = t.strip("/")
            if t.startswith("Category:"):
                name = t[len("Category:"):]
                if name in cat_id:
                    seed[cat_id[name]] += gp; hit = True; n_cat_direct += 1
            else:
                k = art_id.get(t)
                if k is not None:
                    with env.begin(db=ac) as txn:
                        cur = txn.cursor()
                        if cur.set_key(k):
                            any_seeded = False
                            for v in cur.iternext_dup():
                                cid = int(v)
                                if cid in title and not is_admin(title[cid]):
                                    seed[cid] += gp; any_seeded = True
                            if any_seeded:
                                hit = True; n_via_article += 1
        if hit:
            n_grounded += 1
        else:
            n_miss += 1
    print(json.dumps({"queries_with_tree": len(qtree),
                      "grounded_queries": n_grounded,
                      "category_urls": n_cat_direct, "via_article": n_via_article,
                      "grounded_but_unmatched": n_miss,
                      "seed_categories": len(seed)}), flush=True)
    # diffuse over the category graph
    admin_ids = {i for i, t in title.items() if is_admin(t)}
    print(f"[graph] excluding {len(admin_ids)} admin categories from diffusion", flush=True)
    adj = {}
    with open(EDGES) as fh:
        next(fh)
        for ln in fh:
            c, _, p = ln.rstrip("\n").partition("\t")
            c, p = int(c), int(p)
            if c in admin_ids or p in admin_ids:
                continue          # admin cats are universal hubs; routing through them
            adj.setdefault(c, []).append(p); adj.setdefault(p, []).append(c)
    field, cur = dict(seed), dict(seed)
    for _ in range(T_DIFF):
        nxt = defaultdict(float)
        for node, v in cur.items():
            nb = adj.get(node, [])
            if not nb:
                continue
            sh = ALPHA * v / len(nb)
            for x in nb:
                nxt[x] += sh
        for node, v in nxt.items():
            field[node] = field.get(node, 0.0) + v
        cur = dict(nxt)
    pool = [(c, w) for c, w in field.items()
            if c in title and not is_admin(title[c]) and w > 0]
    pool.sort(key=lambda x: -x[1])
    print(f"[pool] diffused support: {len(pool)} categories", flush=True)
    print("[preview] top 15: " +
          ", ".join(title[c][:30] for c, _ in pool[:15]), flush=True)
    rng = np.random.default_rng(SEED)
    ws = np.array([w for _, w in pool]) ** 0.5
    p = ws / ws.sum()
    k = min(SAMPLE_CATS, len(pool))
    sel = rng.choice(len(pool), size=k, replace=False, p=p)
    chosen = {str(pool[i][0]).encode() for i in sel}
    # generation
    art_cats = {}
    with env.begin(db=ac) as txn:
        for kk, v in txn.cursor():
            if v in chosen:
                art_cats.setdefault(kk, []).append(int(v))
    keys = sorted(art_cats)
    if len(keys) > MAX_ARTICLES:
        pick = rng.choice(len(keys), MAX_ARTICLES, replace=False)
        keys = [keys[i] for i in sorted(pick)]
    atitle = {}
    with env.begin(db=meta) as txn:
        for kk in keys:
            t = txn.get(kk, db=meta)
            if t is not None:
                atitle[kk] = t.decode("utf-8", errors="replace")
    keys = [kk for kk in keys if kk in atitle]
    print(f"[gen] articles with metadata: {len(keys)}", flush=True)
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
    members = defaultdict(list)
    for kk in keys:
        for c in art_cats[kk]:
            members[c].append(kk)
    cat_list = sorted({c for v in art_cats.values() for c in v})
    n_rows = defaultdict(int)
    with open(f"{S}/targets_grounded.tsv", "w", encoding="utf-8") as out:
        out.write('# process_expression\tSM-FS GROUNDED targeting: owner-curated '
                  'pearltrees->wikipedia links seed gap field; graph diffusion; no '
                  'popularity pool, no region classifier, no fuzzy matching\n')
        out.write('#   ELEM    = lineage(enwiki,mu=graph,estimand="element_of")\n')
        out.write('#   LINEAGE = lineage(enwiki,decay=0.85,mu=graph,estimand="ancestry")\n')
        out.write('#   SYM     = cowalk(enwiki,walk="sibling",weight="idf_node_size",'
                  'mu=graph,estimand="path")\n')
        out.write("# node\tother\ttarget\top\tkind\n")
        for kk in keys:
            at = atitle[kk].replace("\t", " ").replace("_", " ")
            direct = art_cats[kk]
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
                    if s2 != kk and s2 in atitle:
                        out.write(f"{at}\t{atitle[s2].replace(chr(9),' ').replace('_',' ')}"
                                  f"\t{wgt}\tSYM\tsib\n")
                        n_rows["SYM"] += 1
            for _ in range(EASY):
                e = cat_list[rng.integers(len(cat_list))]
                if e not in direct and e in title and not is_admin(title[e]):
                    out.write(f"{at}\t{title[e].replace('_',' ')}\t0.0\tELEM\teasy\n")
                    n_rows["easy"] += 1
    print(json.dumps({"rows": dict(n_rows), "articles": len(keys),
                      "sampled_cats": k, "out": f"{S}/targets_grounded.tsv"}), flush=True)

run()
