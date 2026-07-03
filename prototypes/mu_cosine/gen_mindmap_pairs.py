#!/usr/bin/env python3
"""Mindmap pair generator (DESIGN_mindmap_lineage.md). Implements the design contract:

  * POSITIVES (LINEAGE / MSE): every (node, ancestor) pair, mu = gamma^(d-1) graded by up-distance d
    (direct parent d=1 -> 1.0). Per-chain weighting (1/#pairs-in-chain) to counter depth bias.
  * NEGATIVES (LINEAGE / MSE): substitution candidates (non-ancestor, non-descendant), structural graph-judge
    mu_graph = max(floor, gamma^hops(c,p) * lca_frac(c,p)); unreachable -> hops = D_max+1 (graceful). Typed by
    hops: 1-2 hard (sibling/cousin), 3-4 medium, >=5/unreachable easy.
  * RANK groups (LINEAGE-RANK / CE): per node, [true parent] + negatives, teacher
    rank_score = (1-beta)*mu_graph + beta*e5_sim_prefix_norm   (convex; e5 on the GRANDPARENT prefix),
    beta = (G/(L+G))*k, G/L in nodes, k<=1. CE target = the true parent.

Maps are UNIONED into one graph (poor-man's fusion for cross-map connectivity; real fusion = fuse_corpus.py +
Pearltrees). Privacy scrubbed upstream (parse_smmx). Judge tag = 'graph'.

Outputs (graded-round compatible):
  <out>_pairs.tsv  : node  root  mu  op  relation  node_type  root_type  corpus  judge   (LINEAGE MSE rows)
  <out>_nodes.tsv  : key  corpus  node_type  title  embed_text
  <out>_rank.jsonl : {"node","true","candidates":[{"key","mu_graph","e5_prefix_sim","rank_score"}...],"beta"}

  python3 gen_mindmap_pairs.py --maps context/*.smmx --out mindmap_lin --gamma 0.6 --k-neg 6
"""
import argparse, glob, json, math, os, random, sys
from collections import Counter, defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_mindmap_lineage import parse_map, lineage

CORPUS = "mindmap"
NT = "mindmap_node"
JUDGE = "graph"


def bfs_dist(adj, src):
    dist, q = {src: 0}, deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return dist


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def descendants(node, children):
    seen, stack = set(), [node]
    while stack:
        for c in children.get(stack.pop(), ()):
            if c not in seen:
                seen.add(c); stack.append(c)
    return seen


def load_e5():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("intfloat/e5-small-v2")


def neg_type(hops, reachable):
    if not reachable or hops >= 5:
        return "easy"
    if hops <= 2:
        return "hard"
    return "medium"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=sorted(glob.glob(
        os.path.join(os.path.dirname(__file__), "context", "*.smmx"))))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "mindmap_lin"))
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--floor", type=float, default=0.02)
    ap.add_argument("--k-neg", type=int, default=6, help="negative candidates per node")
    ap.add_argument("--k-blend", type=float, default=1.0, help="beta blend const k (<=1)")
    ap.add_argument("--temp", type=float, default=0.2, help="teacher softmax temperature (T<1 sharpens the CE target)")
    ap.add_argument("--tmp", default="/tmp/mu_data/mm_parse")
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    rng = random.Random(0)

    # --- UNION all maps into one graph (poor-man's fusion for connectivity) ---
    title, parent = {}, {}
    for smmx in a.maps:
        t, pa, _fb, err = parse_map(smmx, a.tmp)
        if not t:
            print(f"  SKIP {os.path.basename(smmx)}: {err.strip()[:60]}"); continue
        title.update(t)
        for n, p in pa.items():
            parent.setdefault(n, p)                            # first map's structural parent wins
    children = defaultdict(set)
    adj = defaultdict(set)
    for c, p in parent.items():
        children[p].add(c); adj[c].add(p); adj[p].add(c)
    D_max = max((len(lineage(n, parent)) for n in title), default=1)
    print(f"fused graph: {len(title)} nodes, {len(parent)} parent-edges, D_max={D_max}")

    e5 = load_e5()
    keys = sorted(title)
    vecs = {k: v for k, v in zip(keys, e5.encode(
        ["passage: " + title[k] for k in keys], normalize_embeddings=True))}

    def prefix_text(node):                                     # grandparent prefix = root..grandparent
        lin = lineage(node, parent)
        return " / ".join(title.get(k, k) for k in lin[:-1]) if len(lin) >= 2 else title.get(node, node)

    def e5_prefix_sim(c, p):                                   # cosine of grandparent-prefix texts
        va, vb = e5.encode(["passage: " + prefix_text(c), "passage: " + prefix_text(p)],
                           normalize_embeddings=True)
        return float((va * vb).sum())

    pair_rows, rank_groups = [], []
    dist_cache = {}
    depth_pairs, neg_types_w = Counter(), Counter()
    entropies, unreachable = [], 0

    filed = [n for n in title if n in parent]
    for n in filed:
        p = parent[n]
        lin_n = lineage(n, parent)                             # root..n
        L = len(lin_n)
        ancestors = lin_n[:-1]                                 # root..parent (graded POSITIVES)
        # --- POSITIVES (MSE): (n, ancestor) mu = gamma^(d-1), d = up-distance ---
        n_anc = len(ancestors)
        w = 1.0 / max(1, n_anc)                                # per-chain weight (depth-bias)
        depth_pairs[L] += n_anc
        for i, anc in enumerate(reversed(ancestors)):          # i=0 -> parent (d=1)
            mu = round(a.gamma ** i, 4)
            pair_rows.append((n, anc, mu, "LINEAGE", "ancestor", NT, NT, CORPUS, JUDGE))
        # --- NEGATIVES (MSE) + RANK candidates ---
        if p not in dist_cache:
            dist_cache[p] = bfs_dist(adj, p)
        dist = dist_cache[p]
        lin_p = lineage(p, parent)
        forbidden = descendants(n, children) | set(ancestors) | {n}
        # STRATIFIED negative sampling weighted toward HARD (near siblings/cousins) — uniform sampling over the
        # whole graph almost never hits them, inverting the intended hard:medium:easy ratio.
        buckets = {"hard": [], "medium": [], "easy": []}
        for m in title:
            if m in forbidden:
                continue
            buckets[neg_type(dist.get(m, D_max + 1), m in dist)].append(m)
        for b in buckets.values():
            rng.shuffle(b)
        want = {"hard": round(a.k_neg * 0.5), "medium": round(a.k_neg * 0.33)}
        want["easy"] = a.k_neg - want["hard"] - want["medium"]      # ~ 1.5 : 1 : 0.5
        negs = []
        for b in ("hard", "medium", "easy"):
            negs += buckets[b][:want[b]]
        leftover = [m for b in ("hard", "medium", "easy") for m in buckets[b][want[b]:]]
        negs = (negs + leftover)[:a.k_neg]                          # top up if a bucket was short
        G = max(0, L - 2)
        beta = (G / (L + G)) * a.k_blend if (L + G) else 0.0
        cand_list = [{"key": p, "mu_graph": 1.0, "e5_prefix_sim": 1.0,
                      "rank_score": round((1 - beta) * 1.0 + beta * 1.0, 4)}]
        for c in negs:
            reachable = c in dist
            hops = dist.get(c, D_max + 1)
            if not reachable:
                unreachable += 1
            frac = common_prefix_len(lineage(c, parent), lin_p) / max(1, len(lin_p))
            mu_g = max(a.floor, (a.gamma ** hops) * frac)
            t = neg_type(hops, reachable)
            neg_types_w[t] += w
            pair_rows.append((n, c, round(mu_g, 4), "LINEAGE", f"neg_{t}", NT, NT, CORPUS, JUDGE))
            sim = e5_prefix_sim(c, p)
            sim_norm = (sim + 1.0) / 2.0
            rs = (1 - beta) * mu_g + beta * sim_norm
            cand_list.append({"key": c, "mu_graph": round(mu_g, 4),
                              "e5_prefix_sim": round(sim, 4), "rank_score": round(rs, 4)})
        # teacher softmax entropy over candidates AT TEMPERATURE T (the CE target; collapse diagnostic)
        scores = [x["rank_score"] for x in cand_list]
        mx = max(scores); exps = [math.exp((s - mx) / a.temp) for s in scores]; Z = sum(exps)
        probs = [e / Z for e in exps]
        entropies.append(-sum(pp * math.log(pp + 1e-12) for pp in probs))
        rank_groups.append({"node": n, "true": p, "beta": round(beta, 4), "temp": a.temp,
                            "candidates": cand_list})

    # --- write outputs ---
    with open(a.out + "_pairs.tsv", "w", encoding="utf-8") as f:
        f.write("# node\troot\tmu\top\trelation\tnode_type\troot_type\tcorpus\tjudge\n")
        for r in pair_rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    with open(a.out + "_nodes.tsv", "w", encoding="utf-8") as f:
        f.write("# key\tcorpus\tnode_type\ttitle\tembed_text\n")
        for k in keys:
            f.write(f"{k}\t{CORPUS}\t{NT}\t{title[k]}\t{title[k]}\n")
    with open(a.out + "_rank.jsonl", "w", encoding="utf-8") as f:
        for g in rank_groups:
            f.write(json.dumps(g) + "\n")

    npos = sum(1 for r in pair_rows if r[4] == "ancestor")
    nneg = len(pair_rows) - npos
    print(f"\nwrote {len(pair_rows)} MSE rows ({npos} pos / {nneg} neg), {len(rank_groups)} rank groups -> {a.out}_*")
    print(f"  pair-count by depth: {dict(sorted(depth_pairs.items()))}")
    print(f"  WEIGHTED negative-type dist: { {k: round(v,1) for k,v in neg_types_w.items()} }")
    print(f"  teacher entropy mean/min: {sum(entropies)/max(1,len(entropies)):.3f} / {min(entropies):.3f}  "
          f"(low => flat teacher, weak CE grad)")
    print(f"  unreachable negatives (single-map fragmentation, hops=D_max+1): {unreachable}")


if __name__ == "__main__":
    main()
