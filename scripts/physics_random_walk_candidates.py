#!/usr/bin/env python3
"""Random walks DOWN the Wikipedia category graph from a root, to surface candidate
(often deep) topic nodes for LLM physics-classification. Deterministic (seeded).

The downward cone leaks out of physics within a few hops (Wikipedia categories are
associative, not is-a), so walks are a *sampler* of reachable nodes; the LLM filter
(separate step) decides which are genuinely physics. Output: TSV `node<TAB>visits`
sorted by visit count (a rough relevance proxy), candidates only (root excluded).
"""
import argparse, collections, random

def load_children(path):
    ch = collections.defaultdict(list)
    with open(path) as f:
        next(f)
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) >= 2:
                ch[a[1]].append(a[0])  # parent -> child
    return ch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--root", default="Physics")
    ap.add_argument("--walks", type=int, default=400)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--top", type=int, default=200)
    args = ap.parse_args()
    ch = load_children(args.tsv)
    rng = random.Random(args.seed)
    visits = collections.Counter()
    for _ in range(args.walks):
        node = args.root
        for _ in range(args.depth):
            kids = ch.get(node)
            if not kids:
                break
            node = rng.choice(kids)
            visits[node] += 1
    visits.pop(args.root, None)
    for name, c in visits.most_common(args.top):
        print(f"{name}\t{c}")

if __name__ == "__main__":
    main()
