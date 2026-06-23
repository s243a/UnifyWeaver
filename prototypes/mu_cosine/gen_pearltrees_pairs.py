"""Turn a harvested Pearltrees subtree (fetch_pearltrees_tree.py) into scored-pair candidates for the
pearltrees corpus — a gap-fill for topics enwiki has no category for (e.g. Circuit Theory). Membership
edges (collection_of = subtopic, element_of = member page) are graded by Haiku for CENTRALITY to the
parent topic; shortcuts (cross-references) are emitted as weak associations (not Haiku-judged here).
Extended columns carry relation / node-types / corpus / external-url so the row routes to the right
operator + corpus token and keeps its enwiki anchor (the join key). load_pairs ignores extras.

    python3 gen_pearltrees_pairs.py --tree pt_circuit.tsv --out mu_pairs_pt_circuit.tsv
"""
import argparse
import os
import random

from gen_more_sym_pairs import load_existing_keys

ROOT = os.path.dirname(os.path.abspath(__file__))
OFF_DOMAIN = ["Cooking", "Galaxies", "Feudalism", "Comedians", "Chess_openings", "1978_albums"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=os.path.join(ROOT, "pt_circuit.tsv"))
    ap.add_argument("--neg-ratio", type=float, default=2.0)
    ap.add_argument("--dedup-against", default=os.path.join(ROOT, "mu_pairs_scored_cumulative.tsv"))
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs_pt_circuit.tsv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    edges = []   # (parent, child, relation, child_type, url)
    with open(args.tree, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split("\t")
            if len(c) >= 4:
                edges.append((c[0], c[1], c[2], c[3], c[4] if len(c) > 4 else ""))

    existing = load_existing_keys(args.dedup_against) if os.path.exists(args.dedup_against) else set()
    pairs = set(existing)
    rows = []   # a,b,stratum,wl,mu,relation,a_type,b_type,corpus,url
    members = []
    for par, ch, rel, ctype, url in edges:
        if par == ch:
            continue
        k = tuple(sorted((par, ch)))
        if k in pairs:
            continue
        pairs.add(k)
        atype = "collection"
        if rel == "shortcut":
            # cross-reference: weak symmetric association, not Haiku-graded here
            rows.append((par, ch, "assoc_pt", -1, "", "association", atype, ctype, "pearltrees", url))
        else:
            st = "pos_pt_" + par.lower()[:14]
            rows.append((par, ch, st, -1, "", rel, atype, ctype, "pearltrees", url))
            members.append((par, ch))

    # free negatives: members' children vs off-domain topics
    nodes = sorted({b for _, b in members} | {a for a, _ in members})
    npos = len(members)
    nn = 0
    while nn < args.neg_ratio * npos and nodes:
        a = rng.choice(OFF_DOMAIN)
        b = rng.choice(nodes)
        k = tuple(sorted((a, b)))
        if k in pairs:
            continue
        pairs.add(k)
        rows.append((a, b, "neg", -1, "0.0", "element_of", "collection", "page", "pearltrees", ""))
        nn += 1
    rng.shuffle(rows)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# pearltrees membership candidates (gen_pearltrees_pairs.py; corpus=pearltrees). "
                "a=parent topic b=subtopic/page; Haiku grades CENTRALITY. assoc_pt=shortcut cross-refs. "
                "cols: a\tb\tstratum\twl\tmu\trelation\ta_type\tb_type\tcorpus\turl\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    from collections import Counter
    cnt = Counter(r[2] for r in rows)
    print("wrote " + str(len(rows)) + " -> " + args.out)
    for k, v in sorted(cnt.items()):
        print(f"  {k:24} {v}")
    print(f"to Haiku-score (pos_pt_*): {sum(v for k, v in cnt.items() if k.startswith('pos_pt_'))}")


if __name__ == "__main__":
    main()
