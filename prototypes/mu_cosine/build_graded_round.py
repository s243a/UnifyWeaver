#!/usr/bin/env python3
"""Turn one or more PRIVACY-SCRUBBED fused neighbourhoods (fuse_corpus.py → <prefix>_nodes.tsv +
<prefix>_edges.tsv) into a calibrated, tagged GRADED ROUND for the mu-attention trainer.

Each fused edge (src, dst, relation) becomes directional μ targets under one of the existing operators,
with the relation's calibrated μ (the human-judge target) — corpus differences are NOT baked into the
target; they ride the maskable provenance token so the model conditions/marginalizes them
(DESIGN_calibrated_judges.md, DESIGN_provenance_and_representation.md).

  RELATION → OPERATOR + μ (see RELATION_SPEC):
    subtopic / subcategory   → WIKI  (directional narrower-membership): μ(member|container) high, reverse low
    element_of               → ELEM  (page/collection membership):      μ(member|container) high, reverse low
    super_category           → WIKI  (the dst is the BROADER one):       μ(src|dst) high, reverse low
    see_also / assoc / sequence → SYM (symmetric, weak associative):     μ both directions ~low-mid
    bridge                   → SYM  (SAME concept across corpora):       μ both directions high

Edge direction convention from the parsers: (src=container/parent, dst=member/child), except super_category
where dst is the broader node. An item is (node, root, op) ⇒ a target for μ(node|root).

Outputs (consumed by the trainer in the next step; not committed — derived from gitignored fused data):
  <out>_pairs.tsv : node  root  mu  op  relation  node_type  root_type  corpus  judge
  <out>_nodes.tsv : key  corpus  node_type  title  embed_text   (embed_text = the e5 string per node)

    python3 build_graded_round.py --fused /tmp/cyb_fused_2 --fused /tmp/ds_fused_3 --out /tmp/graded
"""
import argparse
import os
from collections import Counter

# corpus prefix (fuse_corpus keys mm:/pt:/wiki:) → CORPORA codebook name, and its default judge.
CORPUS_OF = {"mm": "mindmap", "pt": "pearltrees", "wiki": "enwiki"}
JUDGE_OF = {"mindmap": "human", "pearltrees": "human", "enwiki": "graph"}

# relation → (op, kind, mu_hi, mu_lo). kind: "down" dst is member/narrower of src; "up" dst is broader of
# src; "sym" symmetric. mu_lo = the reverse-direction target (semantic-drift floor) for directional rels.
RELATION_SPEC = {
    "subtopic":       ("WIKI", "down", 0.85, 0.12),
    "subcategory":    ("WIKI", "down", 0.90, 0.10),
    "element_of":     ("ELEM", "down", 0.90, 0.10),
    "super_category": ("WIKI", "up",   0.85, 0.12),
    "see_also":       ("SYM",  "sym",  0.40, 0.40),
    "assoc":          ("SYM",  "sym",  0.30, 0.30),
    "sequence":       ("SYM",  "sym",  0.30, 0.30),
    "bridge":         ("SYM",  "sym",  0.90, 0.90),   # same concept, different corpus/node-type
}


def load_fused(prefix):
    nodes = {}                                            # key → (corpus_prefix, node_type, title)
    with open(prefix + "_nodes.tsv", encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split("\t")               # key, corpus, node_type, title
            if len(c) >= 4:
                nodes[c[0]] = (c[1], c[2], c[3])
    edges = []
    with open(prefix + "_edges.tsv", encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split("\t")               # a_key, b_key, relation
            if len(c) >= 3:
                edges.append((c[0], c[1], c[2]))
    return nodes, edges


def embed_text(corpus, node_type, title, pid=""):
    """The e5 string for a node. Hook for the role/identity prefix (DESIGN §Cheaper-than-a-transform): a
    Pearltrees TEAM would be 'Team <title> <id>'. We have no teams (groups account) harvested yet, so for
    now every node embeds its plain title; the prefix activates when the s243a_groups account is added."""
    return title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", action="append", required=True, help="fused prefix (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    nodes = {}                                            # key → (corpus_prefix, node_type, title)
    edges = []
    for pref in args.fused:
        n, e = load_fused(pref)
        nodes.update(n)
        edges.extend(e)

    def corpus_judge(node, root):
        # Provenance = the corpus that ASSERTED the edge = the HUMAN-curated (non-enwiki) endpoint. A bridge
        # is mm/pt ↔ wiki: enwiki didn't assert it, the mindmap/pearltrees curation did — so BOTH directions
        # get that side's provenance, not enwiki/graph. Priority mindmap > pearltrees > enwiki.
        prefixes = {nodes.get(node, ("",))[0], nodes.get(root, ("",))[0]}
        corpus = ("mindmap" if "mm" in prefixes else
                  "pearltrees" if "pt" in prefixes else "enwiki")
        return corpus, JUDGE_OF[corpus]

    def ntype(key):
        return nodes.get(key, ("", "category", ""))[1]

    rows, skipped = {}, Counter()                         # (node,root,op) → row ; dedup keeps strongest |μ-.5|
    def emit(node, root, mu, op, rel):
        if node == root or node not in nodes or root not in nodes:
            skipped["dangling"] += 1
            return
        corpus, judge = corpus_judge(node, root)          # provenance = the human-curated endpoint's corpus
        k = (node, root, op)
        if k in rows and abs(rows[k][0] - 0.5) >= abs(mu - 0.5):
            return                                        # keep the more decisive existing target
        rows[k] = (mu, rel, ntype(node), ntype(root), corpus, judge)

    for src, dst, rel in edges:
        spec = RELATION_SPEC.get(rel)
        if not spec:
            skipped[f"rel:{rel}"] += 1
            continue
        op, kind, hi, lo = spec
        if kind == "down":                                # dst is a member/narrower of src
            emit(dst, src, hi, op, rel); emit(src, dst, lo, op, rel)
        elif kind == "up":                                # dst is the broader; src belongs to dst
            emit(src, dst, hi, op, rel); emit(dst, src, lo, op, rel)
        else:                                             # symmetric
            emit(dst, src, hi, op, rel); emit(src, dst, hi, op, rel)

    with open(args.out + "_pairs.tsv", "w", encoding="utf-8") as f:
        f.write("# node\troot\tmu\top\trelation\tnode_type\troot_type\tcorpus\tjudge\n")
        for (node, root, op), (mu, rel, nty, rty, corpus, judge) in sorted(rows.items()):
            f.write(f"{node}\t{root}\t{mu:.2f}\t{op}\t{rel}\t{nty}\t{rty}\t{corpus}\t{judge}\n")
    with open(args.out + "_nodes.tsv", "w", encoding="utf-8") as f:
        f.write("# key\tcorpus\tnode_type\ttitle\tembed_text\n")
        for key, (cp, nty, title) in sorted(nodes.items()):
            corpus = CORPUS_OF.get(cp, "mindmap")
            f.write(f"{key}\t{corpus}\t{nty}\t{title}\t{embed_text(corpus, nty, title)}\n")

    op_c = Counter(op for (_, _, op) in rows)
    rel_c = Counter(r[1] for r in rows.values())
    cj_c = Counter((r[4], r[5]) for r in rows.values())
    print(f"{len(nodes)} nodes, {len(edges)} fused edges → {len(rows)} graded targets")
    print(f"  by op:       {dict(op_c)}")
    print(f"  by relation: {dict(rel_c)}")
    print(f"  by corpus⊗judge: { {f'{c}/{j}': n for (c, j), n in cj_c.items()} }")
    if skipped:
        print(f"  skipped: {dict(skipped)}")
    print(f"  wrote {args.out}_pairs.tsv + {args.out}_nodes.tsv")


if __name__ == "__main__":
    main()
