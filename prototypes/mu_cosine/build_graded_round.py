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
import random
from collections import Counter

# corpus prefix (fuse_corpus keys mm:/pt:/wiki:) → CORPORA codebook name, and its default judge.
CORPUS_OF = {"mm": "mindmap", "pt": "pearltrees", "wiki": "enwiki"}
JUDGE_OF = {"mindmap": "human", "pearltrees": "human", "enwiki": "graph"}

# relation → (op, kind, mu_hi, mu_lo). kind: "down" dst is member/narrower of src; "up" dst is broader of
# src; "sym" symmetric. mu_lo = the reverse-direction target (semantic-drift floor) for directional rels.
RELATION_SPEC = {
    "subtopic":       ("HIER", "down", 0.85, 0.12),
    "subcategory":    ("HIER", "down", 0.90, 0.10),
    "element_of":     ("ELEM", "down", 0.90, 0.10),
    "super_category": ("HIER", "up",   0.85, 0.12),
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
            c = ln.rstrip("\n").split("\t")               # a_key, b_key, relation, confidence, raw_text
            if len(c) >= 3:
                conf = float(c[3]) if len(c) > 3 and c[3] else 1.0
                edges.append((c[0], c[1], c[2], conf, c[4] if len(c) > 4 else ""))
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
    ap.add_argument("--bridge-neg-ratio", type=float, default=1.0, help="bridge negatives per positive "
                    "bridge: a bridge source paired with a RANDOM wiki node it is NOT bridged to (μ≈0.1, "
                    "SYM). Teaches discrimination, not just dominance control. 0 disables.")
    ap.add_argument("--bridge-neg-mu", type=float, default=0.1)
    ap.add_argument("--e5-cache", default=None, help="optional e5 table cache (e.g. e5_tables_graded.pt): if "
                    "given, flag bridges whose endpoints are FAR in frozen e5 (likely bad/non-obvious links) "
                    "and quarantine them to <out>_bridge_review.tsv for LLM review before training")
    ap.add_argument("--bridge-min-cos", type=float, default=0.80, help="quarantine a bridge if its endpoints' "
                    "e5 cosine is below this (only with --e5-cache). e5 cosines are COMPRESSED (~0.78–1.0 in "
                    "practice), so this is e5-calibrated — 0.80 flags roughly the suspect bottom decile")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)

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
    row_text = {}                                         # (node,root,op) → raw section-header / annotation text
    def emit(node, root, mu, op, rel, conf=1.0, raw=""):
        if node == root or node not in nodes or root not in nodes:
            skipped["dangling"] += 1
            return
        corpus, judge = corpus_judge(node, root)          # provenance = the human-curated endpoint's corpus
        k = (node, root, op)
        if k in rows and abs(rows[k][0] - 0.5) >= abs(mu - 0.5):
            return                                        # keep the more decisive existing target
        rows[k] = (mu, rel, ntype(node), ntype(root), corpus, judge, conf)
        row_text[k] = raw                                 # carry the raw text alongside (side map)

    for src, dst, rel, conf, raw in edges:
        spec = RELATION_SPEC.get(rel)
        if not spec:
            skipped[f"rel:{rel}"] += 1
            continue
        op, kind, hi, lo = spec
        if kind == "down":                                # dst is a member/narrower of src
            emit(dst, src, hi, op, rel, conf, raw); emit(src, dst, lo, op, rel, conf, raw)
        elif kind == "up":                                # dst is the broader; src belongs to dst
            emit(src, dst, hi, op, rel, conf, raw); emit(dst, src, lo, op, rel, conf, raw)
        else:                                             # symmetric
            emit(dst, src, hi, op, rel, conf, raw); emit(src, dst, hi, op, rel, conf, raw)

    # --- e5-PRIOR bridge sanity gate: a bridge asserts "same concept across corpora"; if its endpoints are
    # FAR in frozen e5, the link is suspect (a bad link, or a non-obvious synonym e5 can't see). Quarantine
    # those for LLM/human review BEFORE training rather than feeding a possibly-wrong μ=0.9 (the user's idea).
    review = []
    if args.e5_cache and os.path.exists(args.e5_cache):
        import torch
        d = torch.load(args.e5_cache, weights_only=False)
        vidx = {n: i for i, n in enumerate(d["names"])}
        vec = d["passage"]                                # unit-normed ⇒ dot product = cosine
        bridge_pairs = {tuple(sorted((n, r))) for (n, r, _op), v in rows.items() if v[1] == "bridge"}
        quarantine = set()
        for a, b in bridge_pairs:
            ia, ib = vidx.get(a), vidx.get(b)
            if ia is None or ib is None:
                continue
            c = float(vec[ia] @ vec[ib])
            if c < args.bridge_min_cos:
                quarantine.add(frozenset((a, b)))
                review.append((a, b, c))
        if quarantine:
            rows = {k: v for k, v in rows.items()
                    if not (v[1] == "bridge" and frozenset((k[0], k[1])) in quarantine)}

    # --- bridge NEGATIVE sampling: each surviving bridge source vs a random wiki node it is NOT bridged to,
    # at μ≈bridge_neg_mu (SYM). Teaches discrimination ("THIS pair is the same, random ones aren't") rather
    # than relying on down-weighting alone. Full-weight (not a "bridge" rel ⇒ trainer won't down-weight them).
    n_neg = 0
    if args.bridge_neg_ratio > 0:
        wiki_nodes = [k for k, v in nodes.items() if v[0] == "wiki"]
        bsrc = {}                                         # mm/pt source → its true wiki bridge targets
        for (n, r, _op), v in rows.items():
            if v[1] == "bridge":
                w, s = (n, r) if nodes.get(n, ("",))[0] == "wiki" else (r, n)
                bsrc.setdefault(s, set()).add(w)
        for s, targets in bsrc.items():
            for _ in range(int(round(args.bridge_neg_ratio))):
                w = rng.choice(wiki_nodes) if wiki_nodes else None
                if not w or w == s or w in targets:
                    continue
                emit(s, w, args.bridge_neg_mu, "SYM", "bridge_neg")
                emit(w, s, args.bridge_neg_mu, "SYM", "bridge_neg")
                n_neg += 1

    if review:
        with open(args.out + "_bridge_review.tsv", "w", encoding="utf-8") as f:
            f.write("# QUARANTINED bridges (e5 cosine < %.2f) — LLM-review before training\n"
                    "# a\tb\te5_cosine\ta_title\tb_title\n" % args.bridge_min_cos)
            for a, b, c in sorted(review, key=lambda x: x[2]):
                f.write(f"{a}\t{b}\t{c:.3f}\t{nodes.get(a,('','',''))[2]}\t{nodes.get(b,('','',''))[2]}\n")

    with open(args.out + "_pairs.tsv", "w", encoding="utf-8") as f:
        f.write("# node\troot\tmu\top\trelation\tnode_type\troot_type\tcorpus\tjudge\tconfidence\traw_text\n")
        for k, (mu, rel, nty, rty, corpus, judge, conf) in sorted(rows.items()):
            node, root, op = k
            raw = row_text.get(k, "").replace("\t", " ").replace("\n", " ")
            f.write(f"{node}\t{root}\t{mu:.2f}\t{op}\t{rel}\t{nty}\t{rty}\t{corpus}\t{judge}\t{conf:.2f}\t{raw}\n")
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
    n_inf = sum(1 for r in rows.values() if r[6] < 1.0)
    print(f"  INFERRED targets (confidence<1.0 → trainer adds operator noise / switch): {n_inf}/{len(rows)}")
    if n_neg:
        print(f"  bridge negatives added: {n_neg} pairs (μ={args.bridge_neg_mu}, full-weight, rel=bridge_neg)")
    if review:
        print(f"  PRIVACY/QUALITY: quarantined {len(review)} far-in-e5 bridges (cos<{args.bridge_min_cos}) "
              f"→ {args.out}_bridge_review.tsv (LLM-review before training)")
    if skipped:
        print(f"  skipped: {dict(skipped)}")
    print(f"  wrote {args.out}_pairs.tsv + {args.out}_nodes.tsv")


if __name__ == "__main__":
    main()
