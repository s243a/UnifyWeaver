#!/usr/bin/env python3
"""Produce a **dense** μ map by running MiniLM directly — no training (kickoff prompt A).

For every category in `data/benchmark/10k/category_parent.tsv`, MiniLM-encode its name
(`sentence-transformers/all-MiniLM-L6-v2`), take the cosine to a single anchor embedding (default
`Physics`), clamp to `[0,1]` (`mu_encoder.to_membership`), and emit `name<TAB>μ` in the format of
`tests/fixtures/wikipedia_physics_fuzzy_nodes.tsv` (a '#'-comment header, then `name<TAB>μ`).

This is the *fastest path to unblock the graph work*: a raw pretrained-embedding cosine, NOT the
trained cosine-μ encoder. So read the μ as a coarse semantic-proximity prior, not an authoritative
membership — it has no domain supervision (the Haiku-scored fixture does). Its job is DENSITY: a μ for
*every* node so `descendant_mu_mass_gated` / `gated_ic` / `lin_from_ic` no longer prune through
unscored connectors. Training the real encoder (the separate project) is what makes μ authoritative.

Two integration guards (see README "Integration with the WAM-Rust core"), both enforced here:
  1. CLAMP cosine ∈ [-1,1] to μ ∈ [0,1] via `to_membership` — the Rust mass functions assume μ ≥ 0.
  2. Emit names VERBATIM from the TSV — the Rust loaders intern names → ids; a mismatch silently
     becomes μ=0. We encode a human-readable form (underscores → spaces, the natural reading of a
     Wikipedia category title) but emit the verbatim underscore name, and report coverage.

Deps: `pip install -r requirements.txt` + HuggingFace egress (for the model weights). Pure-stdlib
fallback is intentionally NOT provided: the point is a real embedding, so a missing model is a hard
error, not a silent random map.

    python3 gen_dense_mu.py --anchor Physics --out dense_mu_physics.tsv
"""
import argparse
import os
import sys

from mu_encoder import to_membership

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
GRAPH = os.path.join(REPO, "data", "benchmark", "10k", "category_parent.tsv")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_category_names(path):
    """Unique category names (nodes) from a `child<TAB>parent` edge list, in first-seen order.

    Nodes appear as both children and parents, so we union both columns. First-seen order keeps the
    emitted map stable/diff-friendly across runs.
    """
    names, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("child"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            for nm in (parts[0], parts[1]):
                if nm and nm not in seen:
                    seen.add(nm)
                    names.append(nm)
    return names


def readable(name):
    """Human-readable form of a Wikipedia category title for the encoder (underscores → spaces)."""
    return name.replace("_", " ")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=GRAPH, help="category_parent.tsv (child<TAB>parent)")
    ap.add_argument("--anchor", default="Physics", help="anchor category name (cosine target)")
    ap.add_argument("--out", default=os.path.join(ROOT, "dense_mu_physics.tsv"),
                    help="output name<TAB>μ TSV")
    ap.add_argument("--model", default=MODEL, help="sentence-transformers model id")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--no-normalize-names", action="store_true",
                    help="encode the raw underscore name instead of the readable (space) form")
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import util as st_util
    except ImportError:
        sys.exit("error: sentence-transformers not installed. Run: "
                 "pip install -r prototypes/mu_cosine/requirements.txt")

    names = load_category_names(args.graph)
    print(f"loaded {len(names)} unique category names from {args.graph}", file=sys.stderr)

    model = SentenceTransformer(args.model)
    texts = names if args.no_normalize_names else [readable(n) for n in names]
    anchor_text = args.anchor if args.no_normalize_names else readable(args.anchor)

    anchor_vec = model.encode(anchor_text, convert_to_tensor=True, normalize_embeddings=True)
    embs = model.encode(texts, batch_size=args.batch_size, convert_to_tensor=True,
                        normalize_embeddings=True, show_progress_bar=True)
    cos = st_util.cos_sim(embs, anchor_vec).reshape(-1)  # one column → flat vector of cosines

    rows = []
    neg = 0
    for name, c in zip(names, cos.tolist()):
        if c < 0.0:
            neg += 1
        rows.append((name, to_membership(c)))

    mus = [m for _, m in rows]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# Dense fuzzy physics-membership μ — MiniLM cosine to the '{args.anchor}' anchor "
                "(NO training).\n")
        f.write(f"# model: {args.model}; clamp: to_membership (cosine∈[-1,1] → μ∈[0,1]); "
                "names verbatim from the category graph.\n")
        f.write("# Provenance: gen_dense_mu.py over data/benchmark/10k/category_parent.tsv. A raw "
                "pretrained-embedding prior (coarse semantic proximity), NOT the trained cosine-μ\n")
        f.write("# encoder and NOT authoritative — its job is DENSITY (a μ for every node) so the "
                "gated cone / IC / Lin machinery stops pruning through unscored connectors.\n")
        f.write("# Format: name<TAB>μ, μ∈[0,1]. '#' lines are comments.\n")
        for name, m in rows:
            f.write(f"{name}\t{m:.4f}\n")

    print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    print(f"  μ stats: min {min(mus):.4f}  max {max(mus):.4f}  mean {sum(mus)/len(mus):.4f}",
          file=sys.stderr)
    print(f"  clamped {neg} negative cosines to 0 ({100*neg/len(rows):.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
