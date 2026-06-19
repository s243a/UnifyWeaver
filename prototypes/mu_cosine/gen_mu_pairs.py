#!/usr/bin/env python3
"""Generate candidate (a, b) category pairs to be LLM-scored for pairwise μ(a, b).

This ONLY emits candidate pairs to a file — it spends **no LLM budget**. The Haiku scoring is a
separate step (stub at the bottom; ask the user before running it).

Design (from the discussion):

  * The cosine-μ training is a *graded* word2vec SGNS — learn embeddings so related pairs have high
    cosine, unrelated pairs low — so the negative-sampling ratio transfers. Mikolov et al. 2013 use
    k = 5–20 negatives per positive (small data); we target ~5:1 (configurable), i.e. ~83% of emitted
    pairs are negatives (expected μ below the cutoff).

  * POSITIVES — a "mesh" grown around a few roots of interest. Random-walk-with-restart from a growing
    frontier (seeded with the roots): pick a start from the frontier (or restart at a seed with prob
    `restart_alpha`), take a short geometric-length walk, emit (start, endpoint), and add the endpoint
    to the frontier so coverage densifies around the seeds and grows outward along coherent paths.
    Steps are **hub-down-weighted** (prob ∝ 1/deg^β): high-degree nodes are the leak conduits
    (`Container_categories`, `Physical_objects`, `Matter`) — word2vec's frequent-word subsampling
    analog — and stepping into one teleports the walk anywhere, breaking the walk-length→relatedness
    relation. Avoiding them keeps positive pairs semantically local (incl. the graded boundary band).

  * NEGATIVES — word2vec's noise: a domain node (`a`, from the frontier) paired with a uniform-random
    category (`b`). On an 8k-node graph where only ~dozens are in-domain, a random `b` is reliably
    unrelated, and uniform sampling is naturally diverse (no hub domination).

Pure stdlib. Output: `name_a<TAB>name_b<TAB>stratum<TAB>walk_len<TAB>mu` (mu blank, for the scorer).

    python3 gen_mu_pairs.py --seeds Physics --n-positives 200 --neg-ratio 5 --out mu_pairs.tsv
"""
import argparse, math, os, random
from collections import deque

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
GRAPH = os.path.join(REPO, "data", "benchmark", "10k", "category_parent.tsv")


def load_graph(path):
    adj = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("child"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            c, p = parts[0], parts[1]
            adj.setdefault(c, set()).add(p)
            adj.setdefault(p, set()).add(c)
    return {k: sorted(v) for k, v in adj.items()}


def weighted_choice(items, weights, rng):
    return rng.choices(items, weights=weights, k=1)[0]


def hub_step(node, adj, deg, beta, prev, rng):
    """Pick a next neighbour, down-weighting high-degree (hub) nodes; avoid immediate backtrack."""
    nbrs = [n for n in adj.get(node, ()) if n != prev] or list(adj.get(node, ()))
    if not nbrs:
        return None
    w = [1.0 / (deg[n] ** beta) for n in nbrs]
    return weighted_choice(nbrs, w, rng)


def walk(start, adj, deg, stop_prob, beta, rng, max_len=8):
    """Geometric-length hub-down-weighted walk; returns (endpoint, path)."""
    node, prev, path = start, None, [start]
    while len(path) <= max_len and rng.random() > stop_prob:
        nxt = hub_step(node, adj, deg, beta, prev, rng)
        if nxt is None:
            break
        prev, node = node, nxt
        path.append(node)
    return node, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="Physics", help="comma-separated root categories of interest")
    ap.add_argument("--n-positives", type=int, default=200)
    ap.add_argument("--neg-ratio", type=float, default=5.0, help="negatives per positive (word2vec k)")
    ap.add_argument("--stop-prob", type=float, default=0.4, help="geometric per-step stop probability")
    ap.add_argument("--hub-beta", type=float, default=1.0, help="step down-weight exponent on degree")
    ap.add_argument("--restart-alpha", type=float, default=0.3, help="prob a positive restarts at a seed")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs.tsv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    adj = load_graph(GRAPH)
    deg = {n: max(1, len(adj.get(n, ()))) for n in adj}
    universe = list(adj.keys())
    seeds = [s for s in args.seeds.split(",") if s in adj]
    missing = [s for s in args.seeds.split(",") if s not in adj]
    if missing:
        print(f"WARNING: seeds not in graph (skipped): {missing}")
    if not seeds:
        raise SystemExit("no valid seeds in the graph")
    print(f"{len(universe)} categories, seeds {seeds}")

    pairs = set()          # canonicalised (a,b) to dedup
    rows = []              # (a, b, stratum, walk_len)
    frontier = list(seeds)  # the growing mesh

    # POSITIVES: RWR over the growing frontier, hub-down-weighted short walks.
    tries = 0
    while sum(1 for r in rows if r[2] == "pos") < args.n_positives and tries < args.n_positives * 50:
        tries += 1
        start = rng.choice(seeds) if rng.random() < args.restart_alpha else rng.choice(frontier)
        end, path = walk(start, adj, deg, args.stop_prob, args.hub_beta, rng)
        if end == start:
            continue
        key = tuple(sorted((start, end)))
        if key in pairs:
            continue
        pairs.add(key)
        rows.append((start, end, "pos", len(path) - 1))
        # grow the mesh with the path's interior/endpoint (keeps coverage connected around the seeds)
        for n in path[1:]:
            if n not in frontier:
                frontier.append(n)

    n_pos = sum(1 for r in rows if r[2] == "pos")

    # NEGATIVES: a domain node × a uniform-random category (word2vec noise).
    n_neg_target = int(round(n_pos * args.neg_ratio))
    tries = 0
    while sum(1 for r in rows if r[2] == "neg") < n_neg_target and tries < n_neg_target * 50:
        tries += 1
        a = rng.choice(frontier)            # the "target" (in/near domain)
        b = rng.choice(universe)            # the noise
        if a == b or b in adj.get(a, ()):   # skip self / direct neighbours (those aren't negatives)
            continue
        key = tuple(sorted((a, b)))
        if key in pairs:
            continue
        pairs.add(key)
        rows.append((a, b, "neg", -1))

    n_neg = sum(1 for r in rows if r[2] == "neg")
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        f.write("# candidate pairs for pairwise μ(a,b) scoring. stratum: pos=meshed walk, neg=noise.\n")
        f.write("# columns: name_a<TAB>name_b<TAB>stratum<TAB>walk_len<TAB>mu   (mu BLANK — fill via scoring)\n")
        for a, b, st, wl in rows:
            f.write(f"{a}\t{b}\t{st}\t{wl}\t\n")

    pos_lens = [wl for a, b, st, wl in rows if st == "pos"]
    print(f"wrote {len(rows)} pairs to {args.out}: {n_pos} pos / {n_neg} neg "
          f"(ratio {n_neg/max(1,n_pos):.1f}:1, target {args.neg_ratio}:1)")
    print(f"mesh frontier grew to {len(frontier)} nodes around {len(seeds)} seed(s); "
          f"positive walk-length mean {sum(pos_lens)/max(1,len(pos_lens)):.1f}")
    print("NEXT: score the `mu` column with a Haiku subagent (see score_stub below) — costs LLM "
          "budget, so confirm with the user first.")


def score_stub(pairs_path):  # noqa: ARG001
    """TODO (spends LLM budget — ask the user first).

    Read `pairs_path`, and for each (name_a, name_b) ask a Haiku subagent: 'On a 0..1 scale, how
    related are the Wikipedia categories <a> and <b>? (1 = essentially the same topic, 0 = unrelated)'
    — batched, with the same prompt discipline as the existing fixtures
    (tests/fixtures/wikipedia_physics_*). Write μ back into the 5th column. Then the dense-μ training
    in train_cosine_mu.py / the transformer uses these pairwise labels across the varied anchors.
    """
    raise NotImplementedError("scoring is the budget-spending step; wire up a Haiku subagent and "
                              "confirm with the user before running")


if __name__ == "__main__":
    main()
