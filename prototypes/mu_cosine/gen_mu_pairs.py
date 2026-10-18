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

  * WALK DIRECTION (depth drift). The baseline positive walk is **undirected** (steps to any
    neighbour) — it drifts deep (children outnumber parents in the tail). `--bidir` switches to a
    **depth-balanced** mix: bidirectional walks (reach siblings/cousins at a representative depth, no
    deep/apex drift) blended with `--child-only` downward walks (in-domain ancestor→descendant), via
    `--bidir-frac`. Depth balance (up vs down) is orthogonal to the hub-down-weighting (1/deg^β within
    a direction); both are kept. See DESIGN_bidirectional_walk.md and REPORT_bidir_walk.md.

Pure stdlib. Output: `name_a<TAB>name_b<TAB>stratum<TAB>walk_len<TAB>mu` (mu blank, for the scorer).

    python3 gen_mu_pairs.py --seeds Physics --n-positives 200 --neg-ratio 5 --out mu_pairs.tsv
    python3 gen_mu_pairs.py --seeds Physics --bidir --bidir-mode coinflip --bidir-frac 0.5 ...
"""
import argparse, math, os, random
from collections import deque

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
GRAPH = os.environ.get("UW_MU_GRAPH", os.path.join(REPO, "data", "benchmark", "10k", "category_parent.tsv"))


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


def load_directed(path):
    """Load the child→parent DAG into separate down (children) and up (parents) adjacency.

    Returns (children, parents): children[n] = nodes one step DOWN (toward leaves), parents[n] =
    nodes one step UP (toward the apex/root). The undirected `load_graph` stays the baseline; the
    bidirectional/child-only walks need the direction split to balance (or restrict) depth drift.
    """
    children, parents = {}, {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("child"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            c, p = parts[0], parts[1]
            parents.setdefault(c, set()).add(p)   # p is UP from c
            children.setdefault(p, set()).add(c)  # c is DOWN from p
            children.setdefault(c, set())
            parents.setdefault(p, set())
    return ({k: sorted(v) for k, v in children.items()},
            {k: sorted(v) for k, v in parents.items()})


def estimate_global_beta(children, parents):
    """The global up-edge weight β = E[c²]/E[p²] (handshake lemma: E[c]=E[p], so the size-biased
    branching ratio reduces to the raw second-moment ratio). A single weight on parent edges that
    balances the depth drift the walk *experiences on average* (Lyons–Pemantle–Peres λ-biased walk at
    criticality). Cheaper than the per-node c/p coin-flip; exact only in aggregate."""
    nodes = set(children) | set(parents)
    n = max(1, len(nodes))
    ec2 = sum(len(children.get(x, ())) ** 2 for x in nodes) / n
    ep2 = sum(len(parents.get(x, ())) ** 2 for x in nodes) / n
    return (ec2 / ep2) if ep2 else 1.0


def hub_pick(nbrs, deg, beta, prev, rng):
    """Hub-down-weighted pick (prob ∝ 1/deg^β) from a candidate set, avoiding immediate backtrack.
    This is the within-direction domain-drift guard, orthogonal to up/down depth balance."""
    cand = [n for n in nbrs if n != prev] or list(nbrs)
    if not cand:
        return None
    w = [1.0 / (deg[n] ** beta) for n in cand]
    return weighted_choice(cand, w, rng)


def walk_child_only(start, children, deg, stop_prob, beta, rng, max_len=8):
    """Downward (child-only) walk: stays in the seed's subtree → depth(endpoint) ≥ depth(seed),
    in-domain ancestor→descendant pairs, zero domain drift. Hub-down-weighted within the down step."""
    node, prev, path = start, None, [start]
    while len(path) <= max_len and rng.random() > stop_prob:
        nxt = hub_pick(children.get(node, ()), deg, beta, prev, rng)
        if nxt is None:
            break
        prev, node = node, nxt
        path.append(node)
    return node, path


def walk_bidir(start, children, parents, deg, stop_prob, beta, rng, mode="coinflip",
               global_beta=1.0, max_len=8):
    """Depth-balanced bidirectional walk (reaches siblings/cousins at a representative depth).

      * coinflip: at each step flip a fair coin for up-vs-down (P(up)=P(down)=½ when both exist),
        then hub-down-weighted pick within that direction. Equivalent to the per-node up-weight
        β=c/p ⇒ E[Δdepth]=0 at every interior node (depth is a martingale), exact zero-drift.
      * global: one up-weight β=E[c²]/E[p²] on parent edges. P(down)=c/(c+βp), P(up)=βp/(c+βp);
        balances depth drift on average (cheaper single-weight approximation).

    Boundary: at a leaf (no children) it must go up; at a root (no parents) it must go down —
    reflecting-ish boundaries, fine for short walks from interior seeds. Hub-down-weighting is kept
    WITHIN the chosen direction (orthogonal domain-drift guard)."""
    node, prev, path = start, None, [start]
    while len(path) <= max_len and rng.random() > stop_prob:
        ch, pa = children.get(node, ()), parents.get(node, ())
        if not ch and not pa:
            break
        if not ch:
            go_down = False
        elif not pa:
            go_down = True
        elif mode == "global":
            c, p = len(ch), len(pa)
            p_down = c / (c + global_beta * p)
            go_down = rng.random() < p_down
        else:  # coinflip: per-node exact zero-drift
            go_down = rng.random() < 0.5
        nxt = hub_pick(ch if go_down else pa, deg, beta, prev, rng)
        if nxt is None:
            break
        prev, node = node, nxt
        path.append(node)
    return node, path


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
    ap.add_argument("--max-frontier", type=int, default=3000, help="cap the mesh so it stays around the seeds")
    ap.add_argument("--bidir", action="store_true",
                    help="enable depth-balanced directed walks (mix of bidirectional + child-only); "
                         "without it the baseline undirected hub-walk is used")
    ap.add_argument("--bidir-mode", choices=["coinflip", "global"], default="coinflip",
                    help="coinflip: per-node fair up/down coin (β=c/p, exact zero-drift); "
                         "global: single up-weight β=E[c²]/E[p²] (cheaper average approximation)")
    ap.add_argument("--bidir-frac", type=float, default=0.5,
                    help="fraction of positive walks that are bidirectional; the rest are child-only "
                         "(downward). 0.0 = pure child-only, 1.0 = pure bidirectional")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs.tsv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    adj = load_graph(GRAPH)
    deg = {n: max(1, len(adj.get(n, ()))) for n in adj}
    universe = list(adj.keys())
    children = parents = None
    global_beta = 1.0
    if args.bidir:
        children, parents = load_directed(GRAPH)
        global_beta = estimate_global_beta(children, parents)
        print(f"directed walks: bidir-mode={args.bidir_mode}, bidir-frac={args.bidir_frac} "
              f"(global up-weight β=E[c²]/E[p²]={global_beta:.2f})")

    def do_walk(start):
        """Dispatch one positive walk per the selected mode (baseline undirected, or the
        bidir-frac mix of depth-balanced bidirectional + child-only downward walks)."""
        if not args.bidir:
            return walk(start, adj, deg, args.stop_prob, args.hub_beta, rng)
        if rng.random() < args.bidir_frac:
            return walk_bidir(start, children, parents, deg, args.stop_prob, args.hub_beta, rng,
                              mode=args.bidir_mode, global_beta=global_beta)
        return walk_child_only(start, children, deg, args.stop_prob, args.hub_beta, rng)
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
        end, path = do_walk(start)
        if end == start:
            continue
        key = tuple(sorted((start, end)))
        if key in pairs:
            continue
        pairs.add(key)
        rows.append((start, end, "pos", len(path) - 1))
        # grow the mesh with the path's interior/endpoint (keeps coverage connected around the seeds).
        # Capped at --max-frontier: an unbounded frontier would, over many walks, drift into a plain
        # random walk from an ever-expanding set rather than a mesh *around* the seeds.
        for n in path[1:]:
            if len(frontier) >= args.max_frontier:
                break
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
        # skip self / DIRECT neighbours (those aren't negatives). Stricter: exclude k-hop neighbours of
        # `a` (a small BFS) so borderline-related pairs don't leak into the negatives — only needed if
        # the scored μ histogram shows negatives creeping above the cutoff.
        if a == b or b in adj.get(a, ()):
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
    """Scoring the candidate pairs (spends LLM budget — ask the user first).

    Realized 2026-06-20: the 200 `stratum=pos` pairs from a `--seeds Physics` run were scored with
    parallel Haiku subagents (graded sameness/relatedness, 0..1; rubric: 1.0 same/nested topic,
    0.5–0.7 same broad domain, 0.2–0.4 loosely related, 0.0 unrelated) and the 1000 `neg` pairs
    assigned μ=0 by construction. The result is committed as `mu_pairs_scored.tsv` (the expensive,
    reusable label asset; the unscored `mu_pairs.tsv` is git-ignored and regenerable). To re-score a
    fresh batch: regenerate with `main()`, split the `pos` rows, hand each batch to a Haiku subagent
    with that rubric, and merge μ back into the 5th column. Then `train_cosine_mu_torch.py --mode
    pairs --minilm` trains the dense-μ encoder on these varied-anchor pairwise labels (the structure
    single-anchor μ(X|Physics) training collapses — see validate_lin_agreement.py).
    """
    raise NotImplementedError("scoring is the budget-spending step; the committed mu_pairs_scored.tsv "
                              "was produced via Haiku subagents — confirm with the user before re-running")


if __name__ == "__main__":
    main()
