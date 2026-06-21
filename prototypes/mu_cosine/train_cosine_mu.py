#!/usr/bin/env python3
"""Runnable proof-of-objective for the μ-cosine embedding idea (pure stdlib — no numpy/torch).

The hypothesis (from the design discussion): a category's fuzzy membership μ(X | root) is the
*cosine* between a learned vector for X and a learned vector for the domain root. If so, we should be
able to learn per-category vectors such that cos(v_X, v_root) ≈ μ(X), using the Haiku-scored μ as the
training target. This script verifies exactly that on the **real** Wikipedia-physics fixture, with the
**distance-biased sampling** from the design (nearby categories sampled more often).

It deliberately uses learnable vectors *directly* (the simplest "encoder") to isolate and prove the
objective. The full architecture — a configurable MLP/MoE encoder producing those vectors from
MiniLM-initialised embeddings — is in `mu_encoder.py` (forward pass), and is the separate project;
this file is the evidence that the objective it optimises is sound.

No dependencies. Closed-form cosine-MSE gradient (finite-difference checked at startup).

    python3 train_cosine_mu.py [--dim 16] [--epochs 4000] [--lr 0.2] [--seed 1]
"""
import argparse, math, random, os

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
GRAPH = os.path.join(REPO, "data", "benchmark", "10k", "category_parent.tsv")
MU = os.path.join(REPO, "tests", "fixtures", "wikipedia_physics_fuzzy_nodes.tsv")
ANCHOR = "Physics"


def load_graph(path):
    """Undirected adjacency over category names (for the BFS 'closeness' distance)."""
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
    return adj


def load_mu(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def bfs_dist(adj, src):
    """Undirected hop distance from src to every reachable node."""
    from collections import deque
    dist = {src: 0}
    q = deque([src])
    while q:
        x = q.popleft()
        for y in adj.get(x, ()):
            if y not in dist:
                dist[y] = dist[x] + 1
                q.append(y)
    return dist


# ---- vector ops ----
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a)) or 1e-12


def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def cos_grad(a, b):
    """d cos(a,b) / d a  (returns the gradient vector w.r.t. a)."""
    na, nb = norm(a), norm(b)
    c = dot(a, b) / (na * nb)
    # ∂c/∂a = b/(‖a‖‖b‖) − c·a/‖a‖²
    return [b[i] / (na * nb) - c * a[i] / (na * na) for i in range(len(a))]


def finite_diff_check():
    """Sanity: the closed-form cos gradient matches a finite difference — BOTH sides (∂cos/∂a, used
    for the node update, and ∂cos/∂b = cos_grad(b, a), used for the anchor update)."""
    random.seed(0)
    d = 5
    a = [random.gauss(0, 1) for _ in range(d)]
    b = [random.gauss(0, 1) for _ in range(d)]
    eps = 1e-6
    ga, gb = cos_grad(a, b), cos_grad(b, a)
    for i in range(d):
        a2 = list(a); a2[i] += eps
        b2 = list(b); b2[i] += eps
        fda = (cosine(a2, b) - cosine(a, b)) / eps
        fdb = (cosine(a, b2) - cosine(a, b)) / eps
        assert abs(fda - ga[i]) < 1e-4, f"∂cos/∂a mismatch dim {i}: {fda} vs {ga[i]}"
        assert abs(fdb - gb[i]) < 1e-4, f"∂cos/∂b mismatch dim {i}: {fdb} vs {gb[i]}"
    return True


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (math.sqrt(sxx * syy) or 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dist-bias", type=float, default=1.0,
                    help="exponent on 1/(1+dist) for sampling; 0 = uniform, higher = nearer-biased")
    args = ap.parse_args()

    assert finite_diff_check()
    random.seed(args.seed)

    adj = load_graph(GRAPH)
    mu = load_mu(MU)
    dist = bfs_dist(adj, ANCHOR)
    # keep scored nodes present in the graph (drop the anchor itself if scored)
    nodes = [n for n in mu if n in adj and n != ANCHOR]
    print(f"loaded {len(nodes)} scored nodes connected to '{ANCHOR}' (μ in [{min(mu[n] for n in nodes):.1f},"
          f"{max(mu[n] for n in nodes):.1f}])")

    # distance-biased sampling weights: nearer to the anchor ⇒ sampled more often (design point).
    def w(n):
        d = dist.get(n, 99)
        return (1.0 / (1.0 + d)) ** args.dist_bias
    weights = [w(n) for n in nodes]

    # learnable vectors: one per node + the anchor. (The full model would *produce* these from a
    # transformer over MiniLM-init embeddings; here they are the parameters directly.)
    def rand_vec():
        return [random.gauss(0, 1.0 / math.sqrt(args.dim)) for _ in range(args.dim)]
    vecs = {n: rand_vec() for n in nodes}
    vecs[ANCHOR] = rand_vec()

    def epoch_loss():
        return sum((cosine(vecs[n], vecs[ANCHOR]) - mu[n]) ** 2 for n in nodes) / len(nodes)

    print(f"initial  MSE {epoch_loss():.4f}   corr(cos, μ) {pearson([cosine(vecs[n], vecs[ANCHOR]) for n in nodes], [mu[n] for n in nodes]):+.3f}")

    for ep in range(args.epochs):
        n = random.choices(nodes, weights=weights, k=1)[0]
        a, b = vecs[n], vecs[ANCHOR]
        c = cosine(a, b)
        err = c - mu[n]
        ga = cos_grad(a, b)
        gb = cos_grad(b, a)
        for i in range(args.dim):
            a[i] -= args.lr * 2 * err * ga[i]
            b[i] -= args.lr * 2 * err * gb[i]  # the anchor also moves (it is learned too)
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            cs = [cosine(vecs[n], vecs[ANCHOR]) for n in nodes]
            print(f"epoch {ep+1:5d}  MSE {epoch_loss():.4f}   corr(cos, μ) {pearson(cs, [mu[n] for n in nodes]):+.3f}")

    # final report
    cs = [cosine(vecs[n], vecs[ANCHOR]) for n in nodes]
    mus = [mu[n] for n in nodes]
    print(f"\nFINAL    MSE {epoch_loss():.4f}   corr(cos, μ) {pearson(cs, mus):+.3f}")
    # a few examples across the μ range
    ex = sorted(nodes, key=lambda n: mu[n])
    print("\nexamples (μ vs learned cosine):")
    for n in [ex[0], ex[len(ex)//4], ex[len(ex)//2], ex[3*len(ex)//4], ex[-1]]:
        print(f"  μ {mu[n]:.2f}  cos {cosine(vecs[n], vecs[ANCHOR]):+.2f}  dist {dist.get(n,'?')}  {n}")


if __name__ == "__main__":
    main()
