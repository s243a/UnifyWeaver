#!/usr/bin/env python3
"""Test whether d_wPow(v) ≈ depth(v) when one endpoint is the root.

Procedure:
1. Load Articles-rooted simplewiki subgraph into memory.
2. Compute b_eff and D directly from the topical subgraph.
3. Sample nodes at various depths from root.
4. For each, enumerate all paths to root within budget B = max(2·depth, 10) at cc=5.
5. Compute d_wPow using the standard weight formula:
       w(p) = (1/D)^N · (1/(b_eff·D))^M
       d_wPow = (Σ w·(h+1)^{-n} / Σ w)^(-1/n)   with n=2
6. Compare d_wPow(v) to depth(v) and report:
   - Mean and stdev of (d_wPow - depth) over sampled v
   - Per-depth-bucket statistics
7. If path enumeration blows up, bail out with smaller budget.

Predicts (under metric-tree-likeness, low TLI):
  d_wPow(v) ≈ depth(v) with tight scatter.
  Deviation = TLI-like quantity per node.
"""
import lmdb, math, random, time, sys
from collections import defaultdict, deque

LMDB_PATH = '/tmp/sw_post_fix_lmdb'
ROOT_ID = 137597
CC = 5.0          # child-hop cost (matches design note benchmark)
N_EXP = 2         # power-mean exponent
RNG_SEED = 42
N_SAMPLES_PER_DEPTH = 5
MAX_PATHS = 500_000   # path-enumeration safety cap

# ============================================================
# Load graph
# ============================================================

print("=== Depth-likeness test for d_wPow on simplewiki Articles topical ===\n")
print("[1/5] Loading edges...")
parents_of = defaultdict(list)
children_of = defaultdict(list)
env = lmdb.open(LMDB_PATH, max_dbs=4, readonly=True, lock=False,
                map_size=2 * 1024 * 1024 * 1024)
main_db = env.open_db(b'main', dupsort=True)
with env.begin() as txn:
    cur = txn.cursor(main_db)
    if cur.first():
        while True:
            try:
                k, v = cur.item()
                c, p = int(k.decode()), int(v.decode())
                parents_of[c].append(p)
                children_of[p].append(c)
            except Exception:
                break
            if not cur.next():
                break
env.close()

# ============================================================
# BFS from root to get depths
# ============================================================

print("[2/5] BFS to compute depths from root...")
depth_from_root = {ROOT_ID: 0}
q = deque([ROOT_ID])
while q:
    n = q.popleft()
    for c in children_of.get(n, ()):
        if c not in depth_from_root:
            depth_from_root[c] = depth_from_root[n] + 1
            q.append(c)
reachable = set(depth_from_root.keys())
N_reach = len(reachable)
max_depth = max(depth_from_root.values())
print(f"  Reachable: {N_reach}, max depth: {max_depth}")

# Build by-depth index
by_depth = defaultdict(list)
for v, d in depth_from_root.items():
    by_depth[d].append(v)

# ============================================================
# Compute D and b_eff on the topical subgraph
# ============================================================

print("[3/5] Computing D, b_eff on Articles topical subgraph...")
# Restrict adjacency to reachable set for in-subgraph stats
sum_d_c = sum_d_c2 = 0
sum_d_p = sum_d_p2 = 0
n_c = n_p = 0
for v in reachable:
    in_subgraph_children = [c for c in children_of.get(v, ()) if c in reachable]
    dc = len(in_subgraph_children)
    if dc > 0:
        sum_d_c += dc
        sum_d_c2 += dc * dc
        n_c += 1
    in_subgraph_parents = [p for p in parents_of.get(v, ()) if p in reachable]
    dp = len(in_subgraph_parents)
    if dp > 0:
        sum_d_p += dp
        sum_d_p2 += dp * dp
        n_p += 1
E_d_c = sum_d_c / n_c
E_d_c2 = sum_d_c2 / n_c
E_d_p = sum_d_p / n_p
E_d_p2 = sum_d_p2 / n_p
D = E_d_c
b_eff = (E_d_c2 / E_d_c) / (E_d_p2 / E_d_p)
print(f"  E[d_c] = {E_d_c:.3f}  E[d_c²] = {E_d_c2:.2f}  (nodes with children: {n_c})")
print(f"  E[d_p] = {E_d_p:.3f}  E[d_p²] = {E_d_p2:.2f}  (nodes with parents:  {n_p})")
print(f"  D = {D:.3f}  b_eff = {b_eff:.3f}  b_eff·D = {b_eff*D:.2f}")

# ============================================================
# Path enumeration + d_wPow
# ============================================================

def compute_d_wPow(v, budget, cc=CC, n_exp=N_EXP):
    """Enumerate paths from v to ROOT_ID within budget, compute d_wPow.

    Uses A* admissible-heuristic pruning: a partial path at node X with cost C
    cannot reach root within budget B unless
        C + depth_from_root[X] * parent_cost ≤ B
    because depth_from_root[X] is the minimum parent-hop count from X to root.
    This is the same pruning rule the F# bidirectional kernel uses (line 173 of
    kernel_bidirectional_ancestor.fs.mustache).

    Edge moves:
      parent hop: cost 1, increments N
      child hop:  cost cc, increments M
    Path stops successfully when current node == ROOT_ID and at least one hop taken.
    Visited set prevents cycles.
    Budget caps cumulative cost.

    Returns (d_wPow, n_paths, capped) where capped=True if MAX_PATHS hit.
    """
    sum_num = 0.0
    sum_w   = 0.0
    n_paths = 0
    capped = False

    # Stack-based DFS: (node, cost, N, M, visited)
    stack = [(v, 0.0, 0, 0, frozenset({v}))]
    while stack:
        node, cost, N, M, vis = stack.pop()
        if node == ROOT_ID and (N + M) > 0:
            h = N + M
            w = (1.0 / D) ** N * (1.0 / (b_eff * D)) ** M
            sum_num += w * (h + 1) ** (-n_exp)
            sum_w   += w
            n_paths += 1
            if n_paths >= MAX_PATHS:
                capped = True
                break
            continue
        # Parent moves (A*-pruned)
        if cost + 1.0 <= budget:
            for p in parents_of.get(node, ()):
                if p not in vis and p in reachable:
                    new_cost = cost + 1.0
                    # A* admissible heuristic: min remaining cost is
                    # depth_from_root[p] parent hops × cost 1 = depth_from_root[p]
                    if new_cost + depth_from_root.get(p, float('inf')) <= budget:
                        stack.append((p, new_cost, N + 1, M, vis | {p}))
        # Child moves (A*-pruned)
        if cost + cc <= budget:
            for c in children_of.get(node, ()):
                if c not in vis and c in reachable:
                    new_cost = cost + cc
                    if new_cost + depth_from_root.get(c, float('inf')) <= budget:
                        stack.append((c, new_cost, N, M + 1, vis | {c}))

    if sum_w <= 0:
        return None, n_paths, capped
    ratio = sum_num / sum_w
    if ratio <= 0:
        return None, n_paths, capped
    d_wPow = ratio ** (-1.0 / n_exp)
    return d_wPow, n_paths, capped


# ============================================================
# Sample nodes per depth and measure
# ============================================================

# Use B = depth: only shortcut-using M ≥ 1 paths admissible.
N_SAMPLES_PER_DEPTH_BD = 20  # more samples since shortcuts are rare per-node

print(f"\n[4/5] Computing d_wPow for sampled nodes per depth (cc={CC}, budget=depth)...")
random.seed(RNG_SEED)
results = []
DEPTHS_TO_TEST = sorted([d for d in by_depth.keys() if 1 <= d <= min(max_depth, 15)])
print(f"  Testing depths: {DEPTHS_TO_TEST}")
print(f"  Samples per depth: {N_SAMPLES_PER_DEPTH_BD}")
print()
print(f"  {'depth':>6} {'node':>10} {'budget':>7} {'paths':>10} {'d_wPow':>10} {'diff':>7}  {'shortcut?':>10}")

for d in DEPTHS_TO_TEST:
    pool = by_depth[d]
    if not pool:
        continue
    sample_size = min(N_SAMPLES_PER_DEPTH_BD, len(pool))
    sample = random.sample(pool, sample_size)
    for v in sample:
        B = d  # exactly the pure-parent budget
        t0 = time.time()
        d_wPow, n_paths, capped = compute_d_wPow(v, B)
        t_elapsed = time.time() - t0
        cap_marker = " (CAPPED)" if capped else ""
        if d_wPow is None:
            print(f"  {d:>6} {v:>10} {B:>7} {n_paths:>10} {'--':>10} {'--':>7}  {'--':>10}{cap_marker}")
            continue
        diff = d_wPow - d
        # n_paths > 1 means at least one shortcut path was found (in addition to M=0 baseline)
        shortcut_marker = "YES" if n_paths > 1 else "no"
        results.append((d, v, B, n_paths, d_wPow, diff, t_elapsed, capped, n_paths > 1))
        print(f"  {d:>6} {v:>10} {B:>7} {n_paths:>10} {d_wPow:>10.4f} {diff:>+7.3f}  {shortcut_marker:>10}{cap_marker}")

# ============================================================
# Summary statistics
# ============================================================

print(f"\n[5/5] Summary")
if not results:
    print("  No results to summarize.")
    sys.exit(0)

# Overall
diffs = [r[5] for r in results]
mean_diff = sum(diffs) / len(diffs)
var_diff = sum((x - mean_diff) ** 2 for x in diffs) / len(diffs)
std_diff = var_diff ** 0.5
print(f"  Overall (n={len(results)}):")
print(f"    mean(d_wPow - depth) = {mean_diff:+.4f}")
print(f"    stdev               = {std_diff:.4f}")
print(f"    min diff            = {min(diffs):+.4f}")
print(f"    max diff            = {max(diffs):+.4f}")
print()

# Per depth (with shortcut frequency)
print(f"  Per-depth (B=depth, shortcut filter):")
print(f"  {'depth':>6} {'n':>4} {'with_shortcut':>14} {'mean(d_wPow)':>14} {'mean(diff)':>12} {'std(diff)':>10}")
for d in DEPTHS_TO_TEST:
    rows = [r for r in results if r[0] == d]
    if not rows:
        continue
    dwp = [r[4] for r in rows]
    df = [r[5] for r in rows]
    has_short = [r[8] for r in rows]
    n_short = sum(has_short)
    m_dwp = sum(dwp) / len(dwp)
    m_df = sum(df) / len(df)
    if len(df) > 1:
        v_df = sum((x - m_df) ** 2 for x in df) / len(df)
        s_df = v_df ** 0.5
    else:
        s_df = 0.0
    pct_short = 100 * n_short / len(rows)
    print(f"  {d:>6} {len(rows):>4}    {n_short}/{len(rows)} ({pct_short:.0f}%) {m_dwp:>14.4f} {m_df:>+12.4f} {s_df:>10.4f}")

# Verdict
print()
if abs(mean_diff) < 0.5 and std_diff < 1.0:
    verdict = "STRONG: metric IS approximately depth-like"
elif abs(mean_diff) < 1.0 and std_diff < 2.0:
    verdict = "MODERATE: metric is roughly depth-like, with measurable scatter"
else:
    verdict = "WEAK: metric deviates substantially from depth"
print(f"  Verdict: {verdict}")
