#!/usr/bin/env python3
"""Variant 3: B = max acyclic parent distance from v to root.

For each sample node v at children-BFS depth d:
1. Compute MAX simple-path parent-distance from v to root via DP in
   children-BFS-depth order:
       max_parent_dist[v] = 1 + max(max_parent_dist[p] for p in parents(v))
   (since all p have BFS depth < v's, by acyclic-graph property)
2. Set B = max_parent_dist[v]
3. Enumerate paths v → root within B, compute d_wPow
4. Compare to:
   - min_dist + 1 = standard depth+1 baseline
   - (min_dist + max_dist)/2 + 1 = average-parent-path baseline
   - max_dist + 1 = longest-parent-path baseline

User's framing: "deeper nesting better reflects depth in a tree hierarchy.
The metric averages over a relevant set of paths constrained to the natural
depth range."

What this should show:
- If d_wPow ≈ min_dist + 1: metric ignores longer ancestor chains (strict-min depth-like)
- If d_wPow ≈ avg + 1: metric averages over chains (depth-like-on-average)
- If d_wPow < min_dist + 1: shortcuts dominate even at this looser budget
"""
import argparse, lmdb, math, os, random, time, sys
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from max_dist_budget import dp_max_dist, load_metric_max_dist

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--max-dist-metric", default=None,
                 help="Phase-1 scoped LMDB dir holding metric_max_dist_to_root; "
                      "use the materialised true-longest budget instead of the "
                      "in-probe DP (which undercounts — see max_dist_budget.py)")
MAX_DIST_METRIC = _ap.parse_known_args()[0].max_dist_metric

LMDB_PATH = '/tmp/sw_post_fix_lmdb'
ROOT_ID = 137597
CC = 5.0
N_EXP = 2
RNG_SEED = 42
N_SAMPLES_PER_DEPTH = 15
DEPTHS_TO_TEST = list(range(1, 13))  # avoid extreme deep depths where path enumeration explodes
MAX_PATHS = 100_000
PER_PAIR_TIME_BUDGET = 5.0

print("=== Variant 3: B = max acyclic parent distance from v to root ===\n")

# ============================================================
# Load graph
# ============================================================

print("[1/6] Loading edges...", flush=True)
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
                k, val = cur.item()
                c, p = int(k.decode()), int(val.decode())
                parents_of[c].append(p)
                children_of[p].append(c)
            except Exception:
                break
            if not cur.next():
                break
env.close()

# ============================================================
# BFS depths
# ============================================================

print("[2/6] BFS from root for min-depth...", flush=True)
min_dist = {ROOT_ID: 0}
q = deque([ROOT_ID])
while q:
    n = q.popleft()
    for c in children_of.get(n, ()):
        if c not in min_dist:
            min_dist[c] = min_dist[n] + 1
            q.append(c)
reachable = set(min_dist.keys())
print(f"  Reachable: {len(reachable):,}")

# ============================================================
# DP for max acyclic parent distance to root
# ============================================================

print("[3/6] max acyclic parent distance to root...", flush=True)
# Build nodes_at_depth (sorted by BFS depth ascending) — used by the sampler.
nodes_at_depth = defaultdict(list)
for n, d in min_dist.items():
    nodes_at_depth[d].append(n)

if MAX_DIST_METRIC:
    # Use the ingest-materialised budget (true longest acyclic parent path).
    print(f"  using materialised max_dist_to_root from {MAX_DIST_METRIC}", flush=True)
    max_dist = load_metric_max_dist(MAX_DIST_METRIC, expected_root=ROOT_ID)
    missing = [n for n in reachable if n not in max_dist]
    if missing:
        # Node ids absent from the metric (e.g. metric scoped to a subtree) fall
        # back to the in-probe DP so the sampler still has a budget for them.
        print(f"  WARNING: {len(missing):,} reachable nodes absent from metric; "
              f"falling back to DP for those", flush=True)
        dp = dp_max_dist(min_dist, parents_of)
        for n in missing:
            if n in dp:
                max_dist[n] = dp[n]
else:
    # Default: the probe's original BFS-depth-monotone DP. NOTE this undercounts
    # nodes whose longest ancestor chain runs through a deeper-BFS parent; pass
    # --max-dist-metric for the corrected (true-longest) budget. See
    # max_dist_budget.py for the characterised divergence.
    max_dist = dp_max_dist(min_dist, parents_of)

# Stats on max_dist vs min_dist
ratios = []
for n in reachable:
    if n in max_dist and min_dist[n] > 0:
        ratios.append(max_dist[n] / min_dist[n])
ratios.sort()
if ratios:
    print(f"  max_dist/min_dist ratio:")
    print(f"    p25 = {ratios[len(ratios)//4]:.2f}")
    print(f"    median = {ratios[len(ratios)//2]:.2f}")
    print(f"    p75 = {ratios[3*len(ratios)//4]:.2f}")
    print(f"    max = {max(ratios):.2f}")

# ============================================================
# Compute D, b_eff
# ============================================================

print("[4/6] Computing D, b_eff...", flush=True)
sum_d_c = sum_d_c2 = 0
sum_d_p = sum_d_p2 = 0
n_c = n_p = 0
for v in reachable:
    dc = sum(1 for c in children_of.get(v, ()) if c in reachable)
    if dc > 0:
        sum_d_c += dc; sum_d_c2 += dc * dc; n_c += 1
    dp = sum(1 for p in parents_of.get(v, ()) if p in reachable)
    if dp > 0:
        sum_d_p += dp; sum_d_p2 += dp * dp; n_p += 1
D = sum_d_c / n_c
b_eff = (sum_d_c2/sum_d_c) / (sum_d_p2/sum_d_p)
print(f"  D = {D:.3f}  b_eff = {b_eff:.3f}")

# ============================================================
# Path enumeration with A* and time budget
# ============================================================

def compute_d_wPow(v, budget, cc=CC, n_exp=N_EXP, time_budget=PER_PAIR_TIME_BUDGET):
    sum_num = 0.0
    sum_w = 0.0
    n_paths = 0
    capped = False
    timeout = False
    deadline = time.time() + time_budget
    iters = 0

    stack = [(v, 0.0, 0, 0, frozenset({v}))]
    while stack:
        iters += 1
        if iters % 5000 == 0 and time.time() > deadline:
            timeout = True
            break
        node, cost, N, M, vis = stack.pop()
        if node == ROOT_ID and (N + M) > 0:
            h = N + M
            w = (1.0/D)**N * (1.0/(b_eff*D))**M
            sum_num += w * (h+1)**(-n_exp)
            sum_w += w
            n_paths += 1
            if n_paths >= MAX_PATHS:
                capped = True
                break
            continue
        # A* prune: remaining cost ≥ min_dist[node] (parent hops)
        if cost + 1.0 <= budget:
            for p in parents_of.get(node, ()):
                if p in vis or p not in reachable:
                    continue
                new_cost = cost + 1.0
                if new_cost + min_dist.get(p, float('inf')) > budget:
                    continue
                stack.append((p, new_cost, N+1, M, vis | {p}))
        if cost + cc <= budget:
            for c in children_of.get(node, ()):
                if c in vis or c not in reachable:
                    continue
                new_cost = cost + cc
                if new_cost + min_dist.get(c, float('inf')) > budget:
                    continue
                stack.append((c, new_cost, N, M+1, vis | {c}))
    if sum_w <= 0 or n_paths == 0:
        return None, n_paths, capped or timeout
    ratio = sum_num/sum_w
    if ratio <= 0:
        return None, n_paths, capped or timeout
    return ratio**(-1.0/n_exp), n_paths, capped or timeout


# ============================================================
# Run per-depth tests
# ============================================================

print(f"\n[5/6] Sampling per depth (n={N_SAMPLES_PER_DEPTH}), B = max_dist[v]...", flush=True)
random.seed(RNG_SEED)
results = []
print(f"  {'depth':>5} {'node':>10} {'min_d':>6} {'max_d':>6} {'B':>4} "
      f"{'paths':>7} {'d_wPow':>8} {'-min':>7} {'-avg':>7} {'-max':>7}  note", flush=True)
for d in DEPTHS_TO_TEST:
    pool = [n for n in nodes_at_depth.get(d, []) if n in max_dist]
    if not pool:
        continue
    sample_size = min(N_SAMPLES_PER_DEPTH, len(pool))
    sample = random.sample(pool, sample_size)
    for v in sample:
        mn = min_dist[v]
        mx = max_dist[v]
        avg = (mn + mx) / 2
        B = mx  # max parent distance to root
        t0 = time.time()
        d_wPow, n_paths, flag = compute_d_wPow(v, B)
        t_elapsed = time.time() - t0
        flag_marker = " (T)" if flag else ""
        if d_wPow is None:
            print(f"  {d:>5} {v:>10} {mn:>6} {mx:>6} {B:>4} {n_paths:>7} {'--':>8} "
                  f"{'--':>7} {'--':>7} {'--':>7}  no_path{flag_marker}", flush=True)
            continue
        diff_min = d_wPow - (mn + 1)
        diff_avg = d_wPow - (avg + 1)
        diff_max = d_wPow - (mx + 1)
        if abs(diff_min) < 0.1:
            note = "MIN-LIKE"
        elif abs(diff_avg) < 0.5:
            note = "AVG-LIKE"
        elif diff_min < -0.5:
            note = "SHORTCUT"
        else:
            note = "stretched"
        results.append((d, v, mn, mx, B, n_paths, d_wPow, diff_min, diff_avg, diff_max, note, flag))
        print(f"  {d:>5} {v:>10} {mn:>6} {mx:>6} {B:>4} {n_paths:>7} {d_wPow:>8.3f} "
              f"{diff_min:>+7.3f} {diff_avg:>+7.3f} {diff_max:>+7.3f}  {note}{flag_marker}  ({t_elapsed:.1f}s)", flush=True)

# ============================================================
# Summary
# ============================================================

print(f"\n[6/6] Summary", flush=True)
n_total = len(results)
if n_total == 0:
    print("  No results.")
    sys.exit(0)

print(f"  Total valid: {n_total}")
# Counts by category
cats = defaultdict(int)
for r in results:
    cats[r[10]] += 1
for cat, count in sorted(cats.items()):
    print(f"    {cat:>10s}: {count:>3} ({100*count/n_total:.1f}%)")
print()

# Per-depth
print(f"  Per-depth (B = max_parent_dist):")
print(f"  {'depth':>5} {'n':>3} {'mean(min_d)':>11} {'mean(max_d)':>11} {'mean(B)':>8} {'mean(d_wPow)':>13} {'-min':>7} {'-avg':>7} {'-max':>7}")
for d in DEPTHS_TO_TEST:
    rows = [r for r in results if r[0] == d]
    if not rows:
        continue
    m_min = sum(r[2] for r in rows) / len(rows)
    m_max = sum(r[3] for r in rows) / len(rows)
    m_B = sum(r[4] for r in rows) / len(rows)
    m_dwp = sum(r[6] for r in rows) / len(rows)
    m_dmin = sum(r[7] for r in rows) / len(rows)
    m_davg = sum(r[8] for r in rows) / len(rows)
    m_dmax = sum(r[9] for r in rows) / len(rows)
    print(f"  {d:>5} {len(rows):>3} {m_min:>11.2f} {m_max:>11.2f} {m_B:>8.2f} {m_dwp:>13.3f} "
          f"{m_dmin:>+7.3f} {m_davg:>+7.3f} {m_dmax:>+7.3f}")
