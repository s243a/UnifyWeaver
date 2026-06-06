#!/usr/bin/env python3
"""Test depth-likeness for arbitrary node pairs (u, v) at tight carrot budget.

Generalises the root-only depth test: for each pair, compute the minimum-cost
carrot path c_min(u, v), set budget B = c_min, enumerate paths u→v within B,
and check whether d_wPow ≈ h_carrot + 1 (no shortcut) or < h_carrot + 1 (shortcut).

The carrot path: u → (parent hops) → common ancestor a → (child hops) → v
Cost = (parent dist from u to a) · 1 + (child dist from a to v) · 5
Hops = (parent dist) + (child dist)

Optimal carrot: minimum cost over common ancestors.

Predicts (under metric-tree-likeness):
  d_wPow(u, v; B=c_min) ≈ h_carrot + 1 for most pairs (carrot dominant)
  Deviation: per-pair shortcut signature
"""
import lmdb, math, random, time, sys
from collections import defaultdict, deque

LMDB_PATH = '/tmp/sw_post_fix_lmdb'
ROOT_ID = 137597
CC = 5.0
N_EXP = 2
RNG_SEED = 42
N_PAIRS = 100
MAX_PATHS = 50_000  # per-pair safety cap
PER_PAIR_TIME_BUDGET = 5.0  # seconds — skip pair if it exceeds this


# ============================================================
# Load graph + BFS
# ============================================================

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

print("[2/5] BFS from root...")
depth_from_root = {ROOT_ID: 0}
q = deque([ROOT_ID])
while q:
    n = q.popleft()
    for c in children_of.get(n, ()):
        if c not in depth_from_root:
            depth_from_root[c] = depth_from_root[n] + 1
            q.append(c)
reachable = set(depth_from_root.keys())
print(f"  Reachable: {len(reachable):,}")

print("[3/5] Computing D, b_eff on Articles topical subgraph...")
sum_d_c = sum_d_c2 = 0
sum_d_p = sum_d_p2 = 0
n_c = n_p = 0
for v in reachable:
    dc = sum(1 for c in children_of.get(v, ()) if c in reachable)
    if dc > 0:
        sum_d_c += dc
        sum_d_c2 += dc * dc
        n_c += 1
    dp = sum(1 for p in parents_of.get(v, ()) if p in reachable)
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
print(f"  D = {D:.3f}  b_eff = {b_eff:.3f}")
print()


# ============================================================
# Ancestor BFS + carrot path computation
# ============================================================

def ancestors_with_distances(node):
    """BFS up from node via parents, returns dict {ancestor: parent_distance}.

    Restricted to reachable set.
    """
    dist = {node: 0}
    q = deque([node])
    while q:
        n = q.popleft()
        for p in parents_of.get(n, ()):
            if p in reachable and p not in dist:
                dist[p] = dist[n] + 1
                q.append(p)
    return dist


def find_optimal_carrot(u, v):
    """Find the common ancestor a* that minimises carrot cost.

    Returns (cost, hops, ancestor, n_up, n_down, anc_v_dict) — None if disconnected.
    Also returns anc_v_dict so path enumeration can use it for A* pruning.
    """
    anc_u = ancestors_with_distances(u)
    anc_v = ancestors_with_distances(v)
    common = anc_u.keys() & anc_v.keys()
    if not common:
        return None
    best = None
    for a in common:
        cost = anc_u[a] + 5 * anc_v[a]
        hops = anc_u[a] + anc_v[a]
        if best is None or cost < best[0]:
            best = (cost, hops, a, anc_u[a], anc_v[a])
    return (*best, anc_v)


# ============================================================
# Path enumeration u → v
# ============================================================

def compute_d_wPow_uv(u, v, budget, anc_v=None, cc=CC, n_exp=N_EXP, time_budget=PER_PAIR_TIME_BUDGET):
    """Enumerate paths from u to v within budget, with A* pruning + time-budget bail-out.

    A* heuristic: if current node n is an ancestor of v (i.e. n in anc_v), the
    minimum remaining cost is anc_v[n] * cc (child hops from n to v). Otherwise,
    we don't have a tight bound, so use 0 (no pruning for that branch).
    """
    sum_num = 0.0
    sum_w   = 0.0
    n_paths = 0
    capped = False
    time_out = False
    deadline = time.time() + time_budget

    stack = [(u, 0.0, 0, 0, frozenset({u}))]
    iter_count = 0
    while stack:
        iter_count += 1
        if iter_count % 10000 == 0 and time.time() > deadline:
            time_out = True
            break
        node, cost, N, M, vis = stack.pop()
        if node == v and (N + M) > 0:
            h = N + M
            w = (1.0 / D) ** N * (1.0 / (b_eff * D)) ** M
            sum_num += w * (h + 1) ** (-n_exp)
            sum_w   += w
            n_paths += 1
            if n_paths >= MAX_PATHS:
                capped = True
                break
            continue
        # Parent moves with A* prune
        if cost + 1.0 <= budget:
            for p in parents_of.get(node, ()):
                if p in vis or p not in reachable:
                    continue
                new_cost = cost + 1.0
                # A* heuristic: if p is an ancestor of v, we need at least anc_v[p]*cc child hops
                if anc_v is not None and p in anc_v:
                    h_lb = anc_v[p] * cc
                    if new_cost + h_lb > budget:
                        continue
                stack.append((p, new_cost, N + 1, M, vis | {p}))
        # Child moves with A* prune
        if cost + cc <= budget:
            for c in children_of.get(node, ()):
                if c in vis or c not in reachable:
                    continue
                new_cost = cost + cc
                # A* heuristic: if c is an ancestor of v, we need at least anc_v[c]*cc child hops
                if anc_v is not None and c in anc_v:
                    h_lb = anc_v[c] * cc
                    if new_cost + h_lb > budget:
                        continue
                stack.append((c, new_cost, N, M + 1, vis | {c}))

    if sum_w <= 0 or n_paths == 0:
        return None, n_paths, capped or time_out
    ratio = sum_num / sum_w
    if ratio <= 0:
        return None, n_paths, capped or time_out
    d_wPow = ratio ** (-1.0 / n_exp)
    return d_wPow, n_paths, capped or time_out


# ============================================================
# Run pair-by-pair experiment
# ============================================================

print(f"[4/5] Sampling {N_PAIRS} random pairs, computing carrot + d_wPow...")
random.seed(RNG_SEED)
reachable_list = list(reachable)
results = []
t_total = time.time()

print(f"  {'#':>3} {'u_depth':>8} {'v_depth':>8} {'carrot_h':>9} {'B':>4} {'paths':>7} {'d_wPow':>9} {'diff':>8} note")

for i in range(N_PAIRS):
    u, v = random.sample(reachable_list, 2)
    carrot = find_optimal_carrot(u, v)
    if carrot is None:
        # No common ancestor — disconnected in topical subgraph
        results.append((u, v, None, None, None, None, None, "no_ca"))
        print(f"  {i+1:>3} {depth_from_root[u]:>8} {depth_from_root[v]:>8} "
              f"{'--':>9} {'--':>4} {0:>7} {'--':>9} {'--':>8} no_ca", flush=True)
        continue
    cost, hops, a, n_up, n_down, anc_v = carrot
    B = cost
    t0 = time.time()
    d_wPow, n_paths, capped = compute_d_wPow_uv(u, v, B, anc_v=anc_v)
    t_elapsed = time.time() - t0
    if d_wPow is None:
        note = "no_path"
        diff_str = "--"
        results.append((u, v, hops, B, n_paths, None, None, note))
    else:
        expected = hops + 1
        diff = d_wPow - expected
        capped_str = " (CAP)" if capped else ""
        if abs(diff) < 0.05:
            note = "EXACT"
        elif diff < -0.5:
            note = "SHORTCUT"
        elif diff > 0.5:
            note = "stretched"
        else:
            note = "near"
        diff_str = f"{diff:+.3f}"
        results.append((u, v, hops, B, n_paths, d_wPow, diff, note + capped_str))
    print(f"  {i+1:>3} {depth_from_root[u]:>8} {depth_from_root[v]:>8} "
          f"{hops if carrot else '--':>9} {B if carrot else '--':>4} "
          f"{n_paths:>7} {(f'{d_wPow:.3f}' if d_wPow else '--'):>9} "
          f"{diff_str:>8} {note}  ({t_elapsed:.1f}s)", flush=True)

t_total = time.time() - t_total
print(f"\n  Total time: {t_total:.1f}s for {N_PAIRS} pairs")
print()

# ============================================================
# Summary
# ============================================================

print("[5/5] Summary")
n_total = len(results)
n_no_ca = sum(1 for r in results if r[7] == "no_ca")
n_no_path = sum(1 for r in results if r[7] == "no_path")
valid_results = [r for r in results if r[6] is not None]
n_valid = len(valid_results)
print(f"  Total pairs:        {n_total}")
print(f"  No common ancestor: {n_no_ca}")
print(f"  No path found:      {n_no_path}")
print(f"  Valid measurements: {n_valid}")
print()

if not valid_results:
    print("  No valid results to summarize.")
    sys.exit(0)

# Categorize
cat_exact = [r for r in valid_results if abs(r[6]) < 0.05]
cat_near = [r for r in valid_results if 0.05 <= abs(r[6]) < 0.5]
cat_shortcut = [r for r in valid_results if r[6] <= -0.5]
cat_stretched = [r for r in valid_results if r[6] >= 0.5]

print(f"  EXACT      (|d_wPow − h_carrot − 1| < 0.05):  {len(cat_exact):>3} ({100*len(cat_exact)/n_valid:.1f}%)")
print(f"  near       (within 0.5):                    {len(cat_near):>3} ({100*len(cat_near)/n_valid:.1f}%)")
print(f"  SHORTCUT   (d_wPow < h_carrot − 0.5+1):     {len(cat_shortcut):>3} ({100*len(cat_shortcut)/n_valid:.1f}%)")
print(f"  stretched  (d_wPow > h_carrot + 0.5+1):     {len(cat_stretched):>3} ({100*len(cat_stretched)/n_valid:.1f}%)")
print()

# Overall stats
diffs = [r[6] for r in valid_results]
mean_diff = sum(diffs) / len(diffs)
var_diff = sum((x - mean_diff) ** 2 for x in diffs) / len(diffs)
std_diff = var_diff ** 0.5
print(f"  Overall:")
print(f"    mean(d_wPow − (h_carrot+1)) = {mean_diff:+.4f}")
print(f"    stdev                        = {std_diff:.4f}")
print(f"    min diff                     = {min(diffs):+.4f}")
print(f"    max diff                     = {max(diffs):+.4f}")
print()

# Verdict
if len(cat_exact) / n_valid > 0.7:
    verdict = "STRONG: most pairs have d_wPow ≈ carrot + 1; metric-tree-like for arbitrary pairs"
elif len(cat_exact) + len(cat_near) >= 0.7 * n_valid:
    verdict = "MODERATE: most pairs near carrot value; some shortcut variation"
else:
    verdict = "WEAK: substantial deviation from carrot baseline; shortcuts common"
print(f"  Verdict: {verdict}")
