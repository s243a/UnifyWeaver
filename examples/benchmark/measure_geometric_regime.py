#!/usr/bin/env python3
"""Task #15: Measure geometric regime of simplewiki Articles-rooted subgraph.

Approach:
- Open the post-fix simplewiki LMDB (rebuilt Jun 2 2026 from the correct-mode
  3-mode ingester, working on the dumps in context/gemini/UnifyWeaver/data/).
- Build adjacency lists in memory (parents and children).
- BFS from Articles root (page_id 137597) via children to identify reachable set.
- Sample K random pairs from reachable, measure undirected BFS distance.
- Compute D from in-reach degree distribution.
- Compare mean distance to three regime predictions:
    tree:        2·log_D(N_reachable)
    small-world: log(N_reachable) / log(D)
    ultra-small: log log(N_reachable)

Companion to design note §5.6 / theory doc §5.6 — settles Conjecture 3.4
(topical homogeneity) and §5.6 regime classification empirically.
"""
import lmdb, struct, random, math, time, sys
from collections import deque, defaultdict

LMDB_PATH = '/tmp/sw_post_fix_lmdb'
ROOT_ID = 137597  # Category:Articles on simplewiki, verified earlier
N_PAIRS = 1000
MAX_BFS_DEPTH = 30
RNG_SEED = 42

print(f"=== Task #15: Geometric regime of simplewiki topical core ===")
print(f"LMDB: {LMDB_PATH}")
print(f"Root: page_id {ROOT_ID} (Category:Articles)")
print(f"N_pairs: {N_PAIRS}, max BFS depth: {MAX_BFS_DEPTH}")
print()

# ============================================================
# Step 0: Load all edges into memory
# ============================================================

print("[0/4] Loading all edges into memory...")
t0 = time.time()
parents_of = defaultdict(list)   # child_id -> list of parent_ids
children_of = defaultdict(list)  # parent_id -> list of child_ids
n_edges = 0

env = lmdb.open(LMDB_PATH, max_dbs=4, readonly=True, lock=False,
                map_size=2 * 1024 * 1024 * 1024)
main_db = env.open_db(b'main', dupsort=True)
with env.begin() as txn:
    cur = txn.cursor(main_db)
    if not cur.first():
        print("ERROR: empty LMDB")
        sys.exit(1)
    while True:
        try:
            k, v = cur.item()
            child = int(k.decode())
            parent = int(v.decode())
            parents_of[child].append(parent)
            children_of[parent].append(child)
            n_edges += 1
        except Exception:
            break
        if not cur.next():
            break
env.close()
t1 = time.time()
all_nodes = set(parents_of.keys()) | set(children_of.keys())
print(f"  Loaded {n_edges:,} edges, {len(all_nodes):,} distinct nodes ({t1-t0:.1f}s)")
print()

# ============================================================
# Step 1: BFS from Articles root via children
# ============================================================

print(f"[1/4] BFS from root {ROOT_ID} via category_child edges...")
t0 = time.time()
dist_from_root = {ROOT_ID: 0}
q = deque([ROOT_ID])
while q:
    n = q.popleft()
    d = dist_from_root[n]
    for c in children_of.get(n, ()):
        if c not in dist_from_root:
            dist_from_root[c] = d + 1
            q.append(c)
t1 = time.time()
N_reach = len(dist_from_root)
print(f"  Reachable: {N_reach:,} nodes ({t1-t0:.1f}s)")

# Depth distribution
depth_counts = defaultdict(int)
for d in dist_from_root.values():
    depth_counts[d] += 1
print(f"  Depth distribution (cumulative):")
cum = 0
for d in sorted(depth_counts.keys()):
    cum += depth_counts[d]
    pct = 100 * cum / N_reach
    if d <= 10 or d % 5 == 0 or d == max(depth_counts):
        print(f"    depth ≤ {d:3d}: {cum:>8,} ({pct:.1f}%)")
print(f"  Max depth from root: {max(depth_counts)}")
print()

reachable_set = set(dist_from_root.keys())

# ============================================================
# Step 2: Estimate D = E[d_child] over reachable nodes
# ============================================================

print("[2/4] Estimating D = E[d_child] over reachable set...")
t0 = time.time()
total_deg = 0
for node in reachable_set:
    total_deg += len(children_of.get(node, ()))
D = total_deg / len(reachable_set)
# Mean over nodes that have at least one child
deg_with_children = [len(children_of.get(n, ())) for n in reachable_set if len(children_of.get(n, ())) > 0]
D_with_children = sum(deg_with_children) / len(deg_with_children) if deg_with_children else 0
t1 = time.time()
print(f"  D = E[d_child] over all reachable nodes:   {D:.3f}  ({t1-t0:.1f}s)")
print(f"  D restricted to nodes with children:       {D_with_children:.3f}  (= 'branching factor' sense)")
print(f"  Nodes with children: {len(deg_with_children):,} / {N_reach:,}")
print()

# ============================================================
# Step 3: Sample random pairs, measure undirected distance
# ============================================================

print(f"[3/4] Sampling {N_PAIRS} random pairs, measuring undirected distance...")
random.seed(RNG_SEED)
reachable_list = list(reachable_set)


def undirected_bfs(u, v, max_depth, restrict_to=None):
    """Bidirectional undirected BFS.

    Treats both parent and child edges as undirected. If restrict_to is given,
    only traverses through nodes in that set (paths cannot leave the subgraph).
    """
    if u == v:
        return 0
    fwd = {u: 0}
    bwd = {v: 0}
    fwd_frontier = [u]
    bwd_frontier = [v]
    fwd_d = bwd_d = 0

    def in_scope(node):
        return restrict_to is None or node in restrict_to

    for _ in range(max_depth):
        if len(fwd_frontier) <= len(bwd_frontier):
            fwd_d += 1
            new_frontier = []
            for n in fwd_frontier:
                for c in parents_of.get(n, ()):
                    if not in_scope(c):
                        continue
                    if c in bwd:
                        return fwd_d + bwd[c]
                    if c not in fwd:
                        fwd[c] = fwd_d
                        new_frontier.append(c)
                for c in children_of.get(n, ()):
                    if not in_scope(c):
                        continue
                    if c in bwd:
                        return fwd_d + bwd[c]
                    if c not in fwd:
                        fwd[c] = fwd_d
                        new_frontier.append(c)
            fwd_frontier = new_frontier
            if not fwd_frontier:
                return None
        else:
            bwd_d += 1
            new_frontier = []
            for n in bwd_frontier:
                for c in parents_of.get(n, ()):
                    if not in_scope(c):
                        continue
                    if c in fwd:
                        return fwd[c] + bwd_d
                    if c not in bwd:
                        bwd[c] = bwd_d
                        new_frontier.append(c)
                for c in children_of.get(n, ()):
                    if not in_scope(c):
                        continue
                    if c in fwd:
                        return fwd[c] + bwd_d
                    if c not in bwd:
                        bwd[c] = bwd_d
                        new_frontier.append(c)
            bwd_frontier = new_frontier
            if not bwd_frontier:
                return None
    return None


print("  Mode A: BFS unrestricted (can leak through admin parents)")
distances_unrestricted = []
not_found_a = 0
t0 = time.time()
for i in range(N_PAIRS):
    u, v = random.sample(reachable_list, 2)
    d = undirected_bfs(u, v, MAX_BFS_DEPTH, restrict_to=None)
    if d is None:
        not_found_a += 1
    else:
        distances_unrestricted.append(d)
t1 = time.time()
print(f"    {len(distances_unrestricted)} found in {t1-t0:.1f}s, mean={sum(distances_unrestricted)/max(1,len(distances_unrestricted)):.2f}")

print("  Mode B: BFS restricted to Articles subgraph (no parent leaks)")
random.seed(RNG_SEED)  # same pairs for fair comparison
distances_restricted = []
not_found_b = 0
t0 = time.time()
for i in range(N_PAIRS):
    u, v = random.sample(reachable_list, 2)
    d = undirected_bfs(u, v, MAX_BFS_DEPTH, restrict_to=reachable_set)
    if d is None:
        not_found_b += 1
    else:
        distances_restricted.append(d)
    if (i + 1) % 200 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        print(f"    Pair {i+1}/{N_PAIRS} ({rate:.1f} pairs/s, restricted-found: {len(distances_restricted)})")
t1 = time.time()
print(f"    {len(distances_restricted)} found in {t1-t0:.1f}s, mean={sum(distances_restricted)/max(1,len(distances_restricted)):.2f}")
print(f"    not found (>depth {MAX_BFS_DEPTH}, possibly disconnected within subgraph): {not_found_b}")
print()

# Use the restricted (in-subgraph) distances as the primary measurement
distances = distances_restricted

# ============================================================
# Step 4: Compute statistics and compare to regime predictions
# ============================================================

if not distances:
    print("ERROR: no distances measured; cannot continue")
    sys.exit(1)

distances.sort()
mean_d = sum(distances) / len(distances)
median_d = distances[len(distances) // 2]
p25 = distances[len(distances) // 4]
p75 = distances[3 * len(distances) // 4]
p95 = distances[int(0.95 * len(distances))]
max_d = max(distances)
min_d = min(distances)

# Regime predictions — using D over all reachable nodes
log_N = math.log(N_reach)
log_log_N = math.log(log_N) if log_N > 1 else 0
log_D = math.log(D) if D > 1 else 0

tree_pred = 2 * log_N / log_D if log_D > 0 else float('inf')
sw_pred = log_N / log_D if log_D > 0 else float('inf')
ultra_pred = log_log_N

# Also with D_with_children for comparison
log_D2 = math.log(D_with_children) if D_with_children > 1 else 0
tree_pred2 = 2 * log_N / log_D2 if log_D2 > 0 else float('inf')
sw_pred2 = log_N / log_D2 if log_D2 > 0 else float('inf')

print("[4/4] Results")
print()
print(f"  Measured distance distribution (over {len(distances)} pairs):")
print(f"    min:    {min_d}")
print(f"    p25:    {p25}")
print(f"    median: {median_d}")
print(f"    mean:   {mean_d:.2f}")
print(f"    p75:    {p75}")
print(f"    p95:    {p95}")
print(f"    max:    {max_d}")
print()
print(f"  Regime predictions:")
print(f"    Using D = {D:.2f} (mean over all reachable):")
print(f"      Tree           (2·log_D(N)):     {tree_pred:.2f}")
print(f"      Small-world    (log(N)/log(D)):  {sw_pred:.2f}")
print(f"      Ultra-small    (log log(N)):     {ultra_pred:.2f}")
print(f"    Using D = {D_with_children:.2f} (mean over nodes with children):")
print(f"      Tree           (2·log_D(N)):     {tree_pred2:.2f}")
print(f"      Small-world    (log(N)/log(D)):  {sw_pred2:.2f}")
print()
print(f"  Closest regime (using D over all reachable):")
ratios = {
    'tree':         abs(math.log(mean_d / tree_pred)) if tree_pred > 0 else float('inf'),
    'small-world':  abs(math.log(mean_d / sw_pred)) if sw_pred > 0 else float('inf'),
    'ultra-small':  abs(math.log(mean_d / ultra_pred)) if ultra_pred > 0 else float('inf'),
}
best = min(ratios, key=ratios.get)
for label, r in sorted(ratios.items(), key=lambda x: x[1]):
    marker = "  <-- best fit" if label == best else ""
    fold = math.exp(r)
    print(f"    {label:14s} |log(measured/predicted)| = {r:.3f} (factor {fold:.2f}×){marker}")
print()
print(f"  Reachable subgraph stats:")
print(f"    Nodes:        {N_reach:,}")
print(f"    D (E[d_c]):   {D:.3f}")
print(f"    D (excl leaf):{D_with_children:.3f}")
print(f"    log(N):       {log_N:.3f}")
print(f"    log(log(N)):  {log_log_N:.3f}")
