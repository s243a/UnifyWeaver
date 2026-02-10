/**
 * Tree building algorithms: MST re-rooting and J-guided from distance matrix.
 * Operates on pre-computed data from physics_bundle.json.
 */

/**
 * Compute hierarchy breadth scores from a distance matrix.
 * High score = hub/parent topic (many close neighbors, high k-NN density),
 * Low score = specific/child topic (fewer close neighbors).
 * Uses inverse mean k-NN distance, normalized to [0, 1].
 */
function computeEntropyScores(distMatrix, k = 30) {
    const n = distMatrix.length;
    const scores = new Float64Array(n);
    const kActual = Math.min(k, n - 1);
    for (let i = 0; i < n; i++) {
        const dists = [];
        for (let j = 0; j < n; j++) {
            if (i !== j) dists.push(distMatrix[i][j]);
        }
        dists.sort((a, b) => a - b);
        let sum = 0;
        for (let d = 0; d < kActual; d++) sum += dists[d];
        scores[i] = 1.0 / (sum / kActual + 1e-10);
    }
    // Normalize to [0, 1]
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < n; i++) {
        if (scores[i] < min) min = scores[i];
        if (scores[i] > max) max = scores[i];
    }
    const range = max - min || 1e-10;
    for (let i = 0; i < n; i++) {
        scores[i] = (scores[i] - min) / range;
    }
    return scores;
}

/**
 * Build adjacency list from MST edges (undirected).
 */
function buildAdjacency(edges, n) {
    const adj = Array.from({ length: n }, () => []);
    for (const e of edges) {
        adj[e.source].push({ node: e.target, weight: e.weight });
        adj[e.target].push({ node: e.source, weight: e.weight });
    }
    return adj;
}

/**
 * BFS from root to orient an undirected tree.
 * Returns { nodes: [{id, parentId, depth}], edges: [{source, target, weight}] }
 */
function bfsFromRoot(adj, rootId, maxDepth = 10, maxBranching = Infinity) {
    const n = adj.length;
    const visited = new Set([rootId]);
    const queue = [{ id: rootId, depth: 0 }];
    const nodes = [{ id: rootId, parentId: -1, depth: 0 }];
    const edges = [];

    let qi = 0;
    while (qi < queue.length) {
        const { id, depth } = queue[qi++];
        if (depth >= maxDepth) continue;

        // Sort neighbors by weight (closest first) for max branching
        const neighbors = adj[id]
            .filter(nb => !visited.has(nb.node))
            .sort((a, b) => a.weight - b.weight);

        let branchCount = 0;
        for (const nb of neighbors) {
            if (branchCount >= maxBranching) break;
            visited.add(nb.node);
            nodes.push({ id: nb.node, parentId: id, depth: depth + 1 });
            edges.push({ source: id, target: nb.node, weight: nb.weight });
            queue.push({ id: nb.node, depth: depth + 1 });
            branchCount++;
        }
    }

    return { nodes, edges, rootId };
}

/**
 * Root an MST at the given node. Uses pre-computed MST edges.
 */
function rootMST(mstEdges, rootId, n, maxDepth = 10, maxBranching = Infinity) {
    const adj = buildAdjacency(mstEdges, n);
    return bfsFromRoot(adj, rootId, maxDepth, maxBranching);
}

/**
 * Root an MST with branching limits enforced during construction.
 * Phase 1: BFS through MST edges, keeping only maxBranching children per node.
 * Phase 2: Orphaned nodes (unreachable due to full parents) are placed using
 *          the distance matrix fallback — attached to the closest visited node
 *          that still has room, pushing them deeper into the tree.
 */
function rootMSTConstrained(mstEdges, distMatrix, rootId, n, maxDepth, maxBranching,
                            entropyScores = null, upwardPenalty = 1.0, cascadeFactor = 1.5) {
    const adj = buildAdjacency(mstEdges, n);

    const visited = new Set([rootId]);
    const queue = [{ id: rootId, depth: 0 }];
    const nodes = [{ id: rootId, parentId: -1, depth: 0 }];
    const edges = [];
    const childCount = {};

    // Entropy-based penalty state
    const taint = {};
    taint[rootId] = 1.0;
    const rootEntropy = entropyScores ? entropyScores[rootId] : 0;
    let maxEntropy = 0;
    if (entropyScores) {
        for (let i = 0; i < n; i++) maxEntropy = Math.max(maxEntropy, entropyScores[i]);
    }
    const entropyRange = maxEntropy - rootEntropy + 1e-10;

    // Phase 1: BFS through MST edges with branching limits
    let qi = 0;
    while (qi < queue.length) {
        const { id, depth } = queue[qi++];
        if (depth >= maxDepth) continue;

        const neighbors = adj[id]
            .filter(nb => !visited.has(nb.node))
            .sort((a, b) => a.weight - b.weight);

        const existing = childCount[id] || 0;
        let added = 0;
        for (const nb of neighbors) {
            if (existing + added >= maxBranching) break;
            visited.add(nb.node);
            nodes.push({ id: nb.node, parentId: id, depth: depth + 1 });
            edges.push({ source: id, target: nb.node, weight: nb.weight });
            queue.push({ id: nb.node, depth: depth + 1 });
            added++;
            // Propagate taint
            if (entropyScores && entropyScores[id] > rootEntropy) {
                taint[nb.node] = (taint[id] || 1.0) * cascadeFactor;
            } else {
                taint[nb.node] = taint[id] || 1.0;
            }
        }
        childCount[id] = existing + added;
    }

    // Phase 2: Attach orphaned nodes using distance matrix (with penalty)
    while (visited.size < n) {
        let bestDist = Infinity;
        let bestFrom = -1, bestTo = -1;
        for (const v of visited) {
            if ((childCount[v] || 0) >= maxBranching) continue;
            for (let j = 0; j < n; j++) {
                if (visited.has(j)) continue;
                let d = distMatrix[v][j];
                // Apply upward penalty
                // Penalize both: connecting TO a hub parent AND placing a hub node as child
                if (entropyScores && upwardPenalty > 1.0) {
                    const parentExcess = Math.max(0, entropyScores[v] - rootEntropy);
                    const nodeExcess = Math.max(0, entropyScores[j] - rootEntropy);
                    const excess = Math.max(parentExcess, nodeExcess);
                    const edgePenalty = 1.0 + (upwardPenalty - 1.0) * excess / entropyRange;
                    d *= edgePenalty * (taint[v] || 1.0);
                }
                if (d < bestDist) {
                    bestDist = d;
                    bestFrom = v;
                    bestTo = j;
                }
            }
        }
        if (bestTo === -1) break;

        const parentNode = nodes.find(nd => nd.id === bestFrom);
        const depth = parentNode ? parentNode.depth + 1 : 1;
        visited.add(bestTo);
        nodes.push({ id: bestTo, parentId: bestFrom, depth });
        edges.push({ source: bestFrom, target: bestTo, weight: bestDist });
        queue.push({ id: bestTo, depth });
        childCount[bestFrom] = (childCount[bestFrom] || 0) + 1;
        // Propagate taint
        if (entropyScores && entropyScores[bestFrom] > rootEntropy) {
            taint[bestTo] = (taint[bestFrom] || 1.0) * cascadeFactor;
        } else {
            taint[bestTo] = taint[bestFrom] || 1.0;
        }
    }

    return { nodes, edges, rootId };
}

/**
 * Build J-guided tree using density-ordered greedy insertion.
 * Matches the original density explorer algorithm:
 * 1. Estimate density for each node (inverse of distance to k-th neighbor)
 * 2. Process nodes from densest to least dense
 * 3. Each node connects to its nearest already-placed node
 *
 * This produces better semantic hierarchies than BFS because dense/important
 * nodes become natural hubs, and each node finds its genuinely closest parent.
 *
 * When a user-selected root is provided and it differs from the densest node,
 * the root is placed first, then remaining nodes follow density order.
 */
function buildJGuided(distMatrix, rootId, n, k = 10, maxDepth = 20, maxBranching = Infinity,
                      entropyScores = null, upwardPenalty = 1.0, cascadeFactor = 1.5) {
    // Step 1: Compute k-NN density for each node
    // density = 1 / (distance to k-th nearest neighbor)
    const kActual = Math.min(k, n - 1);
    const density = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        // Get sorted distances (excluding self)
        const dists = [];
        for (let j = 0; j < n; j++) {
            if (j !== i) dists.push(distMatrix[i][j]);
        }
        dists.sort((a, b) => a - b);
        // Density = inverse of distance to k-th neighbor
        density[i] = 1.0 / (dists[kActual - 1] + 1e-8);
    }

    // Step 2: Sort nodes by density (descending)
    const order = Array.from({ length: n }, (_, i) => i);
    order.sort((a, b) => density[b] - density[a]);

    // Step 3: Place root first, then process remaining in density order
    const placed = new Set([rootId]);
    const depthMap = { [rootId]: 0 };
    const childCount = {};
    const nodes = [{ id: rootId, parentId: -1, depth: 0 }];
    const edges = [];

    // Entropy-based penalty state
    const taint = {};
    taint[rootId] = 1.0;
    const rootEntropy = entropyScores ? entropyScores[rootId] : 0;
    let maxEntropy = 0;
    if (entropyScores) {
        for (let i = 0; i < n; i++) maxEntropy = Math.max(maxEntropy, entropyScores[i]);
    }
    const entropyRange = maxEntropy - rootEntropy + 1e-10;

    // Remove rootId from order (it's already placed)
    const remaining = order.filter(id => id !== rootId);

    // Step 3b: When penalty is active, demote nodes broader than root.
    // Process specific/child topics first (density order), then parent/hub topics last.
    // This prevents hubs like "Physics" from grabbing the root's only slot when
    // they're the first node processed (density-first) and root is the only parent.
    let processOrder = remaining;
    if (entropyScores && upwardPenalty > 1.0) {
        const belowRoot = remaining.filter(id => entropyScores[id] <= rootEntropy);
        const aboveRoot = remaining.filter(id => entropyScores[id] > rootEntropy);
        processOrder = [...belowRoot, ...aboveRoot];
    }

    // Step 4: Greedy insertion — each node connects to nearest placed node
    for (const idx of processOrder) {
        let minDist = Infinity;
        let bestParent = rootId;

        for (const placedNode of placed) {
            // Skip nodes that are full (branching limit)
            if ((childCount[placedNode] || 0) >= maxBranching) continue;
            let d = distMatrix[idx][placedNode];
            // Apply asymmetric upward penalty with taint propagation
            // Penalize both: connecting TO a hub parent AND placing a hub node as child
            if (entropyScores && upwardPenalty > 1.0) {
                const parentExcess = Math.max(0, entropyScores[placedNode] - rootEntropy);
                const nodeExcess = Math.max(0, entropyScores[idx] - rootEntropy);
                const excess = Math.max(parentExcess, nodeExcess);
                const edgePenalty = 1.0 + (upwardPenalty - 1.0) * excess / entropyRange;
                d *= edgePenalty * (taint[placedNode] || 1.0);
            }
            if (d < minDist) {
                minDist = d;
                bestParent = placedNode;
            }
        }

        const parentDepth = depthMap[bestParent] || 0;
        const nodeDepth = parentDepth + 1;

        placed.add(idx);
        depthMap[idx] = nodeDepth;
        childCount[bestParent] = (childCount[bestParent] || 0) + 1;
        nodes.push({ id: idx, parentId: bestParent, depth: nodeDepth });
        edges.push({ source: bestParent, target: idx, weight: minDist });

        // Propagate taint: children of high-entropy parents inherit compounding penalty
        if (entropyScores && entropyScores[bestParent] > rootEntropy) {
            taint[idx] = (taint[bestParent] || 1.0) * cascadeFactor;
        } else {
            taint[idx] = taint[bestParent] || 1.0;
        }
    }

    return { nodes, edges, rootId };
}

/**
 * Filter a tree to maxDepth and maxBranching after building.
 */
function filterTree(tree, maxDepth, maxBranching) {
    const { nodes, edges, rootId } = tree;

    // Filter by depth
    const depthFiltered = new Set(
        nodes.filter(n => n.depth <= maxDepth).map(n => n.id)
    );

    // Filter by branching: for each parent, keep closest maxBranching children
    const childrenByParent = {};
    for (const e of edges) {
        if (!depthFiltered.has(e.source) || !depthFiltered.has(e.target)) continue;
        if (!childrenByParent[e.source]) childrenByParent[e.source] = [];
        childrenByParent[e.source].push(e);
    }

    const keptEdges = [];
    for (const [parent, children] of Object.entries(childrenByParent)) {
        children.sort((a, b) => a.weight - b.weight);
        keptEdges.push(...children.slice(0, maxBranching));
    }

    // Find reachable nodes from root through kept edges
    const reachable = new Set([rootId]);
    const adj = {};
    for (const e of keptEdges) {
        if (!adj[e.source]) adj[e.source] = [];
        adj[e.source].push(e.target);
    }
    const q = [rootId];
    let i = 0;
    while (i < q.length) {
        const node = q[i++];
        for (const child of (adj[node] || [])) {
            if (!reachable.has(child)) {
                reachable.add(child);
                q.push(child);
            }
        }
    }

    return {
        nodes: nodes.filter(n => reachable.has(n.id)),
        edges: keptEdges.filter(e => reachable.has(e.source) && reachable.has(e.target)),
        rootId,
    };
}
