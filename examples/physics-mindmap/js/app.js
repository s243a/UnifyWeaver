/**
 * Physics Mindmap Builder - Main app logic.
 * Loads pre-computed data, manages UI state, renders tree.
 */

let bundle = null;     // Pre-computed data
let currentTree = null; // Current tree (oriented, filtered)
let collapsedNodes = new Set();
let currentEntropyScores = null; // Hierarchy entropy per node

// --- Data Loading ---

let currentDistanceMetric = 'cosine';

async function loadBundle() {
    // Add cache-busting parameter to ensure fresh data
    const resp = await fetch('data/physics_bundle.json?v=' + Date.now());
    bundle = await resp.json();
    console.log(`Loaded ${bundle.metadata.n_points} articles`);

    // Initialize entropy scores from bundle
    currentEntropyScores = bundle.entropy_scores || null;

    // Default: use pre-computed cosine distances from the bundle
    // Only recompute if user switches to a different metric
    if (currentDistanceMetric !== 'cosine' && bundle.embeddings) {
        recomputeDistanceMatrix();
    }

    return bundle;
}

/**
 * Recompute distance matrix when metric changes.
 * For 'cosine', uses the pre-computed bundle distance_matrix.
 */
function recomputeDistanceMatrix() {
    if (!bundle) return;
    if (currentDistanceMetric === 'cosine') {
        bundle.computed_distance_matrix = null; // use bundle.distance_matrix
        currentEntropyScores = bundle.entropy_scores || null;
        console.log('Using pre-computed cosine distance matrix + entropy');
    } else if (bundle.embeddings) {
        bundle.computed_distance_matrix = computeDistances(bundle.embeddings, currentDistanceMetric);
        // Recompute entropy for new distance metric
        const distMat = bundle.computed_distance_matrix;
        currentEntropyScores = computeEntropyScores(distMat);
        console.log(`Computed ${currentDistanceMetric} distance matrix + entropy from embeddings`);
    }
}

/**
 * Normalize embeddings to unit length.
 */
function normalizeEmbeddings(embeddings) {
    const n = embeddings.length;
    const dim = embeddings[0].length;
    const normed = [];
    for (let i = 0; i < n; i++) {
        let norm = 0;
        for (let d = 0; d < dim; d++) norm += embeddings[i][d] * embeddings[i][d];
        norm = Math.sqrt(norm) || 1e-8;
        const vec = new Float64Array(dim);
        for (let d = 0; d < dim; d++) vec[d] = embeddings[i][d] / norm;
        normed.push(vec);
    }
    return normed;
}

/**
 * Compute pairwise distances using the specified metric.
 * @param {string} metric - 'cosine', 'chord', or 'angular'
 */
function computeDistances(embeddings, metric = 'cosine') {
    const n = embeddings.length;
    const dim = embeddings[0].length;
    const normed = normalizeEmbeddings(embeddings);

    const dist = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            // Compute cosine similarity (dot product of unit vectors)
            let dot = 0;
            for (let d = 0; d < dim; d++) dot += normed[i][d] * normed[j][d];
            // Clamp to [-1, 1] to handle floating point errors
            dot = Math.max(-1, Math.min(1, dot));

            let d;
            if (metric === 'angular') {
                // Angular distance: θ = arccos(cos θ)
                d = Math.acos(dot);
            } else if (metric === 'chord') {
                // Chord distance: 2 * sin(θ/2) = sqrt(2 * (1 - cos θ))
                d = Math.sqrt(2 * (1 - dot));
            } else {
                // Cosine distance: 1 - cos θ
                d = 1 - dot;
            }
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    return dist;
}

// --- UI Initialization ---

function initApp() {
    // Show shared controls bar
    document.getElementById('controls-bar').classList.remove('hidden');
    populateRootSelector();
    bindControls();
    switchTab('tree');
    buildAndRenderTree();
}

function populateRootSelector() {
    const select = document.getElementById('rootSelect');
    // Sort titles alphabetically for the dropdown
    const indexed = bundle.titles.map((t, i) => ({ title: t, id: i }));
    indexed.sort((a, b) => a.title.localeCompare(b.title));

    select.innerHTML = '';
    for (const { title, id } of indexed) {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = title;
        select.appendChild(opt);
    }

    // Default to "Physics" if available, else MST default root
    const physicsIdx = bundle.titles.indexOf('Physics');
    select.value = physicsIdx >= 0 ? physicsIdx : bundle.default_root;
}

function bindControls() {
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // Controls collapse toggle
    document.getElementById('controlsHeader').addEventListener('click', toggleControls);

    // Tree controls
    document.getElementById('rootSelect').addEventListener('change', () => {
        collapsedNodes.clear();
        buildAndRenderTree();
        updateVisualization();
    });
    document.getElementById('treeType').addEventListener('change', () => {
        collapsedNodes.clear();
        buildAndRenderTree();
        updateVisualization();
    });
    document.getElementById('maxDepth').addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        document.getElementById('maxDepthValue').textContent = val;
        buildAndRenderTree();
        updateVisualization();
    });
    document.getElementById('maxBranching').addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        document.getElementById('maxBranchingValue').textContent = val >= 20 ? '∞' : val;
        buildAndRenderTree();
        updateVisualization();
    });

    // Upward penalty
    const penaltySlider = document.getElementById('upwardPenalty');
    if (penaltySlider) {
        penaltySlider.addEventListener('input', (e) => {
            document.getElementById('upwardPenaltyValue').textContent =
                parseFloat(e.target.value).toFixed(1);
            buildAndRenderTree();
            updateVisualization();
        });
    }

    // Export
    document.getElementById('exportBtn').addEventListener('click', doExport);

    // Settings: distance metric
    document.getElementById('distanceMetric').addEventListener('change', (e) => {
        currentDistanceMetric = e.target.value;
        recomputeDistanceMatrix();
        collapsedNodes.clear();
        buildAndRenderTree();
        updateVisualization();
    });
}

// --- Controls Collapse ---

function toggleControls() {
    const bar = document.getElementById('controls-bar');
    bar.classList.toggle('collapsed');
}

function setControlsCollapsed(collapsed) {
    const bar = document.getElementById('controls-bar');
    bar.classList.toggle('collapsed', collapsed);
}

function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
    document.querySelectorAll('.screen').forEach(s => s.classList.toggle('active', s.id === `screen-${tabName}`));

    // On small screens, auto-collapse controls for viz tab to maximize plot space
    if (window.innerWidth < 768) {
        setControlsCollapsed(tabName === 'viz');
    }

    if (tabName === 'viz' && bundle) {
        loadVisualization();
    }
}

// --- Tree Building ---

function buildAndRenderTree() {
    const rootId = parseInt(document.getElementById('rootSelect').value);
    const treeType = document.getElementById('treeType').value;
    const n = bundle.metadata.n_points;
    const maxDepth = parseInt(document.getElementById('maxDepth').value);
    const maxBranchingVal = parseInt(document.getElementById('maxBranching').value);
    const maxBranching = maxBranchingVal >= 20 ? Infinity : maxBranchingVal;
    const upwardPenalty = parseFloat(document.getElementById('upwardPenalty')?.value || 2.0);

    // Use computed distance matrix (from embeddings with current metric) or fallback
    const distMatrix = bundle.computed_distance_matrix || bundle.distance_matrix;

    if (treeType === 'mst') {
        // MST: enforce branching during construction so orphaned nodes
        // get pushed deeper via distance-matrix fallback, then filter by depth
        const fullTree = rootMSTConstrained(bundle.mst_edges, distMatrix, rootId, n, 20, maxBranching,
            currentEntropyScores, upwardPenalty);
        currentTree = filterTree(fullTree, maxDepth, Infinity);
    } else {
        // J-guided: density-ordered greedy insertion
        const fullTree = buildJGuided(distMatrix, rootId, n, 10, 20, maxBranching,
            currentEntropyScores, upwardPenalty);
        currentTree = filterTree(fullTree, maxDepth, Infinity);
    }

    renderTree();
}

// --- Tree Rendering ---

function renderTree() {
    if (!currentTree) return;

    const container = document.getElementById('treeList');
    const stats = document.getElementById('treeStats');

    // Stats
    const shown = currentTree.nodes.length;
    stats.textContent = `${shown} nodes (root: ${bundle.titles[currentTree.rootId]})`;

    // Build child map
    const childMap = {};
    for (const e of currentTree.edges) {
        if (!childMap[e.source]) childMap[e.source] = [];
        childMap[e.source].push(e.target);
    }

    // Render tree as indented list
    container.innerHTML = '';
    renderNode(container, currentTree.rootId, 0, childMap);
}

function renderNode(container, nodeId, depth, childMap) {
    const children = childMap[nodeId] || [];
    const hasChildren = children.length > 0;
    const isCollapsed = collapsedNodes.has(nodeId);

    const row = document.createElement('div');
    row.className = `tree-node tree-depth-${Math.min(depth, 4)}`;
    row.dataset.id = nodeId;

    // Mark nodes with higher entropy than root (parent-like)
    if (currentEntropyScores && currentTree) {
        const rootEntropy = currentEntropyScores[currentTree.rootId];
        if (currentEntropyScores[nodeId] > rootEntropy * 1.1) {
            row.classList.add('entropy-parent');
        }
    }

    // Indent
    const indent = document.createElement('span');
    indent.className = 'tree-indent';
    indent.style.width = `${depth * 20}px`;
    row.appendChild(indent);

    // Toggle
    const toggle = document.createElement('span');
    toggle.className = `tree-toggle ${hasChildren ? (isCollapsed ? 'collapsed' : 'expanded') : 'leaf'}`;
    if (hasChildren) {
        toggle.addEventListener('click', (e) => {
            e.stopPropagation();
            if (collapsedNodes.has(nodeId)) {
                collapsedNodes.delete(nodeId);
            } else {
                collapsedNodes.add(nodeId);
            }
            renderTree();
        });
    }
    row.appendChild(toggle);

    // Label with Wikipedia link
    const label = document.createElement('span');
    label.className = 'tree-label';
    const link = document.createElement('a');
    link.href = titleToWikipediaUrl(bundle.titles[nodeId]);
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = bundle.titles[nodeId];
    label.appendChild(link);
    row.appendChild(label);

    // Child count badge
    if (hasChildren) {
        const badge = document.createElement('span');
        badge.className = 'tree-badge';
        badge.textContent = `(${children.length})`;
        row.appendChild(badge);
    }

    // Click row to select as new root
    row.addEventListener('click', () => {
        document.getElementById('rootSelect').value = nodeId;
        collapsedNodes.clear();
        buildAndRenderTree();
    });

    container.appendChild(row);

    // Render children if not collapsed
    if (hasChildren && !isCollapsed) {
        for (const childId of children) {
            renderNode(container, childId, depth + 1, childMap);
        }
    }
}

// --- Export ---

function doExport() {
    const format = document.getElementById('exportFormat').value;
    exportMindmap(format, currentTree, bundle.titles, bundle.coordinates_2d);
}

// --- Visualization (lazy-loaded) ---

let vizLoaded = false;

function loadVisualization() {
    if (vizLoaded) {
        updateVisualization();
        return;
    }

    const container = document.getElementById('viz-container');
    container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading visualization...</div>';

    // Load Plotly on demand
    const script = document.createElement('script');
    script.src = 'https://cdn.plot.ly/plotly-2.27.0.min.js';
    script.onload = () => {
        vizLoaded = true;
        renderVisualization();
    };
    document.head.appendChild(script);
}

function renderVisualization() {
    if (typeof renderDensityExplorer === 'function') {
        renderDensityExplorer(bundle, currentTree);
    }
}

function updateVisualization() {
    if (vizLoaded && typeof updateDensityOverlay === 'function') {
        updateDensityOverlay(currentTree);
    }
}

// --- Boot ---

document.addEventListener('DOMContentLoaded', async () => {
    document.getElementById('screen-tree').innerHTML =
        '<div class="loading"><div class="spinner"></div>Loading physics data...</div>';

    try {
        await loadBundle();
        // Restore the tree screen content
        document.getElementById('screen-tree').innerHTML = document.getElementById('tree-template').innerHTML;
        initApp();
    } catch (e) {
        document.getElementById('screen-tree').innerHTML =
            `<div class="loading" style="color: red;">Error loading data: ${e.message}</div>`;
        console.error(e);
    }
});
