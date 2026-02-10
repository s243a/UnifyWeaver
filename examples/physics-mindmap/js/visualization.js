/**
 * Visualization screen: Plotly-based density explorer (simplified).
 * Lazy-loaded when user switches to the Visualization tab.
 */

let plotDiv = null;

function renderDensityExplorer(bundle, tree) {
    plotDiv = document.getElementById('viz-container');
    plotDiv.innerHTML = '';

    const coords = bundle.coordinates_2d;
    const grid = bundle.density_grid;
    const titles = bundle.titles;

    // Density heatmap
    const heatmap = {
        z: grid.values,
        x: Array.from({ length: grid.grid_size }, (_, i) =>
            grid.x_min + (i / (grid.grid_size - 1)) * (grid.x_max - grid.x_min)),
        y: Array.from({ length: grid.grid_size }, (_, i) =>
            grid.y_min + (i / (grid.grid_size - 1)) * (grid.y_max - grid.y_min)),
        type: 'heatmap',
        colorscale: 'YlOrRd',
        showscale: false,
        opacity: 0.6,
        hoverinfo: 'skip',
    };

    // Points
    const points = {
        x: coords.map(c => c[0]),
        y: coords.map(c => c[1]),
        mode: 'markers',
        type: 'scatter',
        marker: { size: 6, color: '#1e293b', opacity: 0.7 },
        text: titles,
        hoverinfo: 'text',
        name: 'Articles',
    };

    // Tree edges
    const edgeX = [], edgeY = [];
    for (const e of tree.edges) {
        edgeX.push(coords[e.source][0], coords[e.target][0], null);
        edgeY.push(coords[e.source][1], coords[e.target][1], null);
    }

    const edges = {
        x: edgeX,
        y: edgeY,
        mode: 'lines',
        type: 'scatter',
        line: { color: '#06b6d4', width: 1 },
        hoverinfo: 'skip',
        name: 'Tree',
    };

    // Root marker
    const rootMarker = {
        x: [coords[tree.rootId][0]],
        y: [coords[tree.rootId][1]],
        mode: 'markers+text',
        type: 'scatter',
        marker: { size: 12, color: '#dc2626', symbol: 'star' },
        text: [titles[tree.rootId]],
        textposition: 'top center',
        textfont: { size: 11, color: '#dc2626' },
        hoverinfo: 'text',
        name: 'Root',
    };

    // Peaks
    const peakMarkers = {
        x: bundle.peaks.map(p => p.x),
        y: bundle.peaks.map(p => p.y),
        mode: 'markers+text',
        type: 'scatter',
        marker: { size: 8, color: '#7c3aed', symbol: 'diamond' },
        text: bundle.peaks.map(p => p.title),
        textposition: 'bottom center',
        textfont: { size: 10, color: '#7c3aed' },
        hoverinfo: 'text',
        name: 'Peaks',
    };

    const layout = {
        title: { text: 'Wikipedia Physics — Density Manifold', font: { size: 14 } },
        xaxis: { title: 'SVD 1', zeroline: false },
        yaxis: { title: 'SVD 2', zeroline: false, scaleanchor: 'x' },
        showlegend: true,
        legend: { x: 0, y: 1, font: { size: 11 } },
        margin: { l: 40, r: 10, t: 40, b: 40 },
        hovermode: 'closest',
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToInclude: [
            'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d',
            'resetScale2d', 'toImage'
        ],
        modeBarButtonsToRemove: [
            'select2d', 'lasso2d', 'autoScale2d'
        ],
        displaylogo: false,
        scrollZoom: true,
    };

    Plotly.newPlot(plotDiv, [heatmap, edges, points, rootMarker, peakMarkers], layout, config);
}

function updateDensityOverlay(tree) {
    if (!plotDiv || !bundle) return;

    const coords = bundle.coordinates_2d;

    // Update tree edges (trace index 1)
    const edgeX = [], edgeY = [];
    for (const e of tree.edges) {
        edgeX.push(coords[e.source][0], coords[e.target][0], null);
        edgeY.push(coords[e.source][1], coords[e.target][1], null);
    }

    // Update root marker (trace index 3)
    Plotly.restyle(plotDiv, {
        x: [edgeX],
        y: [edgeY],
    }, [1]);

    Plotly.restyle(plotDiv, {
        x: [[coords[tree.rootId][0]]],
        y: [[coords[tree.rootId][1]]],
        text: [[bundle.titles[tree.rootId]]],
    }, [3]);
}
