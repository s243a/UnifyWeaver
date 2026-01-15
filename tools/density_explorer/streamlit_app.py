#!/usr/bin/env python3
"""
Density Manifold Explorer - Streamlit App

Interactive visualization of embedding density manifolds with tree overlays.

Usage:
    streamlit run tools/density_explorer/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools/density_explorer"))

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from shared import load_and_compute, compute_density_manifold, load_embeddings


# Page config
st.set_page_config(
    page_title="Density Manifold Explorer",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Density Manifold Explorer")


# Sidebar controls
st.sidebar.header("Settings")

# Data source
embeddings_path = st.sidebar.text_input(
    "Embeddings path",
    value=str(PROJECT_ROOT / "datasets/wikipedia_physics.npz")
)

top_k = st.sidebar.slider("Number of points", 50, 500, 300, 50)

# Density settings
st.sidebar.subheader("Density")
bandwidth = st.sidebar.slider("KDE Bandwidth", 0.01, 1.0, 0.1, 0.01)
grid_size = st.sidebar.slider("Grid resolution", 50, 200, 100, 10)

# Tree settings
st.sidebar.subheader("Tree Overlay")
show_tree = st.sidebar.checkbox("Show tree", value=True)
tree_type = st.sidebar.selectbox("Tree type", ["mst", "j-guided"], index=0)
max_depth = st.sidebar.slider("Max depth", 1, 20, 10, 1)

# Display settings
st.sidebar.subheader("Display")
show_points = st.sidebar.checkbox("Show points", value=True)
show_peaks = st.sidebar.checkbox("Show peaks", value=True)
n_peaks = st.sidebar.slider("Number of peaks", 1, 10, 5, 1)
show_contours = st.sidebar.checkbox("Show contours", value=True)


@st.cache_data
def load_data(path: str, k: int):
    """Load and cache embeddings."""
    embeddings, titles = load_embeddings(path)
    return embeddings[:k], titles[:k] if titles else None


@st.cache_data
def compute_manifold(embeddings_tuple, titles, bw, gs, tree, tree_t, peaks, n_p):
    """Compute and cache density manifold."""
    embeddings = np.array(embeddings_tuple)
    return compute_density_manifold(
        embeddings=embeddings,
        titles=titles,
        bandwidth=bw,
        grid_size=gs,
        include_tree=tree,
        tree_type=tree_t,
        include_peaks=peaks,
        n_peaks=n_p
    )


# Load data
try:
    embeddings, titles = load_data(embeddings_path, top_k)
    st.sidebar.success(f"Loaded {len(embeddings)} embeddings")
except Exception as e:
    st.error(f"Failed to load embeddings: {e}")
    st.stop()

# Compute manifold
with st.spinner("Computing density manifold..."):
    data = compute_manifold(
        tuple(map(tuple, embeddings)),  # Make hashable for cache
        titles,
        bandwidth,
        grid_size,
        show_tree,
        tree_type,
        show_peaks,
        n_peaks
    )

# Create figure
fig = go.Figure()

# Density heatmap
Z = np.array(data.density_grid.values)
fig.add_trace(go.Heatmap(
    z=Z,
    x=np.linspace(data.density_grid.x_min, data.density_grid.x_max, data.density_grid.grid_size),
    y=np.linspace(data.density_grid.y_min, data.density_grid.y_max, data.density_grid.grid_size),
    colorscale='Viridis',
    showscale=True,
    colorbar=dict(title="Density"),
    hoverinfo='skip',
    opacity=0.8
))

# Contours
if show_contours:
    fig.add_trace(go.Contour(
        z=Z,
        x=np.linspace(data.density_grid.x_min, data.density_grid.x_max, data.density_grid.grid_size),
        y=np.linspace(data.density_grid.y_min, data.density_grid.y_max, data.density_grid.grid_size),
        showscale=False,
        contours=dict(
            coloring='lines',
            showlabels=False
        ),
        line=dict(width=1, color='white'),
        opacity=0.5,
        hoverinfo='skip'
    ))

# Tree edges
if show_tree and data.tree:
    # Filter by depth
    for edge in data.tree.edges:
        source = next(n for n in data.tree.nodes if n.id == edge.source_id)
        target = next(n for n in data.tree.nodes if n.id == edge.target_id)

        if target.depth <= max_depth:
            fig.add_trace(go.Scatter(
                x=[source.x, target.x],
                y=[source.y, target.y],
                mode='lines',
                line=dict(color='cyan', width=1),
                opacity=0.6,
                hoverinfo='skip',
                showlegend=False
            ))

    # Root marker
    root = next(n for n in data.tree.nodes if n.id == data.tree.root_id)
    fig.add_trace(go.Scatter(
        x=[root.x],
        y=[root.y],
        mode='markers',
        marker=dict(size=15, color='cyan', symbol='square'),
        name=f"Root: {root.title[:30]}",
        hoverinfo='text',
        hovertext=root.title
    ))

# Data points
if show_points:
    fig.add_trace(go.Scatter(
        x=[p['x'] for p in data.points],
        y=[p['y'] for p in data.points],
        mode='markers',
        marker=dict(size=6, color='red', opacity=0.7),
        name=f"Points (n={data.n_points})",
        hoverinfo='text',
        hovertext=[p['title'] for p in data.points]
    ))

# Peaks
if show_peaks and data.peaks:
    fig.add_trace(go.Scatter(
        x=[p.x for p in data.peaks],
        y=[p.y for p in data.peaks],
        mode='markers+text',
        marker=dict(size=12, color='yellow', symbol='star'),
        text=[p.title[:20] for p in data.peaks],
        textposition='top center',
        textfont=dict(size=10, color='white'),
        name="Density Peaks",
        hoverinfo='text',
        hovertext=[f"{p.title} (density: {p.density:.2f})" for p in data.peaks]
    ))

# Layout
fig.update_layout(
    title=f"Density Manifold ({data.n_points} points)",
    xaxis_title=f"SVD 1 ({data.projection.variance_explained[0]:.1f}%)",
    yaxis_title=f"SVD 2 ({data.projection.variance_explained[1]:.1f}%)",
    xaxis=dict(scaleanchor="y", scaleratio=1),
    height=700,
    showlegend=True,
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)

# Display
st.plotly_chart(fig, use_container_width=True)

# Info panel
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Points", data.n_points)
    st.metric("Grid Size", f"{data.density_grid.grid_size}x{data.density_grid.grid_size}")

with col2:
    st.metric("Bandwidth", f"{data.density_grid.bandwidth:.3f}")
    st.metric("Tree Type", data.tree.tree_type if data.tree else "None")

with col3:
    st.metric("SVD Var 1", f"{data.projection.variance_explained[0]:.1f}%")
    st.metric("SVD Var 2", f"{data.projection.variance_explained[1]:.1f}%")

# Peak details
if show_peaks and data.peaks:
    st.subheader("Density Peaks")
    for i, peak in enumerate(data.peaks):
        st.write(f"**{i+1}. {peak.title}** - density: {peak.density:.3f}")

# Tree details
if show_tree and data.tree:
    with st.expander("Tree Structure"):
        st.write(f"**Root:** {next(n.title for n in data.tree.nodes if n.id == data.tree.root_id)}")
        st.write(f"**Edges:** {len(data.tree.edges)}")
        st.write(f"**Max depth:** {max(n.depth for n in data.tree.nodes)}")

# Export
with st.expander("Export Data"):
    st.download_button(
        "Download JSON",
        data.to_json(),
        file_name="density_manifold.json",
        mime="application/json"
    )
