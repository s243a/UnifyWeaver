#!/usr/bin/env python3
"""
Plot organizational depth scatter: Wikipedia Physics -> Pearltrees Hierarchy.

Items positioned by organizational distance (n=5 Manhattan).
The X axis correlates with hierarchy depth (r=0.85).

Data: tools/density_explorer/data/wikipedia_physics.json
Output: Interactive HTML + static PNG

Usage:
    python examples/plot_organizational_depth.py
"""
import json
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
from pathlib import Path

# Resolve paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'tools' / 'density_explorer' / 'data' / 'wikipedia_physics.json'
OUTPUT_DIR = PROJECT_ROOT / 'tools' / 'density_explorer' / 'data'

# Load the data
with open(DATA_PATH) as f:
    data = json.load(f)

points = data['points']
print(f"Loaded {len(points)} points")

# Create color map by depth
depth_colors = {
    1: '#1f77b4',   # Blue - root level (science)
    3: '#2ca02c',   # Green - Physics level
    4: '#ff7f0e',   # Orange - Thermodynamics, mechanics
    5: '#d62728',   # Red
    6: '#9467bd',   # Purple - history
    7: '#8c564b',   # Brown - dynamics, metaphysics
    8: '#e377c2',   # Pink
    11: '#7f7f7f',  # Gray
    12: '#bcbd22',  # Olive
}

# Group points by depth
depth_groups = defaultdict(list)
for p in points:
    depth_groups[p['path_length']].append(p)

# Create figure
fig = go.Figure()

# Add traces for each depth level
for depth in sorted(depth_groups.keys()):
    group = depth_groups[depth]

    fig.add_trace(go.Scatter(
        x=[p['x'] for p in group],
        y=[p['y'] for p in group],
        mode='markers',
        name=f'Depth {depth} ({len(group)})',
        marker=dict(
            size=8,
            color=depth_colors.get(depth, '#333333'),
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=[f"{p['label']}<br>Folder: {p['folder']}<br>Depth: {p['path_length']}"
              for p in group],
        hoverinfo='text'
    ))

# Add folder labels at cluster centers
folder_centers = defaultdict(lambda: {'x': [], 'y': [], 'depth': 0})
for p in points:
    folder_centers[p['folder']]['x'].append(p['x'])
    folder_centers[p['folder']]['y'].append(p['y'])
    folder_centers[p['folder']]['depth'] = p['path_length']

# Add annotation for major folders (>= 20 items)
major_folders = [(f, d) for f, d in folder_centers.items() if len(d['x']) >= 20]
for folder, d in major_folders[:15]:
    cx, cy = np.mean(d['x']), np.mean(d['y'])
    fig.add_annotation(
        x=cx, y=cy,
        text=folder[:20],
        showarrow=False,
        font=dict(size=10, color='black'),
        bgcolor='rgba(255,255,255,0.7)',
        borderpad=2
    )

# Layout
fig.update_layout(
    title=('Wikipedia Physics \u2192 Pearltrees Hierarchy'
           '<br><sub>Items positioned by organizational distance (n=5 Manhattan)</sub>'),
    xaxis_title='Projection X (correlates with depth, r=0.85)',
    yaxis_title='Projection Y',
    width=1200,
    height=800,
    legend=dict(
        title='Organizational Depth',
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    hovermode='closest'
)

# Save as interactive HTML
html_path = OUTPUT_DIR / 'wikipedia_physics_interactive.html'
fig.write_html(str(html_path))
print(f"\nSaved interactive visualization to {html_path}")

# Save static image (requires kaleido: pip install kaleido)
try:
    png_path = OUTPUT_DIR / 'wikipedia_physics_scatter.png'
    fig.write_image(str(png_path), scale=2)
    print(f"Saved static image to {png_path}")
except Exception as e:
    print(f"Could not save static image ({e}). Install kaleido: pip install kaleido")

print(f"\nTo view, open the HTML file in a browser:")
print(f"  file://{html_path.absolute()}")
