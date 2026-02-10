#!/usr/bin/env python3
"""
Generate physics_bundle.json for the Physics Mindmap Builder.

Downloads Wikipedia articles from HuggingFace, filters to physics topics,
re-embeds with nomic-embed-text-v1.5, and computes all data needed for the app.

Usage:
    cd examples/physics-mindmap
    python generate_data.py
    python generate_data.py --top-k 200  # fewer articles
    python generate_data.py --max-rows 50000  # faster download

Requirements:
    pip install datasets sentence-transformers numpy scipy
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not installed")
    print("Run: pip install datasets")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers library not installed")
    print("Run: pip install sentence-transformers")
    sys.exit(1)

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.stats import gaussian_kde


# --- Physics keyword lists for filtering ---

PHYSICS_EXACT_TERMS = {
    "physics", "thermodynamics", "entropy", "enthalpy", "quantum",
    "mechanics", "electromagnetism", "magnetism", "relativity",
    "electron", "photon", "proton", "neutron", "quark", "boson",
    "fermion", "lepton", "hadron", "meson", "baryon", "neutrino",
    "atom", "molecule", "nucleus", "ion", "plasma", "isotope",
    "wavelength", "frequency", "amplitude", "oscillation", "resonance",
    "gravity", "gravitation", "mass", "inertia", "torque", "momentum",
    "acceleration", "velocity", "kinetic", "potential", "joule",
    "watt", "volt", "ampere", "ohm", "coulomb", "farad", "henry",
    "tesla", "weber", "hertz", "kelvin", "celsius", "fahrenheit",
    "thermometer", "calorimeter", "spectrometer", "interferometer",
    "laser", "maser", "superconductor", "semiconductor", "insulator",
    "conductor", "dielectric", "capacitor", "inductor", "resistor",
    "circuit", "transistor", "diode", "led", "photodiode",
    "refraction", "diffraction", "interference", "polarization",
    "absorption", "emission", "radiation", "blackbody", "spectrum",
    "doppler", "redshift", "blueshift", "cosmic", "cosmology",
    "astrophysics", "nuclear", "fission", "fusion", "radioactive",
    "half-life", "decay", "isotope", "neutron", "reactor",
}

PHYSICS_COMPOUND_TERMS = [
    "quantum mechanics", "statistical mechanics", "classical mechanics",
    "fluid dynamics", "thermodynamic", "electromagnetic",
    "wave function", "schrodinger equation", "heisenberg uncertainty",
    "newton's law", "maxwell's equations", "boltzmann", "planck",
    "einstein", "bohr model", "pauli exclusion", "fermi",
    "bose-einstein", "lorentz", "minkowski", "lagrangian", "hamiltonian",
    "navier-stokes", "reynolds number", "mach number", "bernoulli",
    "carnot", "otto cycle", "diesel cycle", "rankine cycle",
    "heat engine", "heat pump", "refrigeration", "entropy production",
    "ideal gas", "van der waals", "gibbs free energy", "helmholtz",
    "partition function", "phase transition", "critical point",
    "superconductivity", "superfluidity", "bose condensate",
    "photoelectric", "compton scattering", "pair production",
    "special relativity", "general relativity", "spacetime", "geodesic",
    "black hole", "event horizon", "hawking radiation", "gravitational",
    "standard model", "higgs boson", "symmetry breaking", "gauge theory",
    "string theory", "quantum field", "renormalization",
]


# --- Step 1: Download and filter Wikipedia articles ---

def parse_supabase_body(body):
    """Parse title and content from Supabase body format ('Title: X Content: Y')."""
    if body.startswith("Title: "):
        parts = body.split(" Content: ", 1)
        if len(parts) == 2:
            return parts[0][7:], parts[1]
    return "", body


def load_wikipedia_articles(max_rows=100000):
    """Load Wikipedia articles from HuggingFace (Supabase/wikipedia-en-embeddings)."""
    print(f"Downloading Wikipedia articles from HuggingFace (max {max_rows} rows)...")
    dataset = load_dataset(
        "Supabase/wikipedia-en-embeddings",
        split="train",
        streaming=True,
    )

    titles, texts, embeddings = [], [], []
    for i, row in enumerate(dataset):
        if i >= max_rows:
            break
        body = row.get("body", "")
        title, content = parse_supabase_body(body)
        titles.append(title)
        texts.append(content)
        embeddings.append(row.get("embedding", []))
        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1} rows...")

    print(f"Loaded {len(titles)} articles")
    return titles, texts, np.array(embeddings, dtype=np.float32)


def filter_physics_articles(titles, texts, embeddings, top_k=300):
    """Filter to physics-related articles using keywords + semantic similarity."""
    print("\nFiltering physics articles...")

    # Keyword filtering
    word_pattern = re.compile(
        r'\b(' + '|'.join(re.escape(t) for t in PHYSICS_EXACT_TERMS) + r')\b',
        re.IGNORECASE,
    )
    compound_lower = [t.lower() for t in PHYSICS_COMPOUND_TERMS]

    scores = {}
    for i, title in enumerate(titles):
        score = 0.0
        title_lower = title.lower()

        title_matches = word_pattern.findall(title_lower)
        if title_matches:
            score += len(set(title_matches)) * 2.0

        for ct in compound_lower:
            if ct in title_lower:
                score += 3.0

        if texts[i]:
            text_lower = texts[i].lower()[:1000]
            text_matches = word_pattern.findall(text_lower)
            if text_matches:
                score += len(set(text_matches)) * 0.5
            for ct in compound_lower:
                if ct in text_lower:
                    score += 1.0

        if score > 0:
            scores[i] = score

    print(f"  Keyword matches: {len(scores)}")

    # Semantic similarity using top keyword matches as seeds
    if scores:
        top_seeds = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:50]
        seed_indices = [idx for idx, _ in top_seeds]
        seed_embs = embeddings[seed_indices]

        # Normalize
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_normed = embeddings / (emb_norms + 1e-8)
        seed_norms = np.linalg.norm(seed_embs, axis=1, keepdims=True)
        seed_normed = seed_embs / (seed_norms + 1e-8)

        centroid = seed_normed.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        similarities = emb_normed @ centroid

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        for idx in top_indices:
            scores[int(idx)] = scores.get(int(idx), 0) + float(similarities[idx]) * 5.0

    # Select top-k by combined score
    sorted_by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    selected = [idx for idx, _ in sorted_by_score]

    print(f"  Selected: {len(selected)} articles")
    print(f"  Sample: {', '.join(titles[i] for i in selected[:10])}")

    return selected


# --- Step 2: Re-embed with nomic ---

def embed_with_nomic(texts, model_name="nomic-ai/nomic-embed-text-v1.5", batch_size=8):
    """Embed texts with nomic-embed-text-v1.5."""
    print(f"\nLoading {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    prefixed = [f"search_document: {t}" for t in texts]
    print(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(prefixed, show_progress_bar=True, batch_size=batch_size)

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = (embeddings / norms).astype(np.float32)

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


# --- Step 3: Compute distances and derived data ---

def cosine_distance_matrix(embeddings):
    """Compute pairwise cosine distances."""
    # Already L2-normalized, so dot product = cosine similarity
    sim = embeddings @ embeddings.T
    np.clip(sim, -1, 1, out=sim)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0)
    return dist


def compute_breadth_scores(dist_matrix, k=30):
    """Compute hierarchy breadth scores (inverse mean k-NN distance, normalized to [0,1])."""
    n = len(dist_matrix)
    scores = np.zeros(n)
    k_actual = min(k, n - 1)
    for i in range(n):
        dists = np.sort(dist_matrix[i])
        knn = dists[1:k_actual + 1]
        scores[i] = 1.0 / (knn.mean() + 1e-10)
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)
    return scores


def compute_mst_edges(dist_matrix):
    """Compute MST and return unrooted edge list."""
    sparse = csr_matrix(dist_matrix)
    mst = minimum_spanning_tree(sparse)
    mst_coo = mst.tocoo()

    edges = []
    for i, j, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        edges.append({"source": int(i), "target": int(j), "weight": float(w)})

    from collections import Counter
    degree = Counter()
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    default_root = max(degree, key=degree.get) if degree else 0

    return edges, default_root


def project_to_2d(embeddings):
    """Project embeddings to 2D using SVD."""
    centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    points_2d = centered @ Vt[:2].T
    total_var = np.sum(S ** 2)
    var_explained = [(S[i] ** 2 / total_var) * 100 for i in range(2)]
    return points_2d, S, var_explained


def compute_density_grid(points_2d, grid_size=100):
    """Compute KDE density grid."""
    x, y = points_2d[:, 0], points_2d[:, 1]
    pad = 0.1
    x_min, x_max = x.min() - pad, x.max() + pad
    y_min, y_max = y.min() - pad, y.max() + pad

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kde = gaussian_kde(points_2d.T)
    density = kde(positions).reshape(grid_size, grid_size)

    n = len(points_2d)
    bandwidth = n ** (-1.0 / 6)

    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "grid_size": grid_size,
        "bandwidth": float(bandwidth),
        "values": np.round(density, 6).tolist(),
    }


def find_peaks(density_grid, points_2d, titles, n_peaks=5):
    """Find density peaks and label with nearest article."""
    from scipy.ndimage import maximum_filter

    values = np.array(density_grid["values"])
    local_max = values == maximum_filter(values, size=20)
    peak_coords = np.argwhere(local_max)

    if len(peak_coords) == 0:
        return []

    peak_values = [values[r, c] for r, c in peak_coords]
    top_idx = np.argsort(peak_values)[-n_peaks:]

    gs = density_grid["grid_size"]
    peaks = []
    for idx in top_idx:
        r, c = peak_coords[idx]
        px = density_grid["x_min"] + (c / (gs - 1)) * (density_grid["x_max"] - density_grid["x_min"])
        py = density_grid["y_min"] + (r / (gs - 1)) * (density_grid["y_max"] - density_grid["y_min"])
        dists = np.sqrt((points_2d[:, 0] - px) ** 2 + (points_2d[:, 1] - py) ** 2)
        nearest = int(np.argmin(dists))
        peaks.append({
            "x": round(float(px), 6),
            "y": round(float(py), 6),
            "density": round(float(peak_values[idx]), 6),
            "nearest_id": nearest,
            "title": titles[nearest],
        })

    return peaks


# --- Main pipeline ---

def main():
    parser = argparse.ArgumentParser(description="Generate physics_bundle.json")
    parser.add_argument("--max-rows", type=int, default=100000,
                        help="Max Wikipedia rows to download (default: 100000)")
    parser.add_argument("--top-k", type=int, default=300,
                        help="Number of physics articles to select (default: 300)")
    parser.add_argument("--output", type=str, default="data/physics_bundle.json",
                        help="Output path (default: data/physics_bundle.json)")
    args = parser.parse_args()

    print("=== Physics Mindmap Builder: Data Generation ===\n")

    # 1. Download and filter
    titles, texts, minilm_embeddings = load_wikipedia_articles(max_rows=args.max_rows)
    selected = filter_physics_articles(titles, texts, minilm_embeddings, top_k=args.top_k)

    selected_titles = [titles[i] for i in selected]

    # 2. Re-embed with nomic
    embeddings = embed_with_nomic(selected_titles)

    # 3. Compute distances
    print("\nComputing cosine distance matrix...")
    dist_matrix = cosine_distance_matrix(embeddings)
    print(f"Distance range: [{dist_matrix.min():.3f}, {dist_matrix.max():.3f}]")

    # 4. Breadth scores
    print("\nComputing breadth scores...")
    breadth_scores = compute_breadth_scores(dist_matrix)
    top5 = np.argsort(breadth_scores)[-5:][::-1]
    print(f"Broadest: {', '.join(selected_titles[i] + f' ({breadth_scores[i]:.3f})' for i in top5)}")

    # 5. SVD to 2D
    print("\nProjecting to 2D via SVD...")
    points_2d, singular_values, var_explained = project_to_2d(embeddings)
    print(f"Variance explained: {var_explained[0]:.1f}% + {var_explained[1]:.1f}%")

    # 6. MST
    print("\nBuilding MST...")
    mst_edges, default_root = compute_mst_edges(dist_matrix)
    print(f"MST: {len(mst_edges)} edges, default root: {default_root} ({selected_titles[default_root]})")

    # 7. Density grid
    print("\nComputing density grid...")
    density_grid = compute_density_grid(points_2d)

    # 8. Peaks
    print("\nFinding density peaks...")
    peaks = find_peaks(density_grid, points_2d, selected_titles)
    print(f"Found {len(peaks)} peaks")

    # 9. Bundle
    bundle = {
        "titles": selected_titles,
        "embeddings": np.round(embeddings, 4).tolist(),
        "distance_matrix": np.round(dist_matrix, 4).tolist(),
        "coordinates_2d": np.round(points_2d, 6).tolist(),
        "mst_edges": mst_edges,
        "default_root": default_root,
        "density_grid": density_grid,
        "peaks": peaks,
        "entropy_scores": np.round(breadth_scores, 6).tolist(),
        "metadata": {
            "n_points": len(selected_titles),
            "distance_metric": "cosine",
            "embedding_model": "nomic-embed-text-v1.5",
            "dataset": "Supabase/wikipedia-en-embeddings",
            "entropy_method": "knn_density_breadth",
            "entropy_k": 30,
            "embedding_dim": int(embeddings.shape[1]),
            "variance_explained": [round(float(v), 4) for v in var_explained],
            "singular_values": [round(float(s), 4) for s in singular_values[:2]],
        },
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(bundle, f, separators=(",", ":"))

    size_kb = output_path.stat().st_size / 1024
    print(f"\n=== Saved {output_path} ({size_kb:.0f} KB) ===")
    print(f"\nTo serve the app:")
    print(f"  python -m http.server 8080")
    print(f"  Open http://localhost:8080")


if __name__ == "__main__":
    main()
