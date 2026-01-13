#!/usr/bin/env python3
"""
Fetch physics-related Wikipedia articles from HuggingFace datasets.

This script:
1. Loads pre-computed Wikipedia embeddings from Cohere dataset
2. Uses semantic similarity to filter physics-related articles
3. Outputs filtered subset for testing attachment criterion

Usage:
    python3 scripts/fetch_wikipedia_physics.py --top-k 500 --output datasets/wikipedia_physics.npz
    python3 scripts/fetch_wikipedia_physics.py --categories thermodynamics quantum --top-k 200

Requirements:
    pip install datasets numpy
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not installed")
    print("Run: pip install datasets")
    sys.exit(1)


# Physics-related seed terms for semantic filtering
PHYSICS_SEEDS = [
    "physics thermodynamics statistical mechanics",
    "quantum mechanics wave function Schrödinger",
    "classical mechanics Newton force energy",
    "fluid dynamics Navier-Stokes turbulence",
    "electromagnetism Maxwell equations field",
    "entropy temperature heat transfer",
    "particle physics standard model quarks",
    "relativity spacetime Einstein gravity",
    "condensed matter solid state physics",
    "optics light wave interference",
]

# High-confidence physics terms (standalone words, not substrings)
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

# Compound terms that strongly indicate physics content
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

# Keywords for title-based filtering (kept for compatibility)
PHYSICS_KEYWORDS = PHYSICS_EXACT_TERMS


def filter_by_keywords(
    titles: List[str],
    texts: Optional[List[str]] = None,
    exact_terms: Set[str] = PHYSICS_EXACT_TERMS,
    compound_terms: List[str] = PHYSICS_COMPOUND_TERMS,
    check_text: bool = True
) -> List[Tuple[int, float]]:
    """Filter articles by keyword matches in titles and text.

    Uses word boundary matching for exact terms and substring matching
    for compound terms. Returns (index, score) tuples.

    Args:
        titles: List of article titles
        texts: Optional list of article texts (for content matching)
        exact_terms: Set of single-word physics terms
        compound_terms: List of multi-word physics phrases
        check_text: Whether to also check text content

    Returns:
        List of (index, match_score) tuples
    """
    import re

    matches = []
    exact_lower = {t.lower() for t in exact_terms}
    compound_lower = [t.lower() for t in compound_terms]

    # Build regex patterns for word boundary matching
    word_pattern = re.compile(r'\b(' + '|'.join(re.escape(t) for t in exact_lower) + r')\b', re.IGNORECASE)

    for i, title in enumerate(titles):
        score = 0.0
        title_lower = title.lower()

        # Check title for exact terms (word boundaries)
        title_matches = word_pattern.findall(title_lower)
        if title_matches:
            score += len(set(title_matches)) * 2.0  # Title matches worth more

        # Check title for compound terms
        for ct in compound_lower:
            if ct in title_lower:
                score += 3.0  # Compound terms very strong signal

        # Check text content if available
        if check_text and texts and texts[i]:
            text_lower = texts[i].lower()[:1000]  # First 1000 chars

            text_matches = word_pattern.findall(text_lower)
            if text_matches:
                score += len(set(text_matches)) * 0.5

            for ct in compound_lower:
                if ct in text_lower:
                    score += 1.0

        if score > 0:
            matches.append((i, score))

    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def filter_by_semantic_similarity(
    embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    threshold: float = 0.5,
    top_k: Optional[int] = None
) -> List[Tuple[int, float]]:
    """Filter by cosine similarity to query embeddings.

    Args:
        embeddings: Article embeddings (N x D)
        query_embeddings: Query/seed embeddings (M x D)
        threshold: Minimum similarity threshold
        top_k: If set, return top-k most similar

    Returns:
        List of (index, similarity) tuples
    """
    # Normalize embeddings
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (emb_norms + 1e-8)

    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_norm = query_embeddings / (query_norms + 1e-8)

    # Compute mean query embedding
    query_centroid = query_norm.mean(axis=0)
    query_centroid = query_centroid / (np.linalg.norm(query_centroid) + 1e-8)

    # Compute similarities
    similarities = embeddings_norm @ query_centroid

    if top_k:
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(int(i), float(similarities[i])) for i in top_indices]
    else:
        matches = np.where(similarities >= threshold)[0]
        return [(int(i), float(similarities[i])) for i in matches]


def parse_supabase_body(body: str) -> Tuple[str, str]:
    """Parse title and content from Supabase body format.

    Format: 'Title: X Content: Y'
    """
    if body.startswith("Title: "):
        parts = body.split(" Content: ", 1)
        if len(parts) == 2:
            title = parts[0][7:]  # Remove 'Title: '
            content = parts[1]
            return title, content
    return "", body


def load_wikipedia_embeddings(
    max_rows: int = 100000,
    streaming: bool = True,
    dataset_name: str = "Supabase/wikipedia-en-embeddings"
) -> Tuple[List[str], List[str], np.ndarray]:
    """Load Wikipedia embeddings dataset.

    Uses Supabase/wikipedia-en-embeddings which has:
    - 224K articles from Simple English Wikipedia
    - 384-dim embeddings (MiniLM)
    - Body field contains 'Title: X Content: Y'

    Args:
        max_rows: Maximum rows to load
        streaming: Use streaming mode (memory efficient)
        dataset_name: HuggingFace dataset name

    Returns:
        titles, texts, embeddings
    """
    print(f"Loading {dataset_name} (max {max_rows} rows)...")

    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=streaming
    )

    titles = []
    texts = []
    embeddings = []

    for i, row in enumerate(dataset):
        if i >= max_rows:
            break

        # Parse body field
        body = row.get("body", "")
        title, content = parse_supabase_body(body)

        titles.append(title)
        texts.append(content)
        embeddings.append(row.get("embedding", []))

        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1} rows...")

    print(f"Loaded {len(titles)} articles")
    embeddings = np.array(embeddings, dtype=np.float32)

    return titles, texts, embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Fetch physics-related Wikipedia articles"
    )
    parser.add_argument(
        "--max-rows", type=int, default=100000,
        help="Maximum rows to load from dataset (default: 100000)"
    )
    parser.add_argument(
        "--top-k", type=int, default=500,
        help="Number of physics articles to extract"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4,
        help="Similarity threshold for semantic filtering"
    )
    parser.add_argument(
        "--method", choices=["keyword", "semantic", "both"], default="both",
        help="Filtering method"
    )
    parser.add_argument(
        "--output", type=str, default="datasets/wikipedia_physics.npz",
        help="Output file path"
    )
    parser.add_argument(
        "--output-jsonl", type=str,
        help="Optional JSONL output for metadata"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be filtered without saving"
    )

    args = parser.parse_args()

    # Load dataset
    titles, texts, embeddings = load_wikipedia_embeddings(max_rows=args.max_rows)

    selected_indices = {}  # idx -> score

    # Keyword filtering
    if args.method in ["keyword", "both"]:
        print("\nFiltering by keywords...")
        keyword_matches = filter_by_keywords(titles, texts)
        print(f"  Found {len(keyword_matches)} keyword matches")
        for idx, score in keyword_matches:
            selected_indices[idx] = selected_indices.get(idx, 0) + score

    # Semantic filtering
    if args.method in ["semantic", "both"]:
        print("\nFiltering by semantic similarity...")

        # Use keyword matches as seeds for semantic search
        if selected_indices:
            # Take top keyword matches as seeds
            top_seeds = sorted(selected_indices.items(), key=lambda x: x[1], reverse=True)[:50]
            seed_indices = [idx for idx, _ in top_seeds]
            seed_embeddings = embeddings[seed_indices]
        else:
            # Use physics seed terms (would need to embed them)
            print("  Note: Semantic filtering without keyword seeds not implemented")
            print("  Using keyword matches as semantic seeds")
            keyword_matches = filter_by_keywords(titles, texts)[:50]
            if not keyword_matches:
                print("  Error: No seed articles found")
                return
            seed_indices = [idx for idx, _ in keyword_matches]
            seed_embeddings = embeddings[seed_indices]

        semantic_matches = filter_by_semantic_similarity(
            embeddings, seed_embeddings,
            threshold=args.threshold,
            top_k=args.top_k
        )
        print(f"  Found {len(semantic_matches)} semantic matches")

        for idx, sim in semantic_matches:
            # Add semantic score (normalized to similar range as keyword scores)
            selected_indices[idx] = selected_indices.get(idx, 0) + sim * 5.0

    # Convert to sorted list by score
    sorted_by_score = sorted(selected_indices.items(), key=lambda x: x[1], reverse=True)

    if args.top_k and len(sorted_by_score) > args.top_k:
        sorted_by_score = sorted_by_score[:args.top_k]

    selected_indices = [idx for idx, _ in sorted_by_score]

    print(f"\nTotal selected: {len(selected_indices)} articles")

    # Show sample
    print("\nSample titles:")
    for i in selected_indices[:20]:
        print(f"  - {titles[i][:60]}")
    if len(selected_indices) > 20:
        print(f"  ... and {len(selected_indices) - 20} more")

    if args.dry_run:
        print("\n[Dry run - not saving]")
        return

    # Prepare output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_titles = [titles[i] for i in selected_indices]
    selected_texts = [texts[i] for i in selected_indices]
    selected_embeddings = embeddings[selected_indices]

    # Save NPZ
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        titles=selected_titles,
        texts=selected_texts,
        embeddings=selected_embeddings,
        source="cohere/wikipedia-22-12-en-embeddings",
        filter_method=args.method
    )

    # Optional JSONL output
    if args.output_jsonl:
        jsonl_path = Path(args.output_jsonl)
        print(f"Saving metadata to {jsonl_path}...")
        with open(jsonl_path, 'w') as f:
            for i, idx in enumerate(selected_indices):
                record = {
                    "idx": i,
                    "original_idx": idx,
                    "title": titles[idx],
                    "text_preview": texts[idx][:200] if texts[idx] else ""
                }
                f.write(json.dumps(record) + "\n")

    print("Done!")
    print(f"\nTo test attachment:")
    print(f"  python3 scripts/mindmap/test_foreign_attachment.py \\")
    print(f"    --foreign-embeddings {output_path} --top-k 50")


if __name__ == "__main__":
    main()
